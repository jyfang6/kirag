import torch
import numpy as np 
import torch.nn as nn 
from tqdm import trange
from copy import deepcopy
from torch import Tensor
import torch.distributed as dist
from typing import Optional, Union, Dict, List

from dataset.corpus import Corpus
from retriever.index import Indexer
from dataset.collators import RetrieverCollator
from retriever.encoders import E5Encoder, BGEEncoder
from utils.utils import (
    to_device, 
    get_global_labels_for_inbatchtraining,
    get_global_embeddings_for_inbatchtraining
)

RETRIEVER_MAP = {
    "E5Retriever": E5Encoder,
    "BGERetriever": BGEEncoder, 
}

def load_retriever(retriever_name, model_name_or_path, **kwargs):
    if retriever_name not in RETRIEVER_MAP:
        raise KeyError(f"{retriever_name} is not implemented! Current available retrievers: {list(RETRIEVER_MAP.keys())}")
    print(f"loading {retriever_name} model from {model_name_or_path} ...")
    return RETRIEVER_MAP[retriever_name].from_pretrained(model_name_or_path, **kwargs)


class BaseRetriever(nn.Module):

    def __init__(self, retriever_name, model_name_or_path, retriever_kwargs={}, 
                 temperature=1.0, norm_query=False, norm_doc=False, local_rank=-1, **kwargs):
        
        super().__init__()
        self.encoder = load_retriever(retriever_name, model_name_or_path, **retriever_kwargs, **kwargs)
        self.retriever_name = retriever_name
        self.model_name_or_path = model_name_or_path
        self.retriever_kwargs = retriever_kwargs
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.local_rank = local_rank
        self.world_size = dist.get_world_size() if self.local_rank >= 0 else 1 
        self.temperature = temperature
        self.kwargs = kwargs

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    @property
    def device(self):
        for n, p in self.named_parameters():
            return p.device 
    
    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size
    
    def compute_logits(self, query_embeddings, doc_embeddings, **kwargs):
        if len(query_embeddings.shape) ==  1 and len(doc_embeddings.shape) == 1:
            logits = torch.einsum("d,d->", query_embeddings, doc_embeddings)
        elif len(query_embeddings.shape) ==  1 and len(doc_embeddings.shape) == 2:
            logits = torch.einsum("d,md->m", query_embeddings, doc_embeddings)
        elif len(query_embeddings.shape) == 2 and len(doc_embeddings.shape) == 3:
            assert len(query_embeddings) == len(doc_embeddings)
            logits = torch.einsum("nd,nmd->nm", query_embeddings, doc_embeddings)
        elif len(query_embeddings.shape) == 2 and len(doc_embeddings.shape) == 2:
            logits = torch.einsum("nd,md->nm", query_embeddings, doc_embeddings) 
        else:
            raise ValueError(f"Invalid embedding shape! query_embeddings: {query_embeddings.shape}, doc_embeddings: {doc_embeddings.shape}.")

        return logits

    def score(self, query_embeddings, doc_embeddings, **kwargs):
        if self.temperature == "sqrt":
            scores = self.compute_logits(query_embeddings, doc_embeddings) / np.sqrt(query_embeddings.shape[-1])
        else:
            scores = self.compute_logits(query_embeddings, doc_embeddings) / self.temperature
        return scores
    
    def get_encoder_output(self, args, **kwargs):
        assert len(args["input_ids"].shape) == 2 
        outputs = self.encoder(**args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs 

    def encoder_embed(self, args, **kwargs):

        need_reshape = (len(args["input_ids"].shape) != 2)
        if need_reshape:
            *other_dim, last_dim = args["input_ids"].shape 
            args = {k: v.reshape(-1, last_dim) if torch.is_tensor(v) else v for k, v in args.items()}
        embeddings = self.get_encoder_output(args, **kwargs)
        embedding_size = embeddings.shape[-1]
        if need_reshape:
            embeddings = embeddings.reshape(*other_dim, embedding_size)
        return embeddings 
    
    def query(self, args, **kwargs):
        query_embeddings = self.encoder_embed(args, **kwargs)
        if self.norm_query:
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=-1)
        return query_embeddings
    
    def doc(self, args, **kwargs):
        doc_embeddings = self.encoder_embed(args, **kwargs)
        if self.norm_doc:
            doc_embeddings = torch.nn.functional.normalize(doc_embeddings, dim=-1)
        return doc_embeddings
    
    def save_model(self, save_path):
        self.encoder.save_pretrained(save_path)
    
    def load_model(self, save_path):
        self.encoder = load_retriever(self.retriever_name, save_path, **self.retriever_kwargs, **self.kwargs)
        

class InBatchRetriever(BaseRetriever):

    def forward(self, query_args, doc_args, labels=None, **kwargs):

        query_embeddings = self.query(query_args, **kwargs)
        global_query_embeddings = get_global_embeddings_for_inbatchtraining(self.local_rank, self.world_size, query_embeddings)
        doc_embeddings = self.doc(doc_args, **kwargs)
        global_doc_embeddings = get_global_embeddings_for_inbatchtraining(self.local_rank, self.world_size, doc_embeddings)
        local_doc_size = len(doc_embeddings)
        global_labels = get_global_labels_for_inbatchtraining(self.local_rank, self.world_size, labels, local_doc_size)
        scores = self.score(global_query_embeddings, global_doc_embeddings)

        if global_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(scores, global_labels)
            outputs = (loss, scores, global_query_embeddings, global_doc_embeddings)
        else:
            outputs = (scores, global_query_embeddings, global_doc_embeddings)
        
        return outputs 


class DenseRetriever(nn.Module):

    def __init__(
        self, 
        retriever: BaseRetriever, 
        collator: RetrieverCollator, 
        indexer: Optional[Indexer]=None, 
        corpus: Optional[Corpus]=None, 
        batch_size: int=4,
        **kwargs 
    ):
        super().__init__()
        self.retriever = retriever
        self.device=self.retriever.device
        self.retriever.eval()

        self.collator = collator
        self.indexer = indexer
        self.corpus = corpus

        self.batch_size = batch_size
        self.kwargs = kwargs
    
    def get_documents(self, docid_list: Union[List[str], Dict[str, float]]) -> List[dict]:

        documents = []
        if isinstance(docid_list, list):
            for docid in docid_list:
                documents.append(deepcopy(self.corpus.get_document(docid)))
        elif isinstance(docid_list, dict):
            # rank documents based on scores 
            sorted_docid_list = sorted(docid_list.items(), key=lambda x: x[1], reverse=True)
            for docid, score in sorted_docid_list:
                document = deepcopy(self.corpus.get_document(docid))
                document["score"] = float(score)
                documents.append(document)
        else:
            raise ValueError(f"{type(docid_list)} is not a supported type for \"docid_list\"!")

        return documents
    
    def calculate_query_embeddings(self, queries: List[str], max_length: int=None, verbose: bool=False, **kwargs) -> Tensor:

        queries_embeddings_list = [] 
        assert isinstance(queries, list) and len(queries) > 0 # must provide queries 
        num_batches = (len(queries) - 1) // self.batch_size + 1 
        if verbose:
            progress_bar = trange(num_batches, desc="Calculating Query Embeddings")
        for i in range(num_batches):
            batch_queries = queries[i*self.batch_size: (i+1)*self.batch_size]
            batch_queries_inputs = self.collator.encode_query(batch_queries, max_length=max_length, **kwargs)
            batch_queries_inputs = to_device(batch_queries_inputs, self.device)
            batch_queries_embeddings = self.retriever.query(batch_queries_inputs).detach().cpu()
            queries_embeddings_list.append(batch_queries_embeddings)
            if verbose:
                progress_bar.update(1)
        # queries_embeddings = np.concatenate(queries_embeddings_list, axis=0)
        queries_embeddings = torch.cat(queries_embeddings_list, dim=0)

        return queries_embeddings 
    
    def calculate_document_embeddings(self, documents: List[str], max_length: int=None, verbose: bool=False, **kwargs) -> Tensor:

        documents_embeddings_list = [] 
        assert isinstance(documents, list) and len(documents) > 0 # must provide documents
        num_batches = (len(documents) - 1) // self.batch_size + 1 
        if verbose:
            progress_bar = trange(num_batches, desc="Calculating Document Embeddings")
        for i in range(num_batches):
            batch_documents = documents[i*self.batch_size: (i+1)*self.batch_size]
            batch_documents_inputs = self.collator.encode_doc(batch_documents, max_length=max_length, **kwargs)
            batch_documents_inputs = to_device(batch_documents_inputs, self.device)
            batch_documents_embeddings = self.retriever.doc(batch_documents_inputs).detach().cpu()
            documents_embeddings_list.append(batch_documents_embeddings)
            if verbose:
                progress_bar.update(1)
        # documents_embeddings = np.concatenate(documents_embeddings_list, axis=0)
        documents_embeddings = torch.cat(documents_embeddings_list, dim=0)

        return documents_embeddings 

    def parse_indexer_output(self, indexer_output):

        retrieval_results = []
        for topk_str_indices, topk_score_array in indexer_output:
            one_retrieval_results = [] 
            for docid, score in zip(topk_str_indices, topk_score_array):
                if self.corpus is not None:
                    document = deepcopy(self.corpus.get_document(docid))
                    document["score"] = float(score)
                else:
                    document = {"id": docid, "score": score}
                one_retrieval_results.append(document)
            retrieval_results.append(one_retrieval_results)
        
        return retrieval_results

    def batch_retrieve(self, queries: List[str], topk: int, verbose: bool=False, **kwargs) -> List[dict]:

        queries_embeddings = self.calculate_query_embeddings(queries=queries, verbose=verbose, **kwargs)
        queries_embeddings = queries_embeddings.numpy()

        # knn search 
        knn_results = self.indexer.search_knn(
            query_vectors=queries_embeddings, 
            top_docs=topk, 
            index_batch_size=1024,
            verbose=verbose
        )

        # parse results 
        retrieval_results = []
        for topk_str_indices, topk_score_array in knn_results:
            one_retrieval_results = [] 
            for docid, score in zip(topk_str_indices, topk_score_array):
                if self.corpus is not None:
                    document = deepcopy(self.corpus.get_document(docid))
                    document["score"] = float(score)
                else:
                    document = {"id": docid, "score": score}
                one_retrieval_results.append(document)
            retrieval_results.append(one_retrieval_results)
        return retrieval_results
    
    def forward(self, queries: Union[str, List[str]], topk: int, verbose: bool=False, **kwargs) -> Union[dict, List[dict]]:
        """
        Input:
            queries: str or List[str]
        Output:
            if self.corpus is not None:
                [ [{doc_content_in_corpus(key-value pair), "score": float}, ...topk], ... ]
            else:
                [ [{"id": docid[str], "score": doc_score[float]}, ...], ... ]
        """
        assert self.indexer is not None # must provide indexer 
        if isinstance(queries, str):
            return self.batch_retrieve([queries], topk=topk, verbose=verbose, **kwargs)[0]
        else:
            return self.batch_retrieve(queries, topk=topk, verbose=verbose, **kwargs)
