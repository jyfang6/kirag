import re 
import numpy as np
from nltk.tokenize import sent_tokenize
from typing import Union, Optional, Tuple, List, Dict

import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import LlamaForCausalLM, Qwen2ForCausalLM, T5ForConditionalGeneration
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from dataset.collators import E5Collator, BGECollator
from retriever.e5 import get_e5_embeddings_for_document, get_e5_embeddings_for_query
from generator.generator import Generator
from generator.utils import append_texts_to_decoder_only_generator_inputs
from knowledge_graph.kg_generator import KGGenerator
from retriever.retrievers import BaseRetriever, DenseRetriever
from utils.utils import hash_object
from evaluation.metrics import f1_score


class TripleSelector(nn.Module):

    def __init__(
        self, 
        tokenizer, 
        selector, 
        max_length: int=4096, 
        max_new_tokens: int=5, 
        examplar_type: str="hotpotqa", 
        num_examplars: int=5, 
        adaptive_examplars: bool=True, 
        adaptive_examplars_retriever: str="e5", 
        use_sentences: bool=False, 
        use_triple_filter: bool=True, 
        triple_filter_retriever: str="e5", 
        triple_filter_retriever_path: str=None, 
        triple_filter_reranker: str=None,
        triple_filter_reranker_path: str=None, 
        triple_filter_reranker_num: int=100, 
        num_candidate_triples: int=25, 
        batch_size: int=4, 
        query_reformulator=None, 
        use_title_in_triples: bool=False, 
        maximum_possible_choices: int=100, 
        use_cot: bool=False, 
        **kwargs, 
    ):
        
        super().__init__()
        assert examplar_type in ["hotpotqa", "2wikimultihopqa", "musique", "nq", "tqa", "webqa", "bamboogle", "wikipedia"]

        self.use_cot = use_cot
        if self.use_cot:
            adaptive_examplars = False
            num_examplars = min(num_examplars, 5)

        self.tokenizer = tokenizer
        self.selector = selector
        assert isinstance(selector, (LlamaForCausalLM, Qwen2ForCausalLM))
        self.device = self.selector.device 
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_examplars = num_examplars
        self.adaptive_examplars = adaptive_examplars
        self.adaptive_examplars_retriever = adaptive_examplars_retriever
        self.reasoning_chain_examplars, self.triple_selection_examplars = self.load_examplars(examplar_type)
        if self.adaptive_examplars:
            self.examplars_embeddings = self.calculate_examplars_embeddings()
        self.use_sentences = use_sentences

        # 对可能的选项进行设置
        # self.all_possible_choices = [chr(ord('B')+i) for i in range(25)] + [chr(ord('a')+i) for i in range(26)] # 从B开始因为A预留了
        # self.choices_to_indices = {choice: i for i, choice in enumerate(self.all_possible_choices)}
        # self.maximum_possible_choices = len(self.all_possible_choices)
        self.use_triple_filter = use_triple_filter
        self.triple_filter_retriever = triple_filter_retriever
        if self.triple_filter_retriever in ["finetuned_e5", "finetuned_bge", "finetuned_contriever"]:
            self.triple_retriever_collator, self.triple_retriever = self.load_finetuned_triple_retriever(triple_filter_retriever, triple_filter_retriever_path)
        self.num_candidate_triples = num_candidate_triples
        self.maximum_possible_choices = maximum_possible_choices # 没有用triple_filter的情况下最多的triples
        if triple_filter_reranker is not None:
            self.triple_reranker_collator, self.triple_reranker = self.load_triple_reranker(triple_filter_reranker, triple_filter_reranker_path)
        else:
            self.triple_reranker = None
        self.triple_filter_reranker_num = triple_filter_reranker_num
        # if self.num_candidate_triples > self.maximum_possible_choices: 
        #     print(f"num_candidate_triples={self.num_candidate_triples} exceeds the maximum number of triples, setting it to maximum num {self.maximum_possible_choices}")
        #     self.num_candidate_triples = self.maximum_possible_choices

        self.batch_size = batch_size
        self.query_reformulator = query_reformulator
        if self.query_reformulator is None:
            self.query_reformulator = QuerReformulator(reformulate_type="concat", convert_triple_to_text=False)
        self.use_title_in_triples = use_title_in_triples
        if self.use_cot:
            self.task_instruction = "Select the next knowledge triple (step-by-step) that extends an existing set of knowledge triples to form a coherent reasoning path capable of answering a specified question. "
        else:
            self.task_instruction = "Select the next knowledge triple that extends an existing set of knowledge triples to form a coherent reasoning path capable of answering a specified question. " \
                "If the current reasoning path is sufficient to answer the question, simply output 0. Please only output the choice for the next knowledge triple."
        self.kwargs = kwargs

    def load_examplars(self, examplar_type):

        print(f"loading {examplar_type} examplars for TripleSelector... ")
        # if examplar_type == "hotpotqa":
        if examplar_type in ["hotpotqa", "nq", "tqa", "webqa", "bamboogle", "wikipedia"]:
            from prompts.kg_selection.hotpotqa_demonstrations import reasoning_chains_hotpotqa_examplars, triple_selection_hotpotqa_examplars
            reasoning_chain_examplars = reasoning_chains_hotpotqa_examplars
            triple_selection_examplars = triple_selection_hotpotqa_examplars
        elif examplar_type == "2wikimultihopqa":
            from prompts.kg_selection.wikimultihopqa_demonstrations import reasoning_chains_2wikimultihopqa_examplars, triple_selection_2wikimultihopqa_examplars
            reasoning_chain_examplars = reasoning_chains_2wikimultihopqa_examplars
            triple_selection_examplars = triple_selection_2wikimultihopqa_examplars
        elif examplar_type == "musique":
            from prompts.kg_selection.musique_demonstrations import reasoning_chains_musique_examplars, triple_selection_musique_examplars
            reasoning_chain_examplars = reasoning_chains_musique_examplars
            triple_selection_examplars = triple_selection_musique_examplars
        # elif examplar_type == "wikipedia":
        #     from prompts.kg_construction.wikipedia_demonstrations import generate_knowledge_triples_wikipedia_examplars
        #     examplars = generate_knowledge_triples_wikipedia_examplars
        else:
            raise KeyError(f"{examplar_type} is not a supported examplar type!")
        
        return reasoning_chain_examplars, triple_selection_examplars

    def triple_retriever_for_query(self, query_list: List[str], max_length: int=None):

        original_collator_query_maxlength = self.triple_retriever_collator.query_maxlength
        if max_length is not None:
            self.triple_retriever_collator.query_maxlength = max_length
        
        query_embeddings_list = [] 
        for i in range((len(query_list)-1)//self.batch_size+1):
            batch_query_list = query_list[i*self.batch_size: (i+1)*self.batch_size]
            batch_query_args = self.triple_retriever_collator.encode_query(batch_query_list)
            batch_query_args = to_device(batch_query_args, device=self.device)
            batch_query_embeddings = self.triple_retriever.query(batch_query_args)
            query_embeddings_list.append(batch_query_embeddings.detach().cpu())
        query_embeddings = torch.cat(query_embeddings_list, dim=0)

        if max_length is not None:
            self.triple_retriever_collator.query_maxlength = original_collator_query_maxlength
        
        return query_embeddings
    
    def triple_retriever_for_document(self, document_list: List[str], max_length: int=None):

        original_collator_doc_maxlength = self.triple_retriever_collator.doc_maxlength
        if max_length is not None:
            self.triple_retriever_collator.doc_maxlength = max_length

        document_embeddings_list = [] 
        for i in range((len(document_list)-1) // self.batch_size + 1):
            batch_document_list = document_list[i*self.batch_size: (i+1)*self.batch_size]
            batch_document_args = self.triple_retriever_collator.encode_doc(batch_document_list)
            batch_document_args = to_device(batch_document_args, device=self.device)
            batch_document_embeddings = self.triple_retriever.doc(batch_document_args)
            document_embeddings_list.append(batch_document_embeddings.detach().cpu())
        document_embeddings = torch.cat(document_embeddings_list, dim=0)

        if max_length is not None:
            self.triple_retriever_collator.doc_maxlength = original_collator_doc_maxlength
        
        return document_embeddings 

    def calculate_retriever_query_embeddings(self, retriever: str, queries: List[str], max_length: int=128):
        if retriever == "e5":
            queries_embeddings = get_e5_embeddings_for_query(queries, max_length=max_length)
        elif retriever == "e5_mistral":
            queries_embeddings = get_e5_mistral_embeddings_for_query("retrieve_relevant_triples", query_list=queries, max_length=max_length)
        elif retriever in ["finetuned_e5", "finetuned_bge", "finetuned_contriever"]:
            queries_embeddings = self.triple_retriever_for_query(query_list=queries, max_length=max_length)
        else:
            raise NotImplementedError(f"{retriever} is not a supported retriever!")
        return queries_embeddings
    
    def calculate_retriever_document_embeddings(self, retriever: str, documents: List[str], max_length: int=256):
        if retriever == "e5":
            documents_embeddings = get_e5_embeddings_for_document(documents, max_length=max_length)
        elif retriever == "e5_mistral":
            documents_embeddings = get_e5_mistral_embeddings_for_document(documents, max_length=max_length)
        elif retriever in ["finetuned_e5", "finetuned_bge", "finetuned_contriever"]:
            documents_embeddings = self.triple_retriever_for_document(documents, max_length=max_length)
        else:
            raise NotImplementedError(f"{retriever} is not a supported retriever!")
        return documents_embeddings
    
    def calculate_examplars_embeddings(self):

        questions = [example["question"] for example in self.reasoning_chain_examplars]
        questions_embeddings = self.calculate_retriever_query_embeddings(
            self.adaptive_examplars_retriever, questions, max_length=128
        )
        return questions_embeddings
    
    def load_triple_reranker(self, reranker_name: str, reranker_path: Optional[str]=None):
        DEFAULT_RERANKER_PATH = {
            "bge_reranker": "BAAI/bge-reranker-large",
        }
        if reranker_path is None:
            reranker_path = DEFAULT_RERANKER_PATH[reranker_name]
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_RERANKER_PATH[reranker_name])

        if reranker_name == "bge_reranker":
            collator = BGEKGChainRerankerCollator(tokenizer=tokenizer, maxlength=512)
            print(f"Loading BGE Triple Reranker from {reranker_path} ...")
            reranker = BaseReranker("BGEReranker", model_name_or_path=reranker_path)
        else:
            raise NotImplementedError(f"{reranker_name} reranker is not implemented!")
        
        reranker.to(self.device)
        reranker.eval()
        return collator, reranker
    
    def load_finetuned_triple_retriever(self, retriever_name: str, retriever_path: Optional[str]=None):

        if "e5" in retriever_name.lower():
            default_retriever_path = "intfloat/e5-large-v2"
        elif "bge" in retriever_name.lower():
            default_retriever_path = "BAAI/bge-large-en-v1.5"
        elif "contriever" in retriever_name.lower():
            default_retriever_path = "facebook/contriever"
        else:
            raise NotImplementedError(f"triple filter retriever \"{retriever_name}\" is not implemented!")

        if retriever_path is None:
            retriever_path = default_retriever_path
        tokenizer = AutoTokenizer.from_pretrained(default_retriever_path)
        if retriever_name == "finetuned_e5":
            collator = E5Collator(tokenizer=tokenizer, query_maxlength=256, doc_maxlength=128)
            print(f"loading finetuned E5 retriever from {retriever_path}...")
            retriever = BaseRetriever(retriever_name="E5Retriever", model_name_or_path=retriever_path)
        elif retriever_name == "finetuned_bge":
            collator = BGECollator(tokenizer=tokenizer, query_maxlength=256, doc_maxlength=128)
            print(f"loading finetuned BGE retriever from {retriever_path}...")
            retriever = BaseRetriever(retriever_name="BGERetriever", model_name_or_path=retriever_path)
        elif retriever_name == "finetuned_contriever":
            collator = ContrieverCollator(tokenizer=tokenizer, query_maxlength=256, doc_maxlength=128)
            print(f"loading finetuned Contriever retriever from {retriever_path}...")
            retriever = BaseRetriever(retriever_name="ContrieverRetriever", model_name_or_path=retriever_path)
        else:
            raise NotImplementedError(f"triple filter retriever \"{retriever_name}\" is not implemented!")

        retriever.to(self.device)
        retriever.eval()
        return collator, retriever
    
    def rank_examplars(self, question):

        if not self.adaptive_examplars:
            return list(range(len(self.reasoning_chain_examplars)))
        
        question_embedding = self.calculate_retriever_query_embeddings(
            self.adaptive_examplars_retriever, [question], max_length=128
        )
        similarities = torch.matmul(question_embedding, self.examplars_embeddings.T)
        indices = torch.argsort(similarities, dim=1, descending=True).tolist()[0]
        return indices

    def get_texts_from_documents(self, documents: Union[Dict[str, str], List[Dict[str, str]]]) -> List[str]:

        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]

        texts = [] 
        for doc in documents:
            title = doc["title"]
            text = doc.get("text", None)
            if text is None:
                text = " ".join([sent.strip() for sent in doc["sentences"]])
            texts.append("Title: {}\nText: {}".format(title, text))
        
        if not is_list:
            texts = texts[0]

        return texts
    
    def parse_reasoning_chains(self, reasoning_chains: Optional[List[Dict[str, Union[List, float, bool]]]]):
        """
        Input:
            reasoning_chains: [
                {
                    "triples": [
                        {"text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]},
                    ],
                    "score": float,
                    "finished": bool 
                }
            ]
        Output: 
            chains: [[{"text": "<xx; xx; xx>", "reference": [str, int]}, ...], ...]
            chains_scores: [0.4, ...]
            chains_finished: [False, ...]
        """

        if reasoning_chains is None or len(reasoning_chains) == 0:
            chains, chains_scores, chains_finished = [[]], [1.0], [False]
            return chains, chains_scores, chains_finished

        chains, chains_scores, chains_finished = [], [], [] 
        for reasoning_chain in reasoning_chains:
            chains.append(reasoning_chain["triples"])
            chains_scores.append(reasoning_chain["score"])
            chains_finished.append(reasoning_chain["finished"])
        return chains, chains_scores, chains_finished
    
    def parse_reasoning_chains_triple_filter_scores(self, reasoning_chains: Optional[List[Dict[str, Union[List, float, bool]]]]):
        if reasoning_chains is None or len(reasoning_chains) == 0:
            return [[]]
        chains_triple_filter_scores = [] 
        for reasoning_chain in reasoning_chains:
            num_triples = len(reasoning_chain["triples"])
            chains_triple_filter_scores.append(reasoning_chain.get("triple_filter_scores", [1.0]*num_triples))
        return chains_triple_filter_scores
    
    def parse_reasoning_chains_history(self, reasoning_chains: Optional[List[Dict[str, Union[List, float, bool]]]]):
        if reasoning_chains is None or len(reasoning_chains) == 0:
            return [[]]
        chains_history = [] 
        for reasoning_chain in reasoning_chains:
            chains_history.append(reasoning_chain.get("chain_history", None))
        return chains_history
    
    def get_candidate_triples_from_documents(self, documents: List[Dict[str, Union[str, List]]]):
        """
        Input: 
            documents: [{"id": str, "title": str, "text": str / "sentence": str, "triples": ["text": str, "sentence": int]}]
        Output:
            triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
        """
        triples = [] 

        for doc in documents:
            doc_id = doc["id"]
            title = doc["title"]
            for one_triple_in_doc in doc["triples"]:
                triple = {
                    "title": title,
                    "text": one_triple_in_doc["text"],
                    "reference": [doc_id, one_triple_in_doc["sentence"]]
                }
                triples.append(triple)

            if self.use_sentences:
                sentences = doc.get("sentences", None)
                if sentences is None:
                    sentences = sent_tokenize(doc["text"])
                for sentence_index, sentence in enumerate(sentences):
                    triple = {"title":title, "text": sentence, "reference": [doc_id, sentence_index]}
                    triples.append(triple)

        return triples
    
    def get_triple_text(self, triple: Dict[str, Union[str, List]]):
        """
        Input: 
            triple: {"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}
        """
        # if self.use_title_in_triples:
        #     triple_text = "title: {}, text: {}".format(triple["title"], convert_triples_to_sentences(triple["text"]))
        # else:
        #     triple_text = convert_triples_to_sentences(triple["text"])
        if self.use_title_in_triples:
            triple_text = "title: {}, text: {}".format(triple["title"], triple["text"])
        else:
            triple_text = triple["text"]
        return triple_text
    
    def get_reasoning_chains_texts(self, reasoning_chains: List[List[Dict[str, Union[str, List]]]]):
        """
        Input:
            reasoning_chains: [[{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...], ...]
        Output:
            reasoning_chains_texts: [["<xxx; xxx; xxx>", ...], ...]
        """
        if len(reasoning_chains) == 0:
            return [[]]
        reasoning_chains_texts = [[self.get_triple_text(triple) for triple in chain] for chain in reasoning_chains]
        return reasoning_chains_texts

    def compute_reranker_triple_filter_scores(
        self, 
        question: str, 
        reasoning_chains: List[List[Dict[str, Union[str, List]]]], 
        triples: List[Dict[str, Union[str, List]]],
        queries_triples_similarities: Tensor, 
    ):
        """
        Input:
            question: str, 
            reasoning_chains: [[{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...], ...],
            triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
        Output:
            queries_triples_similarities: Tensor[num_reasoning_chains, num_triples]
        """
        # 首先得到queries_triples_similarities中排在前self.triple_filter_reranker_num的triple
        topk_values, topk_indices = torch.topk(
            queries_triples_similarities, 
            k=min(self.triple_filter_reranker_num, len(triples)), 
            dim=1
        )

        reasoning_chains_texts = self.get_reasoning_chains_texts(reasoning_chains) # [["xxx", "xxx"]]
        candidate_reasoning_chains_texts = [] 
        for i, reasoning_chain_texts in enumerate(reasoning_chains_texts):
            topk_triples = [triples[idx] for idx in topk_indices[i].tolist()]
            for triple in topk_triples:
                candidate_reasoning_chain_texts = reasoning_chain_texts + [self.get_triple_text(triple)]
                candidate_reasoning_chains_texts.append(candidate_reasoning_chain_texts)
        
        args, *others = self.triple_reranker_collator(
            [
                {
                    "index": 0, # placeholder 
                    "questions": [question] * len(candidate_reasoning_chains_texts),
                    "reasoning_chains": candidate_reasoning_chains_texts, 
                    "labels": [0] * len(candidate_reasoning_chains_texts)
                }
            ]
        )

        batch_scores_list = [] 
        for i in range((len(candidate_reasoning_chains_texts)-1) // self.batch_size+1):
            batch_args = to_device({k: v[i*self.batch_size: (i+1)*self.batch_size] for k, v in args.items()}, self.device)
            batch_scores = self.triple_reranker.score(args=batch_args).detach().cpu()
            batch_scores_list.append(batch_scores)
        scores = torch.cat(batch_scores_list, dim=0)
        scores = scores.reshape(len(reasoning_chains), -1)

        new_queries_triples_similarities = torch.zeros_like(queries_triples_similarities).fill_(-1e6)
        for i in range(new_queries_triples_similarities.shape[0]):
            new_queries_triples_similarities[i, topk_indices[i]] = scores[i]

        return new_queries_triples_similarities
        
    def filter_candidate_triples(
        self, 
        question: str, 
        reasoning_chains: List[List[Dict[str, Union[str, List]]]], 
        triples: List[Dict[str, Union[str, List]]],
        num_candidate_triples: int,
    ):
        """
        Input:
            question: str, 
            reasoning_chains: [[{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...], ...],
            triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
        Output:
            candidate_triples_indices: [[int]*num_candidate_triples, ...]
            candidate_triples_scores: [[float]*num_candidate_triples]
        """

        num_triples = len(triples)

        # 首先得到所有的triples的embeddings 
        # triples_texts = [convert_triples_to_sentences(triple["text"]) for triple in triples]
        triples_texts = [self.get_triple_text(triple) for triple in triples]
        triples_embeddings = self.calculate_retriever_document_embeddings(
            self.triple_filter_retriever, triples_texts, max_length=128
        )
        # 然后根据question和现有的reasoning chains来构造queries
        # reasoning_chains_texts = [[triple["text"] for triple in reasoning_chain] for reasoning_chain in reasoning_chains]
        # queries = [
        #     self.query_reformulator.reformulate(question, reasoning_chain_texts) \
        #         for reasoning_chain_texts in reasoning_chains_texts
        # ]
        reasoning_chains_texts = self.get_reasoning_chains_texts(reasoning_chains)
        queries = self.query_reformulator.forward(question=question, reasoning_chains=reasoning_chains_texts)
        # 得到queries的的embeddings
        queries_embeddings = self.calculate_retriever_query_embeddings(
            self.triple_filter_retriever, queries=queries, max_length=256
        )
        # 然后计算query-triple的相似度
        queries_triples_similarities = torch.matmul(queries_embeddings, triples_embeddings.T)

        # 如果self.triple_reranker不为None的话对queries_triples_similarities进行重排
        if self.triple_reranker is not None:
            queries_triples_similarities = self.compute_reranker_triple_filter_scores(question, reasoning_chains, triples, queries_triples_similarities)

        # 构造query-triple mask, 去掉已经在reasoning chain中的triple
        triples_mask = torch.ones_like(queries_triples_similarities)
        for i, reasoning_chain in enumerate(reasoning_chains):
            existing_triples_texts = set([triple["text"] for triple in reasoning_chain])
            for j, triple in enumerate(triples):
                triple_text = triple["text"]
                if triple_text in existing_triples_texts:
                    triples_mask[i, j] = 0 
        
        # 整合mask中的信息来更新相似度
        MIN_VALUE = torch.finfo(queries_triples_similarities.dtype).min
        queries_triples_similarities +=  MIN_VALUE * (1.0 - triples_mask)

        # 得到每一个query最相似的top-num_candidate_triples个triples的indices 
        # topk_relevant_triples_indices = torch.topk(
        #     queries_triples_similarities, 
        #     k = min(num_candidate_triples, num_triples),
        #     dim=1
        # )[1].tolist()
        topk_relevant_triples_scores, topk_relevant_triples_indices = torch.topk(
            queries_triples_similarities, 
            k = min(num_candidate_triples, num_triples),
            dim=1
        )
        topk_relevant_triples_indices = topk_relevant_triples_indices.tolist()
        topk_relevant_triples_scores = topk_relevant_triples_scores.tolist()

        return topk_relevant_triples_indices, topk_relevant_triples_scores
    
    def filter_duplicate_triples(
        self, 
        reasoning_chains: List[List[Dict[str, Union[str, List]]]], 
        triples: List[Dict[str, Union[str, List]]],
        candidate_triples_indices: List[List[int]],
        candidate_triples_scores: List[List[float]]=None, 
    ):
        """
        Input:
            reasoning_chains: [[{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...], ...],
            triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
            candidate_triples_indices: [[int*num_candidate_triples, ...], ...]
            candidate_triples_scores: [[int*num_candidate_triples, ...], ...]
        Output:
            candidate_triples_indices_wo_duplicates: [[int*num_candidates_for_this_chain], ...]
        """
        if candidate_triples_scores is None:
            return_scores = False
            candidate_triples_scores = [[0.0]*len(indices) for indices in candidate_triples_indices]
        else:
            return_scores = True
        
        candidate_triples_indices_wo_duplicates = [] 
        candidate_triples_scores_wo_duplicates = [] 

        for reasoning_chain, chain_candidate_triples_indices, chain_candidate_triples_scores in \
            zip(reasoning_chains, candidate_triples_indices, candidate_triples_scores):

            existing_triples_texts = set([triple["text"] for triple in reasoning_chain])

            new_chain_candidate_triples_indices = [] 
            new_chain_candidate_triples_scores = [] 
            for index, score in zip(chain_candidate_triples_indices, chain_candidate_triples_scores):
                candidate_triple_text = triples[index]["text"]
                if candidate_triple_text not in existing_triples_texts:
                    new_chain_candidate_triples_indices.append(index)
                    new_chain_candidate_triples_scores.append(score)

            candidate_triples_indices_wo_duplicates.append(new_chain_candidate_triples_indices)
            candidate_triples_scores_wo_duplicates.append(new_chain_candidate_triples_scores)

        # if return_scores:
        #     return candidate_triples_indices_wo_duplicates, candidate_triples_scores_wo_duplicates
        # else:
        #     return candidate_triples_indices_wo_duplicates
        return candidate_triples_indices_wo_duplicates, candidate_triples_scores_wo_duplicates

    def convert_examplar_triple_text_to_triple_item(self, triple_text: str):
        """
        Input:
            triple_text: <xxx; xxx; xxx>
        """
        title = triple_text.replace("<", "").replace(">", "").split(";")[0].strip()
        return {"title": title, "text": triple_text}
    
    def convert_examplar_reasoning_chain_to_sentences(self, chain: str):
        """
        Input:
            chain: "<xx1; xx2; xx3>, <xx4; xx5, xx6>" or "B. <xxx; xxx; xxx>" or "A. no need for additional triples"
        Output:
            xx1 xx2 xx3. xx4 xx5 xx6
        """
        # version 1 
        # pattern = r"<(.*?)>"
        # matches = re.findall(pattern, chain)
        # sentences = [string.replace(";", "", 2) for string in matches]

        # version 2 
        pattern = r"<(.*?)>"
        matches = re.findall(pattern, chain)
        triples = [self.convert_examplar_triple_text_to_triple_item(f"<{string}>") for string in matches]
        triples_texts = [self.get_triple_text(triple) for triple in triples]
        return ". ".join(triples_texts)
    
    def convert_examplar_options_to_numbers(self, candidates: List[str]):
        """
        Input:
            candidates: [["A. xxx", "B. xxx", ...], ... ]
        Output:
            new_candidates: [["0. xxx", "1. xxx", ...], ...]
        """
        option2numbers = {chr(ord('A')+i):str(i) for i in range(len(candidates))}
        candidates = [candidate.strip() for candidate in candidates]
        new_candidates = [option2numbers[candidate[0]] + candidate[1:] for candidate in candidates]
        return new_candidates

    def convert_examplar_answer_to_numbers(self, answer: str):
        """
        Input: 
            answer: "A"
        Output:
            new_answer: "0"
        """
        if self.use_cot:
            answer_token = answer.strip()[-2]
            return answer[:-2] + f"{ord(answer_token)-ord('A')}."
        else:
            return str(ord(answer.strip())-ord('A'))
    
    def convert_candidate_triples_to_choices(self, candidates: List[str]):
        """
        Input:
            candidates: ["xxx1 xxx2 xxx3", "xxx4 xxx5 xx6"]
        Output:
            choices: ["0. no need for additional knowledge triples", "1. xxx1 xxx2 xxx3", "2. xxx4 xxx5 xx6"]
        """
        # version: numbers as indices 
        choices = ["0. no need for additional knowledge triples"]
        for i, candidate in enumerate(candidates):
            choices.append(str(i+1) + ". " + candidate)
        return choices

        # choices = ["A. no need for additional knowledge triples"]
        # for i, candidate in enumerate(candidates):
        #     choices.append(self.all_possible_choices[i] + ". " + candidate)
        # return choices

    def get_selector_inputs(
        self, 
        question: str, 
        existing_triples: List[List[str]], 
        candidate_triples: List[List[str]],
        ranked_examplars_indices: List[int],
    ):
        """
        Input:
            question: str
            existing_triples: [[triple_text, ...], ...] # [["<xxx; xxx; xxx>", ...], ....]
            candidate_triples: [[triple_text, ...], ...] # [["<xxx; xxx; xxx>", ...], ....]
        Output:
            instructions: [str, ...]
            inputs: [str, ...]
        """
        def vary_num_examplars_based_on_context_window(instruction, examplars, input):
            final_examplars = None
            while len(examplars) > 0:
                for num in range(len(examplars), 0, -1):
                    possible_prompt = "{} {}\n\n{}".format(
                        instruction, 
                        "\n\n".join(examplars[:num]), 
                        input
                    )
                    possible_prompt_tokens = self.tokenizer.encode(possible_prompt)
                    if len(possible_prompt_tokens) <= self.max_length:
                        final_examplars = examplars[:num]
                        break
                if final_examplars is None:
                    examplars = examplars[1:]
                else:
                    break
            final_examplars = [] if final_examplars is None else final_examplars
            return final_examplars

        instructions, inputs = [], [] 
        for triples, candidates in zip(existing_triples, candidate_triples):

            # 首先得到instruction
            instruction = self.task_instruction
            examplars = [] 

            hop = len(triples)
            if self.num_examplars > 0:
                instruction += "\n\nThe followings are some examples of coherent reasoning paths capable of answering the specified question " \
                f"and how the {hop+1}-th knowledge triples in these paths are selected:\n\n"

                for index in ranked_examplars_indices:
                    reasoning_chain_examplar = self.reasoning_chain_examplars[index]
                    triple_selection_examplar = self.triple_selection_examplars[index]

                    #! 临时代码，有这段主要是因为我只标注了前5个例子
                    if self.use_cot and "cot_answer" not in triple_selection_examplar:
                        break

                    if len(triple_selection_examplar) < hop + 1 :
                        continue

                    examplar = "coherent reasoning path: {}\nquestion: {}\n".format(
                        self.convert_examplar_reasoning_chain_to_sentences(reasoning_chain_examplar["chains"]), 
                        reasoning_chain_examplar["question"]
                    )
                    examplar += "The {}-th triple in the reasoning path is selected as:\n".format(hop+1)
                    one_step_triple_selection = triple_selection_examplar[hop]
                    examplar += "existing knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:{}\n".format(
                        # ". ".join(convert_triples_to_sentences(one_step_triple_selection["triples"])), 
                        ". ".join(
                            [
                                self.convert_examplar_reasoning_chain_to_sentences(triple_text) \
                                    for triple_text in one_step_triple_selection["triples"]
                            ]
                        ),
                        one_step_triple_selection["question"], 
                        # "\n".join(
                        #     self.convert_examplar_options_to_numbers(
                        #         convert_triples_to_sentences(
                        #             one_step_triple_selection["candidate_triples"]
                        #         )
                        #     )
                        # ),
                        "\n".join(
                            self.convert_candidate_triples_to_choices(
                                [
                                    self.convert_examplar_reasoning_chain_to_sentences(ct) \
                                        for ct in one_step_triple_selection["candidate_triples"][1:]
                                ]
                            )
                        ),
                        self.convert_examplar_answer_to_numbers(
                            one_step_triple_selection["cot_answer"] if self.use_cot else one_step_triple_selection["answer"]
                        )
                        # one_step_triple_selection["answer"]
                    )
                    examplars.append(examplar)

                    if len(examplars) >= self.num_examplars:
                        break

            input_text = "The {}-th triple in the reasoning path is selected as:\nexisting knowledge triples: {}\nquestion: {}\ncandidate knowledge triples:\n{}\nthe next possible triple is:".format(
                hop+1,
                # ". ".join(convert_triples_to_sentences(triples)),
                ". ".join(triples), 
                question, 
                "\n".join(
                    self.convert_candidate_triples_to_choices(
                        # convert_triples_to_sentences(candidates)
                        candidates=candidates
                    )
                )
            )
            
            examplars = vary_num_examplars_based_on_context_window(instruction, examplars, input_text)
            instruction += "\n\n".join(examplars)
            instructions.append(instruction)
            inputs.append(input_text)
        
        return instructions, inputs

    def get_selector_prompts_chat_format(self, instructions: List[List[str]], inputs: List[List[str]]):
        """
        Input: 
            instruction: [str], inputs: [str]
        Output:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        """
        prompts = [] 
        for instruction, input in zip(instructions, inputs):
            if isinstance(self.selector, (LlamaForCausalLM, Qwen2ForCausalLM)):
                prompts.append(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": input}
                    ]
                )
            else:
                raise NotImplemented(f"chat format for {type(self.generator)} is not implemented yet!")
        return prompts
    
    def tokenizer_encode_chat_format(self, prompts: List[List[Dict[str, str]]]):
        """
        Input:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        Output:
            {"input_ids": Tensor, "attention_mask": Tensor}
        """
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        batch_dict = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def get_selector_generated_token_ids(self, input_ids: Tensor, token_ids: Tensor) -> Tensor:
        if isinstance(self.selector, T5ForConditionalGeneration):
            generated_token_ids = token_ids[:, 1:] # T5模型第一个token是<bos> token
        else:
            generated_token_ids = token_ids[:, input_ids.shape[1]:]
        return generated_token_ids
    
    def selector_generate(self, inputs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:

        batch_size, max_new_tokens, device = self.batch_size, self.max_new_tokens, self.device
        generated_token_ids_list, generated_token_logits_list = [], [] 
        for i in range((len(inputs["input_ids"])-1)//batch_size+1):
            batch_inputs = {k: v[i*batch_size: (i+1)*batch_size] for k, v in inputs.items()}
            batch_inputs = to_device(batch_inputs, device)
            batch_outputs = self.selector.generate(**batch_inputs, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True, do_sample=False, temperature=1.0) # temperature=1.5, do_sample=True) # 
            # batch_generated_token_ids = batch_outputs.sequences[:, batch_inputs["input_ids"].shape[1]:].detach().cpu()
            batch_generated_token_ids = self.get_selector_generated_token_ids(batch_inputs["input_ids"], batch_outputs.sequences)
            batch_generated_token_ids = batch_generated_token_ids.detach().cpu()
            batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores], dim=1).detach().cpu()
            assert batch_generated_token_ids.shape[1] == batch_generated_token_logits.shape[1] # the shape of token_ids and token_logits is not the same!

            if batch_generated_token_ids.shape[1] < max_new_tokens:
                real_batch_size, num_generated_tokens = batch_generated_token_ids.shape 
                padding_length = max_new_tokens-num_generated_tokens
                padding_token_ids = torch.zeros((real_batch_size, padding_length), dtype=batch_generated_token_ids.dtype).fill_(self.tokenizer.pad_token_id)
                padding_token_logits = torch.zeros((real_batch_size, padding_length, batch_generated_token_logits.shape[-1]), dtype=batch_generated_token_logits.dtype)
                batch_generated_token_ids = torch.cat([batch_generated_token_ids, padding_token_ids], dim=1)
                batch_generated_token_logits = torch.cat([batch_generated_token_logits, padding_token_logits], dim=1)
            
            generated_token_ids_list.append(batch_generated_token_ids)
            generated_token_logits_list.append(batch_generated_token_logits)

        generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
        generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

        return generated_token_ids, generated_token_logits

    def get_option_token_id_to_option_map(self, num_options: int):
        """
        Input:
            num_options: number of options 
        Output:
            option_token_id_to_option_map: {0的token_id: "0"}
        """
        option_token_id_to_option_map = {}
        options = [str(i) for i in range(num_options)]
        # options = ['A'] + self.all_possible_choices[:num_options]
        if isinstance(self.selector, (LlamaForCausalLM, Generator)):
            for option in options:
                option_token_id_to_option_map[self.tokenizer.encode(option, add_special_tokens=False)[-1]] = option
                option_token_id_to_option_map[self.tokenizer.encode(" {}".format(option), add_special_tokens=False)[-1]] = option
        else:
            raise NotImplemented(f"get_option_token_id_to_option_map for {type(self.generator)} is not implemented yet!")
        return option_token_id_to_option_map
    
    def get_option_token_indices(self, token_ids: Tensor, option_token_id_to_option_map: dict):
        """
        Input:
            token_ids: Tensor[num_chains, self.max_new_tokens]
            option_token_id_to_option_map: {0的token_id: "0"}
        Output:
            option_token_indices: Tensor[num_chains]
        """
        option_token_indices = torch.zeros((token_ids.shape[0],), dtype=token_ids.dtype)
        for i in range(token_ids.shape[0]):
            for j in range(token_ids.shape[1]):
                if token_ids[i, j].item() in option_token_id_to_option_map:
                    option_token_indices[i] = j 
                    break
        return option_token_indices

    def get_option_tokens_probs(self, token_ids: Tensor, token_logits: Tensor, maximum_num_options: int):

        """
        Input: 
            token_ids: [num_chains, self.max_new_tokens]
            token_logits: [num_chains, self.max_new_tokens, vocabulary_size]
            maximum_num_options: int 
        Output: 
            option_tokens_list: ["0", "0", ..., "num_option_token_ids"]
            option_probs: [num_chains, num_option_token_ids]
        """
        # {0的token_id: "0"}
        option_token_id_to_option_map = self.get_option_token_id_to_option_map(maximum_num_options)
        # Tensor[num_chains]
        option_token_indices = self.get_option_token_indices(token_ids, option_token_id_to_option_map)
        option_token_logits = token_logits.gather(
            1, 
            option_token_indices[:, None, None].expand(-1, -1, token_logits.shape[-1])
        )
        # Tensor[num_chains, vocabulary_size] 
        option_token_logits = option_token_logits.squeeze(1)

        # 只得到选项token的概率，去掉不相关的token的概率
        option_token_ids_list = list(option_token_id_to_option_map.keys())
        option_tokens_list = [option_token_id_to_option_map[token_id] for token_id in option_token_ids_list]
        option_tokens_probs = F.softmax(option_token_logits[:, option_token_ids_list], dim=1)

        return option_tokens_list, option_tokens_probs

    def forward(
        self, 
        question: str, 
        documents: List[Dict[str, Union[str, List]]],
        existing_reasoning_chains: Optional[List[Dict[str, Union[List, float, bool]]]]=None,
        num_beams: int=5, 
        max_num_chains: Optional[int]=None, # 设置为None返回所有可能的chains
        min_triple_prob: Optional[float]=1e-4, 
        return_triple_filter_scores: Optional[bool]=True, 
        return_chain_history: Optional[bool]=False,
        **kwargs, 
    ):
        """
        Input: 
            question: str, 
            documents: [{"id": str, "title": str, "text": str / "sentence": str, "triples": ["text": str, "sentence": int]}]
            existing_reasoning_chains: [
                {
                    "triples": [
                        {"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]},
                    ],
                    "score": float,
                    "finished": bool,
                    ("triple_filter_scores": [float, float, ..],) optional 
                    ("chain_history": {"question": str, "existing_triples": [str,], "candidate_triples": [str], "selected_triple_idx": int or -1}) optional
                }
            ]
        Output: 
            new_reasoning_chains: the same format as existing_reasoning_chains
        """
        if return_triple_filter_scores:
            assert self.use_triple_filter is True # you must use triple filter when setting "return_triple_filter_scores=True"

        if existing_reasoning_chains is not None and len(existing_reasoning_chains) == 0:
            existing_reasoning_chains = None
        
        # chains: [[{"title": str, "text": "<xx; xx; xx>", "reference": [str, int]}, ...], ...]
        # chains_scores: [0.4, ...]
        # chains_finished: [False, ...]
        chains, chains_scores, chains_finished = self.parse_reasoning_chains(existing_reasoning_chains)
        if existing_reasoning_chains is not None and np.sum(chains_finished) == len(existing_reasoning_chains):
            # 表明所有的chain都已经结束了，直接返回现有的结果
            return existing_reasoning_chains
        
        # 如果self.adaptive_examplars为True的话按照相似度对demonstrations进行排序，否则直接返回从0开始的一个列表
        ranked_examplars_indices = self.rank_examplars(question)

        # 得到document中所有的candidate triples，如果self.use_sentence为True的话会把sentence也当做triples
        # all_triples: [{"title": str, "text": "<xx; xx; xx>", "reference": [str(document id), int(sentence index)]}, ...]
        all_triples = self.get_candidate_triples_from_documents(documents)

        if self.use_triple_filter:
            # 如果选择使用triple_filter的话先用adaptive_examplars_retriever过滤掉一些不相关的triples
            # [[int, int, ...], ...]
            candidate_triples_indices, candidate_triples_scores = self.filter_candidate_triples(question, chains, all_triples, self.num_candidate_triples)
        else:
            # 否则把所有的triples当做candidate_triples 
            # candidate_triples_indices = [list(range(len(all_triples))) for _ in range(len(chains))]
            # 选择最多前self.maximum_possible_choices个triples当做candidates
            candidate_triples_indices = [list(range(min(len(all_triples), self.maximum_possible_choices)))for _ in range(len(chains))]
            candidate_triples_scores = None 
        
        # 过滤掉candidate_triples_indices中和现有的reasoning chain重复的部分，
        # 注意这样不同reasoning_chains的candidate数量可能不一样，而且有可能为空; 如果return_triple_filter_scores为False的话, candidate_triples_scores_wo_duplicates仍然有值, 但是都是1.0,没有意义
        candidate_triples_indices_wo_duplicates, candidate_triples_scores_wo_duplicates = self.filter_duplicate_triples(chains, all_triples, candidate_triples_indices, candidate_triples_scores)
        maximum_num_candidate_triples = max([len(candidate_indices) for candidate_indices in candidate_triples_indices_wo_duplicates])

        # 得到selector模型的输入
        # existing_triples = [[triple["text"] for triple in reasoning_chain] for reasoning_chain in chains]
        # candidate_triples_wo_duplicates = [[all_triples[idx]["text"] for idx in triples_indices] for triples_indices in candidate_triples_indices_wo_duplicates]
        existing_triples = self.get_reasoning_chains_texts(chains)
        candidate_triples_wo_duplicates = [
            [self.get_triple_text(all_triples[idx]) for idx in triples_indices] \
                for triples_indices in candidate_triples_indices_wo_duplicates
        ]
        instructions, inputs = self.get_selector_inputs(
            question, 
            existing_triples = existing_triples, 
            candidate_triples = candidate_triples_wo_duplicates, 
            ranked_examplars_indices=ranked_examplars_indices,
        )
        prompts = self.get_selector_prompts_chat_format(instructions, inputs)
        selector_inputs = self.tokenizer_encode_chat_format(prompts)
        generated_token_ids, generated_token_logits = self.selector_generate(selector_inputs)

        # 解析模型的生成结果 (+1是因为除了candidate triples之外加了一个截止的选项,注意：如果要用字母版本的选项的话记得去掉+1)
        # option_tokens_list: ["0", "0", ..."num_option_token_ids"]
        # option_probs: [num_chains, num_option_token_ids]
        option_tokens_list, option_tokens_probs = self.get_option_tokens_probs(
            generated_token_ids, generated_token_logits, maximum_num_candidate_triples + 1 
        )
        # from pdb import set_trace; set_trace()
        # beam search for generating new chains 
        chains_triple_filter_scores = self.parse_reasoning_chains_triple_filter_scores(existing_reasoning_chains)
        chains_history = self.parse_reasoning_chains_history(existing_reasoning_chains)

        new_chains, new_chains_scores, new_chains_finished = [], [], [] 
        new_chains_triple_filter_scores = [] 
        new_chains_history = [] 
        topk_options_probs, topk_options_indices = torch.topk(
            option_tokens_probs, 
            k=min(option_tokens_probs.shape[1], num_beams), 
            dim=1
        )

        for i in range(len(chains)):
            candidate_triples_indices_for_current_chain = candidate_triples_indices_wo_duplicates[i]
            candidate_triples_scores_for_current_chain = candidate_triples_scores_wo_duplicates[i]
            candidate_triples_for_current_chain = [self.get_triple_text(all_triples[idx]) for idx in candidate_triples_indices_for_current_chain]
            # 如果当前路径已经结束或者没有任何的candidate triples的话，直接复制当前路径并结束
            if chains_finished[i] or len(candidate_triples_indices_for_current_chain) == 0:
                new_chains.append(chains[i])
                new_chains_scores.append(chains_scores[i])
                new_chains_finished.append(True)
                new_chains_triple_filter_scores.append(chains_triple_filter_scores[i])
                new_chains_history.append(chains_history[i])
                # new_chains_history.append(
                #     {
                #         "question": question, 
                #         "existing_triples": existing_triples[i], 
                #         "candidate_triples": candidate_triples_for_current_chain, 
                #         "selected_triple_idx": -1, # -1表示没有选择candidate_triples中的任意一个triple
                #     }
                # )
                continue
            # 如果当前路径所有选项对应的概率都是nan的时候, 说明模型没有生成答案(可能是max_new_tokens设置得过小), 直接复制当前路径
            if torch.all(torch.isnan(topk_options_probs[i])):
                print("No choice in generated results! generated text: {}".format(self.tokenizer.decode(generated_token_ids[i])))
                new_chains.append(chains[i])
                new_chains_scores.append(chains_scores[i])
                new_chains_finished.append(False)
                new_chains_triple_filter_scores.append(chains_triple_filter_scores[i])
                new_chains_history.append(chains_history[i])
                # new_chains_history.append(
                #     {
                #         "question": question, 
                #         "existing_triples": existing_triples[i], 
                #         "candidate_triples": candidate_triples_for_current_chain, 
                #         "selected_triple_idx": -1, # -1表示没有选择candidate_triples中的任意一个triple
                #     }
                # )
                continue
            # 如果没有结束的话, 对下一个可能的triple进行beam search
            for b in range(min(option_tokens_probs.shape[1], num_beams)):
                # 如果topk的选项的概率是nan或者概率值过小的话，跳过这个选项
                if torch.isnan(topk_options_probs[i, b]) or topk_options_probs[i, b].item() < min_triple_prob:
                    continue
                current_choice = option_tokens_list[topk_options_indices[i, b].item()]
                # 如果模型给的选项超过了当前可能的选项的时候跳过, 如果candidate_triples一共有两个，那么option就是["0", "1", "2"] 
                if current_choice != '0' and int(current_choice) > len(candidate_triples_indices_for_current_chain):
                # if current_choice != 'A' and self.choices_to_indices[current_choice] >= len(candidate_triples_indices_for_current_chain):
                    continue
                new_chains_scores.append(chains_scores[i]*topk_options_probs[i, b].item())
                if current_choice == "0":
                # if current_choice == "A":
                    new_chains.append(chains[i])
                    new_chains_finished.append(True)
                    new_chains_triple_filter_scores.append(chains_triple_filter_scores[i])
                    new_chains_history.append(
                        {
                            "question": question, 
                            "existing_triples": existing_triples[i], 
                            "candidate_triples": candidate_triples_for_current_chain, 
                            "selected_triple_idx": -1, # -1表示没有选择candidate_triples中的任意一个triple
                        }
                    )
                else:
                    # -1是因为当choice为1的时候，实际上是candidate_triples_indices_for_current_chain中index为0的triple
                    new_chains.append(
                        chains[i]+[all_triples[candidate_triples_indices_for_current_chain[int(current_choice)-1]]]
                    )
                    # new_chains.append(
                    #     chains[i]+[all_triples[candidate_triples_indices_for_current_chain[self.choices_to_indices[current_choice]]]]
                    # )
                    new_chains_finished.append(False)
                    new_chains_triple_filter_scores.append(chains_triple_filter_scores[i]+[candidate_triples_scores_for_current_chain[int(current_choice)-1]])
                    new_chains_history.append(
                        {
                            "question": question, 
                            "existing_triples": existing_triples[i], 
                            "candidate_triples": candidate_triples_for_current_chain, 
                            "selected_triple_idx": int(current_choice)-1,
                        }
                    )
        
        assert len(new_chains) == len(new_chains_scores)
        assert len(new_chains) == len(new_chains_finished)
        assert len(new_chains) == len(new_chains_triple_filter_scores)
        assert len(new_chains) == len(new_chains_history)

        new_chains_sorted_indices = sorted(range(len(new_chains_scores)), key=lambda x: new_chains_scores[x], reverse=True)
        topk_new_chains_sorted_indices = new_chains_sorted_indices[:max_num_chains]

        chains = [new_chains[idx] for idx in topk_new_chains_sorted_indices]
        chains_scores = [new_chains_scores[idx] for idx in topk_new_chains_sorted_indices]
        chains_finished = [new_chains_finished[idx] for idx in topk_new_chains_sorted_indices]
        chains_triple_filter_scores = [new_chains_triple_filter_scores[idx] for idx in topk_new_chains_sorted_indices]
        chains_history = [new_chains_history[idx] for idx in topk_new_chains_sorted_indices]

        results = []
        for chain, chain_score, chain_finished, chain_triple_filter_score, chain_history in \
            zip(chains, chains_scores, chains_finished, chains_triple_filter_scores, chains_history):
            result_item = {"triples": chain, "score": chain_score, "finished": chain_finished}
            if return_triple_filter_scores:
                assert len(chain) == len(chain_triple_filter_score) # chain and chain_triple_filter_score is not of equal length
                result_item["triple_filter_scores"] = chain_triple_filter_score
            if return_chain_history:
                result_item["chain_history"] = chain_history
            results.append(result_item)
        
        return results
    
    def forward_wo_selector(
        self, 
        question: str, 
        documents: List[Dict[str, Union[str, List]]],
        existing_reasoning_chains: Optional[List[Dict[str, Union[List, float, bool]]]]=None,
        num_beams: int=5, 
        max_num_chains: Optional[int]=None, # 设置为None返回所有可能的chains
        return_triple_filter_scores: Optional[bool]=False, 
        **kwargs, 
    ):
        if existing_reasoning_chains is not None and len(existing_reasoning_chains) == 0:
            existing_reasoning_chains = None
        chains, chains_scores, chains_finished = self.parse_reasoning_chains(existing_reasoning_chains)
        if existing_reasoning_chains is not None and np.sum(chains_finished) == len(existing_reasoning_chains):
            return existing_reasoning_chains
        
        # 得到document中所有的candidate triples，如果self.use_sentence为True的话会把sentence也当做triples
        all_triples = self.get_candidate_triples_from_documents(documents)
        ############ 添加一个EOC triple ############
        # all_triples.append({"title":"", "text": "<END>", "reference": None})
        ###########################################

        # 使用triple_filter_retriever(和triple_filter_reranker)对triples进行排序
        candidate_triples_indices, candidate_triples_scores = \
            self.filter_candidate_triples(question, chains, all_triples, self.num_candidate_triples)
        candidate_triples_indices_wo_duplicates, candidate_triples_scores_wo_duplicates = \
            self.filter_duplicate_triples(chains, all_triples, candidate_triples_indices, candidate_triples_scores)

        chains_triple_filter_scores = self.parse_reasoning_chains_triple_filter_scores(existing_reasoning_chains)

        new_chains, new_chains_scores, new_chains_finished = [], [], []
        new_chains_triple_filter_scores = [] 
        for i in range(len(chains)):
            # 当前路径已经结束或者没有candidate triples的时候
            if chains_finished[i] or len(candidate_triples_indices_wo_duplicates[i]) == 0:
                new_chains.append(chains[i])
                new_chains_scores.append(chains_scores[i])
                new_chains_finished.append(True)
                new_chains_triple_filter_scores.append(chains_triple_filter_scores[i])
                continue 
            for j in range(num_beams):
                chain_candidate_triple_idx = candidate_triples_indices_wo_duplicates[i][j]
                chain_candidate_triple_score = candidate_triples_scores_wo_duplicates[i][j]
                if chain_candidate_triple_score <= -1e4:
                    break
                ################## 没有EOC Triple #########################
                new_chains_scores.append(chains_scores[i]+chain_candidate_triple_score)
                new_chains.append(chains[i]+[all_triples[chain_candidate_triple_idx]])
                new_chains_finished.append(False)
                new_chains_triple_filter_scores.append(chains_triple_filter_scores[i]+[chain_candidate_triple_score])
                ###########################################################
                ################## 添加一个EOC triple#######################
                # num = len(chains[i])
                # new_chains_scores.append(((chains_scores[i]*num)+chain_candidate_triple_score)/(num+1))# 计算avg score
                # if all_triples[chain_candidate_triple_idx]["text"] == "<END>":
                #     new_chains.append(chains[i])
                #     new_chains_finished.append(True) 
                # else:
                #     new_chains.append(chains[i]+[all_triples[chain_candidate_triple_idx]])
                #     new_chains_finished.append(False)
                ###########################################
        
        assert len(new_chains) == len(new_chains_scores)
        assert len(new_chains) == len(new_chains_finished)
        assert len(new_chains) == len(new_chains_triple_filter_scores)

        new_chains_sorted_indices = sorted(range(len(new_chains_scores)), key=lambda x: new_chains_scores[x], reverse=True)
        topk_new_chains_sorted_indices = new_chains_sorted_indices[:max_num_chains]

        chains = [new_chains[idx] for idx in topk_new_chains_sorted_indices]
        chains_scores = [new_chains_scores[idx] for idx in topk_new_chains_sorted_indices]
        chains_finished = [new_chains_finished[idx] for idx in topk_new_chains_sorted_indices]
        chains_triple_filter_scores = [new_chains_triple_filter_scores[idx] for idx in topk_new_chains_sorted_indices]

        results = []
        for chain, chain_score, chain_finished in zip(chains, chains_scores, chains_finished):
            result_item = {"triples": chain, "score": chain_score, "finished": chain_finished}
            if return_triple_filter_scores:
                result_item["triple_filter_scores"] = chains_triple_filter_scores
            results.append(result_item)

        return results


class KiRAG(nn.Module):

    def __init__(
        self, 
        retriever: DenseRetriever,
        kg_generator: KGGenerator, 
        constructor: Generator,
        examplar_type: str="hotpotqa", 
        num_examplars: int=5, 
        adaptive_examplars: bool=True, 
        adaptive_examplars_retriever: str="e5", 
        aligner_model: str="e5", 
        aligner_model_name_or_path: str = None, 
        num_turns: int=5, 
        topk: int=10,
        num_candidate_triples: int=20, 
        maximum_possible_choices: int=100, 
        use_title_in_triples: bool=False,
        **kwargs,
    ):
        super().__init__()

        assert examplar_type in ["hotpotqa", "2wikimultihopqa", "musique", "nq", "tqa", "webqa", "bamboogle", "wikipedia"]

        self.retriever = retriever
        self.kg_generator = kg_generator
        self.constructor = constructor
        self.device = self.constructor.device
        self.tokenizer = self.constructor.tokenizer
        self.max_length = self.constructor.max_length
        self.num_examplars = num_examplars
        self.adaptive_examplars = adaptive_examplars
        self.adaptive_examplars_retriever = adaptive_examplars_retriever

        self.reasoning_chain_examplars, self.triple_selection_examplars = self.load_examplars(examplar_type)
        if self.adaptive_examplars:
            self.examplars_embeddings = self.calculate_examplars_embeddings()
        self.aligner = self.load_aligner(aligner_model, aligner_model_name_or_path)

        self.num_turns = num_turns
        self.topk = topk 
        self.num_candidate_triples = num_candidate_triples
        self.maximum_possible_choices = maximum_possible_choices
        self.use_title_in_triples = use_title_in_triples
        self.kwargs = kwargs

        self.task_instruction = "Select the next knowledge triple that extends an existing set of knowledge triples to form a coherent reasoning path capable of answering a specified question. " \
            "If the current reasoning path is sufficient to answer the question, simply output 0. Please only output the choice for the next knowledge triple."

    def load_examplars(self, examplar_type):

        print(f"loading {examplar_type} examplars for Reasoning Chain Constructor ... ")
        if examplar_type in ["hotpotqa", "nq", "tqa", "webqa", "bamboogle", "wikipedia"]:
            from prompts.kg_selection.hotpotqa_demonstrations import reasoning_chains_hotpotqa_examplars, triple_selection_hotpotqa_examplars
            reasoning_chain_examplars = reasoning_chains_hotpotqa_examplars
            triple_selection_examplars = triple_selection_hotpotqa_examplars
        elif examplar_type == "2wikimultihopqa":
            from prompts.kg_selection.wikimultihopqa_demonstrations import reasoning_chains_2wikimultihopqa_examplars, triple_selection_2wikimultihopqa_examplars
            reasoning_chain_examplars = reasoning_chains_2wikimultihopqa_examplars
            triple_selection_examplars = triple_selection_2wikimultihopqa_examplars
        elif examplar_type == "musique":
            from prompts.kg_selection.musique_demonstrations import reasoning_chains_musique_examplars, triple_selection_musique_examplars
            reasoning_chain_examplars = reasoning_chains_musique_examplars
            triple_selection_examplars = triple_selection_musique_examplars
        else:
            raise KeyError(f"{examplar_type} is not a supported examplar type!")
        
        return reasoning_chain_examplars, triple_selection_examplars

    def calculate_retriever_query_embeddings(self, retriever: str, queries: List[str], max_length: int=128):
        if retriever == "e5":
            queries_embeddings = get_e5_embeddings_for_query(queries, max_length=max_length)
        else:
            raise NotImplementedError(f"{retriever} is not a supported retriever!")
        return queries_embeddings
    
    def calculate_retriever_document_embeddings(self, retriever: str, documents: List[str], max_length: int=256):
        if retriever == "e5":
            documents_embeddings = get_e5_embeddings_for_document(documents, max_length=max_length)
        else:
            raise NotImplementedError(f"{retriever} is not a supported retriever!")
        return documents_embeddings
    
    def calculate_examplars_embeddings(self):

        questions = [example["question"] for example in self.reasoning_chain_examplars]
        questions_embeddings = self.calculate_retriever_query_embeddings(
            self.adaptive_examplars_retriever, questions, max_length=128
        )
        return questions_embeddings
    
    def load_aligner(self, aligner_name: str, aligner_path: Optional[str]=None) -> DenseRetriever:

        if "e5" in aligner_name.lower():
            default_aligner_path = "intfloat/e5-large-v2"
        elif "bge" in aligner_name.lower():
            default_aligner_path = "BAAI/bge-large-en-v1.5"
        else:
            raise NotImplementedError(f"The aligner with backbone \"{aligner_name}\" is not implemented!")
        
        if aligner_path is None:
            aligner_path = default_aligner_path
        tokenizer = AutoTokenizer.from_pretrained(default_aligner_path)

        query_maxlength, doc_maxlength = 256, 64 
        if "e5" in aligner_name.lower():
            collator = E5Collator(tokenizer=tokenizer, query_maxlength=query_maxlength, doc_maxlength=doc_maxlength)
            print(f"loading E5 Aligner from {aligner_name}...")
            aligner = BaseRetriever(retriever_name="E5Retriever", model_name_or_path=aligner_path)
        elif "bge" in aligner_name.lower():
            collator = BGECollator(tokenizer=tokenizer, query_maxlength=query_maxlength, doc_maxlength=doc_maxlength)
            print(f"loading BGE Aligner from {aligner_name}...")
            aligner = BaseRetriever(retriever_name="BGERetriever", model_name_or_path=aligner_path)
        else:
            raise NotImplementedError(f"The aligner with backbone \"{aligner_name}\" is not implemented!")
        
        aligner.to(self.device)
        aligner.eval()

        dense_retriever = DenseRetriever(retriever=aligner, collator=collator)

        return dense_retriever 

    def rank_examplars(self, question):
        if not self.adaptive_examplars:
            return list(range(len(self.reasoning_chain_examplars)))
        question_embedding = self.calculate_retriever_query_embeddings(
            self.adaptive_examplars_retriever, [question], max_length=128
        )
        similarities = torch.matmul(question_embedding, self.examplars_embeddings.T)
        indices = torch.argsort(similarities, dim=1, descending=True).tolist()[0]
        return indices
    
    def update_retrieved_documents(self, docids_to_scores: dict, retrieved_documents: List[List[dict]]):

        for one_retrieval_results in retrieved_documents:
            for doc in one_retrieval_results:
                docid, doc_score = doc["id"], doc["score"]
                docids_to_scores[docid] = max(docids_to_scores.get(docid, -1e9), doc_score)
        
        return docids_to_scores

    def get_candidate_triples_from_documents(self, documents: List[Dict[str, Union[str, List]]]):

        triples = [] 
        for doc in documents:
            doc_id = doc["id"]
            title = doc["title"]
            for one_triple_in_doc in doc["triples"]:
                triple = {
                    "title": title,
                    "text": one_triple_in_doc["text"],
                    "reference": [doc_id, one_triple_in_doc["sentence"]]
                }
                triples.append(triple)
        
        return triples

    def get_triple_text(self, triple: Dict[str, Union[str, List]]):
        if self.use_title_in_triples:
            triple_text = "title: {}, text: {}".format(triple["title"], triple["text"])
        else:
            triple_text = triple["text"]
        return triple_text
    
    def get_reasoning_chains_texts(self, reasoning_chains: List[List[Dict[str, Union[str, List]]]]):
        if len(reasoning_chains) == 0:
            return [[]]
        reasoning_chains_texts = [[self.get_triple_text(triple) for triple in chain] for chain in reasoning_chains]
        return reasoning_chains_texts
    
    def update_retrieved_triples(
        self, 
        id2score: Dict[str, float], 
        id2triple: Dict[str, dict], 
        triples: List[dict], 
        triples_indices: List[List[int]], 
        triples_scores: List[List[float]]
    ) -> Tuple[Dict[str, float], Dict[str, dict]]:
        
        if triples_scores is None:
            triples_scores = [[1.0]*len(indices) for indices in triples_indices]
        
        for indices, scores in zip(triples_indices, triples_scores):
            for idx, score in zip(indices, scores):
                triple = triples[idx]
                triple_id = hash_object(triple)[:20]
                id2score[triple_id] = max(id2score.get(triple_id, -1e9), score)
                id2triple[triple_id] = triple
        
        return id2score, id2triple
    
    def update_reasoning_chains_triples_based_on_string(
        self, 
        id2score: Dict[str, float], 
        id2triple: Dict[str, dict], 
        triples: List[str], 
        **kwargs
    ):
        if len(triples) == 0:
            return id2score, id2triple
        
        def equal(s1, s2):
            return s1.strip().lower() == s2.strip().lower()
        
        def compare_triple(t1, t2):
            try:
                t1_head, t1_relation, t1_tail = t1.replace("<", "").replace(">", "").split(";")
                t2_head, t2_relation, t2_tail = t2.replace("<", "").replace(">", "").split(";")
            except:
                return False
            if equal(t1_head, t2_head) and equal(t1_relation, t2_relation) and equal(t1_tail, t2_tail):
                return True 
            return False
        
        def find_triple_id(triple):
            for triple_id, existing_triple_item in id2triple.items():
                if compare_triple(triple, existing_triple_item["text"]):
                    return triple_id
            return None 

        for triple in triples:
            parsed_triple = self.kg_generator.parse_triples_text(triple) # 检测是否有triples
            if len(parsed_triple) == 0:
                continue
            triple_id = find_triple_id(parsed_triple[0])
            if triple_id is not None:
                print("Found Triple in id2triple: {} ==> {}".format(parsed_triple, id2triple[triple_id]))
                id2score[triple_id] += 0.5 
        
        return id2score, id2triple
        
    def update_reasoning_chains_triples_based_on_similarity(
        self, 
        id2score: Dict[str, float], 
        id2triple: Dict[str, dict], 
        triples: List[str], 
        index2id: Dict[int, str], 
        embeddings: Tensor, 
        **kwargs
    ):
        current_triple_ids_with_embeddings = set(index2id.values())
        triples_texts_wo_embeddings = [] 
        for triple_id, triple in id2triple.items():
            if triple_id in current_triple_ids_with_embeddings:
                continue
            index = len(index2id)
            index2id[index] = triple_id
            triples_texts_wo_embeddings.append(triple["text"])
        
        if len(triples_texts_wo_embeddings) > 0:
            triples_embeddings = self.calculate_retriever_document_embeddings(
                self.adaptive_examplars_retriever, triples_texts_wo_embeddings, max_length=64
            )
            if embeddings is not None:
                embeddings = torch.cat([embeddings, triples_embeddings], dim=0)
            else:
                embeddings = triples_embeddings

        assert len(index2id) == len(embeddings)
        triples_embeddings = self.calculate_retriever_document_embeddings(
            self.adaptive_examplars_retriever, triples, max_length=64, 
        )
        similarities = torch.matmul(triples_embeddings, embeddings.T)
        indices = similarities.argmax(1).tolist()
        for index, triple in zip(indices, triples):
            if len(triple) == 0:
                continue
            if "answer is: yes" in triple.lower():
                continue
            if "answer is: no" in triple.lower():
                continue
            matched_triple_id = index2id[index]
            print("Found triples: {} ==> {}\n".format(triple, id2triple[matched_triple_id]))
            id2score[matched_triple_id] += 0.5 
        
        return id2score, id2triple, index2id, embeddings

    def update_reasoning_chains_triples_based_on_f1_score(
        self, 
        id2score: Dict[str, float], 
        id2triple: Dict[str, dict], 
        triples: List[str], 
        **kwargs
    ):
        if len(triples) == 0:
            return id2score, id2triple
            
        sorted_triple_ids = sorted(list(id2score.keys()), key=lambda x: id2score[x], reverse=True)
        for triple in triples:
            if len(triple) == 0 or "answer is: yes" in triple.lower() or "answer is: no" in triple.lower():
                continue
            if len(self.kg_generator.parse_triples_text(triple)) > 0:
                scores = [f1_score(triple, id2triple[triple_id]["text"])[0] for triple_id in sorted_triple_ids]
                if max(scores) <0.6:
                    continue
                index = scores.index(max(scores))
                triple_id = sorted_triple_ids[index]
                id2score[triple_id] += 0.5 
            if "answer is:" in triple:
                answer = triple.split("answer is:")[1].strip()
                for triple_id in sorted_triple_ids:
                    if answer.lower() in id2triple[triple_id]["text"].lower():
                        id2score[triple_id] += 0.5 
                        break
        
        return id2score, id2triple
              
    def get_triples(self, id2score: Dict[str, float], id2triple: Dict[str, dict]) -> List[dict]:

        sorted_tid_list = sorted(id2score.items(), key=lambda x: x[1], reverse=True)
        triples = [id2triple[tid] for tid, _ in sorted_tid_list]
        return triples
    
    def get_docids_to_scores_from_triples(self, id2score: Dict[str, float], id2triple: Dict[str, dict], max_num_docs: int=None) -> Dict[str, float]:

        docids_to_scores = {} 
        for tid, score in id2score.items():
            triple = id2triple[tid]
            docid = triple["reference"][0]
            docids_to_scores[docid] = max(docids_to_scores.get(docid, -1e9), score)
        sorted_docids_scores = sorted(docids_to_scores.items(), key=lambda x: x[1], reverse=True)
        if max_num_docs:
            sorted_docids_scores = sorted_docids_scores[:max_num_docs]
        sorted_docids_to_scores = {docid: score for docid, score in sorted_docids_scores}

        return sorted_docids_to_scores
    
    def filter_candidate_triples(
        self, 
        question: str, 
        reasoning_chains: List[List[Dict[str, Union[str, List]]]], 
        triples: List[Dict[str, Union[str, List]]],
        num_candidate_triples: int,
    ):
        num_triples = len(triples)

        reasoning_chains_texts = self.get_reasoning_chains_texts(reasoning_chains)
        queries = []
        for texts in reasoning_chains_texts:
            query = "{}\nknowledge triples: {}.".format(question, ". ".join(texts))
            queries.append(query)
        queries_embeddings = self.aligner.calculate_query_embeddings(queries=queries, max_length=256)

        triples_texts = [self.get_triple_text(triple) for triple in triples]
        triples_embeddings = self.aligner.calculate_document_embeddings(documents=triples_texts, max_length=128)
        queries_triples_similarities = torch.matmul(queries_embeddings, triples_embeddings.T)

        topk_relevant_triples_scores, topk_relevant_triples_indices = torch.topk(
            queries_triples_similarities, 
            k = min(num_candidate_triples, num_triples),
            dim=1
        )
        topk_relevant_triples_indices = topk_relevant_triples_indices.tolist()
        topk_relevant_triples_scores = topk_relevant_triples_scores.tolist()

        return topk_relevant_triples_indices, topk_relevant_triples_scores

    def get_constructor_inputs(self, question, triples, ranked_examplars_indices):

        instruction = "Follow the examples to answer the input question by reasoning step-by-step. Output both reasoning steps and the answer."

        if self.num_examplars > 0:
            examplars = [] 
            for index in ranked_examplars_indices[:self.num_examplars]:
                item = self.reasoning_chain_examplars[index]
                thought = ". ".join([triple.strip() for triple in item["chains"].split(",")])
                examplar = "Question: {}\nThought: {}. So the answer is: {}".format(item["question"], thought, item["answer"]) 
                examplars.append(examplar)
            instruction += ("\n\nExamples:\n\n" + "\n\n".join(examplars))
        
        user_input = "\n".join([self.get_triple_text(triple) for triple in triples])
        user_input += "\n\nQuestion: {}".format(question)
        user_input = user_input.strip()

        return [instruction], [user_input]
    
    def get_constructor_documents_inputs(self, question, documents, ranked_examplars_indices):

        instruction = "Follow the examples to answer the input question by reasoning step-by-step. Output both reasoning steps and the answer."

        if self.num_examplars > 0:
            examplars = [] 
            for index in ranked_examplars_indices[:self.num_examplars]:
                item = self.reasoning_chain_examplars[index]
                thought = ". ".join([triple.strip() for triple in item["chains"].split(",")])
                examplar = "Question: {}\nThought: {}. So the answer is: {}".format(item["question"], thought, item["answer"]) 
                examplars.append(examplar)
            instruction += ("\n\nExamples:\n\n" + "\n\n".join(examplars))
        
        user_input = "\n\n".join(
            [
                "Wikipedia Title: {}\n{}".format(
                    doc["title"], 
                    doc["text"] if "text" in doc else " ".join(doc["sentences"])
                ) 
                for doc in documents
            ]
        )
        user_input += "\n\nQuestion: {}".format(question)
        user_input = user_input.strip()

        return [instruction], [user_input]
    
    def constructor_generate(self, instructions, inputs, reasoning_chains):
        
        texts = ["Thought: " + ". ".join(chain) for chain in reasoning_chains]
        return self.constructor.generator_generate(instructions=instructions, inputs=inputs, current_generated_texts=texts)

    def one_retrieval(
        self, 
        question: str, 
        triples: List[dict], 
        reasoning_chains: List[List[str]], 
        documents: Optional[List[dict]]=None, 
        ranked_examplars_indices: Optional[List[int]]=None, 
    ):
        instructions, inputs = self.get_constructor_inputs(
            question=question, 
            triples=triples, 
            ranked_examplars_indices=ranked_examplars_indices
        )
        if documents is not None:
            document_instruction, document_input = self.get_constructor_documents_inputs(
                question = question, 
                documents = documents, 
                ranked_examplars_indices=ranked_examplars_indices
            )
            instructions.extend(document_instruction)
            inputs.extend(document_input)
        generated_token_ids, _ = self.constructor_generate(instructions, inputs, reasoning_chains)
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        generated_texts = [text.strip().lstrip(".,;").strip() for text in generated_texts]

        for chain, text in zip(reasoning_chains, generated_texts):
            generated_triples = self.kg_generator.parse_triples_text(text)
            if len(generated_triples) > 0:
                chain.append(generated_triples[0])
            else:
                if len(text) > 0:
                    chain.append(sent_tokenize(text)[0])
                else:
                    chain.append(text)
        return reasoning_chains
    
    def retrieve(self, question: str, num_beams: int=1, **kwargs):

        assert num_beams in [1, 2], "num_beams must be chosen from [1, 2]"
        ranked_examplars_indices = self.rank_examplars(question=question)
        reasoning_chains = [[] for _ in range(num_beams)]
        docids_to_scores = {}
        triple_ids_to_scores, triple_ids_to_triple = {}, {}

        for i in range(self.num_turns):

            if i > 0 and all([len(chain[-1])==0 for chain in reasoning_chains]):
                break
            queries = [question]*num_beams if i == 0 else \
                [question + " " + chain[-1] if len(chain)>0 else question for chain in reasoning_chains]
            retrieved_documents = self.retriever(queries, topk=self.topk)
            docids_to_scores = self.update_retrieved_documents(docids_to_scores, retrieved_documents)
            documents = self.retriever.get_documents(docids_to_scores)
            documents_with_kgs = self.kg_generator(documents)
            all_triples = self.get_candidate_triples_from_documents(documents_with_kgs)

            chains = [[{"title": "", "text": triple} for triple in chain] for chain in reasoning_chains]
            candidate_triples_indices, candidate_triples_scores = self.filter_candidate_triples(
                question=question, reasoning_chains=chains, triples=all_triples, num_candidate_triples=self.num_candidate_triples
            )

            triple_ids_to_scores, triple_ids_to_triple = self.update_retrieved_triples(
                triple_ids_to_scores, triple_ids_to_triple, all_triples, candidate_triples_indices, candidate_triples_scores
            )
            candidate_triples = self.get_triples(triple_ids_to_scores, triple_ids_to_triple)
            candidate_triples = candidate_triples[:self.maximum_possible_choices] 
            reasoning_chains = self.one_retrieval(
                question=question, 
                triples=candidate_triples, 
                reasoning_chains=reasoning_chains, 
                documents=documents if num_beams==2 else None, 
                ranked_examplars_indices=ranked_examplars_indices,
            )
            reasoning_chains_triples = [] 
            for chain in reasoning_chains:
                if len(chain) > i:
                    reasoning_chains_triples.append(chain[i])
            triple_ids_to_scores, triple_ids_to_triple = self.update_reasoning_chains_triples_based_on_f1_score(
                triple_ids_to_scores, triple_ids_to_triple, reasoning_chains_triples,
            )
        docids_to_scores = self.get_docids_to_scores_from_triples(triple_ids_to_scores, triple_ids_to_triple)
        return reasoning_chains, docids_to_scores