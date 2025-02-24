import torch 
import logging 
import numpy as np 
from tqdm import tqdm

from trainer.base_trainer import BaseTrainer
from utils.utils import get_global_tensors, get_global_tensor_list

logger = logging.getLogger(__name__)


class RetrieverWithPosNegsTrainer(BaseTrainer):

    def save_model_checkpoint(self, ckpt_path):
        model = self.model.module if self.use_ddp else self.model 
        model.save_model(ckpt_path)
        logger.info(f"Rank: {self.local_rank}, Saving checkpoint to {ckpt_path}")
    
    def load_model_checkpoint(self, ckpt_path):
        logger.info(f"Rank: {self.local_rank}, Loading checkpoint from {ckpt_path}")
        model = self.model.module if self.use_ddp else self.model 
        model.load_model(ckpt_path)
        model = model.to(self.device)

    def training_step(self, model, batch):

        query_args, doc_args, positive_doc_indices, index = batch
        loss = model(query_args, doc_args, labels=positive_doc_indices)[0]

        return loss

    def evaluate_epoch_start(self, dataloader, **kwargs):

        if self.local_rank <= 0:
            logger.info("Start Evaluating!")
            progress_bar = tqdm(total=len(dataloader), desc="Calculating All Doc Embeddings")

        global_doc_embeddings_list = [] 
        self.global_index_to_pos_index = {}

        model = self.model.module if self.use_ddp else self.model
        for i, batch in enumerate(dataloader):
            batch = self.setup_batch(batch) # 移到GPU中
            query_args, doc_args, batch_pos_doc_indices, index = batch
            batch_doc_size = len(doc_args["input_ids"])
            local_doc_embeddings = model.doc(doc_args).contiguous()
            batch_global_embeddings = get_global_tensors(self.local_rank, self.world_size, local_doc_embeddings) 
            batch_global_index = get_global_tensor_list(self.local_rank, self.world_size, index)
            batch_global_pos_doc_index = get_global_tensor_list(self.local_rank, self.world_size, batch_pos_doc_indices)
            
            global_doc_embeddings_list.append(batch_global_embeddings.detach().cpu())
            for j, (one_local_index, one_local_global_pos_doc_index) in enumerate(zip(batch_global_index, batch_global_pos_doc_index)):
                assert len(one_local_index) == len(one_local_global_pos_doc_index)
                for idx, pos_doc_idx in zip(one_local_index.tolist(), one_local_global_pos_doc_index.tolist()):
                    self.global_index_to_pos_index[idx] = \
                        i * batch_doc_size * self.world_size + j * batch_doc_size + pos_doc_idx
            if self.local_rank <= 0:
                progress_bar.update(1)
        
        if self.local_rank <= 0:
            progress_bar.close()

        self.global_doc_embeddings = torch.cat(global_doc_embeddings_list, dim=0)


    def evaluate_step(self, model, batch, dataloader):

        query_args, _, _, index = batch
        batch_size = len(index)
        query_embeddings = model.query(query_args).detach().cpu()
        scores = model.score(query_embeddings, self.global_doc_embeddings)
        pos_doc_index = torch.tensor([self.global_index_to_pos_index[idx] for idx in index.detach().cpu().tolist()], dtype=torch.long)
        argsort_results = torch.argsort(scores, dim=-1, descending=True)
        rank = torch.empty_like(argsort_results)
        for i in range(len(argsort_results)):
            rank[i][argsort_results[i]] = torch.arange(1, argsort_results.shape[-1]+1).type_as(rank).to(argsort_results.device)
        pos_doc_rank = rank.gather(dim=1, index=pos_doc_index.unsqueeze(1)).squeeze(1)
        pos_doc_mean_reciprocal_rank = torch.reciprocal(pos_doc_rank.float()).mean()

        return pos_doc_mean_reciprocal_rank, batch_size
    

class AlignerTrainer(RetrieverWithPosNegsTrainer):

    def evaluate_epoch_start(self, dataloader, **kwargs):
        pass

    def evaluate_step(self, model, batch, dataloader):

        query_args, doc_args, positive_doc_indices, *others = batch 
        batch_size = 4 
        query_embeddings_list = [] 
        for i in range((len(query_args["input_ids"])-1) // batch_size+1):
            batch_query_args = {k: v[i*batch_size: (i+1)*batch_size] if torch.is_tensor(v) else v for k, v in query_args.items()}
            batch_query_embeddings = model.query(batch_query_args).detach().cpu()
            query_embeddings_list.append(batch_query_embeddings)
        query_embeddings = torch.cat(query_embeddings_list, dim=0)

        doc_embeddings_list = [] 
        for i in range((len(doc_args["input_ids"])-1)//batch_size+1):
            batch_doc_args = {k: v[i*batch_size: (i+1)*batch_size] if torch.is_tensor(v) else v for k, v in doc_args.items()}
            batch_doc_embeddings = model.doc(batch_doc_args).detach().cpu()
            doc_embeddings_list.append(batch_doc_embeddings)
        doc_embeddings = torch.cat(doc_embeddings_list, dim=0)

        positive_doc_indices = positive_doc_indices.tolist()
        reciprocal_rank_list = [] 
        for i in range(len(positive_doc_indices)):
            doc_start_idx = positive_doc_indices[i]
            doc_end_idx = positive_doc_indices[i+1] if i+1<len(positive_doc_indices) else len(doc_embeddings)
            query_specific_doc_embeddings = doc_embeddings[doc_start_idx:doc_end_idx]
            query_doc_scores = torch.matmul(query_embeddings[i:i+1], query_specific_doc_embeddings.T)
            positive_doc_rank = (torch.argsort(query_doc_scores[0], descending=True)==0).nonzero(as_tuple=True)[0][0] + 1 
            reciprocal_rank_list.append(torch.reciprocal(positive_doc_rank.float()).item())
        
        return np.mean(reciprocal_rank_list), len(reciprocal_rank_list)