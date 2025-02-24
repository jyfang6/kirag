import os
import torch
import pickle
import logging
import argparse
from tqdm import tqdm
import torch.distributed as dist
from transformers import AutoTokenizer 
from torch.nn.parallel import DistributedDataParallel as DDP

from utils.utils import (
    cleanup,
    to_device, 
    gpu_setup,
    setup_logger,
    get_dataloader,
    get_global_tensor_list_to_main_process
)
from utils.const import COLLATOR_MAP, CORPUS_MAP
from retriever.retrievers import InBatchRetriever


logger = logging.getLogger(__file__)

def setup_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)

    # dataset settings
    parser.add_argument("--corpus", type=str, default="2wikimultihopqa", help="the name of the corpus")
    parser.add_argument("--query_maxlength", type=int, default=512, help="the maxinum number of query tokens")
    parser.add_argument("--doc_maxlength", type=int, default=512, help="the maxinum number of doc tokens")

    # retriever settings
    parser.add_argument("--retriever_name", type=str, default="E5Retriever", choices=["E5Retriever", "BGERetriever"], help="the name of the retriever")
    parser.add_argument("--retriever_model_name_or_path", type=str, default="intfloat/e5-large-v2", help="the name or path of the retriever model")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="intfloat/e5-large-v2", help="the name or path of the tokenizer")

    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--name", type=str, default="e5_retriever")
    parser.add_argument("--index_folder", type=str, default="2wikimultihopqa")
    parser.add_argument("--per_gpu_batch_size", type=int, default=8)
    parser.add_argument("--num_passage_per_index_file", type=int, default=1000000)

    args = parser.parse_args()
    return args 


def cal_doc_embeddings(args, model, corpus_dataloader, collator):

    device = torch.device("cuda:0") if args.local_rank < 0 else torch.device(f"cuda:{args.local_rank}")

    # setup model 
    model = model.to(device)
    model.eval()

    if args.local_rank >= 0:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    main_process = (args.local_rank<=0)

    if main_process:
        progress_bar = tqdm(total=len(corpus_dataloader), desc="Calculating All Corpus Embeddings")
        save_embedding_folder = os.path.join(args.save_dir, args.name, args.index_folder)
        os.makedirs(save_embedding_folder, exist_ok=True)
    
    global_corpus_embeddings = None 
    num_saved_corpus_embeddings = 0 

    global_embedding_idx_to_corpus_idx = []
    global_passage_id_list = []

    unwrap_model = model.module if hasattr(model, "module") else model
    world_size = 1 if args.local_rank < 0 else dist.get_world_size()

    for i, batch in enumerate(corpus_dataloader):
        batch_local_corpus_idx = batch["index"].to(device).contiguous()
        batch_local_inputs = collator.encode_doc(batch["passage"])
        batch_local_doc_args = to_device(batch_local_inputs, device)
        batch_local_doc_embeddings = unwrap_model.doc(batch_local_doc_args).contiguous()

        batch_global_doc_embeddings_list = get_global_tensor_list_to_main_process(args.local_rank, world_size, batch_local_doc_embeddings)
        batch_global_corpus_idx_list = get_global_tensor_list_to_main_process(args.local_rank, world_size, batch_local_corpus_idx)
        
        if args.local_rank >= 0:
            dist.barrier()

        if main_process:

            batch_global_doc_embeddings = torch.cat([item.detach().cpu() for item in batch_global_doc_embeddings_list], dim=0)
            batch_global_corpus_index = torch.cat([item.detach().cpu() for item in batch_global_corpus_idx_list], dim=0).tolist()

            if global_corpus_embeddings is None:
                global_corpus_embeddings = batch_global_doc_embeddings
            else:
                global_corpus_embeddings = torch.cat([global_corpus_embeddings, batch_global_doc_embeddings], dim=0)
            global_embedding_idx_to_corpus_idx.extend(batch_global_corpus_index)
            global_passage_id_list.extend([corpus_dataloader.dataset.index_to_passage_id[index] for index in batch_global_corpus_index])

            if len(global_corpus_embeddings) >= args.num_passage_per_index_file or i == (len(corpus_dataloader) - 1):

                assert len(global_corpus_embeddings) == len(global_embedding_idx_to_corpus_idx)
                assert len(global_corpus_embeddings) == len(global_passage_id_list)
                
                corpus_idx_to_embedding_idx_map = {corpus_idx: embedding_idx for embedding_idx, corpus_idx in enumerate(global_embedding_idx_to_corpus_idx)}
                corpus_idx_start = num_saved_corpus_embeddings
                corpus_idx_end = min(num_saved_corpus_embeddings+len(global_corpus_embeddings), len(corpus_dataloader.dataset))
                indices = [corpus_idx_to_embedding_idx_map[idx] for idx in range(corpus_idx_start, corpus_idx_end)]
                new_global_corpus_embeddings = global_corpus_embeddings[indices]
                new_global_passage_id_list = [global_passage_id_list[idx] for idx in indices]

                logger.info(f"Finished calculating embeddings from {corpus_idx_start} to {corpus_idx_end-1}. Saving embeddings to {save_embedding_folder} ...")
                pickle.dump(new_global_corpus_embeddings, open(os.path.join(save_embedding_folder, f"corpus_embeddings_{corpus_idx_start}_{corpus_idx_end-1}.pkl"), "wb"))
                pickle.dump(new_global_passage_id_list, open(os.path.join(save_embedding_folder, f"passage_id_list_{corpus_idx_start}_{corpus_idx_end-1}.pkl"), "wb"))

                num_saved_corpus_embeddings += len(global_corpus_embeddings)
                global_corpus_embeddings = None 
                global_embedding_idx_to_corpus_idx = []
                global_passage_id_list = []

            progress_bar.update(1)
        
        if args.local_rank >= 0:
            dist.barrier()

    if main_process:
        progress_bar.close()
        
    if args.local_rank >= 0:
        dist.barrier()
    
    if args.local_rank >= 0:
        cleanup()


if __name__ == "__main__":

    args = setup_parser()
    gpu_setup(args.local_rank, random_seed=42)

    # setup logging
    checkpoint_path = os.path.join(args.save_dir, args.name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    setup_logger(args.local_rank, os.path.join(checkpoint_path, "cal_doc_embedding.log"))

    # load tokenizer
    retriever_name = args.retriever_name 
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.warning("Missing padding token, adding a new pad token!")
        tokenizer.add_special_tokens({"pad_token": '[PAD]'})

    # load collator
    collator = COLLATOR_MAP[retriever_name](tokenizer=tokenizer, query_maxlength=args.query_maxlength, doc_maxlength=args.doc_maxlength)

    # load dataset
    corpus_dataset = CORPUS_MAP[args.corpus](title_prefix="title: ", passage_prefix="text: ")
    corpus_dataloader = get_dataloader(args.local_rank, corpus_dataset, args.per_gpu_batch_size, shuffle=False, drop_last=False)

    # load retriever
    model = InBatchRetriever(
        retriever_name=retriever_name, 
        model_name_or_path=args.retriever_model_name_or_path, 
        local_rank=args.local_rank,
        temperature=0.01
    )

    cal_doc_embeddings(args, model, corpus_dataloader, collator)