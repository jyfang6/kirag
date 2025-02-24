import os
import glob 
import pickle
import logging
import argparse
import numpy as np
import torch.distributed as dist
from transformers import AutoTokenizer 

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
from dataset.datasets import KGChainRetrieverSeqSampleDataset
from trainer.aligner_trainer import AlignerTrainer


logger = logging.getLogger(__file__)

def setup_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)

    # dataset 
    parser.add_argument("--data_folders", nargs="+", help="the folders' of training data")
    parser.add_argument("--query_maxlength", type=int, default=256, help="the maxinum number of query tokens")
    parser.add_argument("--doc_maxlength", type=int, default=64, help="the maxinum number of doc tokens")

    # aligner setting
    parser.add_argument("--backbone", type=str, default="E5Retriever", choices=["E5Retriever", "BGERetriever"], help="the backbone model of the Aligner model")
    parser.add_argument("--backbone_model_name", type=str, default="intfloat/e5-large-v2")
    parser.add_argument("--backbone_tokenizer_name", type=str, default="intfloat/e5-large-v2")

    # trainer setting 
    parser.add_argument("--per_gpu_batch_size", type=int, default=8)
    parser.add_argument("--eval_per_gpu_batch_size", type=int, default=None)

    # experiment setting
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoint/", help="root folder for training")
    parser.add_argument("--name", type=str, default="reasoning_chain_aligner", help="the folder to save checkpoint, the checkpoint and logging will be saved to save_dir/name")
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--accumulate_grad_batches", type=int, default=2)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--warmup_steps", type=float, default=0.1)
    
    parser.add_argument("--val_every_n_steps", type=int, default=200)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_checkpoint_every_n_steps", type=int, default=10000)

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = setup_parser()
    gpu_setup(args.local_rank, args.seed)

    # setup logging
    checkpoint_path = os.path.join(args.save_dir, args.name)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path, exist_ok=True)
    setup_logger(args.local_rank, os.path.join(checkpoint_path, "trainer.log"))

    # load tokenizer
    model_name = args.backbone
    tokenizer = AutoTokenizer.from_pretrained(args.backbone_tokenizer_name)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.warning("Missing padding token, adding a new pad token!")
        tokenizer.add_special_tokens({"pad_token": '[PAD]'})

    # load collator 
    collator =  COLLATOR_MAP[model_name](
        tokenizer=tokenizer, query_maxlength=args.query_maxlength, doc_maxlength=args.doc_maxlength
    )

    # get dataset and dataloader
    batch_size = args.per_gpu_batch_size
    eval_batch_size = args.per_gpu_batch_size if args.eval_per_gpu_batch_size is None else args.eval_per_gpu_batch_size
    if not args.test_only:
        train_dataset = KGChainRetrieverSeqSampleDataset(is_train=True, data_folders = args.data_folders)
        dev_dataset = KGChainRetrieverSeqSampleDataset(is_train=False, data_folders = args.data_folders)
        train_dataloader = get_dataloader(args.local_rank, train_dataset, batch_size, shuffle=True, collate_fn=collator)
        dev_dataloader = get_dataloader(args.local_rank, dev_dataset, eval_batch_size, shuffle=False, collate_fn=collator)
    
    # load model 
    model = InBatchRetriever(
        retriever_name = args.backbone, 
        model_name_or_path = args.backbone_model_name, 
        local_rank = args.local_rank, 
        temperature = 0.01
    )

    # load trainer
    trainer =  AlignerTrainer(
        model=model,
        default_root_dir=checkpoint_path,
        learning_rate=args.learning_rate, 
        per_gpu_batch_size=args.per_gpu_batch_size, 
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        weight_decay=args.weight_decay, 
        warmup_steps=args.warmup_steps, 
        max_epochs=args.max_epochs, 
        val_every_n_steps=args.val_every_n_steps, 
        log_every_n_steps=args.log_every_n_steps,
        save_checkpoint_every_n_steps=args.save_checkpoint_every_n_steps, 
        seed = args.seed, 
        local_rank=args.local_rank, 
        world_size=dist.get_world_size() if args.local_rank >=0 else 1,
        bf16=True, 
        use_amp=True, 
        debug=True # supress wandb 
    )

    if not args.test_only:
        trainer.train(train_dataloader, dev_dataloader)
    
    if args.local_rank >= 0:
        dist.barrier()

    if args.local_rank >= 0:
        cleanup()
