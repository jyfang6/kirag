import os
import glob 
import pickle
import logging
import argparse

from tqdm import tqdm
from retriever.index import Indexer

logger = logging.getLogger(__file__)


def setup_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--index_name", type=str, default="ip_indexer")
    parser.add_argument("--index_folder", type=str, default=None)
    parser.add_argument("--embedding_size", type=int, default=1024)
    args = parser.parse_args()

    return args 

def sort_embedding_files(files):
    files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    return files 

def build_faiss_index(args):

    indexer = Indexer(args.embedding_size, metric="inner_product")

    embedding_files = glob.glob(os.path.join(args.index_folder, "corpus_embeddings_*.pkl"))
    embedding_files = sort_embedding_files(embedding_files)
    passage_id_list_files = glob.glob(os.path.join(args.index_folder, "passage_id_list_*.pkl"))
    assert len(embedding_files) == len(passage_id_list_files)
    embedding_passage_id_files = []
    for file in embedding_files:
        embedding_end_passage_id_str = file.split(".")[0].split("_")[-1]
        for passage_id_file in passage_id_list_files:
            if embedding_end_passage_id_str in passage_id_file:
                embedding_passage_id_files.append((file, passage_id_file))
                break 

    for embedding_file, passage_id_file in tqdm(embedding_passage_id_files, desc="Build Index", total=len(embedding_passage_id_files)):
        embeddings = pickle.load(open(embedding_file, "rb"))
        passage_ids = pickle.load(open(passage_id_file, "rb"))
        indexer.index_data(passage_ids, embeddings.cpu().numpy())
            
    logger.info(f"Saving index to {args.index_folder} ... ")
    indexer.serialize(args.index_folder)

    for file in embedding_files + passage_id_list_files:
        os.remove(file)


if __name__ == "__main__":

    # 设置超参数
    args = setup_parser()

    # index 
    build_faiss_index(args)
