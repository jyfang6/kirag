import os
import torch 
import logging
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from knowledge_graph.kg_generator import KGGenerator
from knowledge_graph.models import KiRAG

from retriever.index import Indexer
from retriever.retrievers import InBatchRetriever, DenseRetriever
from generator.generator import Generator
from utils.const import COLLATOR_MAP, CORPUS_MAP
from utils.utils import gpu_setup, setup_logger, load_json, save_json
from utils.pipeline_utils import load_llm_tokenizer_and_model, get_retrieved_documents

logger = logging.getLogger(__file__)

device = torch.device("cuda")

def setup_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1)

    # dataset settings
    parser.add_argument("--dataset", required=True, type=str, help="dataset name")
    parser.add_argument("--query_file", required=True, type=str, help="query file")
    parser.add_argument("--corpus", type=str, default="2wikimultihopqa", help="the name of the corpus")
    parser.add_argument("--query_maxlength", type=int, default=512, help="the maxinum number of query tokens")
    parser.add_argument("--doc_maxlength", type=int, default=512, help="the maxinum number of doc tokens")
    parser.add_argument("--index_folder", type=str, default="checkpoint/e5_retriever/2wikimultihopqa", help="the path of the corpus index")
    parser.add_argument("--embedding_size", type=int, default=1024)

    # retriever settings
    parser.add_argument("--retriever_name", type=str, default="E5Retriever", choices=["E5Retriever", "BGERetriever"], help="the name of the retriever")
    parser.add_argument("--retriever_model_name_or_path", type=str, default="intfloat/e5-large-v2", help="the name or path of the retriever model")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="intfloat/e5-large-v2", help="the name or path of the tokenizer")

    parser.add_argument("--hf_token", type=str, required=True, help="the Huggingface Token used to access Llama3 model.")
    parser.add_argument("--llm", type=str, required=True, help="the LLM used in the Reasoning Chain Constructor.")
    parser.add_argument("--cached_kg_triples_file", type=str, default=None)
    parser.add_argument("--aligner_model", type=str, default="e5", choices=["e5", "bge"], help="the backbone of the Reasoning Chain Aligner model.")
    parser.add_argument("--aligner_model_name_or_path", type=str, required=True, help="The huggingface name or path of the Aligner model.")
    parser.add_argument("--num_beams", type=int, default=1, choices=[1, 2], help="number of reasoning chains. 1 denotes using only triples and 2 denotes using both triples and documents.")

    parser.add_argument("--per_gpu_batch_size", type=int, default=8)
    parser.add_argument("--save_dir", type=str, default="checkpoint")
    parser.add_argument("--name", type=str, default="e5_retriever")
    parser.add_argument("--save_file", required=True, type=str, help="save retrieval results")
    
    args = parser.parse_args()
    return args 


def convert_chains_to_qa_format(example, reasoning_chains, documents):

    ctxs = [] 
    document_ids_to_ranked_indices = {} 
    for i, doc in enumerate(documents):
        if "sentences" not in doc:
            doc["sentences"] = sent_tokenize(doc["text"])
        ctxs.append(doc)
        document_ids_to_ranked_indices[doc["id"]] = i
    
    example["ctxs"] = ctxs
    example["paths"] = reasoning_chains

    return example

def retrieve(args, questions, model, corpus_dataset):
    
    retrieval_results = []
    for example in tqdm(questions, total=len(questions), desc="Retrieval Progress"):
        question = example["question"]
        reasoning_chains, retrieved_documents_ids_to_scores = model.retrieve(question=question, num_beams=args.num_beams)
        retrieved_documents = get_retrieved_documents(retrieved_documents_ids_to_scores, corpus_dataset)
        retrieval_results.append(convert_chains_to_qa_format(example, reasoning_chains, retrieved_documents))

    if args.cached_kg_triples_file is not None:
        kirag.kg_generator.save_cached_kg_triples(args.cached_kg_triples_file)
    
    return retrieval_results

def setup_retriever_model(args):

    # load tokenizer
    retriever_name = args.retriever_name
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        logger.warning("Missing padding token, adding a new pad token!")
        tokenizer.add_special_tokens({"pad_token": '[PAD]'})

    # load collator
    collator = COLLATOR_MAP[retriever_name](tokenizer=tokenizer, query_maxlength=args.query_maxlength, doc_maxlength=args.doc_maxlength)

    # load retriever
    retriever = InBatchRetriever(
        retriever_name=retriever_name, 
        model_name_or_path=args.retriever_model_name_or_path, 
        local_rank=args.local_rank,
        temperature=0.01
    )

    # load dataset
    logger.info(f"Loading corpus from {args.corpus} ...")
    corpus_dataset = CORPUS_MAP[args.corpus](title_prefix="title: ", passage_prefix="text: ")

    # load index 
    logger.info(f"Loading index from {args.index_folder} ...")
    indexer = Indexer(args.embedding_size, metric="inner_product")
    indexer.deserialize_from(args.index_folder)

    # load zero-shot retriever
    dense_retriever = DenseRetriever(retriever=retriever, collator=collator, indexer=indexer, corpus=corpus_dataset, batch_size=args.per_gpu_batch_size)

    return dense_retriever, corpus_dataset

def setup_kirag_model(args):

    llm_tokenizer, llm_model = load_llm_tokenizer_and_model(args.llm, hf_token=args.hf_token, device=device)
    constructor = Generator(llm_tokenizer, llm_model, max_length=4096, max_new_tokens=64, batch_size=4)

    if args.llm == "llama3":
        kg_generator = KGGenerator(tokenizer=llm_tokenizer, generator=llm_model, examplar_type=args.dataset, batch_size=4)
    else:
        llama3_tokenizer, llama3_model = load_llm_tokenizer_and_model("llama3", device=device)
        kg_generator = KGGenerator(tokenizer=llama3_tokenizer, generator=llama3_model, examplar_type=args.dataset, batch_size=4)
    
    if args.cached_kg_triples_file is not None:
        kg_generator.load_cached_kg_triples(args.cached_kg_triples_file)
    
    kirag = KiRAG(
        retriever=dense_retriever,
        kg_generator=kg_generator, 
        constructor=constructor, 
        examplar_type=args.dataset,
        aligner_model=args.aligner_model, 
        aligner_model_name_or_path=args.aligner_model_name_or_path
    )

    return kirag


if __name__ == "__main__":

    args = setup_parser()

    gpu_setup(args.local_rank, random_seed=42)

    # logging
    checkpoint_path = os.path.join(args.save_dir, args.name)
    os.makedirs(checkpoint_path, exist_ok=True)
    setup_logger(args.local_rank, os.path.join(checkpoint_path, "kg_adaptive_retrieve.log"))

    # setup retriever model 
    dense_retriever, corpus_dataset = setup_retriever_model(args)

    # setup kirag model 
    kirag = setup_kirag_model(args)

    # 加载query的数据
    logger.info(f"loading query data from {args.query_file} ...")
    questions = load_json(args.query_file)[:5]

    # retrieve
    retrieval_results = retrieve(
        args,
        questions = questions,
        model = kirag,
        corpus_dataset = corpus_dataset,
    )

    # save results 
    save_file_path = os.path.join(checkpoint_path, args.save_file)
    logger.info(f"Writing retrieval results to {save_file_path} ...")
    save_json(retrieval_results, save_file_path, use_indent=True)
