import argparse
import numpy as np 
from tqdm import tqdm
from utils.utils import load_json
from evaluation.metrics import has_answer


def setup_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, required=True, help="the name of the dataset")
    parser.add_argument("--save_file", type=str, required=True, help="the saved retrieval result file")
    parser.add_argument("--qrels", type=str, help="the qrels for multi-hop QA datasets")
    parser.add_argument("--k", type=int, default=3, help="top-k documents")
    args = parser.parse_args()
    return args 


def load_tsv_file(file):
    results = {}
    with open(file, encoding="utf-8", mode="r") as fin:
        for line in fin:
            qid, docno, score = line.strip().split("\t")
            if qid not in results:
                results[qid] = [] 
            results[qid].append((docno, float(score)))
    for key, item in results.items():
        item.sort(key=lambda x: x[1], reverse=True)
    return results


def evaluate_retrieval_performance_with_qrels(args):

    retrieval_results = load_json(args.save_file)
    qrels = load_tsv_file(args.qrels)

    num_documents_at_k, precision_at_k, recall_at_k, f1_at_k = [], [], [], [] 
    for example in tqdm(retrieval_results, desc="Calculating Retrieval Metrics"):

        qid = example["id"]
        ctxs = example["ctxs"]
        if len(ctxs) == 0:
            continue
        # the documents are already ranked
        topk_ranked_document_ids = [ctx["id"] for ctx in ctxs[:args.k]]
        retrieved_docnos = set(topk_ranked_document_ids)
        qid_qrels = set([docno for docno, score in qrels[qid] if score > 0])
        true_positives = retrieved_docnos & qid_qrels
        if len(true_positives) == 0:
            precision, recall, f1 = 0.0, 0.0, 0.0 
        else:
            precision = len(true_positives) / len(retrieved_docnos)
            if len(qid_qrels) == 0:
                recall = 0.0 
            else:
                recall = len(true_positives) / len(qid_qrels)
            f1 = 2*precision*recall / (precision+recall)
        precision_at_k.append(precision)
        recall_at_k.append(recall)
        f1_at_k.append(f1)
        num_documents_at_k.append(len(topk_ranked_document_ids))
    
    metrics = {}
    metrics["Precision@{}".format(args.k)] = np.mean(precision_at_k)
    metrics["Recall@{}".format(args.k)] = np.mean(recall_at_k)
    metrics["F1@{}".format(args.k)] = np.mean(f1_at_k)
    metrics["NumDoc@{}".format(args.k)] = np.mean(num_documents_at_k)
    return metrics


def evaluate_retrieval_performance(args):

    retrieval_results = load_json(args.save_file)
    has_answers_at_k = [] 
    for example in tqdm(retrieval_results, desc="Calculating Retrieval Metrics"):
        answers = example["answers"]
        ctxs = example["ctxs"]
        ctx_has_answers = 0.0 
        for ctx in ctxs[:args.k]:
            relevant = has_answer(answers, "title: {} text: {}".format(ctx["title"], ctx["text"]))
            if relevant:
                ctx_has_answers = 1.0 
                break
        has_answers_at_k.append(ctx_has_answers)

    metrics = {} 
    metrics["Recall@{}".format(args.k)] = np.mean(has_answers_at_k)

    return metrics



if __name__ == "__main__":
    
    args = setup_parser()
    if args.dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        metrics = evaluate_retrieval_performance_with_qrels(args)
    else:
        metrics = evaluate_retrieval_performance(args)
    print(metrics)