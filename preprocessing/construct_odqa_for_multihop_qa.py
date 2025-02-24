import os 
import nltk 
import argparse
from tqdm import tqdm 
from collections import OrderedDict
from utils.utils import (
    load_json, 
    save_json, 
    hash_object, 
    load_tsv_str_format, 
    save_tsv, 
    seed_everything
)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab' tokenizer...")
    nltk.download('punkt_tab')

LOAD_DATA_FOLDER = {
    "hotpotqa": "/nfs/common/data/hotpotqa/raw_data", 
    "2wikimultihopqa": "/nfs/common/data/2wikimultihopqa/raw_data", 
    "musique": "/nfs/common/data/musique/raw_data", 
    "webqa": "/nfs/common/data/webqa/raw_data", 
    "bamboogle": "/nfs/common/data/bamboogle/raw_data", 
}

SAVE_DATA_FOLDER = {
    "hotpotqa": "/nfs/common/data/hotpotqa/open_domain_data", 
    "2wikimultihopqa": "/nfs/common/data/2wikimultihopqa/open_domain", 
    "musique": "/nfs/common/data/musique/open_domain_data_github", 
    "webqa": "/nfs/common/data/webqa/open_domain_data_github", 
    "bamboogle": "/nfs/common/data/bamboogle/open_domain_data", 
}

def setup_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--num_dev_data", type=int, default=500)
    args = parser.parse_args()
    return args


def load_2wikimultihopqa_corpus():

    print("loading corpus from 2wikimultihop train, dev, test sets ... ")
    corpus = OrderedDict()
    for file in ["train.json", "dev.json", "test.json"]:
        file = os.path.join(LOAD_DATA_FOLDER["2wikimultihopqa"], file)
        print(f"loading context from {file} ... ")
        data = load_json(file)
        for example in data:
            for title, sentences in example["context"]:
                doc_obj = {"title": title, "sentences": sentences}
                doc_hash_id = hash_object(doc_obj)
                if doc_hash_id in corpus:
                    continue
                corpus[doc_hash_id] = doc_obj
    
    print(f"Successfully load {len(corpus)} unique documents!")
    doc_hash_id_to_doc_id = {} 
    corpus_with_doc_ids = [] 
    for i, (doc_hash_id, doc_obj) in enumerate(corpus.items()):
        doc_id = str(i)
        doc_hash_id_to_doc_id[doc_hash_id] = doc_id
        corpus_with_doc_ids.append(
            {
                "id": doc_id, 
                "title": doc_obj["title"], 
                "sentences": doc_obj["sentences"]
            }
        )
    
    return doc_hash_id_to_doc_id, corpus_with_doc_ids


def convert_2wikimultihopqa_to_odqa_data(doc_hash_id_to_doc_id, corpus):

    qrels, orig_train_qa_pairs, orig_dev_qa_pairs = [], [], [] 
    print("Converting train, dev sets in 2wikimultihopqa into open domain QA data ...")
    for file in ["train.json", "dev.json"]:
        qa_pairs = orig_train_qa_pairs if "train" in file else orig_dev_qa_pairs
        file = os.path.join(LOAD_DATA_FOLDER["2wikimultihopqa"], file)
        print(f"loading data from {file} ... ")
        data = load_json(file)
        for example in tqdm(data, desc="convert to ODQA data"):
            qid = example["_id"]

            qrels_per_example, supporting_facts = [], []
            for sf_title, sf_sentence_idx in example["supporting_facts"]:
                relevant_doc = None 
                for title, sentences in example["context"]:
                    if sf_title.strip().lower() == title.strip().lower():
                        relevant_doc = {"title": title, "sentences": sentences}
                        break
                assert relevant_doc is not None
                relevant_doc_hash_id = hash_object(relevant_doc)
                relevant_doc_id = doc_hash_id_to_doc_id[relevant_doc_hash_id]
                supporting_facts.append((relevant_doc_id, sf_sentence_idx))
                qrel_item = (qid, relevant_doc_id, 1)
                if qrel_item not in qrels_per_example:
                    qrels_per_example.append(qrel_item)
            
            qrels.extend(qrels_per_example)
            qa_pairs.append(
                {
                    "id": qid, 
                    "question": example["question"], 
                    "answers": [example["answer"]],
                    "supporting_facts": supporting_facts, 
                }
            )
    
    # data splits 
    import numpy as np 
    seed_everything(0)
    indices = np.random.permutation(len(orig_train_qa_pairs))
    train_qa_pairs = [orig_train_qa_pairs[idx] for idx in indices[:-args.num_dev_data]]
    dev_qa_pairs = [orig_train_qa_pairs[idx] for idx in indices[-args.num_dev_data:]]
    test_qa_pairs = orig_dev_qa_pairs

    return qrels, train_qa_pairs, dev_qa_pairs, test_qa_pairs


def load_musique_corpus():

    from nltk.tokenize import sent_tokenize

    print("loading corpus from musique train, dev, test sets ... ")

    corpus = OrderedDict()
    for file in ["musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl", "musique_ans_v1.0_test.jsonl"]:
        file = os.path.join(LOAD_DATA_FOLDER["musique"], file)
        print(f"loading context from {file} ... ")
        data = load_json(file, type="jsonl")
        for example in data:
            for context_item in example["paragraphs"]:
                sentences = sent_tokenize(context_item["paragraph_text"])
                doc_obj = {"title": context_item["title"], "sentences": sentences}
                doc_hash_id = hash_object(doc_obj)
                if doc_hash_id in corpus:
                    continue
                corpus[doc_hash_id] = doc_obj
    
    print(f"Successfully load {len(corpus)} unique documents!")
    doc_hash_id_to_doc_id = {} 
    corpus_with_doc_ids = [] 
    for i, (doc_hash_id, doc_obj) in enumerate(corpus.items()):
        doc_id = str(i)
        doc_hash_id_to_doc_id[doc_hash_id] = doc_id
        corpus_with_doc_ids.append(
            {
                "id": doc_id, 
                "title": doc_obj["title"], 
                "sentences": doc_obj["sentences"]
            }
        )
    
    return doc_hash_id_to_doc_id, corpus_with_doc_ids


def convert_musique_to_odqa_data(doc_hash_id_to_doc_id, corpus):

    from nltk.tokenize import sent_tokenize

    qrels, orig_train_qa_pairs, orig_dev_qa_pairs = [], [], [] 
    print("Converting train, dev sets in MuSiQue into open domain QA data ...")
    for file in ["musique_ans_v1.0_train.jsonl", "musique_ans_v1.0_dev.jsonl"]:
        qa_pairs = orig_train_qa_pairs if "train" in file else orig_dev_qa_pairs
        file = os.path.join(LOAD_DATA_FOLDER["musique"], file)
        print(f"loading data from {file} ... ")
        data = load_json(file, type="jsonl")
        for example in tqdm(data, desc="convert to ODQA data"):
            qid = example["id"]
            qrels_per_example, supporting_facts = [], [] 
            for question_decomposition_item in example["question_decomposition"]:
                sf_idx = question_decomposition_item["paragraph_support_idx"]
                sf_title = example["paragraphs"][sf_idx]["title"]
                sf_sentences = sent_tokenize(example["paragraphs"][sf_idx]["paragraph_text"])
                relevant_doc = {"title": sf_title, "sentences": sf_sentences}
                relevant_doc_hash_id = hash_object(relevant_doc)
                relevant_doc_id = doc_hash_id_to_doc_id[relevant_doc_hash_id]

                qrel_item = (qid, relevant_doc_id, 1)
                if qrel_item not in qrels_per_example:
                    qrels_per_example.append(qrel_item)

                sf_sentence_idx = 0 
                for i, sentence in enumerate(sf_sentences):
                    if question_decomposition_item["answer"].lower() in sentence.lower():
                        sf_sentence_idx = i 
                        break
                supporting_facts.append((relevant_doc_id, sf_sentence_idx))
            
            qrels.extend(qrels_per_example)
            qa_pairs.append(
                {
                    "id": qid, 
                    "question": example["question"], 
                    "answers": [example["answer"]],
                    "supporting_facts": supporting_facts, 
                }
            )
    
    # data splits 
    import numpy as np 
    seed_everything(0)
    indices = np.random.permutation(len(orig_train_qa_pairs))
    train_qa_pairs = [orig_train_qa_pairs[idx] for idx in indices[:-args.num_dev_data]]
    dev_qa_pairs = [orig_train_qa_pairs[idx] for idx in indices[-args.num_dev_data:]]
    test_qa_pairs = orig_dev_qa_pairs

    return qrels, train_qa_pairs, dev_qa_pairs, test_qa_pairs


def load_hotpotqa_corpus():

    import bz2
    import glob 
    from tqdm import tqdm
    import json 

    corpus_folder_name = "enwiki-20171001-pages-meta-current-withlinks-abstracts"
    corpus_folder = os.path.join(LOAD_DATA_FOLDER["hotpotqa"], corpus_folder_name)
    print(f"loading HotPotQA Corpus from {corpus_folder} ... ")

    doc_hash_id_to_doc_id = {} 
    corpus_with_doc_ids = [] 
    for filepath in tqdm(glob.glob(os.path.join(corpus_folder, "*", "wiki_*.bz2"))):
        for datum in bz2.BZ2File(filepath).readlines():
            instance = json.loads(datum.strip())
            doc_item = {"title": instance, "sentences": instance["text"]}
            doc_hash_id = hash_object(doc_item)
            if doc_hash_id in doc_hash_id_to_doc_id:
                if instance["id"] == doc_hash_id_to_doc_id[doc_hash_id]:
                    # 文本一样id也一样
                    print("duplicate document!")
                    continue
                else:
                    # 文本一样但是id不一样
                    print("Documents with same texts but different ids")
                    doc_hash_id = doc_hash_id + "_v2"
            doc_hash_id_to_doc_id[doc_hash_id] = instance["id"]
            corpus_with_doc_ids.append(
                {
                    "id": instance["id"], 
                    "title": instance["title"], 
                    "sentences": instance["text"]
                }
            )
    
    corpus_with_doc_ids = sorted(corpus_with_doc_ids, key=lambda x: int(x["id"]))
    return doc_hash_id_to_doc_id, corpus_with_doc_ids


def convert_hotpotqa_to_odqa_data(doc_hash_id_to_doc_id, corpus):

    title_to_doc = {}
    for doc in corpus:
        title_to_doc[doc["title"]] = doc 

    qrels, orig_train_qa_pairs, orig_dev_qa_pairs = [], [], [] 
    print("Converting train, dev sets in HotPotQA into open domain QA data ...")
    for file in ["hotpot_train_v1.1.json", "hotpot_dev_distractor_v1.json"]:
        qa_pairs = orig_train_qa_pairs if "train" in file else orig_dev_qa_pairs
        file = os.path.join(LOAD_DATA_FOLDER["hotpotqa"], file)
        print(f"loading data from {file} ... ")
        data = load_json(file)
        for example in tqdm(data, desc="convert to ODQA data"):
            qid = example["_id"]
            qrels_per_example, supporting_facts = [], [] 
            for sf_title, sf_sentence_idx in example["supporting_facts"]:
                relevant_doc_id = title_to_doc[sf_title]["id"]
                supporting_facts.append((relevant_doc_id, sf_sentence_idx))

                qrel_item = (qid, relevant_doc_id, 1)
                if qrel_item not in qrels_per_example:
                    qrels_per_example.append(qrel_item)

            qrels.extend(qrels_per_example)
            qa_pairs.append(
                {
                    "id": qid, 
                    "question": example["question"], 
                    "answers": [example["answer"]],
                    "supporting_facts": supporting_facts, 
                }
            )
    
    # data splits 
    import numpy as np 
    seed_everything(0)
    indices = np.random.permutation(len(orig_train_qa_pairs))
    train_qa_pairs = [orig_train_qa_pairs[idx] for idx in indices[:-args.num_dev_data]]
    dev_qa_pairs = [orig_train_qa_pairs[idx] for idx in indices[-args.num_dev_data:]]
    test_qa_pairs = orig_dev_qa_pairs

    return qrels, train_qa_pairs, dev_qa_pairs, test_qa_pairs


def convert_webqa_to_odqa_data():

    dataset = "webqa"
    load_data_folder = LOAD_DATA_FOLDER[dataset]
    save_data_folder = SAVE_DATA_FOLDER[dataset]
    os.makedirs(save_data_folder, exist_ok=True)
    for file in ["webquestions-test.qa.csv"]:
        if file.endswith(".json"):
            """
            {"question": str, "answers": [str], ... }
            """
            data = load_json(os.path.join(load_data_folder, file))
        else:
            data = load_tsv_str_format(os.path.join(load_data_folder, file))
            uniform_data = [] 
            for question, answers_str in data:
                uniform_data.append(
                    {
                        "question": question, 
                        "answers": eval(answers_str)
                    }
                )
            data = uniform_data
        
        qa_pairs = [] 
        for example in tqdm(data, desc="convert to ODQA data"):
            qa_pairs.append(
                {
                    "question": example["question"], 
                    "answers": example["answers"]
                }
            )
        
        save_file_type = "train"
        if "dev" in file:
            save_file_type = "dev"
        if "test" in file:
            save_file_type = "test"
        save_file = "{}_qa_pairs.json".format(save_file_type)
        save_json(qa_pairs, os.path.join(save_data_folder, save_file), use_indent=True)


def convert_bamboogle_to_odqa_data():
    dataset = "bamboogle"
    load_data_folder = LOAD_DATA_FOLDER[dataset]
    save_data_folder = SAVE_DATA_FOLDER[dataset]
    os.makedirs(save_data_folder, exist_ok=True)

    test_file = os.path.join(load_data_folder, "Bamboogle_Prerelease.tsv")
    test_data = load_tsv_str_format(test_file)[1:]
    test_qa_pairs = [] 
    for example in tqdm(test_data, desc="convert to ODQA data"):
        test_qa_pairs.append(
            {
                "question": example[0], 
                "answers": [example[-1]], 
            }
        )
    save_json(test_qa_pairs, os.path.join(save_data_folder, "test_qa_pairs.json"), use_indent=True)


CORPUS_MAP = {
    "hotpotqa": load_hotpotqa_corpus, 
    "2wikimultihopqa": load_2wikimultihopqa_corpus, 
    "musique": load_musique_corpus, 
}

PROCESS_MAP = {
    "hotpotqa": convert_hotpotqa_to_odqa_data, 
    "2wikimultihopqa": convert_2wikimultihopqa_to_odqa_data, 
    "musique": convert_musique_to_odqa_data, 
}


def construct_open_domain_data(args):

    dataset = args.dataset
    input_data_folder = LOAD_DATA_FOLDER[dataset]
    save_data_folder = SAVE_DATA_FOLDER[dataset]
    os.makedirs(save_data_folder, exist_ok=True)

    # obtain corpus
    doc_hash_id_to_doc_id, corpus = CORPUS_MAP[dataset]()

    # split data 
    qrels, train, dev, test = PROCESS_MAP[dataset](doc_hash_id_to_doc_id, corpus)
    save_tsv(qrels, os.path.join(save_data_folder, "qrels.tsv"))
    save_json(train, os.path.join(save_data_folder, "train_qa_pairs.json"))
    save_json(dev, os.path.join(save_data_folder, "dev_qa_pairs.json"))
    save_json(test, os.path.join(save_data_folder, "test_qa_pairs.json"))
    save_json(corpus, os.path.join(save_data_folder, "corpus.json"))


if __name__ == "__main__":

    args = setup_parser()
    if args.dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        construct_open_domain_data(args)
    elif args.dataset in ["bamboogle"]:
        convert_bamboogle_to_odqa_data()
    elif args.dataset in ["webqa"]:
        convert_webqa_to_odqa_data()
    else:
        raise ValueError(f"`{args.dataset}` is an unknown dataset!")
    