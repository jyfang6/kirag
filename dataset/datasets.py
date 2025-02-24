import os
import random
import pickle 
import itertools
import numpy as np 
from typing import Union, List
from torch.utils.data import Dataset

from utils.utils import load_json, convert_triples_to_sentences


class RetrieverDataset(Dataset):

    def __init__(self, data_files, question_prefix='question:', title_prefix='title:', passage_prefix='context:', **kwargs):
        
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix 
        self.passage_prefix = passage_prefix
        self.kwargs = kwargs 

        self.data = self.load_data(data_files)
    
    def load_data(self, data_files):
        if isinstance(data_files, str):
            data_files = [data_files]

        data = [] 
        for data_file in data_files:
            print(f"Loading data from {data_file} ... ")
            data.extend(load_json(data_file, type="json"))
        
        total_num = len(data)
        new_data = []
        for example in data:
            if "positive_ctxs" not in example or len(example["positive_ctxs"]) == 0:
                continue
            new_data.append(example)
        print(f"Successfully load {len(new_data)} examples. Total Number of data: {total_num}, filtered number of data: {total_num-len(new_data)}!")
        return new_data

    def sort_context_based_on_score(self, context_list):
        if context_list is None or len(context_list) == 0 or "score" not in context_list[0] or context_list[0]["score"] is None:
            return
        context_list.sort(key=lambda x: float(x["score"]), reverse=True)

    def get_passages(self, passage_list, num_passages, random=False):
        if passage_list is None:
            return [] 
        num_possible_passages = min(len(passage_list), num_passages)
        passage_indices = np.random.permutation(len(passage_list))[:num_possible_passages] if random else np.arange(num_possible_passages)
        passages = [passage_list[idx] for idx in passage_indices]
        return passages
    
    def get_passage_id(self, one_passage_item):
        for key in one_passage_item.keys():
            if "id" in key:
                return key
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    

class KGChainRetrieverDataset(RetrieverDataset):

    def __init__(
        self, 
        data_with_posnegs_files: Union[str, List[str]], 
        question_prefix: str='question:', 
        use_title: bool=False, 
        convert_triple_to_sentence: bool=False, 
        chain_type: str="triple", 
        title_prefix: str='title:', 
        passage_prefix: str='context:', 
        num_positives: int=4, 
        num_negatives: int=1, 
        **kwargs
    ):
        
        self.question_prefix = question_prefix
        self.use_title = use_title
        self.convert_triple_to_sentence = convert_triple_to_sentence
        self.chain_type = chain_type

        self.title_prefix = title_prefix 
        self.passage_prefix = passage_prefix
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.kwargs = kwargs 
        self.is_train = self.kwargs.get("is_train", True) 

        self.query_template_with_chain = "{question_prefix} {question}\nknowledge triples: {chain}."
        self.query_template_wo_chain = "{question_prefix} {question}"

        self.data = self.load_data(data_with_posnegs_files)
    
    def load_data(self, data_files: Union[str, List[str]]):
        if isinstance(data_files, str):
            data_files = [data_files]
        data = [] 
        for data_file in data_files:
            print(f"Loading data from {data_file} ... ")
            data.extend(load_json(data_file, type="json"))
        print("Successfully loaded {} examples!".format(len(data)))
        return data

    def get_triples_texts(self, triples: List[dict]):
        texts = [] 
        for triple in triples:
            text = ""
            if self.use_title:
                text += "title: {} text: ".format(triple["title"])
            if self.chain_type == "triple":
                if self.convert_triple_to_sentence:
                    text += convert_triples_to_sentences(triple["text"])
                else:
                    text += triple["text"]
            else:
                raise ValueError(f"Invalid value '{self.chain_type}' detected for `chain_type`. Only support 'triple'.")
            texts.append(text)
        
        return texts
    

class KGChainRetrieverSeqSampleDataset(KGChainRetrieverDataset):

    """
    Each data_folder should have: train_aligner.json, dev_aligner.json, is_comparison_map.pkl (for hotpotqa and 2wikimultihopqa)
    """

    def __init__(
        self, 
        is_train: bool, 
        data_folders: List[str], 
        question_prefix: str = '', 
        use_title: bool = False, 
        convert_triple_to_sentence: bool = False, 
        chain_type: str = "triple", title_prefix: str = 'title:', 
        passage_prefix: str = 'text:', 
        num_positives: int = 2, num_negatives: int = 10, 
        **kwargs
    ):  
        load_data_files = [] 
        for folder in data_folders:
            load_data_files.append(os.path.join(folder, "train_aligner.json" if is_train else "dev_aligner.json"))
        
        super().__init__(load_data_files, question_prefix, use_title, convert_triple_to_sentence, chain_type, title_prefix, passage_prefix, num_positives, num_negatives, is_train=is_train, **kwargs)
        print("loading comparison_question_ids ...")
        self.comparison_question_ids = self.load_comparison_question_ids(data_folders)
        print("Number of comparison questions: {}".format(len(self.comparison_question_ids)))
    
    def load_comparison_question_ids(self, data_folders: List[str]):

        comparison_question_ids = set()
        for folder in data_folders:
            if "hotpotqa" in folder or "2wikimultihopqa" in folder:
                is_comparison_map = pickle.load(open(os.path.join(folder, "is_comparison_map.pkl"), "rb"))
                for question_id, is_comparison in is_comparison_map.items():
                    if is_comparison:
                        comparison_question_ids.add(question_id)
        return comparison_question_ids

    def __getitem__(self, index: int):

        example = self.data[index]
        negative_triples_key = "hard_negative_triples"

        all_positive_chain_combinations = [] 
        num_hops = len(example["supporting_triples"])
        for hop in range(num_hops):
            if example["id"] in self.comparison_question_ids:
                if hop == 0:
                    all_positive_chain_combinations.extend([(i, ) for i in range(num_hops)])
                    continue
                all_positive_chain_combinations.extend(list(itertools.permutations(range(hop+1))))
            else:
                all_positive_chain_combinations.append(list(range(hop+1)))
        
        if self.is_train:
            positive_chain_combinations = random.sample(
                all_positive_chain_combinations,
                min(self.num_positives, len(all_positive_chain_combinations))
            )
        else:
            positive_chain_combinations = all_positive_chain_combinations
        
        results = []
        for combination in positive_chain_combinations:
            positive_triples = [example["supporting_triples"][pos] if isinstance(pos, int) else pos for pos in combination]
            if len(positive_triples) == 1:
                query = self.query_template_wo_chain.format(
                    question_prefix = self.question_prefix, 
                    question = example["question"]
                ).strip()
            else:
                query = self.query_template_with_chain.format(
                    question_prefix = self.question_prefix, 
                    question = example["question"], 
                    chain = ". ".join(self.get_triples_texts(positive_triples[:-1]))
                ).strip()
            
            positive_passage = self.get_triples_texts(positive_triples[-1:])[0]

            candidate_negative_triples = []
            for key, values in example[negative_triples_key].items():
                candidate_negative_triples.extend(values)
            
            while len(candidate_negative_triples) < self.num_negatives:
                candidate_negative_triples.append(random.choice(candidate_negative_triples))
            
            if self.is_train:
                negative_triples = random.sample(
                    candidate_negative_triples, 
                    min(self.num_negatives, len(candidate_negative_triples))
                )
            else:
                negative_triples = candidate_negative_triples

            negative_passages = [] 
            for triple in negative_triples:
                negative_passage = self.get_triples_texts([triple])[0]
                negative_passages.append(negative_passage)
            
            results.append(
                {
                    "index": index, 
                    "question": query, 
                    "answers": example["answers"], 
                    "positive_passage": positive_passage, 
                    "negative_passages": negative_passages
                }
            )
        
        return results 
