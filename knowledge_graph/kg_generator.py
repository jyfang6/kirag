import os
import re 
import nltk
import pickle
import numpy as np
from copy import deepcopy 
from tqdm import trange
from nltk.tokenize import sent_tokenize
from typing import Union, Tuple, List, Dict

import torch 
import torch.nn as nn

from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from utils.utils import to_device
from retriever.e5 import get_e5_embeddings_for_document

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading 'punkt_tab' tokenizer...")
    nltk.download('punkt_tab')


class KGGenerator(nn.Module):

    def __init__(self, tokenizer, generator, max_length=4096, max_new_tokens=512, examplar_type="hotpotqa", num_examplars=5, adaptive_examplars=True, adaptive_examplars_retriever="e5", batch_size=4, verbose=False, **kwargs):
        
        super().__init__()
        assert examplar_type in ["hotpotqa", "2wikimultihopqa", "musique", "nq", "tqa", "webqa", "bamboogle", "wikipedia"]

        self.tokenizer = tokenizer
        self.generator = generator
        assert isinstance(generator, (LlamaForCausalLM, Qwen2ForCausalLM)) # currently only support using LLaMA3 or Gemma2 as the generator
        self.device = self.generator.device 
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.num_examplars = num_examplars
        self.adaptive_examplars = adaptive_examplars
        self.adaptive_examplars_retriever = adaptive_examplars_retriever
        self.examplars = self.load_examplars(examplar_type)
        if self.adaptive_examplars:
            self.examplars_embeddings = self.calculate_examplars_embeddings()
        self.batch_size = batch_size
        self.verbose = verbose
        self.cached_kg_triples = None 
        self.task_instruction = "You are a knowledge graph constructor tasked with extracting knowledge triples in the form of <head entity; relation; tail entity> from a document. "\
            "Each triple denotes a specific relationship between entities or an event. The head entity and tail entity can be the provided title or phrases in the text. "\
                "If multiple tail entities share the same relation with a head entity, aggregate these tail entities using commas. "\
                    "Format your output in the form of <head entity; relation; tail entity>."
        self.kwargs = kwargs
    
    def load_examplars(self, examplar_type):

        print(f"loading {examplar_type} examplars for KGGenerator ... ")
        if examplar_type == "hotpotqa":
            from prompts.kg_construction.hotpotqa_demonstrations import generate_knowledge_triples_hotpotqa_examplars
            examplars = generate_knowledge_triples_hotpotqa_examplars
        elif examplar_type == "2wikimultihopqa":
            from prompts.kg_construction.wikimultihopqa_demonstrations import generate_knowledge_triples_2wikimultihopqa_examplars
            examplars = generate_knowledge_triples_2wikimultihopqa_examplars
        elif examplar_type == "musique":
            from prompts.kg_construction.musique_demonstrations import generate_knowledge_triples_musique_examplars
            examplars =  generate_knowledge_triples_musique_examplars
        elif examplar_type in ["wikipedia", "nq", "tqa", "webqa", "bamboogle"]:
            from prompts.kg_construction.wikipedia_demonstrations import generate_knowledge_triples_wikipedia_examplars
            examplars = generate_knowledge_triples_wikipedia_examplars
        else:
            raise KeyError(f"{examplar_type} is not a supported examplar type!")
        
        return examplars
    
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
    
    def calculate_examplars_embeddings(self):

        texts = self.get_texts_from_documents(self.examplars)
        if self.adaptive_examplars_retriever == "e5":
            print("calculating embeddings for demonstrations ...")
            with torch.no_grad():
                examplars_embeddings = get_e5_embeddings_for_document(texts, max_length=256)
        else:
            raise KeyError(f"{self.adaptive_examplars_retriever} is not a supported retriever!")
        return examplars_embeddings
    
    def rank_examplars(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:

        """
        Input: [{"title": str, "text": str / "sentences": str}]
        Output: [{"title": str, "text": str / "sentences": str, "ranked_examplars_indices": [str]}]
        """
        texts = self.get_texts_from_documents(documents)
        if self.adaptive_examplars_retriever == "e5":
            with torch.no_grad():
                texts_embeddings = get_e5_embeddings_for_document(texts, max_length=256)
        else:
            raise KeyError(f"{self.adaptive_examplars_retriever} is not a supported retriever!")
        
        similarities = torch.matmul(texts_embeddings, self.examplars_embeddings.T)
        indices = torch.argsort(similarities, dim=1, descending=True).tolist()
        for doc, ranked_examplars_indices in zip(documents, indices):
            doc["ranked_examplars_indices"] = ranked_examplars_indices
        return documents
    
    def load_cached_kg_triples(self, paths: Union[str, List[str]]):
        
        if isinstance(paths, str):
            paths = [paths]
        
        if self.cached_kg_triples is None:
            print("Initializing a new KG triples cache ...")
            self.cached_kg_triples = {}
        
        for path in paths:
            if os.path.exists(path):
                print(f"loading cached KG triples from {path} ...")
                self.cached_kg_triples.update(pickle.load(open(path, "rb")))
    
    def save_cached_kg_triples(self, path):
        
        if self.cached_kg_triples is not None:
            print(f"saving cached KG triples to {path} ...")
            pickle.dump(self.cached_kg_triples, open(path, "wb"))

    def get_documents_inputs(self, documents: List[Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """
        Input: [{"title": str, "text": str / "sentences": str, (optional: "ranked_examplars_indices": [int])}]
        Output: instructions: [str], inputs: [str]
        """
        def vary_num_examplars_based_on_context_window(examplars, doc):
            final_examplars = None
            while len(examplars) > 0:
                for num in range(len(examplars), 0, -1):
                    candidate_examplars = examplars[:num]
                    candidate_prompt = self.task_instruction + "\n\n" + "\n\n".join(candidate_examplars) + "\n\n" + self.get_texts_from_documents(doc)
                    candidate_prompt_tokens = self.tokenizer.encode(candidate_prompt)
                    if len(candidate_prompt_tokens) <= self.max_length:
                        final_examplars = examplars[:num]
                        break
                if final_examplars is None:
                    examplars = examplars[1:]
                else:
                    break
            if final_examplars is None:
                final_examplars = [] 
            return final_examplars

        instructions, inputs = [], []
        for doc in documents:
            ranked_examplars_indices = doc.get("ranked_examplars_indices", None)
            if ranked_examplars_indices is None:
                ranked_examplars_indices = list(range(len(self.examplars)))
            doc_specific_examplars = [self.examplars[idx] for idx in ranked_examplars_indices[:self.num_examplars]]
            doc_specific_examplars = [
                "{}\nKnowledge Triples: {}".format(
                    self.get_texts_from_documents(example), 
                    example["triples"]
                )
                for example in doc_specific_examplars
            ]
            doc_specific_examplars = vary_num_examplars_based_on_context_window(doc_specific_examplars, doc)

            instructions.append(self.task_instruction+"\n\n"+"\n\n".join(doc_specific_examplars))
            # inputs.append(
            #     "Extract knowledge triples from the following document according to the task instruction.\n\n" + \
            #     self.get_texts_from_documents(doc)
            # )
            inputs.append(self.get_texts_from_documents(doc))
        return instructions, inputs

    def get_documents_prompts_chat_format(self, instructions: List[str], inputs: List[str]):
        """
        Input: instruction: [str], inputs: [str]
        """
        prompts = [] 
        for instruction, input in zip(instructions, inputs):
            if isinstance(self.generator, (LlamaForCausalLM, Qwen2ForCausalLM)):
                prompts.append(
                    [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": input}
                    ]
                )
            # elif isinstance(self.generator, Gemma2ForCausalLM):
            #     # content = instruction + "\n\nPlease help me extract knowledge triples for the following document.\n\n" + input
            #     content = instruction + "\n\n" + input
            #     prompts.append(
            #         [
            #             {"role": "user", "content": content}
            #         ]
            #     )
            else:
                raise NotImplemented(f"chat format for {type(self.generator)} is not implemented yet!")
        return prompts
    
    def tokenizer_encode_chat_format(self, prompts: List[List[Dict[str, str]]]):
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True)
        batch_dict = self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def generator_generate(self, inputs):
        inputs = to_device(inputs, self.device)
        generated_token_ids = self.generator.generate(**inputs, max_new_tokens=self.max_new_tokens, temperature=1.0, do_sample=False)
        return generated_token_ids

    def parse_triples_text(self, triples_text: str):
        results = [] 
        for one_triple_text in re.findall(r'<([^>]*)>', triples_text):
            if "head entity" in one_triple_text or "tail entity" in one_triple_text:
                continue
            results.append("<{}>".format(one_triple_text.strip()))
        return results
    
    def find_sentence_for_one_triple(self, doc: Dict[str, str], triple: str):

        def get_common_word_count(substring, text): 
            return np.sum([word in text for word in substring.split()])
        
        sentences = doc.get("sentences", None)
        if sentences is None:
            sentences = sent_tokenize(doc["text"])
        common_word_counts = [get_common_word_count(triple, sentence) for sentence in sentences]
        index = int(np.argmax(common_word_counts))
        return index

    def parse_generator_outputs(self, documents: List[Dict[str, str]], generator_outputs: List[str]):
        """
        Input: documents: [{"title": str, "text": str / "sentences": str}], generator_outputs: ["xxx<xxx>\n<xx>", ...]
        Output: [{"title": str, "text": str / "sentences": str, "triples": [{"text": str, "sentence": int}]}]
        """
        for doc, one_doc_generator_output in zip(documents, generator_outputs):
            triples = [] 
            triples_texts = self.parse_triples_text(one_doc_generator_output)
            for one_triple in triples_texts:
                sentence = self.find_sentence_for_one_triple(doc, one_triple)
                triples.append({"text": one_triple, "sentence": sentence})
            doc["triples"] = triples 
        
        return documents

    def generate_kg_triples_wo_cache(self, documents: Union[Dict[str, str], List[Dict[str, str]]]):
        
        """
        Input: {"title": str, "text": str / "sentences": str} or [{"title": str, "text": str / "sentences": str}]
        Output: {"title": str, "text": str / "sentences": str, "triples": ["text": str, "sentence": int]}, or its list version 
        """
        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]
        if self.adaptive_examplars:
            documents = self.rank_examplars(documents)

        if self.verbose:
            progress_bar = trange((len(documents)-1) // self.batch_size + 1, desc="Generating Knowledge Triples")

        generated_contents = [] 
        for i in range((len(documents)-1) // self.batch_size + 1):
            batch_document = documents[i*self.batch_size: (i+1)*self.batch_size]
            batch_instructions, batch_inputs = self.get_documents_inputs(batch_document)
            batch_prompts = self.get_documents_prompts_chat_format(batch_instructions, batch_inputs)
            batch_generator_inputs = self.tokenizer_encode_chat_format(batch_prompts)
            batch_generated_token_ids = self.generator_generate(batch_generator_inputs)
            batch_input_ids = batch_generator_inputs["input_ids"]
            batch_generated_token_ids = batch_generated_token_ids[:, batch_input_ids.shape[1]:]
            batch_generated_texts = self.tokenizer.batch_decode(batch_generated_token_ids, skip_special_tokens=True)
            generated_contents.extend(batch_generated_texts)
            if self.verbose:
                progress_bar.update(1)
        
        # parser model outputs 
        documents_with_triples = self.parse_generator_outputs(documents, generated_contents)
        if not is_list:
            documents_with_triples = documents_with_triples[0]

        return documents_with_triples
    
    def generate_kg_triples_with_cache(self, documents: Union[Dict[str, str], List[Dict[str, str]]]):

        assert self.cached_kg_triples is not None # muse use "load_cached_kg_triples(path)" function to load or initialize KG cache!

        is_list = isinstance(documents, list)
        if not is_list:
            documents = [documents]

        all_docids = [doc["id"] for doc in documents]
        docs_wo_cached_kg_triples = [doc for docid, doc in zip(all_docids, documents) if docid not in self.cached_kg_triples]
        docs_wo_cached_kg_triples = deepcopy(docs_wo_cached_kg_triples)
        if len(docs_wo_cached_kg_triples) > 0:
            docs_with_kgs = self.generate_kg_triples_wo_cache(docs_wo_cached_kg_triples)
        else:
            docs_with_kgs = []
        # update cache 
        self.cached_kg_triples.update({doc["id"]: doc for doc in docs_with_kgs})
        # get kg triples for all input documents 
        all_docs_with_kgs = [self.cached_kg_triples[docid] for docid in all_docids]
        if not is_list:
            all_docs_with_kgs = all_docs_with_kgs[0]
        
        return all_docs_with_kgs
    
    def forward(self, documents: Union[Dict[str, str], List[Dict[str, str]]]):

        if self.cached_kg_triples is None:
            return self.generate_kg_triples_wo_cache(documents=documents)
        else:
            return self.generate_kg_triples_with_cache(documents=documents)
