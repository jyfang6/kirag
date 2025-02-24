import torch 

def encode_question_passages(batch_text_passages, tokenizer, max_length):

    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding="max_length",
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)

    return passage_ids, passage_masks


def truncate_to_max_sequence(input_ids, attention_mask, labels=None):

    assert len(input_ids.shape) == 2 # input_ids  must be 2-dimensional 
    assert len(attention_mask.shape) == 2 # attention_mask must be 2-dimensional 

    num_tokens = attention_mask.sum(0)
    padding_type = "right" if num_tokens[0].item() > num_tokens[-1].item() else "left"
    if padding_type == "right":
        max_num_tokens = (attention_mask!=0).max(0)[0].nonzero(as_tuple=False)[-1].item()+1
        input_ids = input_ids[..., :max_num_tokens]
        attention_mask = attention_mask[..., :max_num_tokens]
        if labels is not None:
            labels = labels[..., :max_num_tokens]
    else:
        min_padding_length = (attention_mask!=0).max(0)[0].nonzero(as_tuple=False)[0].item()
        input_ids = input_ids[..., min_padding_length:]
        attention_mask = attention_mask[..., min_padding_length:]
        if labels is not None:
            labels = labels[..., min_padding_length:]
    
    if labels is not None:
        return input_ids, attention_mask, labels
    else:
        return input_ids, attention_mask
    

class RetrieverCollator:

    def __init__(self, tokenizer, query_maxlength, doc_maxlength, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        self.tokenizer = tokenizer
        self.query_maxlength = query_maxlength
        self.doc_maxlength = doc_maxlength
        self.query_padding = query_padding # must be chosen from ["max_length", "max_sequence"]
        self.doc_padding = doc_padding
        self.kwargs = kwargs
    
    def encode(self, text_list, maxlength, padding, **kwargs):

        assert padding in ["max_length", "max_sequence"] # padding must be chosen from ["max_length", "max_sequence"]

        if text_list is None or (isinstance(text_list, (tuple, list)) and len(text_list) == 0):
            raise ValueError("text_list is None or an empty tuple/list!")
        
        if isinstance(text_list, str) or isinstance(text_list[0], str):
            padding_scheme = padding if padding == "max_length" else True
            outputs = self.tokenizer(text_list, max_length=maxlength, padding=padding_scheme, truncation=True, return_tensors='pt')
            input_ids, attention_mask = outputs["input_ids"], outputs["attention_mask"]
        elif isinstance(text_list[0], (list, tuple)):
            input_ids, attention_mask = encode_question_passages(text_list, self.tokenizer, maxlength)
            if padding == "max_sequence":
                *input_ids_shape, _ = input_ids.shape
                *attention_mask_shape, _ = attention_mask.shape
                input_ids, attention_mask = truncate_to_max_sequence(input_ids.reshape(-1, maxlength), attention_mask.reshape(-1, maxlength))
                input_ids = input_ids.reshape(*input_ids_shape, input_ids.shape[-1])
                attention_mask = attention_mask.reshape(*attention_mask_shape, attention_mask.shape[-1])
        else:
            raise ValueError(f"Unrecognised type for {text_list}!")
        
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode_query(self, query_list, **kwargs):
        query_maxlength = kwargs.get("max_length", None) or self.query_maxlength
        return self.encode(query_list, query_maxlength, self.query_padding, **kwargs)

    def encode_doc(self, doc_list, **kwargs):
        doc_maxlength = kwargs.get("max_length", None) or self.doc_maxlength
        return self.encode(doc_list, doc_maxlength, self.doc_padding, **kwargs)

    def __call__(self):
        raise NotImplementedError("__call__ is not implemented for the base RetrieverCollator!")


class RetrieverWithPosNegsCollator(RetrieverCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        if doc_maxlength is None:
            doc_maxlength = query_maxlength
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding=query_padding, doc_padding=doc_padding, **kwargs)

    def __call__(self, batch): 

        """
        batch = [
            {
                "index": int, 
                "question": str, 
                "answers": [str],
                "positive_passage": str, 
                "negative_passages": [str]
            }
        ]
        """
        if isinstance(batch[0], list):
            batch = sum(batch, []) 
        query_list = [example["question"] for example in batch]
        doc_list, positive_doc_indices = [], [] 
        for example in batch:
            positive_doc_indices.append(len(doc_list))
            doc_list.append(example["positive_passage"])
            doc_list.extend(example["negative_passages"])
        
        query_args = self.encode_query(query_list)
        doc_args = self.encode_doc(doc_list)
        positive_doc_indices = torch.tensor(positive_doc_indices, dtype=torch.long)
        index = torch.tensor([example["index"] for example in batch], dtype=torch.long)

        return query_args, doc_args, positive_doc_indices, index 
    

class E5Collator(RetrieverWithPosNegsCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding ="max_sequence", **kwargs):
        if doc_maxlength is None:
            doc_maxlength = query_maxlength
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding=query_padding, doc_padding=doc_padding, **kwargs)
    
    def encode_query(self, query_list, **kwargs):
        query_list = ["query: " + query for query in query_list]
        return super().encode_query(query_list=query_list, **kwargs)
    
    def encode_doc(self, doc_list, **kwargs):
        doc_list = ["passage: "+ doc for doc in doc_list]
        return super().encode_doc(doc_list=doc_list, **kwargs)

    
class BGECollator(RetrieverWithPosNegsCollator):

    def __init__(self, tokenizer, query_maxlength, doc_maxlength=None, query_padding="max_sequence", doc_padding="max_sequence", **kwargs):
        super().__init__(tokenizer, query_maxlength, doc_maxlength, query_padding, doc_padding, **kwargs)
    
    def encode_query(self, query_list, **kwargs):
        instruction = "Represent this sentence for searching relevant passages:"
        query_list = [instruction + " " + query for query in query_list]
        return super().encode_query(query_list=query_list, **kwargs)