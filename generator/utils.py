from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch 
from torch import Tensor
from transformers import AutoTokenizer

def pad_token_ids(token_ids: Tensor, max_length: int, pad_token_id: int) -> Tensor:

    batch_size, num_tokens = token_ids.shape 
    if num_tokens >= max_length:
        return token_ids[:, :max_length]
    padding_length = max_length - num_tokens
    dtype = token_ids.dtype
    device = token_ids.device 
    padding_tensor = torch.zeros((batch_size, padding_length), dtype=dtype).fill_(pad_token_id).to(device)
    padded_token_ids = torch.cat([token_ids, padding_tensor], dim=1)

    return padded_token_ids

def pad_token_logits(token_logits: Tensor, max_length: int) -> Tensor:

    batch_size, num_tokens, vocab_size = token_logits.shape 
    if num_tokens >= max_length:
        return token_logits[:, :max_length]
    padding_length = max_length-num_tokens
    dtype, device = token_logits.dtype, token_logits.device
    padding_tensor = torch.zeros((batch_size, padding_length, vocab_size), dtype=dtype).to(device)
    padded_token_logits = torch.cat([token_logits, padding_tensor], dim=1)
    
    return padded_token_logits


def infer_padding_side(attention_mask: Tensor) -> str: 

    assert len(attention_mask.shape) == 2 
    batch_size = attention_mask.shape[0]
    sum_attention_mask = attention_mask.sum(0)
    if sum_attention_mask[0] == batch_size:
        padding = "right"
    elif sum_attention_mask[-1] == batch_size:
        padding = "left"
    else:
        padding = "unknow"
    
    return padding


def get_position_ids(attention_mask: Tensor) -> Tensor:

    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.masked_fill(attention_mask==0, 1)
    return position_ids


def append_texts_to_decoder_only_generator_inputs(tokenizer: AutoTokenizer, inputs: Dict[str, Tensor], texts: List[str]):

    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    batch_size, num_tokens = input_ids.shape
    inputs_length = attention_mask.sum(1)

    assert len(texts) == batch_size # number of texts muse be equal to the batch_size

    padding_side = tokenizer.padding_side 

    texts_token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    max_num_texts_token_ids = max([len(token_ids) for token_ids in texts_token_ids])

    if max_num_texts_token_ids == 0:
        return inputs 

    new_input_ids = torch.zeros((batch_size, num_tokens+max_num_texts_token_ids), dtype=input_ids.dtype).fill_(tokenizer.pad_token_id).to(input_ids.device)
    new_attention_mask = torch.zeros((batch_size, num_tokens+max_num_texts_token_ids), dtype=attention_mask.dtype).to(attention_mask.device)
    for i in range(batch_size):

        text_tensor = torch.tensor(texts_token_ids[i], dtype=input_ids.dtype).to(input_ids.device)
        num_text_token_ids = len(texts_token_ids[i])

        if num_text_token_ids == 0:
            if padding_side == "left":
                new_input_ids[i, -inputs_length[i]:] = input_ids[i, -inputs_length[i]:]
                new_attention_mask[i, -inputs_length[i]:] = attention_mask[i, -inputs_length[i]:]
            elif padding_side == "right":
                new_input_ids[i, :inputs_length[i]] = input_ids[i, :inputs_length[i]]
                new_attention_mask[i, :inputs_length[i]] = attention_mask[i, :inputs_length[i]]
            else:
                raise NotImplementedError(f"pad_texts_to_chat_format_generator_inputs for \"padding_side={padding_side}\" is not implemented!")
            continue

        if padding_side == "left":
            new_input_ids[i, -num_text_token_ids:] = text_tensor
            new_input_ids[i, -num_text_token_ids-inputs_length[i]:-num_text_token_ids] = input_ids[i, -inputs_length[i]:]
            new_attention_mask[i, -num_text_token_ids:] = 1 
            new_attention_mask[i, -num_text_token_ids-inputs_length[i]:-num_text_token_ids] = attention_mask[i, -inputs_length[i]:]
        elif padding_side == "right":
            new_input_ids[i, :inputs_length[i]] = input_ids[i, :inputs_length[i]]
            new_input_ids[i, inputs_length[i]: inputs_length[i]+num_text_token_ids] = text_tensor
            new_attention_mask[i, :inputs_length[i]] = attention_mask[i, :inputs_length[i]]
            new_attention_mask[i, inputs_length[i]: inputs_length[i]+num_text_token_ids] = 1 
        else:
            raise NotImplementedError(f"pad_texts_to_chat_format_generator_inputs for \"padding_side={padding_side}\" is not implemented!")
    
    return {"input_ids": new_input_ids, "attention_mask": new_attention_mask}


def append_texts_to_encoder_decoder_generator_inputs(tokenizer: AutoTokenizer, inputs: Dict[str, Tensor], texts: List[str], decoder_start_token_id: Union[int, Tensor], padding_side: str="left"):

    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
    batch_size = input_ids.shape[0]
    assert len(texts) == batch_size # number of texts muse be equal to the batch_size

    texts_token_ids = [tokenizer.encode(text.strip(), add_special_tokens=False) for text in texts]
    max_num_texts_token_ids = max([len(token_ids) for token_ids in texts_token_ids])

    decoder_input_ids = torch.zeros((batch_size, max_num_texts_token_ids+1), dtype=input_ids.dtype).fill_(tokenizer.pad_token_id).to(input_ids.device)
    decoder_attention_mask = torch.zeros_like(decoder_input_ids).to(attention_mask.dtype)

    for i in range(batch_size):

        text_tensor = torch.tensor(texts_token_ids[i], dtype=input_ids.dtype).to(input_ids.device)
        num_text_token_ids = len(texts_token_ids[i])

        if padding_side == "left":
            decoder_input_ids[i, -num_text_token_ids:] = text_tensor
            decoder_attention_mask[i, -num_text_token_ids:] = 1 
            decoder_input_ids[i, -num_text_token_ids-1: -num_text_token_ids] = decoder_start_token_id
            decoder_attention_mask[i, -num_text_token_ids-1: -num_text_token_ids] = 1 
        elif padding_side == "right":
            decoder_input_ids[i, 0] = decoder_start_token_id
            decoder_attention_mask[i, 0] = 1 
            decoder_input_ids[i, 1: 1+num_text_token_ids] = text_tensor
            decoder_attention_mask[i, 1: 1+num_text_token_ids] = 1 
        else:
            raise NotImplementedError(f"pad_texts_to_chat_format_generator_inputs for \"padding_side={padding_side}\" is not implemented!")
    
    inputs["decoder_input_ids"] = decoder_input_ids
    inputs["decoder_attention_mask"] = decoder_attention_mask

    return inputs

def convert_batch_tokens_to_input_ids(tokenizer: AutoTokenizer, tokens_list: List[List[str]], max_length: int=None, padding_side: str="left") -> Tuple[Tensor, Tensor]:

    token_ids_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list]
    max_length = max([len(token_ids) for token_ids in token_ids_list]) if max_length is None else max_length

    batch_size = len(tokens_list)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long).fill_(tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    
    for i, token_ids in enumerate(token_ids_list):
        token_ids_tensor = torch.tensor(token_ids, dtype=input_ids.dtype)
        num_token_ids = len(token_ids)
        if padding_side == "left":
            input_ids[i, -num_token_ids:] = token_ids_tensor
            attention_mask[i, -num_token_ids:] = 1
        elif padding_side == "right":
            input_ids[i, :num_token_ids] = token_ids_tensor
            attention_mask[i, :num_token_ids] = 1 
        else:
            raise NotImplementedError(f"Unknown padding side: {padding_side}!")
    
    return input_ids, attention_mask


def get_attention_mask_from_generated_token_ids(tokenizer: AutoTokenizer, token_ids: Tensor) -> Tensor:

    attention_mask = (token_ids != tokenizer.pad_token_id).long().to(token_ids.device)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        row_indices, col_indices = (token_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)
        for i in range(len(token_ids)):
            if len(col_indices[row_indices==i]) == 0:
                continue
            attention_mask[i, col_indices[row_indices==i][0]] = 1 
    return attention_mask


@dataclass
class Block:
    text: str = None
    tokens: List[str] = None  
    token_ids: List[int] = None 
    words: List[str] = None
    range_: List[Tuple[int, int]] = None  
    @property
    def len_tokens(self):
        return len(self.tokens)
    @property
    def len_words(self):
        return len(self.range_)


def tokenize_with_word_range(tokenizer: AutoTokenizer, text: str, add_special_tokens: bool=False):

    space_symbol_dict = {
        "meta-llama/Llama-2-7b-chat-hf": '▁',
        "meta-llama/Meta-Llama-3-8B-Instruct": 'Ġ', 
        "Qwen/Qwen2.5-7B-Instruct": 'Ġ',
        "google/gemma-2-9b-it": "▁",
    }
    new_line_symbol_dict = {
        "meta-llama/Llama-2-7b-chat-hf": '<0x0A>',
        "meta-llama/Meta-Llama-3-8B-Instruct": 'Ċ', 
        "Qwen/Qwen2.5-7B-Instruct": 'Ċ',
        "google/gemma-2-9b-it": "\n",
    }

    if tokenizer.name_or_path not in space_symbol_dict:
        raise NotImplementedError(f"tokenize_with_word_range for {tokenizer.name_or_path} is not implemented! Current available choices: {list(space_symbol_dict.keys())}")

    tokenizer_outputs = tokenizer(text, add_special_tokens=add_special_tokens)
    token_ids = tokenizer_outputs["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    word_start_idx = None
    range_ = []  
    space_symbol = space_symbol_dict[tokenizer.name_or_path]
    new_line_symbol = new_line_symbol_dict[tokenizer.name_or_path]

    prefix = 0 
    while prefix < len(tokens) and tokens[prefix] == tokenizer.bos_token:
        prefix += 1 

    for token_idx, token in enumerate(tokens[prefix:]):
        if token_idx == 0 or token.startswith(space_symbol) or token.startswith(new_line_symbol) or token == tokenizer.eos_token \
            or (token_idx-1>=0 and tokens[token_idx-1+prefix].endswith(new_line_symbol)):
            if word_start_idx is not None:
                range_.append([word_start_idx+prefix, token_idx+prefix])
            word_start_idx = token_idx

    if word_start_idx is not None:
        range_.append([word_start_idx+prefix, len(tokens)])
    
    words, word_range = [], [] 
    for i, (l, r) in enumerate(range_):
        word = tokenizer.convert_tokens_to_string(tokens[l: r])
        if tokenizer.name_or_path in ["meta-llama/Llama-2-7b-chat-hf"]:
            if i>0 and tokens[l].startswith(space_symbol):
                word = " " + word
        words.append(word)
        word_range.append([l, r])
    
    return Block(text=text, tokens=tokens, token_ids=token_ids, words=words, range_=word_range)
