import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model_in_4bit(cls, model_name_or_path, hf_token, device=None):

    from transformers import BitsAndBytesConfig
    model = cls.from_pretrained(
        model_name_or_path, 
        cache_dir=None, 
        device_map = "auto" if device is None else device, 
        # max_memory = {0: "800000MB"}, 
        max_memory = {i: "800000MB" for i in range(torch.cuda.device_count())},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        ),
        token=hf_token, 
        torch_dtype=torch.bfloat16, 
    )
    return model

def load_llm_tokenizer_and_model(model_name, hf_token, padding_side="left", dtype=torch.bfloat16, load_in_4bit=False, device=None, **kwargs):

    device = device or torch.device("cuda")

    MODEL_MAP = {
        # llama
        "llama2_instruct": "meta-llama/Llama-2-7b-chat-hf",
        "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
        "llama3_8b": "meta-llama/Meta-Llama-3-8B", 
        "llama3_70b_instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1_8b_instruct": "meta-llama/Llama-3.1-8B", 
        "llama3.1_70b_instruct": "meta-llama/Llama-3.1-70B-Instruct",
        # Mistral 
        "mistral_7b": "mistralai/Mistral-7B-v0.1", 
        "mistral_7b_instruct": "mistralai/Mistral-7B-Instruct-v0.2", 
        # Qwen 
        "qwen2": "Qwen/Qwen2-7B-Instruct",
        "qwen2.5_7b": "Qwen/Qwen2.5-7B",
        "qwen2.5_7b_instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5_32b_instruct": "Qwen/Qwen2.5-32B-Instruct",
        "qwen2.5_72b_instruct": "Qwen/Qwen2.5-72B-Instruct",
        # Gemma
        "gemma2_2b": "google/gemma-2-2b", 
        "gemma2_2b_itstruct": "google/gemma-2-2b-it",
        "gemma2_9b": "google/gemma-2-9b", 
        "gemma2_9b_instruct": "google/gemma-2-9b-it",
        "gemma2_27b_instruct": "google/gemma-2-27b-it",
    }

    if model_name not in MODEL_MAP:
        raise ValueError(f"{model_name} is not a valid model name. Current available models: {list(MODEL_MAP.keys())}")
    
    model_name_or_path = MODEL_MAP[model_name]
    
    padding_side = "left"
    print(f"loading tokenizer for \"{model_name_or_path}\" with padding_side: \"{padding_side}\"")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side, token=hf_token)
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
        print("Missing padding token, setting padding token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if load_in_4bit:
        print(f"loading \"{model_name_or_path}\" model in 4-bits ...")
        model = load_model_in_4bit(AutoModelForCausalLM, model_name_or_path, hf_token=hf_token, device=device)
    else:
        print(f"loading \"{model_name_or_path}\" model in {dtype} ...")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, token=hf_token)
        model.to(device)
    model.eval()

    return tokenizer, model 

def get_retrieved_documents(retrieved_documents_ids_to_scores, corpus_dataset):
    ranked_retrieved_documents_ids_with_scores = sorted(retrieved_documents_ids_to_scores.items(), key=lambda x: x[1], reverse=True)
    retrieved_documents = [] 
    for docid, score in ranked_retrieved_documents_ids_with_scores:
        doc = deepcopy(corpus_dataset.get_document(docid))
        doc["score"] = float(score)
        retrieved_documents.append(doc)
    return retrieved_documents