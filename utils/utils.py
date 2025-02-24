import json 
import torch
import random 
import logging
import numpy as np
import torch.distributed as dist 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_json(path, type="json"):
    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        outputs = json.loads(open(path, "r", encoding="utf-8").read())
    elif type == "jsonl":
        outputs = []
        with open(path, "r", encoding="utf-8") as fin:
            for line in fin:
                outputs.append(json.loads(line))
    else:
        outputs = []
        
    return outputs


def save_json(data, path, type="json", use_indent=False):

    assert type in ["json", "jsonl"] # only support json or jsonl format
    if type == "json":
        with open(path, "w", encoding="utf-8") as fout:
            if use_indent:
                fout.write(json.dumps(data, indent=4))
            else:
                fout.write(json.dumps(data))

    elif type == "jsonl":
        with open(path, "w", encoding="utf-8") as fout:
            for item in data:
                fout.write("{}\n".format(json.dumps(item)))

    return path

def hash_object(o) -> str:
    
    """Returns a character hash code of arbitrary Python objects."""
    import hashlib
    import io
    import dill
    import base58

    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()

def load_tsv_str_format(path):
    data = [] 
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            data.append(tuple([str(piece) for piece in line.strip().split("\t")]))
    return data


def save_tsv(data, path):
    with open(path, "w", encoding="utf-8") as fout:
        for item in data:
            new_item = [] 
            for x in item:
                if isinstance(x, str):
                    new_item.append(x)
                elif isinstance(x, int):
                    new_item.append(str(x))
                elif isinstance(x, float):
                    new_item.append("{:.6f}".format(x))
            item_str = "\t".join(new_item)
            fout.write("{}\n".format(item_str))
        fout.close()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def gpu_setup(local_rank, random_seed):
    if local_rank >= 0:
        dist.init_process_group(backend='nccl')
        seed = random_seed+local_rank
    else:
        seed = random_seed
    seed_everything(seed=seed)
    print(f"GPU Setup, Local Rank: {local_rank}, Random Seed: {seed} ... ")

def setup_logger(local_rank, log_file):

    fh = logging.FileHandler(log_file)
    # fh = MFileHandler(log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    logger.info(f"Rank: {local_rank}, Saving log file to {log_file} ...")


def get_dataloader(local_rank: int, dataset: Dataset, batch_size: int, shuffle: bool, collate_fn=None, drop_last: bool=False):
    
    if local_rank >= 0:
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank, drop_last=drop_last, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=False, num_workers=0, sampler=sampler, drop_last=drop_last, collate_fn=collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, collate_fn=collate_fn)

    return dataloader


def get_global_tensor_list(local_rank, world_size, tensor):
    if local_rank < 0:
        return [tensor]
    global_tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(global_tensor_list, tensor)
    return global_tensor_list


def get_global_tensors(local_rank, world_size, tensor):
    if local_rank < 0:
        return tensor
    global_tensor_list = get_global_tensor_list(local_rank, world_size, tensor)
    global_tensor = torch.cat(global_tensor_list, dim=0)
    return global_tensor


def get_global_tensor_list_to_main_process(local_rank, world_size, tensor):
    if local_rank < 0:
        return [tensor]
    if local_rank == 0:
        global_tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list=global_tensor_list, dst=0)
        results = global_tensor_list
    else:
        dist.gather(tensor, dst=0)
        results = None
    return results


def get_global_embeddings_for_inbatchtraining(local_rank, world_size, local_embeddings):

    if local_rank < 0:
        return local_embeddings
    
    embeddings_to_send = local_embeddings.detach().clone()
    gather_embeddings_list = get_global_tensor_list(local_rank, world_size, embeddings_to_send)

    global_embeddings_list = [] 
    for i, embeddings in enumerate(gather_embeddings_list):
        if i!=local_rank:
            global_embeddings_list.append(embeddings.to(local_embeddings.device))
        else:
            global_embeddings_list.append(local_embeddings)
    global_embeddings = torch.cat(global_embeddings_list, dim=0)

    return global_embeddings


def get_global_labels_for_inbatchtraining(local_rank, world_size, local_labels, local_doc_size):

    if local_rank < 0:
        return local_labels
    if local_labels is None:
        return None
    gather_labels_list = get_global_tensor_list(local_rank, world_size, local_labels)
    global_labels_list = [] 
    for i, labels in enumerate(gather_labels_list):
        global_labels_list.append(labels+i*local_doc_size)
    global_labels = torch.cat(global_labels_list, dim=0).to(local_labels.device)
    return global_labels

def to_device(inputs, device):

    def dict_to_device(data):
        return {k: item.to(device) if torch.is_tensor(item) else item for k, item in data.items()}
    
    if isinstance(inputs, (tuple, list)):
        new_data = [] 
        for item in inputs:
            if isinstance(item, dict):
                new_data.append(dict_to_device(item))
            elif torch.is_tensor(item):
                new_data.append(item.to(device))
            else:
                new_data.append(item)
    elif isinstance(inputs, dict):
        new_data =dict_to_device(inputs)
    else:
        raise TypeError(f"Currently do not support using <{type(inputs)}> as the type of a batch")

    return new_data

def cleanup():
    dist.destroy_process_group()