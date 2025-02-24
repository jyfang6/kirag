import os
import torch 
import logging
import argparse
from tqdm import trange

from dataset.corpus import load_psg_data, CORPUS_PATH
from knowledge_graph.kg_generator import KGGenerator
from utils.pipeline_utils import load_llm_tokenizer_and_model
from utils.utils import load_json


logger = logging.getLogger(__file__)

device = torch.device("cuda")

def setup_parser():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input & output data 
    parser.add_argument("--hf_token", type=str, required=True, help="the Huggingface Token used to access Llama3 model.")
    parser.add_argument("--dataset", type=str, required=True, help="the name of the dataset")
    parser.add_argument("--cached_kg_triples_file", type=str, required=True, help="the file to save the extracted KG.")

    args = parser.parse_args() 
    return args 


def main(data, dataset, save_file):

    tokenizer, llm = load_llm_tokenizer_and_model("llama3", hf_token=args.hf_token)
    batch_size = 4
    kg_generator = KGGenerator(tokenizer=tokenizer, generator=llm, examplar_type=dataset, batch_size=batch_size)
    
    # load existing results 
    kg_generator.load_cached_kg_triples(paths=[save_file])

    for i in trange((len(data)-1)//batch_size+1, desc="KG Generation Progress"):
        kg_generator(data[i*batch_size: (i+1)*batch_size])
        if i>0 and i%1000 == 0:
            kg_generator.save_cached_kg_triples(path=save_file)
    
    # save cached kg triples 
    kg_generator.save_cached_kg_triples(path=save_file)


if __name__ == "__main__":

    args = setup_parser()

    corpus_file = CORPUS_PATH[args.dataset]
    print("loading corpus data from {} ...".format(corpus_file))
    if args.dataset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        corpus_data = load_json(corpus_file)
    else:
        corpus_data = load_psg_data(corpus_file)
    print("Successfully load {} data!".format(len(corpus_data)))

    main(
        data = corpus_data, 
        dataset = args.dataset, 
        save_file = args.cached_kg_triples_file
    )

