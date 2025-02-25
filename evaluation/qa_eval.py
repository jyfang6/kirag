import os 
import logging
import argparse
import numpy as np 
from tqdm import trange

import torch 
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

from utils.utils import load_json
from evaluation.metrics import ems, f1_score
from generator.generator import AnswerGenerator
from utils.pipeline_utils import load_llm_tokenizer_and_model


logger = logging.getLogger(__file__)

def setup_parser():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hf_token", type=str, required=True) 
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--reader", type=str, default="llama3")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()
    return args


def main(args, reader: AnswerGenerator):
    
    print(f"loading data from {args.save_file}")
    data = load_json(args.save_file)
    question_list = [] 
    context_list = None if args.k <=0 else []
    answers_list = [] 
    for example in data:
        question_list.append(example["question"])
        answers = example["answers"]
        answers_list.append(answers if isinstance(answers, list) else [answers])
        if context_list is not None:
            context = [] 
            for ctx in example["ctxs"][:args.k]:
                text = ctx["text"] if "text" in ctx else " ".join(ctx["sentences"])
                if "title" in ctx:
                    context.append("title: {}, text: {}".format(ctx["title"], text))
                else:
                    context.append(text)            
            context_list.append(context)

    pred_answers_list = []
    for i in trange((len(question_list)-1)//args.batch_size+1, desc="Answer Prediction Progress"):
        batch_question_list = question_list[i*args.batch_size: (i+1)*args.batch_size]
        batch_context_list = context_list[i*args.batch_size: (i+1)*args.batch_size]
        batch_pred_answers_list = reader.generate_answer(question=batch_question_list, context=batch_context_list)
        pred_answers_list.extend(batch_pred_answers_list)
    
    em_scores, f1_scores = [], [] 
    for pred_answer, gold_answers in zip(pred_answers_list, answers_list):
        ems_score = ems(pred_answer, gold_answers)
        f1 = f1_score(pred_answer, gold_answers[0])[0]
        em_scores.append(ems_score)
        f1_scores.append(f1)
    
    avg_ems = np.mean(em_scores)
    avg_f1 = np.mean(f1_scores)

    print("==================== Evaluation Result ====================")
    print(">>>> File: {}".format(args.save_file))
    print(">>>> Topk: {}".format(args.k))
    print(">>>> Reader: {}".format(args.reader))
    print(">>>> EM: {:.5f}".format(avg_ems))
    print(">>>> F1: {:.5f}".format(avg_f1))
    print("===========================================================")
     

if __name__ == "__main__":
    
    args = setup_parser()

    tokenizer, model = load_llm_tokenizer_and_model(args.reader, hf_token=args.hf_token, dtype=torch.bfloat16)
    reader = AnswerGenerator(tokenizer=tokenizer, generator=model, max_new_tokens=32, batch_size=args.batch_size)

    main(args, reader)
    