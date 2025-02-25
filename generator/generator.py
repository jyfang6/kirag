from tqdm import trange
from typing import List, Dict, Union, Optional, Any, Tuple

import torch.nn as nn 
from torch import Tensor
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    LlamaForCausalLM, 
    Qwen2ForCausalLM,
    MistralForCausalLM,
    Gemma2ForCausalLM, 
    T5ForConditionalGeneration,
    StoppingCriteriaList,
)

from utils.utils import * 
from generator.utils import (
    pad_token_ids, 
    pad_token_logits,
    append_texts_to_decoder_only_generator_inputs,
    append_texts_to_encoder_decoder_generator_inputs
)

SUPPORTED_DECODER_ONLY_GENERATORS = [LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Gemma2ForCausalLM]
SUPPORTED_ENCODER_DECODER_GENERATORS = [T5ForConditionalGeneration]


class Generator(nn.Module):

    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        generator: AutoModelForCausalLM, 
        max_length: int=4096, 
        max_new_tokens: int=128,
        batch_size: int=4, 
        **kwargs
    ):
        super().__init__()

        supported_generator_types = tuple(SUPPORTED_DECODER_ONLY_GENERATORS+SUPPORTED_ENCODER_DECODER_GENERATORS)
        assert isinstance(generator, supported_generator_types) 

        self.tokenizer = tokenizer
        self.generator = generator
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.check_tokenizer_padding()

        self.is_chat = kwargs.get("is_chat", None) or self.init_is_chat()
        self.is_encoder_decoder = kwargs.get("is_encoder_decoder", None) or self.init_is_encoder_decoder()
        self.config = self.generator.config 
        self.config.update(kwargs)
    
    @property
    def device(self):
        return self.generator.device 
    
    @property
    def dtype(self):
        return self.generator.dtype

    def init_is_chat(self):
        model_name_or_path = self.generator.config._name_or_path.lower()
        if "instruct" in model_name_or_path or "chat" in model_name_or_path or "-it" in model_name_or_path:
            is_chat = True
        else:
            is_chat = False
        return is_chat
    
    def init_is_encoder_decoder(self):
        if isinstance(self.generator, tuple(SUPPORTED_ENCODER_DECODER_GENERATORS)):
            is_encoder_decoder = True 
        elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            is_encoder_decoder = False
        else:
            raise ValueError(f"{type(self.generator)} is an unknow generator!")
        return is_encoder_decoder
    
    def check_tokenizer_padding(self):
        if isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            if self.tokenizer.pad_token is None or self.tokenizer.pad_token_id is None:
                raise ValueError("pad_token or pad_token_id is None in the tokenizer of generator. suggest to set pad_token and pad_token_id to eos_token and eos_token_id respectively.")
            if self.tokenizer.padding_side == "right":
                raise ValueError("Dected right padding using decoder-only transformers as the generator, which may cause some errors. It is suggested to use \"left\" padding!")
    
    def get_generator_prompts_chat_format(
        self, 
        instructions: List[str], 
        messages: Union[List[List[dict]], List[str]],
        **kwargs
    ) -> List[List[Dict[str, str]]]:
        """
        Input: 
            instruction: [str]
            messages: [str] or [[{"user": "user_content"}, {"assistant": "assistant_content"}],...]
        Output:
            prompts: [[{"role": xxx, "content": xxx}, {"role": xxx, "content": xxx}]]
        """
        prompts = [] 
        assert len(instructions) == len(messages) # number of instructions shoule be the same as messages 
        for instruction, message_list in zip(instructions, messages):
            if isinstance(self.generator, (LlamaForCausalLM, Qwen2ForCausalLM)):
                one_prompt = [{"role": "system", "content": instruction}]
                if isinstance(message_list, str):
                    one_prompt.append({"role": "user", "content": message_list})
                elif isinstance(message_list, list):
                    assert "user" in message_list[0] # # the first message must comes from user in the form of: {"user": "user_message"}
                    for message in message_list:
                        if "user" in message:
                            one_prompt.append({"role": "user", "content": message["user"]})
                        # if "system" in message:
                        #     one_prompt.append({"role": "system", "content": message["system"]})
                        if "assistant" in message:
                            one_prompt.append({"role": "assistant", "content": message["assistant"]})
                else:
                    raise ValueError(f"Invalid message type: {type(message_list)}. Only support str or List[dict] messages")
                prompts.append(one_prompt)
            elif isinstance(self.generator, (MistralForCausalLM, Gemma2ForCausalLM)):
                # Mistral Don't have System Role 
                if isinstance(message_list, str):
                    one_prompt = [{"role": "user", "content": instruction + "\n\n" + message_list}]
                elif isinstance(message_list, list):
                    assert "user" in message_list[0] # the first message must comes from user in the form of: {"user": "user_message"}
                    one_prompt = [{"role": "user", "content": instruction + "\n\n" + message_list[0]["user"]}]
                    for message in message_list[1:]:
                        if "user" in message:
                            one_prompt.append({"role": "user", "content": message["user"]})
                        if "assistant" in message:
                            one_prompt.append({"role": "assistant", "content": message["assistant"]})
                else:
                    raise ValueError(f"Invalid message type: {type(message_list)}. Only support str or List[dict] messages")
                prompts.append(one_prompt)
            else:
                raise NotImplemented(f"chat format for {type(self.generator)} is not implemented yet!")
        return prompts
    
    def tokenizer_encode_chat_format(self, prompts: List[List[Dict[str, str]]], max_length: int=None, add_generation_prompt: bool=True, **kwargs) -> Dict[str, Tensor]:
        max_length = self.max_length if max_length is None else max_length
        texts = self.tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=add_generation_prompt) 
        batch_dict = self.tokenizer(texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def tokenizer_encode(self, prompts: List[str], max_length: int=None, **kwargs) -> Dict[str, Tensor]:
        max_length = self.max_length if max_length is None else max_length
        batch_dict = self.tokenizer(prompts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        tokenizer_outputs = {"input_ids": batch_dict["input_ids"], "attention_mask": batch_dict["attention_mask"]}
        return tokenizer_outputs
    
    def get_generated_token_ids(self, input_ids: Tensor, token_ids: Tensor) -> Tensor:
        if isinstance(self.generator, T5ForConditionalGeneration): 
            generated_token_ids = token_ids[:, 1:] 
        elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
            generated_token_ids = token_ids[:, input_ids.shape[1]:]
        else:
            raise NotImplementedError(f"get_generated_token_ids is not implemented for {type(self.generator)}!")
        return generated_token_ids
    
    def get_stop_symbols_stopping_criteria(self, prompt_size: int, stop_words: Union[str, List[str]]) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        from generator.stop_word_criteria import StopWordCriteria
        criteria.append(
            StopWordCriteria(tokenizer=self.tokenizer, prompt_size=prompt_size, stop_words=stop_words)
        )
        return criteria 
    
    def greedy_generate(
        self, 
        inputs: Dict[str, Tensor],
        pad_to_max_new_tokens: bool=False,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        Inputs: 
        {"input_ids": Tensor, "attention_mask": Tensor}

        Outputs:
        Tensor, Tensor
        """
        device = self.device
        batch_size = kwargs.get("batch_size", None) or self.batch_size
        max_new_tokens = kwargs.get("max_new_tokens", None) or self.max_new_tokens
        stopping_criteria = kwargs.get("stopping_criteria", None)
        verbose = kwargs.get("verbose", False)
        
        if verbose:
            progress_bar = trange((len(inputs["input_ids"])-1)//batch_size+1, desc="Generation Progress")

        generated_token_ids_list, generated_token_logits_list = [], [] 
        for i in range((len(inputs["input_ids"])-1)//batch_size+1):
            batch_inputs = {k: v[i*batch_size: (i+1)*batch_size] for k, v in inputs.items()}
            batch_inputs = to_device(batch_inputs, device)
            batch_outputs = self.generator.generate(
                **batch_inputs, 
                max_new_tokens=max_new_tokens, 
                output_scores=True, 
                return_dict_in_generate=True, 
                do_sample=False, 
                temperature=1.0,
                stopping_criteria=stopping_criteria,
            )
            batch_generated_token_ids = self.get_generated_token_ids(batch_inputs["input_ids"], batch_outputs.sequences).detach().cpu()
            batch_generated_token_logits = torch.cat([token_scores.unsqueeze(1) for token_scores in batch_outputs.scores], dim=1).detach().cpu()
            
            generated_token_ids_list.append(batch_generated_token_ids)
            generated_token_logits_list.append(batch_generated_token_logits)
            if verbose:
                progress_bar.update(1)
        
        max_generation_length = max_new_tokens if pad_to_max_new_tokens else \
            max([x.shape[-1] for x in generated_token_ids_list])
        generated_token_ids_list = [
            pad_token_ids(
                token_ids, 
                max_length=max_generation_length, 
                pad_token_id=self.tokenizer.pad_token_id
            ) 
            for token_ids in generated_token_ids_list
        ]
        generated_token_logits_list = [
            pad_token_logits(
                token_logits, 
                max_length=max_generation_length
            ) 
            for token_logits in generated_token_logits_list
        ]

        generated_token_ids = torch.cat(generated_token_ids_list, dim=0)
        generated_token_logits = torch.cat(generated_token_logits_list, dim=0)

        return generated_token_ids, generated_token_logits
        
    def generate(self, inputs, **kwargs) -> Tuple[Tensor, Tensor]:
        max_new_tokens = kwargs.get("max_tokens", None) or kwargs.get("max_new_tokens", None)
        if max_new_tokens is None:
            kwargs["max_new_tokens"] = self.max_new_tokens
        batch_size = kwargs.get("batch_size", None)
        if batch_size is None:
            kwargs["batch_size"] = self.batch_size

        stopping_criteria = None
        if kwargs.get("stop_words", None) is not None:
            if isinstance(self.generator, T5ForConditionalGeneration):
                prompt_size = 1 # <bos> token
            elif isinstance(self.generator, tuple(SUPPORTED_DECODER_ONLY_GENERATORS)):
                prompt_size = inputs["input_ids"].shape[-1] 
            else:
                raise ValueError(f"{type(self.generator)} is not a supported generator!")
            stopping_criteria = self.get_stop_symbols_stopping_criteria(prompt_size, kwargs["stop_words"])
        kwargs["stopping_criteria"] = stopping_criteria

        return self.greedy_generate(inputs, **kwargs)

    def prompt(self, instructions: List[str], inputs: Union[List[List[dict]], List[str]], **kwargs) -> List[str]:

        assert len(instructions) == len(inputs)
        if not self.is_encoder_decoder and self.is_chat:
            prompts_chat_format = self.get_generator_prompts_chat_format(
                instructions=instructions, messages=inputs, **kwargs
            )
            prompts = self.tokenizer.apply_chat_template(prompts_chat_format, tokenize=False, add_generation_prompt=True)
        else:
            assert all([isinstance(user_input, str) for user_input in inputs]) 
            prompts = [inst + "\n\n" + user_input for inst, user_input in zip(instructions, inputs)]
        return prompts
        
    def generator_generate(self, instructions: List[str], inputs: List[str], current_generated_texts: List[str]=None, **kwargs):

        assert len(instructions) == len(inputs)
        if current_generated_texts is not None:
            assert len(instructions) == len(current_generated_texts)
        
        if self.is_encoder_decoder:
            prompts = [inst + "\n\n" + user_input for inst, user_input in zip(instructions, inputs)]
            generator_inputs = self.tokenizer_encode(prompts)
            if current_generated_texts is not None:
                generator_inputs = append_texts_to_encoder_decoder_generator_inputs(
                    tokenizer=self.tokenizer, inputs=generator_inputs, texts=current_generated_texts,
                        decoder_start_token_id=self.config.decoder_start_token_id
                )
        else:
            if self.is_chat:
                prompts_chat_format = self.get_generator_prompts_chat_format(
                    instructions=instructions, messages=inputs, **kwargs
                )
                generator_inputs = self.tokenizer_encode_chat_format(prompts_chat_format, **kwargs)
                if current_generated_texts is not None:
                    generator_inputs = append_texts_to_decoder_only_generator_inputs(
                        tokenizer=self.tokenizer, inputs=generator_inputs, texts=current_generated_texts
                    )
            else:
                prompts = [inst + "\n\n" + user_input for inst, user_input in zip(instructions, inputs)]
                if current_generated_texts is not None:
                    prompts = [prompt + " " + text for prompt, text in zip(prompts, current_generated_texts)]
                generator_inputs = self.tokenizer_encode(prompts)
        
        generated_token_ids, generated_token_logits = self.generate(generator_inputs, **kwargs)
        return generated_token_ids, generated_token_logits
    

class AnswerGenerator(Generator):

    def __init__(self, tokenizer: AutoTokenizer, generator: AutoModelForCausalLM, max_length: int = 4096, max_new_tokens: int = 128, batch_size: int = 4, **kwargs):

        super().__init__(tokenizer, generator, max_length, max_new_tokens, batch_size, **kwargs)
        self.task_instruction = kwargs.get("task_instruction", None) 
        self.task_instruction_wo_context = "Given a question, please only output the answer to the question."
        self.task_instruction_with_context = "Given some context and a question, please only output the answer to the question."
        self.task_instruction_cot = "Answer the following question by reasoning step-by-step. After reasoning, you MUST use \"So the answer is:\" to output the answer."
        self.use_cot = False
        self.answer_prefix = "The answer is:" if not self.use_cot else "Thought:"

    def get_generator_inputs(
        self, 
        questions: List[str], 
        contexts: Optional[List[List[str]]]=None, 
        task_instructions: Optional[Union[str, List[str]]]=None, 
        **kwargs, 
    ) -> Tuple[List[str], List[str]]:
        
        if task_instructions is None:
            if self.task_instruction is not None:
                task_instructions = [self.task_instruction] * len(questions)
            else:
                if self.use_cot:
                    instruction = self.task_instruction_cot
                    instruction += "\n\nExamples:\n{}".format("\n\n".join(self.cot_examplars))
                else:
                    instruction = self.task_instruction_wo_context if contexts is None else self.task_instruction_with_context
                task_instructions = [instruction] * len(questions)
        
        user_inputs = [] 
        for i, question in enumerate(questions):
            user_input = "" 
            if contexts is not None:
                context = contexts[i]
                context_text = "\n\n".join(["{}. {}".format(j+1, text) for j, text in enumerate(context)])
                user_input += f"context:\n\n{context_text}\n\n"
            user_input += f"question: {question}\n{self.answer_prefix}"
            user_inputs.append(user_input)
        
        return task_instructions, user_inputs

    def parse_generated_answers(self, texts: List[str]) -> List[str]:

        def parse_answer(answer: str) -> str:
            candidate_answers = answer.split("\n")
            answer = ""
            i = 0 
            while len(answer) < 1 and i<len(candidate_answers):
                answer = candidate_answers[i].strip()
                i += 1 
            if "answer is" in answer:
                idx = answer.find("answer is")
                answer = answer[idx+len("answer is"): ].strip()
                if answer.startswith(":"):
                    answer = answer[1:].strip()
            return answer
        
        return [parse_answer(text) for text in texts]

    def batch_generate_answers(
        self, 
        questions: List[str], 
        contexts: Optional[List[List[str]]]=None,
        task_instructions: Optional[Union[str, List[str]]]=None, 
        **kwargs, 
    ):
        if task_instructions is not None and isinstance(task_instructions, str):
            task_instructions = [task_instructions]*len(questions)
        if contexts is not None:
            assert len(questions) == len(contexts)
        if task_instructions is not None:
            assert len(questions) == len(task_instructions)
        
        instructions, user_inputs = self.get_generator_inputs(
            questions=questions, 
            contexts=contexts, 
            task_instructions=task_instructions,
            **kwargs
        )
        generated_token_ids, _ = self.generator_generate(
            instructions=instructions, 
            inputs=user_inputs, 
            **kwargs
        )
        generated_texts = self.tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)
        answers = self.parse_generated_answers(generated_texts)

        return answers
    
    def generate_answer(
        self, 
        question: Union[List[str], str], 
        context: Optional[Union[List[List[str]], List[str]]]=None,
        task_instruction: Optional[Union[str, List[str]]]=None, 
        **kwargs, 
    ):
        single_question = False
        if isinstance(question, str):
            single_question = True
            question=[question]
            context = [context] if context is not None else None
        answers = self.batch_generate_answers(
            questions=question, 
            contexts=context, 
            task_instructions=task_instruction,
            **kwargs
        )
        results = answers[0] if single_question else answers

        return results

