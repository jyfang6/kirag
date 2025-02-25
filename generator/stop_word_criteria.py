
from typing import List
import torch
from transformers import StoppingCriteria, AutoTokenizer


class StopWordCriteria(StoppingCriteria):

    def __init__(self, tokenizer: AutoTokenizer, prompt_size: int, stop_words: List[str] = [], check_every: int = 1):
        """
        Initializes the StopWordCriteria with the necessary parameters for checking stop words during text generation.
        
        Parameters:
            tokenizer (AutoTokenizer): The tokenizer for encoding prompts and stop words.
            # prompts (List[str]): Initial prompts used for generation, needed to determine where generated text begins.
            prompt_size (int): used to determine where the generated text begins. (目前只支持left padding)
            stop_words (List[str]): Words that trigger the stopping of generation when detected.
            check_every (int): Frequency of checking for stop words in the token stream (a performance optimization, use 1 to cut it out).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_size = prompt_size

        self.stop_words = stop_words
        self.max_stop_word_size = max((self.tokenizer.encode(word, return_tensors="pt").size(-1) for word in stop_words), default=0)
        self.check_every = check_every

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Determines whether to stop generation based on the presence of stop words.
        
        Stops if a stop word is found in *all* batch elements *and* the sequence length is a multiple of `check_every`.
        Note: Delay in stopping may occur if `check_every > 1`.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            scores (torch.FloatTensor): Generation scores for each token. Not used here.
        
        Returns:
            bool: True to stop generation, False to continue.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Skip check if no stop words are defined or it is not yet time to check
        results = torch.zeros((input_ids.shape[0],), dtype=torch.bool).to(device)

        if (len(self.stop_words) == 0) or (seq_len % self.check_every != 0):
            return results
        
        for i in range(batch_size):
            # Calculate starting index for new tokens
            prompt_size = self.prompt_size
            max_new_tokens = (2 * self.max_stop_word_size) + self.check_every
            latest_tokens = input_ids[i, prompt_size:][-max_new_tokens:]
            if any([word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True) for word in self.stop_words]):
                results[i] = True
            
        return results
