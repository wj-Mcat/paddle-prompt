from __future__ import annotations
from curses import wrapper
from typing import List
from dataclasses import dataclass
from paddle_prompt.plms.ernie import ErnieForMLM
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer

def _load_ernie_plm(model_name_or_path: str):
    return ErnieTokenizer.from_pretrained(model_name_or_path), ErnieForMLM.from_pretrained(model_name_or_path)
    

def load_plm(model_name: str,  model_name_or_path: str, special_tokens: List[str] = []):
    assert model_name == 'ernie'
    return _load_ernie_plm(model_name_or_path=model_name_or_path)



class TokenizerWrapper:
    def __init__(
        self,
        tokenizer: PretrainedTokenizer, 
        mask_token: str = '[MASK]'
    ) -> None:
        self.mask_token = mask_token
        self.tokenizer = tokenizer
