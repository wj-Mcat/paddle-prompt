"""
Paddle Prompt Learning - https://github.com/wj-Mcat/paddle-prompt

Authors:    Jingjing WU (吴京京) <https://github.com/wj-Mcat>


2022-now @ Copyright wj-Mcat

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations
from curses import textpad
import inspect
from typing import List, Dict, Any, Optional, Tuple, Union
from paddlenlp.data.vocab import Vocab
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.bert.tokenizer import BertTokenizer
from paddlenlp.transformers.bart.tokenizer import BartTokenizer
from paddlenlp.transformers.t5.tokenizer import T5Tokenizer


class TokenzierWrapper:
    def __init__(self, tokenizer: Union[str, PretrainedTokenizer]) -> None:

        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

    def __call__(self,
                 text,
                 text_pair=None,
                 max_seq_len: Optional[int] = None,
                 stride=0,
                 is_split_into_words=False,
                 pad_to_max_seq_len=False,
                 truncation_strategy="longest_first",
                 return_position_ids=False,
                 return_token_type_ids=True,
                 return_attention_mask=False,
                 return_length=False,
                 return_overflowing_tokens=False,
                 return_special_tokens_mask=False):
        return self.tokenizer(
            text,
            text_pair=text_pair,
            max_seq_len=max_seq_len,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_max_seq_len=pad_to_max_seq_len,
            truncation_strategy=truncation_strategy,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask
        )

    def encode(self,
               text,
               text_pair=None,
               max_seq_len: Optional[int] = None,
               stride=0,
               is_split_into_words=False,
               pad_to_max_seq_len=False,
               truncation_strategy="longest_first",
               return_position_ids=False,
               return_token_type_ids=True,
               return_attention_mask=False,
               return_length=False,
               return_overflowing_tokens=False,
               return_special_tokens_mask=False):
        return self.tokenizer(
            text,
            text_pair=text_pair,
            max_seq_len=max_seq_len,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_max_seq_len=pad_to_max_seq_len,
            truncation_strategy=truncation_strategy,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask
        )

    def batch_encode(self,
                     batch_texts_or_text_pairs: List[Union[str, Tuple[str, str]]],
                     max_seq_len: Optional[int] = None,
                     stride=0,
                     is_split_into_words=False,
                     pad_to_max_seq_len=False,
                     truncation_strategy="longest_first",
                     return_position_ids=False,
                     return_token_type_ids=True,
                     return_attention_mask=False,
                     return_length=False,
                     return_overflowing_tokens=False,
                     return_special_tokens_mask=False):
        return self.tokenizer.batch_encode(
            batch_texts_or_text_pairs,
            max_seq_len=max_seq_len,
            stride=stride,
            is_split_into_words=is_split_into_words,
            pad_to_max_seq_len=pad_to_max_seq_len,
            truncation_strategy=truncation_strategy,
            return_position_ids=return_position_ids,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_length=return_length,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask
        )

    def get_unused_tokens(self) -> List[str]:
        methods = [
            member[1] for member in inspect.getmembers(self)
            if inspect.ismethod(member[1]) and member[0].startswith('_get_unsed_tokens_of_')
        ]
        if not methods:
            raise ValueError(
                f'can"t find unused tokens of {type(self.tokenizer)}')

        for method in methods:
            tokens = method()
            if tokens is None:
                continue
            return tokens
        raise ValueError(f'unsupported tokenizer of {type(self.tokenizer)}')

    def _get_unsed_tokens_of_bert(self,):
        from paddlenlp.transformers.bert.tokenizer import BertTokenizer
        if not isinstance(self.tokenizer, BertTokenizer):
            return None
        vocab: Vocab = self.tokenizer.vocab
        return [token for token in vocab._token_to_idx.keys() if token.startswith('[unused')]

    def _get_unsed_tokens_of_bart(self,):
        # TODO: there is no unused tokens in bart tokenizer
        from paddlenlp.transformers.bart.tokenizer import BartTokenizer
        if not isinstance(self.tokenizer, BartTokenizer):
            return None
        encoder: Dict[str, int] = self.tokenizer.encoder

        # select last 10 words as the unused tokens, the performance is inefficient
        return [token for token in encoder.keys()][-10:]

    def _get_unsed_tokens_of_t5(self,):
        if not isinstance(self.tokenizer, T5Tokenizer):
            return None
        return [token for token in self.tokenizer.all_special_tokens if token.startswith('<extra_id_')]
