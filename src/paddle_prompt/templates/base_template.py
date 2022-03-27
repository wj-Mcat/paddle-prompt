"""Base Abstract Template class"""
from __future__ import annotations

import json
from abc import ABC
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import paddle
from paddle import nn
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from paddle_prompt.config import Config
from paddle_prompt.schema import InputExample, InputFeature
from paddle_prompt.templates.engine import JinjaEngine
from paddle_prompt.utils import extract_and_stack_by_fields, lists_to_tensors


def _resize_prediction_mask(text: str, label_size: int) -> str:
    mask_str = '[MASK]'
    return text.replace(mask_str, ''.join([mask_str] * label_size))


def _load_label2words(file: str) -> Dict[str, List[str]]:
    label2words = OrderedDict()
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for label, label_obj in data.items():
            label2words[label] = label_obj['labels']
    return label2words


class SoftMixin:
    """Soft Template Mixin object which can handle the soft token"""

    def soft_token_ids(self) -> List[int]:
        """
        This function identifies which tokens are soft tokens.

        Sometimes tokens in the template are not from the vocabulary,

        but a sequence of soft tokens.
        In this case, you need to implement this function
        """
        raise NotImplementedError


class Template(nn.Layer):
    """
    abstract class for templates in prompt

    TODO: how to handle -> fill the target label in the mask place
    """

    def __init__(
            self,
            tokenizer: PretrainedTokenizer,
            config: Config,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.render_engine = JinjaEngine.from_file(config.template_file)
        self.tokenizer: PretrainedTokenizer = tokenizer
        self.config: Config = config
        self.label2words: Dict[str, List[str]] = _load_label2words(
            config.template_file
        )
        self._init_max_token_num()

    def _init_max_token_num(self):
        max_token_num = 0
        for words in self.label2words.values():
            for word in words:
                max_token_num = max(max_token_num, len(word))
        self.config.max_token_num = max_token_num

    def _get_mask_id(self) -> int:
        # TODO: to be removed, this code is to fix the issue of paddlenlp
        special_tokens = [token for token in self.tokenizer.all_special_tokens if token != self.config.mask_token]
        special_ids: List[int] = self.tokenizer.convert_tokens_to_ids(special_tokens)
        ids = self.tokenizer.convert_tokens_to_ids([self.config.mask_token])
        ids = [id for id in ids if id not in special_ids]
        assert len(ids) == 1, 'can"t get [MASK] id from tokenizer'
        return ids[0]

        
    def wrap_examples(
        self,
        examples: List[InputExample],
        label2idx: Dict[str, int] = None
    ):
        """wrap examples with template and convert them to features
            which can be feed into MLM

        Args:
            examples (List[InputExample]): the examples object
            label2idx (Dict[str, int], optional): label to index mapper.
                Defaults to None.

        Returns:
            List[Tensor]: the features which will be feed into MLM
        """
        if not label2idx:
            label2idx = self.config.label2idx

        # 1. construct text or text pair dataset
        texts = [self.render_engine.render(example) for example in examples]
        texts = [_resize_prediction_mask(
            text, self.config.max_token_num) for text in texts]
        encoded_features = self.tokenizer.batch_encode(
            texts,
            max_seq_len=self.config.max_seq_length,
            pad_to_max_seq_len=True,
            return_token_type_ids=True,
        )
        fields = ['input_ids', 'token_type_ids']

        # 2. return different data based on label
        has_label = examples[0].label is not None
        if not has_label:
            return extract_and_stack_by_fields(encoded_features, fields)

        label_ids = []
        is_multi_class = isinstance(examples[0].label, list)
        if not is_multi_class:
            label_ids = [label2idx[example.label] for example in examples]
        else:
            for example in examples:
                example_label_ids = [label2idx[label]
                                     for label in example.label]
                label_ids.append(example_label_ids)

        features = extract_and_stack_by_fields(encoded_features, fields)

        # 3. construct prediction mask
        mask_token_id = self._get_mask_id()
        
        mask_label_mask = np.array(features[0]) == mask_token_id
        np_prediction_mask = np.argwhere(mask_label_mask)
        prediction_mask = []
        for pre_mask in np_prediction_mask:
            prediction_mask.append(
                pre_mask[0] * self.config.max_seq_length + pre_mask[1])
        features.append(np.array(prediction_mask))

        # 4. construct mask_label_ids
        mask_label_ids = []
        for example in examples:
            mask_label_ids.extend(
                self.tokenizer.convert_tokens_to_ids(
                    # TODO: to handle the multiple words?
                    list(self.label2words[example.label][0])
                )
            )
        features.append(np.array(mask_label_ids))

        # 4. add label ids data
        features.append(
            np.array(label_ids)
        )

        features = lists_to_tensors(features, self.config.place())
        return features

    def forward(self, *args, **kwargs) -> Any:
        """should handle the template mainforce logit

        Returns:
            Any: any result. TODO: define the forward result data structure.
        """

