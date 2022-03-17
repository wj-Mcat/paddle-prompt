from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Union
from collections import OrderedDict
from attr import attr

from paddle.fluid.initializer import NumpyArrayInitializer
from paddle import ParamAttr
import numpy as np

from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddle_prompt.config import Tensor
from paddle_prompt.schema import InputFeature

from paddle_prompt.config import Tensor, Config
from paddle_prompt.verbalizers.base_verbalizer import Verbalizer
import paddle


class ManualVerbalizer(Verbalizer):
    def __init__(
        self,
        tokenizer: PretrainedTokenizer,
        label_map: Dict[str, Union[str, List[str]]],
        config: Config,
        prefix: str = ''
    ) -> None:
        assert isinstance(label_map, OrderedDict), 'label_map object must be OrderedDict'

        self.add_prefix(label_map, prefix)
        super().__init__(tokenizer, label_map)
        self.generate_parameters()
        self.config = config 
        
    def add_prefix(self, label_map: Dict[str, Union[str, List[str]]], prefix: str):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.
        
        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        for label, words in label_map.items():
            if isinstance(words, str):
                words = [words]
            label_map[label] = []
            for word in words:
                label_map[label].append(
                    prefix + word
                )
        
    def generate_parameters(self):
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token. 
        """
        # 获取每个word的最大长度
        max_len  = max([max([len(word_ids) for word_ids in words_ids]) for words_ids in self.label_words_ids_dict.values()])
        # 获取每个标签下单词数量的最大长度
        max_num_label_words = max([len(words_ids) for words_ids in self.label_words_ids_dict.values()])                
        words_ids_mask = [[[1]*len(word_ids) + [0]*(max_len-len(word_ids)) for word_ids in words_ids]
                             + [[0]*max_len]*(max_num_label_words-len(words_ids)) 
                             for words_ids in self.label_words_ids_dict.values()]
    
        words_ids = [[word_ids + [0]*(max_len-len(word_ids)) for word_ids in words_ids]
                             + [[0]*max_len]*(max_num_label_words-len(words_ids)) 
                             for words_ids in self.label_words_ids_dict.values()]

        words_ids = np.array(words_ids)
        words_ids_mask = np.array(words_ids_mask)
        
        """
         [
             [356, 246, 456, 0],
             [2356, 3456, 0, 0],
         ]
        """
        self.label_words_ids = paddle.create_parameter(
            words_ids.shape,
            dtype='int32',
            default_initializer=NumpyArrayInitializer(
                words_ids
            ),
            attr=ParamAttr(
                trainable=False,
            )
        )   # [label_num, label_words_num, character_num]
        self.words_ids_mask = paddle.create_parameter(
            words_ids_mask.shape, 
            dtype='int32',
            default_initializer=NumpyArrayInitializer(words_ids_mask),
            attr=ParamAttr(trainable=False)
        ) # [label_num, label_words_num, character_num] the same as the label-words-ids tensor

        # TODO: to be updated
        # self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
                
    def project(
        self,
        logits: Tensor,
    ) -> Tensor:
        r"""
        logits: [batch_size, max_token_num, vocab_size]
        """
        batch_size, max_token_num = len(logits), self.config.max_token_num

        # 1. create the label mask
        label_words_logits = paddle.ones(shape=(batch_size, len(self.label_words_ids)))
        
        # 2. compute the join distribution of labels
        for index in range(max_token_num):
            # [batch_size, token_num, label_num]
            label_logit = paddle.index_select(
                logits,
                index=self.label_words_ids[:, 0, index],
                axis=-1
            )
            # [batch_size, label_num]
            label_words_logits *= label_logit[:, index, :]
        
        return label_words_logits

    def process_logits(self, logits: Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps: 

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The orginal logits.
        
        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project logits to the label space
        label_words_logits = self.project(logits, **kwargs)
        return label_words_logits
        
        if self.post_log_softmax:
            # normalize
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)

            # convert to logits
            label_words_logits = paddle.log(label_words_probs+1e-15)

        # aggreate
        label_logits = self.aggregate(label_words_logits)
        return label_logits 