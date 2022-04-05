"""Ernie Pretrain Mask Language Model"""
from __future__ import annotations
from audioop import reverse
from dataclasses import dataclass
from lib2to3.pgen2.tokenize import tokenize
from typing import Optional, Type, Union

import paddle
from paddle import nn
import paddle.nn.functional as F

from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from paddlenlp.transformers.ernie.modeling import (
    ErniePretrainedModel,
    ErniePretrainingHeads,
    ErnieModel
)
from paddlenlp import transformers
from paddle_prompt.config import Tensor, Config


class ErnieForMLM(ErniePretrainedModel):
    """Ernie Pretrain Mask Language Model"""

    def __init__(self, config: Config):
        super(ErnieForMLM, self).__init__()
        self.ernie = ErnieModel.from_pretrained(config.pretrained_model)
        self.head = ErniePretrainingHeads(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight
        )
        self.apply(self.init_weights)

    def forward(
            self,
            input_ids: Tensor,
            predict_mask: Tensor
    ):
        with paddle.static.amp.fp16_guard():
            outputs: Tensor = self.ernie(input_ids)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, _ = self.head(
                sequence_output,
                pooled_output,
                predict_mask
            )
            return prediction_scores


class ErnieMLMCriterion(paddle.nn.Layer):
    """Criterion for Ernie masked language model"""

    def forward(
            self,
            prediction_scores: Tensor,
            mask_label_ids: Tensor,
            masked_lm_scale: float = 1.0
    ):
        """compute loss for masked language model

        Args:
            prediction_scores (Tensor): _description_
            mask_label_ids (Tensor): _description_
            masked_lm_scale (float, optional): _description_. Defaults to 1.0.

        Returns:
            _type_: _description_
        """
        # shape = [batch_size * max_token_num, 1]   
        mask_label_ids = paddle.reshape(mask_label_ids, shape=[-1, 1])

        # pylint: disable=E1129
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.softmax_with_cross_entropy(
                prediction_scores,
                mask_label_ids,
                ignore_index=-1
            )
            masked_lm_loss = masked_lm_loss / masked_lm_scale
        return paddle.mean(masked_lm_loss)
