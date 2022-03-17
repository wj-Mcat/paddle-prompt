from __future__ import annotationss
from turtle import forward

import paddle
from paddle import nn
from paddle_prompt.config import Tensor


class JointDistributionLoss(nn.Layer):
    """labels joint distribution loss
    """
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, prediction_logits: Tensor, label_ids: Tensor):
        pass
        