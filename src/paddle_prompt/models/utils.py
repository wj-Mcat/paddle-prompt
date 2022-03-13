from __future__ import annotations
from typing import List

import random
import numpy as np
import paddle
from paddle import nn
from paddle.fluid.framework import Parameter

def freeze_module(module: nn.Layer):
    """Freeze all parameters in the moudle"""
    parameters: List[Parameter] = module.parameters()
    for parameter in parameters:
        parameter.stop_gradient = True

def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
