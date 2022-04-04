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


from collections import OrderedDict
from typing import Dict, List, Union

import numpy as np
import paddle
from paddle import nn
from paddle import ParamAttr
from paddle.fluid.initializer import ConstantInitializer
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.model_utils import PretrainedModel

from paddle_prompt.config import Tensor, Config
from paddle_prompt.verbalizers.base_verbalizer import Verbalizer


class SoftVerbalizer(Verbalizer):
    """SoftVerbalizer
    implement the method of paper: [WARP: Word-level Adversarial ReProgramming](https://aclanthology.org/2021.acl-long.381/)
    """

    def __init__(
        self,
        plm: PretrainedModel,
        tokenizer: PretrainedTokenizer,
        label_map: Dict[str, Union[str, List[str]]],
        config: Config, 
        prefix: str = '',
        multi_token_handler: str = 'first'
    ) -> None:
        super().__init__(tokenizer, label_map, config, multi_token_handler)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        
        linear_model: nn.Linear = self._find_linear_from_plm(plm)
        if linear_model is None:
            raise ValueError('the model must contain a linear model')

        self.hidden_to_vocab: Tensor = linear_model.weight

        self.head = nn.Linear(
            plm.config.hidden_size,
            self.num_labels, 
            bias_attr=ParamAttr(
                initializer=ConstantInitializer(0.0),
                trainable=False,
            )
        ) 

    def _find_linear_from_plm(self, model: PretrainedModel) -> bool:
        """TODO: check if there is linear model [hidden_dim, num_class] in the plm 

        Args:
            model (PretrainedModel): the pretrained model

        Returns:
            bool: if it exist in plm
        """
        return True
    
    def _init_parameters(self):
        
        
    
