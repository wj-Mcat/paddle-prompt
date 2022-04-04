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
from __future__ import annotationss

from paddle import nn

from paddle_prompt.config import Tensor


class JointDistributionLoss(nn.Layer):
    """labels joint distribution loss
    """

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, prediction_logits: Tensor, label_ids: Tensor):
        pass
