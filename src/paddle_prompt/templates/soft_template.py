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
from ctypes import Union
from typing import Optional
import paddle

from paddle import ParamAttr
from paddle.fluid.initializer import NumpyArrayInitializer

from paddlenlp.transformers.auto.modeling import AutoModel

from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from numpy import ndarray

from paddle_prompt.config import Config, Tensor
from paddle_prompt.templates.base_template import Template


class SoftTemplate(Template):
    """Soft Template that use the word embedding from the pretrained model
    implement method of paper: [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)
    """
    def __init__(
        self,
        model_or_path: Union[PretrainedModel],
        config: Config,
        tokenizer: PretrainedTokenizer,
        soft_embeddings: Union[Tensor, ndarray] = None,
        num_tokens: Optional[int] = None,
        **kwargs
    ):
        super().__init__(tokenizer, config, **kwargs)

        # 1. check the parameter
        if soft_embeddings is None and num_tokens is None:
            raise ValueError('one of soft_embeddings or num_tokens must be specified')

        # 2. init the model
        if isinstance(model_or_path, str):
            model: PretrainedModel = AutoModel.from_pretrained(model_or_path, config=config)
        else:
            model = model_or_path

        self.vocab_embedding = model.get_input_embeddings()
        self.num_tokens = num_tokens

        # 3. init the soft embedding as the parameter
        if paddle.is_tensor(soft_embeddings):
            soft_embeddings = soft_embeddings.numpy()

        self.soft_embeddings = self.create_parameter(
            soft_embeddings.shape,
            dtype='int32',
            default_initializer=NumpyArrayInitializer(
                soft_embeddings
            ),
            attr=ParamAttr(
                trainable=False,
            )
        )  # [num_soft_tokens, embedding_size]
        
    
    def forward(self, input_ids) -> Tensor:
        """process the input_ids and output the input embeddings

        Args:
            input_ids (Tensor): the input ids of sentence
        """
        # 1. get the input embedding from pretrained model

        input_embedding = self.vocab_embedding(input_ids)
        batch_size = input_ids.shape[0]

        # 2. cat the prefix embedding into the input embedding
        if self.num_tokens > 0:
            soft_embedding = self.soft_embeddings.tile([batch_size, 1, 1])
            input_embedding = paddle.concat([soft_embedding, input_embedding], axis=1)

        return input_embedding

    def post_process(self, logits: Tensor):
        """post process the logits

        Args:
            logits (Tensor): the logit of moddel

        Returns:
            Tensor: the post processed logits
        """
        # TODO: check if the pretrained model is encoder-decoder based
        if self.num_tokens > 0:
            logits = logits[:, self.num_tokens:,: ]
        return logits