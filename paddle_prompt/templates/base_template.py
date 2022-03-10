from __future__ import annotations

from abc import ABC
from typing import List, Optional
import paddle
from paddle import nn
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.model_utils import PretrainedModel

from paddle_prompt.data import InputExample, InputFeature
from paddle_prompt.models.utils import freeze_module
from paddle_prompt.templates.engine import Engine


class Template(nn.Layer, ABC):
    """
    abstract class for templates in prompt
    """
    
    def __init__(
        self,
        plm: PretrainedModel,
        tokenizer: PretrainedTokenizer,
        render_engine: Engine,
        freeze: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if freeze:
            freeze_module(plm)

        self.tokenizer: PretrainedTokenizer = tokenizer
        self.plm: PretrainedModel = plm
        self.render_engine = render_engine

    def wrap_example(self, example: InputExample) -> InputFeature:
        pass    

    def wrap_feature(self, feature: InputFeature) -> InputFeature:
        pass

    def wrap_features(self, features: List[InputFeature]) -> List[InputFeature]:
        pass
