from __future__ import annotations

from abc import ABC
from distutils.command.config import config
import json
from typing import Dict, List, Optional
from copy import deepcopy
from matplotlib.pyplot import text
import numpy as np
import paddle
from paddle import nn
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.model_utils import PretrainedModel

from paddle_prompt.data.schema import InputExample, InputFeature
from paddle_prompt.models.utils import freeze_module
from paddle_prompt.templates.engine import Engine, JinjaEngine
from paddle_prompt.config import Config
from paddle_prompt.data.utils import extract_and_stack_by_fields

def _resize_prediction_mask(text: str, label_size: int) -> str:
    mask_str = '[MASK]'
    return text.replace(mask_str, ''.join([mask_str] * label_size))

def _load_label2words(file: str) -> Dict[str, str]:
    label2words = {}
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for label, label_obj in data.items():
            label2words[label] = label_obj['labels'][0]
    return label2words

class Template(nn.Layer, ABC):
    """
    abstract class for templates in prompt
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
        self.label2words = _load_label2words(config.template_file)

    def wrap_examples(self, examples: List[InputExample], label2idx: Dict[str, int] = None):
        if not label2idx:
            label2idx = self.config.label2idx

        # 1. construct text or text pair dataset
        texts = [self.render_engine.render(example) for example in examples] 
        texts = [_resize_prediction_mask(text, self.config.label_size) for text in texts]
        encoded_features = self.tokenizer.batch_encode(
            texts,
            max_seq_len=self.config.max_seq_length,
            pad_to_max_seq_len=True,
            return_token_type_ids=True
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
                example_label_ids = [label2idx[label] for label in example.label]
                label_ids.append(example_label_ids)
        
        features = extract_and_stack_by_fields(encoded_features, fields)

        # 3. construct prediction mask
        mask_token_id = self.tokenizer.mask_token_id
        mask_label_mask = np.array(features[0]) == mask_token_id
        np_prediction_mask = np.argwhere(mask_label_mask)
        prediction_mask = []
        for pre_mask in np_prediction_mask:
            prediction_mask.append(pre_mask[0] * self.config.max_seq_length + pre_mask[1])
        features.append(np.array(prediction_mask))

        # 4. constrct mask_label_ids
        mask_label_ids = []
        for example in examples:
            mask_label_ids.extend(
                self.tokenizer.convert_tokens_to_ids(
                    list(self.label2words[example.label])
                )
            )
        features.append(np.array(mask_label_ids))
        
        # 4. add label ids data
        features.append(
            np.array(label_ids)
        )
        return features

    

    def wrap_feature(self, feature: InputFeature) -> InputFeature:
        pass

    def wrap_features(self, features: List[InputFeature]) -> List[InputFeature]:
        pass
