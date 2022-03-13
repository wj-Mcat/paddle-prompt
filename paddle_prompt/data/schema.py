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

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from collections import OrderedDict
from anyio import sleep_until
import numpy as np

from dataclasses_json import dataclass_json 

from paddle.io import Dataset


@dataclass_json
@dataclass
class InputExample:
    """Input Example Data Structure for training data
    """
    text: str                                   # source sentence 
    label: Union[str, List[str]]                # label field

    guid: Optional[Union[int, str]] = None      # store the union id for example
    text_pair: Optional[str] = None             # for sentence pair task
    target_text: Optional[str] = None           # for generation task
    meta: Dict[str, Any] = field(default_factory=dict)  # store the meta data of training example

    @property
    def text_or_pairs(self):
        if self.text_pair:
            return self.text, self.text_pair
        return self.text


class ExampleDataset(Dataset):
    def __init__(self, examples: List[InputExample]):
        super().__init__()
        self.examples: List[InputExample] = examples
        self.label2idx: Dict[str, int] = OrderedDict()
        
        for example in examples:
            if example.label not in self.label2idx:
                self.label2idx[example.label] = len(self.label2idx)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int):
        return self.examples[idx]


@dataclass_json
@dataclass
class InputFeature:
    """Input Feature which should be wrapped into PLMs
    """
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]

    label_id: int
