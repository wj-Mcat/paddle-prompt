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
from typing import Dict, List, Any

import numpy as np
import paddle
from paddle.io import Dataset
from paddle.metric import Metric, Accuracy, Precision, Recall

from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer

from paddle_prompt.config import Tensor


def convert_example(example: dict,
                    tokenizer: PretrainedTokenizer,
                    label2idx: Dict[str, int],
                    max_seq_length: int = 512,
                    mode: str = 'train'
                    ):
    """convert single example to input related data

    Args:
        example (InputExample): Single Input Example object
        tokenizer (PretrainedTokenizer): pretrained tokenizer
        max_seq_length (int, optional): max sequence length. Defaults to 512.
        mode (str, optional): the mode of model. Defaults to 'train'.
    """
    encoded_inputs = tokenizer(
        text=example.get('text'),
        text_pair=example.get('text_pair', None),
        max_seq_len=max_seq_length,
        pad_to_max_seq_len=True
    )
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    if mode == 'test':
        return input_ids, token_type_ids

    label_id = label2idx[example['label']]
    label = np.array([label_id], dtype="int64")
    return input_ids, token_type_ids, label


def create_dataloader(dataset: Dataset,
                      mode: str = 'train',
                      batch_size: int = 16,
                      collate_fn=None,
                      trans_fn=None):
    """create dataloader based on dataset

    Args:
        dataset (Dataset): Dataset
        mode (str, optional): mode of model. Defaults to 'train'.
        batch_size (int, optional): batch size in trining epoch. Defaults to 16.
        collate_fn (_type_, optional): transfer data to Tuple data. Defaults to None.
        trans_fn (_type_, optional): convert dataset into features. Defaults to None.
    """
    if trans_fn:
        dataset = dataset.map(trans_fn)

    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=True)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        return_list=True
    )


def extract_and_stack_by_fields(
    encoded_features: List[dict], fields: List[str]
) -> list:
    """if the paddle version is too low, it should extract features from list-dict

    [
        {input_ids: ..., token_type_ids: ...},
        {input_ids: ..., token_type_ids: ...},
        {input_ids: ..., token_type_ids: ...},
    ]

    Args:
        encoded_features (List[dict]): the feature like above example
        fields (List[str]): feature fields
    """
    # 1. check the value type
    if isinstance(encoded_features, dict):
        return [encoded_features[field] for field in fields]

    # 2. extract features from old version output from tokenizer
    tensors = {}
    for field in fields:
        data = [feature[field] for feature in encoded_features]
        tensors[field] = np.array(data)

    return [tensors[field] for field in fields]


def num(tensor_like: Any) -> float:
    """ convert tensor loss to the num
    """
    if paddle.is_tensor(tensor_like):
        tensor_like = paddle.sum(tensor_like)
        return tensor_like.detach().cpu().numpy().item()
    return tensor_like


def to_list(tensor_like) -> List[float]:
    """convert tensors to the list

    Args:
        tensor_like (_type_): tensor object

    Returns:
        list: the python list object
    """
    if paddle.is_tensor(tensor_like):
        return tensor_like.detach().cpu().numpy().tolist()
    return list(tensor_like)


def lists_to_tensors(list_features, place=None) -> List[Tensor]:
    """

    Args:
        list_features (_type_): _description_
        place (_type_, optional): _description_. Defaults to None.

    Returns:
        List[Tensor]: _description_
    """
    tensors = []
    kwargs = {'place': place} if place else {}
    for list_feature in list_features:
        tensors.append(paddle.to_tensor(list_feature, **kwargs))
    return tensors


def get_metric(name: str, **kwargs) -> Metric:
    """

    Args:
        name (str): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Metric: _description_
    """
    if name == 'acc':
        return Accuracy(**kwargs)
    if name == 'precision':
        return Precision(**kwargs)
    if name == 'recall':
        return Recall(**kwargs)

    raise NotImplementedError
