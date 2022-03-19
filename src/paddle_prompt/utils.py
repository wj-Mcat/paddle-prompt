"""Base Data Processors"""
from __future__ import annotations
from typing import Dict, Optional, List

import numpy as np
import paddle
from paddle.io import Dataset
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddle.metric import Metric, Accuracy, Precision, Recall


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
                      collate_fn = None,
                      trans_fn = None):
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


def extract_and_stack_by_fields(encoded_features: List[dict], fields: List[str]) -> set:
    tensors = {}
    for field in fields:
        data = [feature[field] for feature in encoded_features]
        tensors[field] = np.array(data)
    
    return [tensors[field] for field in fields]


def num(tensor_like):
    if paddle.is_tensor(tensor_like):
        return tensor_like.detach().cpu().numpy().item()
    return tensor_like


def to_list(tensor_like):
    if paddle.is_tensor(tensor_like):
        return tensor_like.detach().cpu().numpy().tolist()
    return list(tensor_like)


def get_metric(name: str, **kwargs) -> Metric:
    if name == 'acc':
        return Accuracy(**kwargs)
    if name == 'precision':
        return Precision(**kwargs)
    if name == 'recall':
        return Recall(**kwargs)

    raise NotImplementedError
