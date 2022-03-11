"""
Implement Paper: https://arxiv.org/abs/2001.07676
"""
from __future__ import annotations

import argparse
import os
import sys
import random
import time
import json
from functools import partial

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from tap import Tap
from loguru import logger
from paddle_prompt.plms.ernie import ErnieForPretraining
from paddle_prompt.models.utils import set_seed


from .data import create_dataloader, transform_fn_dict, convert_example

from model import ErnieMLMCriterion

from evaluate import do_evaluate, do_evaluate_chid
from predict import do_predict, do_predict_chid, predict_file, write_fn


class Config(Tap):
    task_name: str                  # The task_name to be evaluated
    batch_size: int = 32            # Batch size per GPU/CPU for training.
    learning_rate: float = 1e-5     # The initial learning rate for Adam.
    save_dir: str = './checkpoint'  # The output directory where the model checkpoints will be written.
    max_seq_length: int = 128       # The maximum total input sequence length after tokenization. 
    weight_decay: float = 0.0       # Weight decay if we apply some.
    epochs: int = 30                # Total number of training epochs to perform.
    warmup_proportion: float = 0.0  # Linear warmup proption over the training process.
    pattern_id: int = 0             # pattern id of pet
    init_from_ckpt: bool = False    # The path of checkpoint to be loaded.
    seed: int = 1000                # random seeds for initialization
    output_dir: str = './output'    # The output directory where to save output
    device: str = 'gpu'             # Select which device to train model, defaults to gpu.
    save_steps: int= 10000         # Inteval steps to save checkpoint
    index: str = '0'                # must be in [0, 1, 2, 3, 4, all]
    
    pretrained_model: str = 'ernie-1.0' # 
    rdrop_coef: float = 0.0         # The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works 

def eval():
    pass

def train_epoch():
    pass

def train():
    pass

def main():
    # 1. load base configuration
    config: Config = Config().parse_args(known_only=True)
    logger.info('loading configuration ...')
    logger.info(config)

    paddle.set_device(config.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(config.seed)
    label_normalize_json = os.path.join("./label_normalized", config.task_name + ".json")

    # Ernie Model
    model = ErnieForPretraining.from_pretrained(config.pretrained_model)
    tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(config.pretrained_model)

    # map y
    label_norm_dict = None
    with open(label_normalize_json, 'r', encoding="utf-8") as f:
        label_norm_dict = json.load(f)

    convert_example_fn = convert_example

    evaluate_fn = do_evaluate

    predict_fn = do_predict 

    # load dataset
    train_ds, public_test_ds, test_ds = load_dataset(
        "fewclue",
        name=config.task_name,
        splits=("train_0", "test_public", "test"))

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_fn = partial(
        transform_fn_dict[config.task_name],
        label_normalize_dict=label_norm_dict,
        pattern_id=config.pattern_id)

    # Task related transform operations, eg: numbert label -> text_label, english -> chinese
    transform_test_fn = partial(
        transform_fn_dict[config.task_name],
        label_normalize_dict=label_norm_dict,
        is_test=True,
    )

    train_ds = train_ds.map(transform_fn, lazy=False)
    public_test_ds = public_test_ds.map(transform_fn, lazy=False)
    test_ds = test_ds.map(transform_test_fn, lazy=False)

    # dataloader： [input_ids, token_type_ids, masked_positions, masked_lm_labels]
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64"),  # masked_positions
        Stack(dtype="int64"),  # masked_lm_labels
    ): [data for data in fn(samples)]
    batchify_test_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
        Stack(dtype="int64"),  # masked_positions
    ): [data for data in fn(samples)]

    trans_func = partial(
        convert_example_fn,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length)

    trans_test_func = partial(
        convert_example_fn,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        is_test=True)

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=config.batch_size,
        collate_fn=batchify_fn,
        trans_fn=trans_func)

    public_test_data_loader = create_dataloader(
        public_test_ds,
        mode='eval',
        batch_size=config.batch_size,
        collate_fn=batchify_fn,
        trans_fn=trans_func)

    test_data_loader = create_dataloader(
        test_ds,
        mode='eval',
        batch_size=config.batch_size,
        collate_fn=batchify_test_fn,
        trans_fn=trans_test_func)

    num_training_steps = len(train_data_loader) * config.epochs

    lr_scheduler = LinearDecayWithWarmup(config.learning_rate, num_training_steps,
                                         config.warmup_proportion)
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=config.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    # load model if there is
    if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
        state_dict = paddle.load(config.init_from_ckpt)
        model.set_dict(state_dict)
        print("warmup from:{}".format(config.init_from_ckpt))

    mlm_loss_fn = ErnieMLMCriterion()
    rdrop_loss = ppnlp.losses.RDropLoss()
    max_test_acc = 0.0
    global_step = 0
    tic_train = time.time()