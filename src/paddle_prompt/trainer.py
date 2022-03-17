"""
refer to: https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification
"""
from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional
from matplotlib.pyplot import axis
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.metric.metrics import Metric, Accuracy
from paddle.io import DataLoader
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from paddle.nn import Layer
from paddle.amp.grad_scaler import AmpScaler


import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.model_utils import PretrainedModel
from loguru import logger
from visualdl import LogWriter

from tqdm import tqdm
from paddle_prompt.schema import InputExample, InputFeature
from paddle_prompt.config import Config, Tensor
from paddle_prompt.processors.base_processor import DataProcessor
from paddle_prompt.utils import create_dataloader, extract_and_stack_by_fields, num
from paddle_prompt.plms.ernie import ErnieMLMCriterion
from paddle_prompt.templates.base_template import Template
from paddle_prompt.verbalizers import compute_mask_label_logits
from paddle_prompt.verbalizers.base_verbalizer import Verbalizer


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class ContextContainer(dict):
    
    def __init__(self) -> None:
        self.train_step: int = 0
        self.dev_step: int = 0
        self.epoch: int = 0

        self.train_acc: float = None
        self.dev_acc: float = 0

        self.loss = None
        self.dev_loss = None
        self.logits = None
        self.labels = None

        self._cache = defaultdict(int) 

    def __getitem__(self, key):
        return self._cache[key] 

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value


class Trainer:
    def __init__(self, config: Config, processor: DataProcessor, tokenizer: PretrainedTokenizer, mlm: PretrainedModel, criterion: Layer, template: Template, verbalizer: Verbalizer) -> None:
        self.config = config
        self.set_device()

        # 2. build data related objects
        self.train_dataset = processor.get_train_dataset()
        self.dev_dataset = processor.get_dev_dataset()
        self.test_dataset = processor.get_test_dataset()
        
        self.train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.template.wrap_examples(examples, self.train_dataset.label2idx),
        )
        self.dev_dataloader = create_dataloader(
            self.dev_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.template.wrap_examples(examples, self.dev_dataset.label2idx),
        )
        self.test_dataloader = create_dataloader(
            self.test_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.template.wrap_examples(examples, self.test_dataset.label2idx),
        )

        # 3. init model related
        self.model = mlm
        self.lr_scheduler: LRScheduler = LinearDecayWithWarmup(
            config.learning_rate, 
            total_steps=len(self.train_dataloader) * config.epochs,
            warmup=config.warmup_proportion
        )
        if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
            state_dict = paddle.load(config.init_from_ckpt)
            self.model.set_dict(state_dict)

        self.tokenizer = tokenizer
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.

        decay_params = [
            p.name for n, p in self.model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        self.optimizer: Optimizer = paddle.optimizer.AdamW(
            learning_rate=self.lr_scheduler,
            parameters=self.model.parameters(),
            weight_decay=config.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        self.criterion = criterion

        self.metric: Metric = paddle.metric.Accuracy()

        self.context_data = ContextContainer()
        self._init_output_dir()
        self.writer: LogWriter = LogWriter(logdir=config.output_dir)

        self.template: Template = template 
        self.verbalizer: Verbalizer = verbalizer

    def _init_output_dir(self):
        if os.path.exists(self.config.output_dir):
            shutil.rmtree(self.config.output_dir) 
        os.makedirs(self.config.output_dir)

    def set_device(self):
        paddle.set_device(self.config.device)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

    @paddle.no_grad()
    def evalute(self, dataloader: DataLoader, mode: str = 'dev'):
        logger.success(f'{mode} stage ...')
        self.model.eval()
        self.metric.reset()

        # 1. predict the labels based on dataloader
        all_loss = 0
        pre_label_ids, truth_label_ids = [], []
        
        bar = tqdm(total=len(dataloader))
        for batch in dataloader:
            input_ids, _, prediction_mask, mask_label_ids, labels = batch
            
            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                logits: Tensor = self.model(
                    input_ids=input_ids, 
                    predict_mask=prediction_mask
                )
                batch_size, vocab_size = len(input_ids), logits.shape[-1]
                # [batch_size, label_num]
                label_logits = self.verbalizer.process_logits(
                    paddle.reshape(logits, shape=(batch_size, -1, vocab_size))
                )
                loss = self.criterion(logits, mask_label_ids).detach().numpy().item()
                all_loss += loss
                
            # Get max probs label's index
            y_pred_index = label_logits.argmax(axis=-1).detach().numpy().tolist()
            pre_label_ids.extend(y_pred_index) 
            labels = labels.detach().numpy().tolist()
            truth_label_ids.extend(labels)

            sub_acc = sum([y_pred_index[index] == labels[index] for index in range(len(labels))]) / len(labels)
            
            bar.update()
            bar.set_description('loss: %.5f \t acc: %.5f' % (loss, sub_acc))

        # 2. compute the metric
        assert len(pre_label_ids) == len(truth_label_ids)
        acc = sum([pre_label_ids[index] == truth_label_ids for index in range(len(pre_label_ids))]) / len(pre_label_ids)
        self.context_data.dev_acc = acc
        self.context_data.dev_loss = all_loss

        logger.info("eval accuracy: %.5f \t loss: %.5f" % (acc, all_loss))

        self.model.train()
        self.metric.reset()

        self.context_data.dev_step += 1
        self.writer.add_scalar(tag='eval-acc', value=acc, step=self.context_data.dev_step)

        if acc > self.context_data.dev_step:
            self.context_data.dev_acc = acc
            logger.success(f'saving the best model ...')
            best_model_file = os.path.join(self.config.output_dir, 'best.pdparams')
            paddle.save(self.model.state_dict(), best_model_file)

    def _update_bar_info(self):
        bar_info = []
        bar_info.append('train-loss: %.4f' % num(self.context_data.loss))
        bar_info.append('train-acc: %.4f' % self.context_data.train_acc)
        bar_info.append('dev-acc: %.4f' % self.context_data.dev_acc)

        self.train_bar.set_description('\t'.join(bar_info))

    def on_batch_end(self):
        # 1. update global step
        self.context_data.train_step += 1
        self.train_bar.update()

        # 2. compute acc on training dataset
        self.writer.add_scalar('train-loss', step=self.context_data.train_step, value=self.context_data.loss)
        self.writer.add_scalar('train-acc', step=self.context_data.train_step, value=self.context_data.train_acc)
        self._update_bar_info()

        # 3. step the grad 
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()

        # 4. eval on dev dataset
        if self.context_data.train_step % self.config.valid_steps == 0:
            self.evalute(self.dev_dataloader)
            logger.info(f'saving the model state dict in step: {self.context_data.train_step} ...')
            last_model_file = os.path.join(self.config.output_dir, 'last.pdparams')
            paddle.save(self.model.state_dict(), last_model_file)

    def on_batch_start(self):
        self.metric.reset()
    
    def train_epoch(self, epoch: int):
        self.model.train()
        logger.info(f'training epoch<{epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        for batch in self.train_dataloader:
            input_ids, _, prediction_mask, mask_label_ids, labels = batch

            self.on_batch_start()

            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"], ):
                logits = self.model(
                    input_ids=input_ids, 
                    predict_mask=prediction_mask
                )
                loss = compute_mask_label_logits(logits, mask_label_ids).mean()
                acc = 
                self.context_data.logits = logits
                self.context_data.loss = loss
                self.context_data.labels = labels

            loss.backward()
            self.on_batch_end()

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            if self.config.do_train:
                self.train_epoch(epoch)

            if self.config.do_dev:
                self.evalute(self.test_dataloader, mode='test')
    
    def predict(self):
        pass

    
