"""
refer to:
    https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/text_classification
"""
from __future__ import annotations

import os
import random
import shutil
from collections import defaultdict
from typing import Any

import numpy as np
import paddle
from loguru import logger
from paddle.io import DataLoader
from paddle.metric.metrics import Metric
from paddle.nn import Layer
from paddle.optimizer import Optimizer
from paddle.optimizer.lr import LRScheduler
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from tqdm import tqdm
from visualdl import LogWriter

from paddle_prompt.config import Config, Tensor
from paddle_prompt.processors.base_processor import DataProcessor
from paddle_prompt.schema import InputExample
from paddle_prompt.templates.base_template import Template
from paddle_prompt.utils import create_dataloader, num
from paddle_prompt.verbalizers import compute_mask_label_logits
from paddle_prompt.verbalizers.base_verbalizer import Verbalizer


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


class ContextContainer:
    """Context data container for training
    """
    def __init__(self) -> None:
        """init the variables"""
        self.train_step: int = 0
        self.dev_step: int = 0
        self.epoch: int = 0

        self.train_acc: float = 0
        self.dev_acc: float = 0

        self.loss = 0
        self.dev_loss = 0
        self.logits = 0
        self.labels = 0

        self._cache = defaultdict(int)

    def __getitem__(self, key):
        return self._cache[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._cache[key] = value


class Trainer:
    """Trainer which can handle the tarin/eval/test/predict stage of the model
    """
    def __init__(
        self, config: Config,
        processor: DataProcessor, tokenizer: PretrainedTokenizer,
        mlm: PretrainedModel, criterion: Layer,
        template: Template, verbalizer: Verbalizer
    ) -> None:
        self.config = config
        self.set_device()

        # 2. build data related objects
        self.train_dataset = processor.get_train_dataset()
        self.dev_dataset = processor.get_dev_dataset()
        self.test_dataset = processor.get_test_dataset()

        self.train_dataloader = create_dataloader(
            self.train_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.template.wrap_examples(
                examples, self.train_dataset.label2idx),
        )
        self.dev_dataloader = create_dataloader(
            self.dev_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.template.wrap_examples(
                examples, self.dev_dataset.label2idx),
        )
        self.test_dataloader = create_dataloader(
            self.test_dataset,
            batch_size=config.batch_size,
            collate_fn=lambda examples: self.template.wrap_examples(
                examples, self.test_dataset.label2idx),
        )

        # 3. init model related
        self.model = mlm
        self.model.to(device=config.place())

        self.lr_scheduler: LRScheduler = LinearDecayWithWarmup(
            config.learning_rate,
            total_steps=len(self.train_dataloader) * config.epochs,
            warmup=config.warmup_proportion
        )
        if config.init_from_ckpt and os.path.isfile(config.init_from_ckpt):
            state_dict = paddle.load(config.init_from_ckpt)
            self.model.set_dict(state_dict)

        self.train_bar = None

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
        """set paddle device
        """
        paddle.set_device(self.config.device)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.init_parallel_env()

    @paddle.no_grad()
    def evaluate(self, dataloader: DataLoader, mode: str = 'dev'):
        """handle the evaluation based on dataloader

        Args:
            dataloader (DataLoader): the source of dataloader
            mode (str, optional): dev/test. Defaults to 'dev'.
        """
        logger.success(f'{mode} stage ...')
        self.model.eval()
        self.metric.reset()

        # 1. predict the labels based on dataloader
        all_loss = 0
        pre_label_ids, truth_label_ids = [], []

        progress_bar = tqdm(total=len(dataloader))
        for batch in dataloader:
            input_ids, _, prediction_mask, mask_label_ids, labels = batch

            # pylint: disable=E1129
            with paddle.amp.auto_cast(
                self.config.use_amp,
                custom_white_list=["layer_norm", "softmax", "gelu"]
            ):
                logits: Tensor = self.model(
                    input_ids=input_ids,
                    predict_mask=prediction_mask
                )
                batch_size, vocab_size = len(input_ids), logits.shape[-1]
                # [batch_size, label_num]
                label_logits = self.verbalizer.process_logits(
                    paddle.reshape(logits, shape=(batch_size, -1, vocab_size))
                )
                loss = self.criterion(
                    logits, mask_label_ids).detach().numpy().item()
                all_loss += loss

            # Get max probs label's index
            y_pred_index = label_logits.argmax(
                axis=-1).detach().numpy().tolist()
            pre_label_ids.extend(y_pred_index)
            labels = labels.detach().numpy().tolist()
            truth_label_ids.extend(labels)

            sub_acc = sum([y_pred_index[index] == labels[index]
                          for index in range(len(labels))]) / len(labels)

            progress_bar.update()
            progress_bar.set_description(
                f'loss: {loss:10.4f} acc: {sub_acc: 10.4f}'
            )

        # 2. compute the metric
        assert len(pre_label_ids) == len(truth_label_ids)
        acc = sum([pre_label_ids[index] == truth_label_ids for index in range(
            len(pre_label_ids))]) / len(pre_label_ids)
        self.context_data.dev_acc = acc
        self.context_data.dev_loss = all_loss

        logger.info(f"eval accuracy: {acc:10.4f} loss: {all_loss:10.4f}")

        self.model.train()
        self.metric.reset()

        self.context_data.dev_step += 1
        self.writer.add_scalar(tag='eval-acc', value=acc,
                               step=self.context_data.dev_step)

        if acc > self.context_data.dev_step:
            self.context_data.dev_acc = acc
            logger.success('saving the best model ...')
            best_model_file = os.path.join(
                self.config.output_dir, 'best.pdparams')
            paddle.save(self.model.state_dict(), best_model_file)

    def _update_bar_info(self):
        bar_info = []
        bar_info.append(f'train-loss: {num(self.context_data.loss):10.6f}')
        bar_info.append(f'train-acc: {self.context_data.train_acc:10.6f}')
        bar_info.append(f'dev-acc: {self.context_data.dev_acc:10.6f}')

        self.train_bar.set_description('\t'.join(bar_info))

    def on_batch_end(self):
        """handle the on batch training is ending logits
        """
        # 1. update global step
        self.context_data.train_step += 1
        self.train_bar.update()

        # 2. compute acc on training dataset
        self.writer.add_scalar(
            'train-loss',
            step=self.context_data.train_step,
            value=self.context_data.loss
        )
        self.writer.add_scalar(
            'train-acc',
            step=self.context_data.train_step,
            value=self.context_data.train_acc
        )
        self._update_bar_info()

        # 3. step the grad
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.clear_grad()

        # 4. eval on dev dataset
        if self.context_data.train_step % self.config.valid_steps == 0:
            self.evaluate(self.dev_dataloader)
            logger.info(
                'saving the model state dict in step: '
                f'{self.context_data.train_step} ...'
            )
            last_model_file = os.path.join(
                self.config.output_dir, 'last.pdparams')
            paddle.save(self.model.state_dict(), last_model_file)
        self.context_data.epoch += 1

    def on_batch_start(self):
        """handle the logit of batch start"""
        self.metric.reset()

    def train_epoch(self):
        """handle the logit of training epoch

        Args:
            epoch (int): _description_
        """
        self.model.train()
        logger.info(f'training epoch<{self.context_data.epoch}> ...')
        self.train_bar = tqdm(total=len(self.train_dataloader))

        for batch in self.train_dataloader:
            input_ids, _, prediction_mask, mask_label_ids, labels = batch
            self.on_batch_start()

            # pylint: disable=E1129
            with paddle.amp.auto_cast(
                    self.config.use_amp,
                    custom_white_list=["layer_norm", "softmax", "gelu"]):
                logits = self.model(
                    input_ids=input_ids,
                    predict_mask=prediction_mask
                )
                loss = compute_mask_label_logits(logits, mask_label_ids).mean()
                self.context_data.logits = logits
                self.context_data.loss = loss
                self.context_data.labels = labels

            loss.backward()
            self.on_batch_end()

    def train(self):
        """the main train epoch"""
        for _ in range(self.config.epochs):
            if self.config.do_train:
                self.train_epoch()

            if self.config.do_dev:
                self.evaluate(self.test_dataloader, mode='test')

    def predict(self, example: InputExample):
        """predict the example"""
