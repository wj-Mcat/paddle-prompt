"""Configuration for paddle_prompt."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import paddle
import paddle.tensor as Tensor
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
from tabulate import tabulate
from tap import Tap

from paddle_prompt.utils import to_list


class TrainConfigMixin(Tap):
    """Train Config Mixin"""
    batch_size: int = 32  # Batch size per GPU/CPU for training.
    learning_rate: float = 5e-5  # The initial learning rate for Adam.
    weight_decay: float = 0.0  # Weight decay if we apply some.
    warmup_proportion: float = 0.0  # Linear warmup proption over the training process.

    valid_steps: int = 100  # The interval steps to evaluate model performance.

    init_from_ckpt: Optional[str] = None  # The path of checkpoint to be loaded.

    epochs: int = 3
    seed: int = 1000  # random seed for initialization
    device: str = 'gpu'  # Select which device to train model, defaults to gpu.
    use_amp: bool = False  # Enable mixed precision training.

    scale_loss: float = 2 ** 15  # The value of scale_loss for fp16.

    label2idx: Dict[str, int] = None

    do_train: bool = True
    do_dev: bool = True
    do_test: bool = True

    @property
    def label_num(self) -> int:
        """get label number"""
        return len(self.label2idx)


class TemplateConfigMixin(Tap):
    """Template Config Mixin"""
    freeze_plm: bool = False  # if freeze the parameters in PLM
    mask_token: str = '[MASK]'
    max_token_num: int = 2  # max number of tokens in label slot
    max_span_num: int = 1  # the max number of words in label domain
    render_engine: str = 'jinja2'  # the template render engine which convert the label-template to input strings with customized functions
    template_file: str = './glue_data/tnews/manual_template.json'


class VerbalizerConfigMixin(Tap):
    """Verbalizer Config Mixin"""
    metric_name: str = 'acc'  # the name of metric


class Config(TrainConfigMixin, TemplateConfigMixin, VerbalizerConfigMixin):
    """Global Configuration
    """
    def __init__(self, file: str = None, **kwargs):
        if file and os.path.exists(file):
            file = [file]
        else:
            file = None
        super().__init__(config_files=file, **kwargs)

    pretrained_model: str = 'ernie-1.0'
    data_dir: str = './glue_data/tnews/'
    output_dir: str = './output'
    task: str = 'tnews'  # Dataset for classfication tasks.
    max_seq_length: int = 128  # The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
    template: str = 'manual_template_0.json'  # the file name of template file

    def place(self):
        """get the device place

        Returns:
            _type_: The device place
        """
        if self.device == 'cpu':
            return paddle.CPUPlace()
        return paddle.CUDAPlace(0)


@dataclass
class MetricReport:
    """Metric Report"""
    acc: float = 0
    precision: float = 0
    recall: float = 0
    f1_score: float = 0
    micro_f1_score: float = 0
    macro_f1_score: float = 0

    @staticmethod
    def from_sequence(truth: List, predicted: List):
        """get the metric report from sequence"""
        predicted, truth = to_list(predicted), to_list(truth)
        metric = dict(
            acc=accuracy_score(truth, predicted),
            precision=precision_score(truth, predicted),
            recall=recall_score(truth, predicted),
            f1_score=f1_score(truth, predicted),
            micro_f1_score=f1_score(truth, predicted, average='micro'),
            macro_f1_score=f1_score(truth, predicted, average='macro'),
        )
        return MetricReport(**metric)

    def __str__(self) -> str:
        """get the string format of the metric report
        """
        return 'acc: %.5f \t precision: %.5f \t  recall: %.5f \t  f1_score: %.5f \t  micro_f1_score: %.5f \t  macro_f1_score: %.5f \t ' % (
            self.acc, self.precision, self.recall, self.f1_score, self.micro_f1_score, self.macro_f1_score)

    def tabulate(self) -> str:
        """use tabulate to make a great metric format"""
        headers = ['acc', 'precision', 'reclal', 'f1_score', 'micro_f1_score', 'macro_f1_score']
        return tabulate(
            [[
                self.acc, self.precision, self.recall, self.f1_score, self.micro_f1_score, self.macro_f1_score
            ]],
            headers=headers,
            tablefmt='grid',
            floatfmt='.4f',
        )


__all__ = [
    'Tensor',

    'TrainConfigMixin',
    'TemplateConfigMixin',
    'VerbalizerConfigMixin',
    'Config',
    "MetricReport"
]
