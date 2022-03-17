from collections import defaultdict
from dataclasses import Field
import os
from typing import Dict, Optional
from tap import Tap
import paddle.tensor as Tensor

class TrainConfigMixin(Tap):
    """Train Config Mixin"""
    batch_size: int = 32                # Batch size per GPU/CPU for training.
    learning_rate: float = 5e-5         # The initial learning rate for Adam.
    weight_decay: float = 0.0           # Weight decay if we apply some.
    warmup_proportion: float = 0.0      # Linear warmup proption over the training process.

    valid_steps: int = 100              # The interval steps to evaluate model performance.
    save_steps: int = 100               # The interval steps to save checkppoints.

    init_from_ckpt: Optional[str] = None# The path of checkpoint to be loaded.

    epochs: int = 3
    seed: int = 1000                    # random seed for initialization
    device: str = 'gpu'                 # Select which device to train model, defaults to gpu.
    use_amp: bool = False               # Enable mixed precision training.

    scale_loss: float = 2**15           # The value of scale_loss for fp16.
    
    label2idx: Dict[str, int] = None

    do_train: bool = True
    do_dev: bool = True
    do_test: bool = True

    @property
    def label_num(self) -> int:
        return len(self.label2idx)


class TemplateConfigMixin(Tap):
    freeze_plm: bool = False    # if freeze the parameters in PLM
    mask_token: str = '[MASK]'
    max_token_num: int = 2      # max number of tokens in label slot
    max_span_num: int = 1       # the max number of words in label domain
    render_engine: str = 'jinja2'   # the template render engine which convert the label-template to input strings with customized functions
    template_file: str = './glue_data/tnews/manual_template.json'


class Config(TrainConfigMixin, TemplateConfigMixin):
    def __init__(self, file: str = None, **kwargs):
        if file and os.path.exists(file):
            file = [file]
        else:
            file = None 
        super().__init__(config_files=file, **kwargs)

    """Configuration for Training"""
    pretrained_model: str = 'ernie-1.0'
    output_dir: str = './output'
    task: str = 'tnews'                     # Dataset for classfication tasks.
    max_seq_length: int = 128               # The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
    template: str = 'manual_template_0.json'# the file name of template file