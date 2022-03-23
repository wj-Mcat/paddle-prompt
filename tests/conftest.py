from __future__ import annotations
import pytest

from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer

from paddle_prompt.config import Config
from paddle_prompt.plms.ernie import ErnieForMLM, ErnieMLMCriterion
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.verbalizers.manual_verbalizer import ManualVerbalizer
from paddle_prompt.trainer import Trainer
from paddle_prompt.processors.tnews import TNewsDataProcessor


@pytest.fixture(scope='session')
def config() -> Config:
    # 1. load base configuration
    config: Config = Config().parse_args(known_only=True)
    config.device = 'cpu'
    return config


@pytest.fixture(scope='session')
def tokenizer(config: Config) -> ErnieTokenizer:
    tokenizer = ErnieTokenizer.from_pretrained(config.pretrained_model)
    return tokenizer


@pytest.fixture(scope='session')
def tnews_processor() -> TNewsDataProcessor:
    return TNewsDataProcessor(data_dir='./glue_data/tnews')



