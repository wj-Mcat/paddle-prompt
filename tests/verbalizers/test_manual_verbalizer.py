import pytest
import os
os.environ['PYTHONPATH'] = '/home/users/wujingjing/projects/bupt-nlp/paddle-prompt/src'
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.verbalizers.manual_verbalizer import ManualVerbalizer
from paddle_prompt.plms.ernie import ErnieForMLM
from paddle_prompt.config import Config
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer

model_name = 'ernie-1.0'


@pytest.fixture(scope='module')
def config() -> Config:
    config = Config().parse_args(known_only=True)
    config.pretrained_model = model_name
    return config


@pytest.fixture(scope='module')
def tokenizer() -> ErnieTokenizer:
    return ErnieTokenizer.from_pretrained(model_name)


@pytest.fixture(scope='module')
def mlm(config: Config) -> ErnieForMLM:
    return ErnieForMLM(config)

@pytest.fixture(scope='module')
def template(config: Config, tokenizer: ErnieTokenizer) -> ManualTemplate:
    return ManualTemplate(config=config, tokenizer=tokenizer)


def test_mlm_logits(tokenizer, mlm):
    sentence = ''
    pass