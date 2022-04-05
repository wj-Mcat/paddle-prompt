"""unit test for manual verbalier"""
# pylint: disable=redefined-outer-name

import pytest
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.plms.ernie import ErnieForMLM
from paddle_prompt.config import Config

MODEL_NAME = ['ernie-1.0']


@pytest.fixture
@pytest.mark.parametrize('model_name', MODEL_NAME)
def config(model_name: str) -> Config:
    config_fixture = Config().parse_args(known_only=True)
    config_fixture.pretrained_model = model_name
    return config_fixture


@pytest.fixture
def tokenizer() -> ErnieTokenizer:
    return ErnieTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def mlm(config: Config) -> ErnieForMLM:
    return ErnieForMLM(config)

@pytest.fixture
def template(config: Config, tokenizer: ErnieTokenizer) -> ManualTemplate:
    return ManualTemplate(config=config, tokenizer=tokenizer)


# def test_mlm_logits(tokenizer, mlm):
#     pass
