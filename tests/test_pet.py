"""
Implement Paper: https://arxiv.org/abs/2001.07676
"""
from __future__ import annotations

import pytest
from paddlenlp.transformers import ErnieTokenizer

from paddle_prompt.config import Config
from paddle_prompt.plms import ErnieForMLM
from paddle_prompt.plms.ernie import ErnieMLMCriterion
from paddle_prompt.processors.base_processor import DataProcessor
from paddle_prompt.processors.tnews import TNewsDataProcessor
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.trainer import Trainer
from paddle_prompt.verbalizers.manual_verbalizer import ManualVerbalizer


@pytest.fixture()
def processor() -> DataProcessor:
    processor = TNewsDataProcessor(data_dir='data/text_classification', index='')
    return processor


def test_pet(processor: DataProcessor, tokenizer: ErnieTokenizer, config: Config):
    config.template_file = 'data/text_classification/manual_template.txt'
    template = ManualTemplate(tokenizer, config)

    verbalizer = ManualVerbalizer(
        tokenizer, label_map=template.label2words, config=config
    )
    trainer = Trainer(
        config=config,
        processor=processor,
        mlm=ErnieForMLM(config),
        tokenizer=tokenizer,
        criterion=ErnieMLMCriterion(),
        template=template,
        verbalizer=verbalizer
    )
    trainer.train()
