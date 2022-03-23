"""
Implement Paper: https://arxiv.org/abs/2001.07676
"""
from __future__ import annotations

import os.path

from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer

from paddle_prompt.config import Config
from paddle_prompt.plms.ernie import ErnieForMLM, ErnieMLMCriterion
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.verbalizers.manual_verbalizer import ManualVerbalizer
from paddle_prompt.trainer import Trainer
from paddle_prompt.processors.tnews import TNewsDataProcessor

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def main():
    # 1. load base configuration
    config: Config = Config().parse_args(known_only=True)
    config.device = 'cpu'
    config.template_file = os.path.join(root, 'tests/data/text_classification/manual_template.json')

    processor = TNewsDataProcessor(
        data_dir=os.path.join(root, 'tests/data/text_classification'),
        index=''
    )
    tokenizer = ErnieTokenizer.from_pretrained(config.pretrained_model)
    template = ManualTemplate(tokenizer, config)
    verbalizer = ManualVerbalizer(tokenizer, label_map=template.label2words, config=config)
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
    

if __name__ == '__main__':
    main()
