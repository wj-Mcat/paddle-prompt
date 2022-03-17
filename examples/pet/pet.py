"""
Implement Paper: https://arxiv.org/abs/2001.07676
"""
from __future__ import annotations

from paddle_prompt.config import Config
from paddle_prompt.plms.ernie import ErnieForMLM, ErnieTokenizer, ErnieMLMCriterion
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.verbalizers.manual_verbalizer import ManualVerbalizer
from paddle_prompt.trainer import Trainer
from paddle_prompt.processors.tnews import TNewsDataProcessor


def main():
    # 1. load base configuration
    config: Config = Config().parse_args(known_only=True)
    config.device = 'cpu'
    config.do_train = False
    
    processor = TNewsDataProcessor(
        data_dir='./glue_data/tnews'
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