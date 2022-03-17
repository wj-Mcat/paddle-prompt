import os
from paddle_prompt.trainer import Trainer
from paddle_prompt.config import Config
from paddle_prompt.processors.tnews import TNewsDataProcessor
from paddle_prompt.plms.ernie import ErnieForMLM

from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer

root_dir = '/home/users/wujingjing/projects/bupt-nlp/paddle-prompt'

def debug_pet():
    config: Config = Config().parse_args(known_only=True)
    config.device = 'cpu'
    processor = TNewsDataProcessor(data_dir=os.path.join(root_dir, 'glue_data/tnews'))
    tokenizer = ErnieTokenizer.from_pretrained(config.pretrained_model)
    classifier = ErnieForMLM(config)
    trainer = Trainer(config=config, processor=processor, mlm=classifier, tokenizer=tokenizer)
    trainer.train()

debug_pet()