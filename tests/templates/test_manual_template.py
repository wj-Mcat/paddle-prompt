"""Unit test for manual template"""
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.processors.tnews import TNewsDataProcessor
from paddle_prompt.config import Config


def test_simple_simple_example():
    """test simple example"""
    config: Config = Config().parse_args(known_only=True)
    config.device = 'cpu'

    config.data_dir = 'tests/data/text_classification'
    config.template_file = 'tests/data/text_classification/manual_template.json'
    processor = TNewsDataProcessor(data_dir=config.data_dir, index='')

    tokenizer = ErnieTokenizer.from_pretrained(config.pretrained_model)
    
    template = ManualTemplate(
        tokenizer=tokenizer,
        config=config
    )
    train_dataset = processor.get_train_dataset()
    
    batch = template.wrap_examples(train_dataset.examples, label2idx=train_dataset.label2idx)
    input_ids, token_type_ids, prediction_mask, mask_label_ids, labels = batch
    assert input_ids.shape == token_type_ids.shape, 'the shape of input_ids and token_type_ids should be the same'
    assert len(prediction_mask.shape) == 1, 'the shape length of prediction_mask should be 1'
    
    assert prediction_mask.shape == mask_label_ids.shape, 'the shape of prediction_mask and mask_label_ids should be the same'
    assert input_ids.shape[0] == labels.shape[0], 'the length of input_ids and labels should be the same'
