"""Unit test for manual template"""
#pylint: disable=W0621
import pytest
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from paddle_prompt.schema import InputExample
from paddle_prompt.templates.manual_template import ManualTemplate
from paddle_prompt.processors.tnews import TNewsDataProcessor
from paddle_prompt.config import Config


@pytest.fixture
def tokenizer():
    return ErnieTokenizer.from_pretrained('ernie-1.0')


@pytest.fixture
def config() -> Config:
    config_fixture = Config().parse_args(known_only=True)
    config_fixture.device = 'cpu'
    config_fixture.data_dir = 'tests/data/text_classification'
    config_fixture.template_file = 'tests/data/text_classification/manual_template.json'
    return config_fixture


@pytest.fixture
def template(config, tokenizer) -> ManualTemplate:
    return ManualTemplate(config=config, tokenizer=tokenizer)


def test_simple_simple_example(config: Config, template: ManualTemplate):
    """test simple example"""

    processor = TNewsDataProcessor(data_dir=config.data_dir, index='')
    train_dataset = processor.get_train_dataset()
    
    batch = template.wrap_examples(train_dataset.examples, label2idx=train_dataset.label2idx)
    input_ids, token_type_ids, prediction_mask, mask_label_ids, labels = batch
    assert input_ids.shape == token_type_ids.shape, 'the shape of input_ids and token_type_ids should be the same'
    assert mask_label_ids.shape[0] == input_ids.shape[0] * config.max_token_num, 'the shape of mask_label_ids is not correct'
    assert prediction_mask.shape == input_ids.shape, 'the prediction mask should be the same as input_ids'
    assert input_ids.shape[0] == labels.shape[0], 'the length of input_ids and labels should be the same'


def test_template_from_label2words(config, tokenizer) -> ManualTemplate:
    label2words = {
        'clock': '订闹钟',
        'greet': '打招呼',
        'weather': '查天气'
    }
    label2template = {
        'clock': '{{text_a}}；这是在[MASK]',
        'greet': '{{text_a}}；这是在[MASK]',
        'weather': '{{text_a}}；这是在[MASK]'
    }
    template = ManualTemplate(
        config=config, tokenizer=tokenizer,
        label2words=label2words,
        prompt_template=label2template
    )
    label2idx = {"clock": 0, "greet": 1, "weather": 2}

    features = template.wrap_examples(
        [
            InputExample(text_a='您好', text_b=None, label='greet'),
            InputExample(text_a='明天早上叫我起床', text_b=None, label='clock')
        ],
        label2idx=label2idx
    )
    assert len(features) == 5
    
