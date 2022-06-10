"""unit test for tokenzier-wrapper"""
import pytest

from paddle_prompt.tokenzier_wrapper import TokenzierWrapper

from paddlenlp.transformers.bert.tokenizer import BertTokenizer


PRETRAINED_MODEL = ['bert-base-uncased', 'bart-base', 't5-base']

@pytest.mark.parametrize('model_name', PRETRAINED_MODEL)
def test_unused_tokens_of_bert(model_name: str):
    # TODO: T5 Tokenizer is not supported in PaddleNLP
    tokenizer_wrapper = TokenzierWrapper(model_name)
    unused_tokens = tokenizer_wrapper.get_unused_tokens()
    assert len(unused_tokens) > 0


@pytest.mark.parametrize('model_name', PRETRAINED_MODEL)
def test_single_tokenize(model_name: str):
    text = 'I love paddlepaddle'
    tokenizer = TokenzierWrapper(model_name)
    encoded_features = tokenizer(
        text,  return_attention_mask=True, return_token_type_ids=True
    )
    assert 'input_ids' in encoded_features
    assert 'token_type_ids' in encoded_features
    assert 'attention_mask' in encoded_features

@pytest.mark.parametrize('model_name', PRETRAINED_MODEL)
def test_batch_tokenize(model_name: str):
    text = ['I love paddlepaddle', 'I love PaddleNLP']
    tokenizer = TokenzierWrapper(model_name)
    encoded_features = tokenizer(
        text,  return_attention_mask=True, return_token_type_ids=True
    )
    assert 'input_ids' in encoded_features
    assert 'token_type_ids' in encoded_features
    assert 'attention_mask' in encoded_features

    assert len(encoded_features['input_ids']) == 2
