"""unit test for soft verbalizer"""
import pytest

from typing import Type
from paddlenlp.transformers.model_utils import PretrainedModel 
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer
from paddlenlp.transformers.bert.modeling import BertForMaskedLM 
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration
from paddlenlp.transformers.bart.modeling import BartForConditionalGeneration

from paddle_prompt.plms import LMWrapper


# @pytest.mark.parametrize('model_name', ["t5-small"])
# def test_find_t5_linear_head(model_name: str):
#     model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
    
#     tokenizer = T5Tokenizer.from_pretrained(model_name)


@pytest.mark.parametrize("model_class,model_name", [
    (BertForMaskedLM, 'bert-base-uncased'),
    (T5ForConditionalGeneration, 't5-small'),
    (BartForConditionalGeneration, 'bart-base'),
])
def test_find_linear_head(model_class: Type[PretrainedModel], model_name: str):
    model = model_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm_wrapper = LMWrapper(model, tokenizer)

    head = lm_wrapper.find_lm_head()
    assert head is not None


@pytest.mark.parametrize("model_class,model_name", [
    (BertForMaskedLM, 'bert-base-uncased'),
    (T5ForConditionalGeneration, 't5-small'),
    (BartForConditionalGeneration, 'bart-base'),
])
def test_find_hidden_size(model_class: Type[PretrainedModel], model_name: str):
    model = model_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm_wrapper = LMWrapper(model, tokenizer)

    hidden_size = lm_wrapper.get_hidden_size()
    assert hidden_size is not None

