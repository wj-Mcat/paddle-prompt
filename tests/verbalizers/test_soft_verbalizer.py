"""unit test for soft verbalizer"""
from __future__ import annotations
from typing import Type

import pytest

from paddlenlp.transformers.model_utils import PretrainedModel 
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer
from paddlenlp.transformers.bert.modeling import BertForMaskedLM 
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration
from paddlenlp.transformers.bart.modeling import BartForConditionalGeneration

from paddle_prompt.config import Config
from paddle_prompt.plms import LMWrapper
from paddle_prompt.plms.ernie import ErnieForMLM


# @pytest.mark.parametrize('model_name', ["t5-small"])
# def test_find_t5_linear_head(model_name: str):
#     model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(model_name)
    
#     tokenizer = T5Tokenizer.from_pretrained(model_name)


@pytest.mark.parametrize("model_class,model_name", [
    (ErnieForMLM, 'ernie-1.0')
    # (BertForMaskedLM, 'bert-base-uncased'),
    # (T5ForConditionalGeneration, 't5-small'),
    # (BartForConditionalGeneration, 'bart-base'),
])
def test_find_linear_head(model_class: Type[PretrainedModel], model_name: str):
    config = Config().parse_args(known_only=True)
    config.pretrained_model = model_name
    model = model_class(config)

    head_embedding = model.get_head_embedding()
    assert head_embedding is not None

@pytest.mark.parametrize("model_class,model_name, hidden_size", [
    (BertForMaskedLM, 'bert-base-uncased', 768),
    (T5ForConditionalGeneration, 't5-small', 512),
    (BartForConditionalGeneration, 'bart-base', 768),
])
def test_find_hidden_size(model_class: Type[PretrainedModel], model_name: str, hidden_size: int):
    model = model_class.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm_wrapper = LMWrapper(model, tokenizer)
    fetched_hidden_size = lm_wrapper.get_hidden_size()
    assert fetched_hidden_size == hidden_size
