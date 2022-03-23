import pytest
import paddle
import numpy as np

from paddle_prompt.config import Config
from paddle_prompt.plms import ErnieForMLM
from paddle_prompt.plms.ernie import ErnieMLMCriterion


@pytest.fixture
def input_ids():
    return paddle.to_tensor([
        [1, 104, 345, 13],
        [1, 104, 345, 13],
        [1, 104, 345, 13],
        [1, 104, 345, 13],
    ])


@pytest.fixture
def predict_mask():
    return paddle.to_tensor([
        1, 2, 5, 6, 9, 10, 14, 15
    ])


def test_ernie_mlm(config: Config, input_ids, predict_mask):
    mlm = ErnieForMLM(config)
    batch_size = len(input_ids)
    assert len(predict_mask) // batch_size > 0

    logits = mlm.forward(input_ids, predict_mask=predict_mask)
    assert paddle.is_tensor(logits)


def test_ernie_criterion():
    criterion = ErnieMLMCriterion()
    prediction_scores = paddle.randn(shape=(4, 18000))
    mask_label_ids = paddle.to_tensor([2, 3, 4, 5])

    loss = criterion(prediction_scores, mask_label_ids)
    assert paddle.is_tensor(loss)
