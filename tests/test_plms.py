import pytest
import paddle

from paddle_prompt.config import Config, Tensor
from paddle_prompt.plms import ErnieForMLM
from paddle_prompt.plms.ernie import ErnieMLMCriterion
from paddle_prompt.utils import get_position_from_mask


@pytest.fixture
def input_ids():
    return paddle.to_tensor([
        [1, 104, 345, 13],
        [1, 104, 345, 13],
        [1, 104, 345, 13],
        [1, 104, 345, 13],
    ])


@pytest.fixture
def predict_position() -> Tensor:
    return paddle.to_tensor([
        1, 2, 5, 6, 9, 10, 13, 14
    ])


@pytest.fixture
def predict_mask() -> Tensor:
    return paddle.to_tensor([
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
    ])


def test_predict_position(predict_position: Tensor, predict_mask: Tensor):
    target_position = get_position_from_mask(predict_mask)
    assert paddle.equal_all(predict_position, target_position).numpy().item()


def test_ernie_mlm(config: Config, input_ids, predict_position):
    mlm = ErnieForMLM(config)
    batch_size = len(input_ids)
    assert len(predict_position) // batch_size > 0

    logits = mlm.forward(input_ids, predict_mask=predict_position)
    assert paddle.is_tensor(logits)

def test_ernie_mlm_with_prediction_mask(config: Config, input_ids, predict_mask):
    mlm = ErnieForMLM(config)
    logits = mlm.forward(input_ids, predict_mask=predict_mask)
    assert paddle.is_tensor(logits)


def test_ernie_criterion():
    criterion = ErnieMLMCriterion()
    prediction_scores = paddle.randn(shape=(4, 18000))
    mask_label_ids = paddle.to_tensor([2, 3, 4, 5])

    loss = criterion(prediction_scores, mask_label_ids)
    assert paddle.is_tensor(loss)
