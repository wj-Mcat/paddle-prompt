"""Unit test for utils"""
import paddle
from paddle_prompt.utils import get_mask_with_ids


def test_simple_get_mask():
    input_ids = paddle.to_tensor([2,3,4])
    
    target_mask = paddle.to_tensor([1,0,0])
    mask = get_mask_with_ids(input_ids, 2)
    assert paddle.equal_all(target_mask, mask), 'the mask should be the same'

    target_mask = paddle.to_tensor([1,0,0])
    mask = get_mask_with_ids(input_ids, [2])
    assert paddle.equal_all(target_mask, mask), 'the mask should be the same'


def test_multi_dim_get_mask():
    input_ids = paddle.to_tensor([[2,3,4],[2,3,4]])

    target_mask = paddle.to_tensor([[1,0,0],[1,0,0]])
    mask = get_mask_with_ids(input_ids, [2])
    assert paddle.equal_all(target_mask, mask), 'the mask should be the same'

# def test_gather():
#     tensor = paddle.to_tensor([
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]
#     ])
#     index = paddle.to_tensor([[0], [1], [2]])
    
#     result = paddle.gather(tensor, index)
#     assert result.shape == [2, 3]

