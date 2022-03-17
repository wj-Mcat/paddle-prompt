import paddle
import paddle.nn.functional as F
from paddle_prompt.config import Tensor


def compute_mask_label_logits(prediction_logit: Tensor, mask_label_ids: Tensor) -> Tensor:
    """compute logits with mask label

    Args:
        prediction_logit (Tensor): logits from mask positions
            [batch_size, max_token_num, vocab_size]
        mask_label_ids (Tensor): label ids
            [batch_size * max_token_num]

    Returns:
        Tensor: the final loss
    """
    vocab_size = prediction_logit.shape[-1]

    # (batch_size * max_token_num, vocab_size)
    predict_logp: Tensor = F.log_softmax(prediction_logit, axis=-1).reshape((-1, vocab_size))

    # [batch_size, ]
    target_logp = predict_logp.index_select(index=mask_label_ids, axis=-1)
    target_logp = target_logp - 1e32  # Apply mask
    target_logp = paddle.log(
        paddle.sum(
            paddle.exp(target_logp),
            axis=-1
        )
    )
    return -target_logp
