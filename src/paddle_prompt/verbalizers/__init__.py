import paddle
import paddle.nn.functional as F
from paddle_prompt.config import Tensor


def compute_mask_label_logits(prediction_logit: Tensor, mask_label_ids: Tensor) -> Tensor:
    """compute logits with mask label

    Args:
        prediction_logit (Tensor): logits from mask positions
        mask_label_ids (Tensor): label ids

    Returns:
        Tensor: the final loss
    """
    predict_logp: Tensor = F.log_softmax(prediction_logit, axis=-1)
    target_logp = predict_logp.gather(index=mask_label_ids, axis=0)
    target_logp = target_logp - 1e32  # Apply mask
    target_logp = paddle.log(
        paddle.sum(
            paddle.exp(target_logp),
            axis=-1
        )
    )
    return -target_logp
