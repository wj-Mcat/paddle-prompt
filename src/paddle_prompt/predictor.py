from __future__ import annotations
import paddle
from paddle.nn import Layer

from dataclasses import dataclass
from paddle_prompt.config import Tensor


@dataclass
class ModelInputs:
    input_ids: Tensor
    predict_mask: Tensor
    trigger_mask: Tensor


class PredictWrapper:
    """
    model wrapper. Handles necc. preprocessing of inputs for triggers experiments.
    """
    def __init__(self, model: Layer):
        self._model = model

    def predict(self, model_inputs: ModelInputs, trigger_ids: Tensor):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        output = self._model(**model_inputs)
        logits = output.logits
        # logits, *_ = self._model(**model_inputs)
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits