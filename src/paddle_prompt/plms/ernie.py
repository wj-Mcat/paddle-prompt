from __future__ import annotations
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers.ernie.modeling import ErniePretrainedModel, ErniePretrainingHeads, ErnieLMPredictionHead, ErnieModel
from paddle_prompt.config import Tensor, Config


class ErnieForMLM(ErniePretrainedModel):
    def __init__(self, config: Config):
        super(ErnieForMLM, self).__init__()
        self.ernie = ErnieModel.from_pretrained(config.pretrained_model)
        self.head = ErniePretrainingHeads(
            self.ernie.config["hidden_size"],
            self.ernie.config["vocab_size"],
            self.ernie.config["hidden_act"],
            embedding_weights=self.ernie.embeddings.word_embeddings.weight
        )
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids: Tensor,
        predict_mask: Tensor
    ):
        with paddle.static.amp.fp16_guard():
            outputs: Tensor = self.ernie(input_ids)
            sequence_output, pooled_output = outputs[:2]
            prediction_scores, _ = self.head(
                sequence_output,
                pooled_output,
                predict_mask
            )
            return prediction_scores

    def predict(
        self,
        input_ids: Tensor,
        predict_mask: Tensor
    ):
        prediction_logits = self.forward(
            input_ids=input_ids,
            masked_positions=predict_mask
        )
        return F.softmax(prediction_logits)


class ErnieMLMCriterion(paddle.nn.Layer):
    def forward(
        self,
        prediction_scores: Tensor,
        masked_lm_labels: Tensor,
        masked_lm_scale: float =1.0
    ):
        masked_lm_labels = paddle.reshape(masked_lm_labels, shape=[-1, 1])
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.softmax_with_cross_entropy(
                prediction_scores,
                masked_lm_labels,
                ignore_index=-1
            )
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            return paddle.mean(masked_lm_loss)
