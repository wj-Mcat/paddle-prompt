from __future__ import annotations
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers.ernie.modeling import ErniePretrainedModel, ErniePretrainingHeads, ErnieLMPredictionHead, ErnieModel
from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
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


class ErnieMLMCriterion(paddle.nn.Layer):
    def forward(
        self,
        prediction_scores: Tensor,
        mask_label_ids: Tensor,
        masked_lm_scale: float =1.0
    ):
        # shape = [batch_size * max_token_num, 1]
        mask_label_ids = paddle.reshape(mask_label_ids, shape=[-1, 1])
        with paddle.static.amp.fp16_guard():
            masked_lm_loss = F.softmax_with_cross_entropy(
                prediction_scores,
                mask_label_ids,
                ignore_index=-1
            )
            masked_lm_loss = masked_lm_loss / masked_lm_scale
            return paddle.mean(masked_lm_loss)

