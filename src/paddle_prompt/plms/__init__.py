from __future__ import annotations
from curses import wrapper
from typing import List, Type, Union, Optional
from dataclasses import dataclass
import inspect

from paddle import nn

from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.auto.modeling import AutoModel
from paddlenlp.transformers.auto.tokenizer import AutoTokenizer

from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
from paddlenlp.transformers.ernie.modeling import ErnieForMaskedLM

from paddlenlp.transformers.bert.modeling import BertForMaskedLM
from paddlenlp.transformers.bart.modeling import BartForConditionalGeneration
from paddlenlp.transformers.t5.modeling import T5ForConditionalGeneration


from paddle_prompt.plms.ernie import ErnieForMLM

# _MODEL_CLASSES = {
#     'bert': ModelClass(**{
#         'config': BertConfig,
#         'tokenizer': BertTokenizer,
#         'model':BertForMaskedLM,
#         'wrapper': MLMTokenizerWrapper,
#     }),
#     'roberta': ModelClass(**{
#         'config': RobertaConfig,
#         'tokenizer': RobertaTokenizer,
#         'model':RobertaForMaskedLM,
#         'wrapper': MLMTokenizerWrapper
#     }),
#     'albert': ModelClass(**{
#         'config': AlbertConfig,
#         'tokenizer': AlbertTokenizer,
#         'model': AlbertForMaskedLM,
#         'wrapper': MLMTokenizerWrapper
#     }),
#     'gpt': ModelClass(**{
#         'config': OpenAIGPTConfig,
#         'tokenizer': OpenAIGPTTokenizer,
#         'model': OpenAIGPTLMHeadModel,
#         'wrapper': LMTokenizerWrapper
#     }),
#     'gpt2': ModelClass(**{
#         'config': GPT2Config,
#         'tokenizer': GPT2Tokenizer,
#         'model': GPT2LMHeadModel,
#         'wrapper': LMTokenizerWrapper
#     }),
#     't5':ModelClass(**{
#         'config': T5Config,
#         'tokenizer': T5Tokenizer,
#         'model': T5ForConditionalGeneration,
#         'wrapper': T5TokenizerWrapper
#     }),
#     't5-lm':ModelClass(**{
#         'config': T5Config,
#         'tokenizer': T5Tokenizer,
#         'model': T5ForConditionalGeneration,
#         'wrapper': T5LMTokenizerWrapper,
#     }),
# }

def _load_ernie_plm(model_name_or_path: str):
    return ErnieTokenizer.from_pretrained(model_name_or_path), ErnieForMLM.from_pretrained(model_name_or_path)
    

def load_plm(model_name: str,  model_name_or_path: str, special_tokens: List[str] = []):
    assert model_name == 'ernie'
    return _load_ernie_plm(model_name_or_path=model_name_or_path)



class TokenizerWrapper:
    def __init__(
        self,
        tokenizer: PretrainedTokenizer, 
        mask_token: str = '[MASK]'
    ) -> None:
        self.mask_token = mask_token
        self.tokenizer = tokenizer

@dataclass
class ModelMaps:
    """supported model map"""
    model_class: Type[PretrainedModel]

    
class LMWrapper(nn.Layer):
    """Wrapper for Ernie pretrained model"""
    def __init__(self, pretrained_model: Union[str, PretrainedModel], tokenizer: Optional[PretrainedTokenizer]) -> None:
        super().__init__()
        if isinstance(pretrained_model, str):
            pretrained_model = AutoModel.from_pretrained(pretrained_model)
        self.plm = pretrained_model
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(pretrained_model)


    def get_hidden_size(self) -> Optional[int]:
        """get_hidden_size of plm, at this time, it can't get in dynamic mode.
            refer to the issue: https://github.com/PaddlePaddle/PaddleNLP/issues/1899

        Returns:
            Optional[int]: the size of the hidden size 
        """
        vocab_size = len(self.tokenizer)
        
        # TODO:  we only support bert, bart, t5 model
        for parameter in self.plm.parameters():
            pass

        if isinstance(self.plm, ErnieForMaskedLM):
            return self.plm.ernie.encoder.layer[1].output_layer.weight.shape
        
        if isinstance(self.plm, BertForMaskedLM):
            return self.plm.bert.encoder.layer[-1].output_layer.weight.shape
        
        if isinstance(self.plm, BartForConditionalGeneration):
            return self.plm.bart.encoder.layer[-1].output_layer.weight.shape

        if isinstance(self.plm, T5ForConditionalGeneration):
            return self.plm.t5.encoder.layer[-1].output_layer.weight.shape
            
        return None

    def find_lm_head(self) -> nn.Linear:
        """head of lm is usualy a linear map function"""
        # 
        vocab_size, hidden_size = self.tokenizer.vocab_size, self.plm.config["hidden_size"]
        def find_target_head(module: nn.Layer, deep: int = 0) -> nn.Linear:
            if deep == 5:
                return None
            if isinstance(module, nn.Linear) and module.weight.shape == (hidden_size, vocab_size):
                return module
            
            for _, child in reversed(module.named_children()):
                result = find_target_head(child, deep + 1)
                if result is not None:
                    return result
            return None
        
        return find_target_head(self.plm)
