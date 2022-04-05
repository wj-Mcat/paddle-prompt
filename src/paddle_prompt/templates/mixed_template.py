"""
Paddle Prompt Learning - https://github.com/wj-Mcat/paddle-prompt

Authors:    Jingjing WU (吴京京) <https://github.com/wj-Mcat>


2022-now @ Copyright wj-Mcat

Licensed under the Apache License, Version 2.0 (the 'License');
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an 'AS IS' BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import annotations

from typing import Dict, List, Optional,  Union
from paddlenlp.transformers.model_utils import PretrainedModel
from paddlenlp.transformers.tokenizer_utils import PretrainedTokenizer
from paddle_prompt.templates.base_template import Template
from paddle_prompt.config import Config


class MixedTemplate(Template):
    """Mixed template which can handle the soft token with template"""
    def __init__(
        self,
        tokenizer: PretrainedTokenizer, config: Config,
        plm: PretrainedModel,
        label2words: Optional[Dict[str, List[str]]] = None,
        prompt_template: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ):
        super().__init__(tokenizer, config, label2words=label2words, prompt_template=prompt_template, **kwargs)

        self.word_embeddings = plm.get_input_embeddings()
