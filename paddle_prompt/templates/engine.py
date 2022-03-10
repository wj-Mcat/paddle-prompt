from __future__ import annotations

from abc import ABC, abstractmethod
import json
import os
from typing import Dict
from jinja2 import Template as Jinja2Template


class Engine(ABC):
    """Abstract Template Style Engine"""
    def __init__(self, label_templates: Dict[str, str]) -> None:
        self.label_templates = label_templates

    @abstractmethod    
    def render(self, label: str, template_string: str) -> str:
        raise NotImplementedError

    @classmethod 
    def from_file(cls, file: str) -> Engine:
        if not os.path.exists(file):
            raise FileNotFoundError(f'label template file not found, the path is: {file}')
        with open(file, 'r', encoding='utf-8') as file_handler:
            label_template = json.load(file_handler)
        return cls(label_template)

class JinjaEngine(Engine):
    def __init__(self, label_templates: Dict[str, str]) -> None:
        super().__init__(label_templates)
        
        self._label_jinja2_template: Dict[str, Jinja2Template] = {
            label: Jinja2Template(text) for label, text in label_templates.item()
        }
    
    def render(self, label: str, example_dict: dict) -> str:
        """render the template based on label template"""
        if label not in self._label_jinja2_template:
            raise ValueError(f'label<{label}> is not supported') 

        template: Jinja2Template = self.label_templates[label]
        return template.render(example_dict)
