from __future__ import annotations
from cProfile import label
import json
import os

from typing import List
from matplotlib.pyplot import text

from paddlenlp.datasets import load_dataset

from paddle_prompt.schema import InputExample, ExampleDataset
from paddle_prompt.processors.base_processor import DataProcessor

    
class TNewsDataProcessor(DataProcessor):
    """Tnews Data Processor
    refer to: https://github.com/CLUEbenchmark/FewCLUE/tree/main/datasets/tnews
    """
    def __init__(self, data_dir: str, index: int = 0) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.data_index = index
        self.train_labels = []

    def _read(self, mode: str) -> List[InputExample]:
        file_path = os.path.join(self.data_dir, f'{mode}_{self.data_index}.json')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'target file not found: <{file_path}> ...')

        examples = []
        with open(file_path, 'r', encoding='utf-8') as file_handler:
            for line in file_handler:
                data = json.loads(line)
                examples.append(InputExample(
                    text=data['sentence'],
                    guid=data['id'],
                    label=data['label_desc'],
                    meta={
                        "keywords": data['keywords']
                    }
                ))
        return examples

    
    def get_train_dataset(self):
        examples = self._read('train')
        return ExampleDataset(examples)

    def get_dev_dataset(self):
        examples = self._read('dev')
        return ExampleDataset(examples)

    def get_test_dataset(self):
        examples = self._read('test')
        return ExampleDataset(examples)
    
    def get_labels(self) -> List[str]:
        return self.train_labels
