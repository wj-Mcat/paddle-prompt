"""Base Data Processors"""
from __future__ import annotations
from typing import List

from abc import ABC
from paddle_prompt.schema import ExampleDataset


class DataProcessor(ABC):
    """Abstract Data Processor Class which handle the different corpus"""    
    def get_train_dataset(self) -> ExampleDataset:
        """get_train_dataset
        """
        raise NotImplementedError

    def get_test_dataset(self) -> ExampleDataset:
        raise NotImplementedError

    def get_dev_dataset(self) -> ExampleDataset:
        raise NotImplementedError
    
    def get_labels(self) -> List[str]:
        pass

    def get_train_labels(self) -> List[str]:
        return self.get_labels()
    
    def get_test_labels(self) -> List[str]:
        return self.get_labels()
        
    def get_dev_labels(self) -> List[str]:
        return self.get_labels()



