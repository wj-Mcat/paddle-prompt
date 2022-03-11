from __future__ import annotations

import pytest

from paddle.io import Dataset, DataLoader
from paddlenlp.datasets import load_dataset


@pytest.fixture
def tc_examples() -> Dataset:
    """text classification dataset"""
    pass