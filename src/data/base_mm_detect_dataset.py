from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset


class BaseMMDetectDataset(Dataset, ABC):
    """Abstract dataset for MM-DETECT (paired original / back-translated captions)."""

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Return one sample as a dict with keys expected by ``MMDetectMethod``:

          - ``audio``: float waveform tensor
          - ``target_original``: keyword to predict in the original caption
          - ``masked_original``: original caption with one keyword replaced by [MASK]
          - ``target_back``: keyword to predict in the back-translated caption
          - ``masked_back``: back-translated caption with one keyword replaced by [MASK]
        """
        raise NotImplementedError
