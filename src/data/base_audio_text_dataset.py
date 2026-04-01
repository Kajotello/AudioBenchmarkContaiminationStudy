from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class BaseAudioTextDataset(Dataset, ABC):
    """Abstract dataset for audio and paired description text."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        """
        Return one sample as:
          - audio tensor
          - description text
        """
        raise NotImplementedError
