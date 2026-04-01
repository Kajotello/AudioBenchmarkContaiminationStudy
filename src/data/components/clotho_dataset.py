from __future__ import annotations

from typing import Any

import torch
from datasets import Audio, load_dataset

from src.data.base_audio_text_dataset import BaseAudioTextDataset


class ClothoAudioTextDataset(BaseAudioTextDataset):
    """Hugging Face CLAPv2/Clotho dataset wrapper."""

    def __init__(
        self,
        split: str = "train",
        dataset_id: str = "CLAPv2/Clotho",
        caption_index: int = 1,
        sampling_rate: int | None = None,
    ) -> None:
        """
        Args:
            split: Dataset split, e.g. 'train', 'validation', 'test'.
            dataset_id: Hugging Face dataset identifier.
            caption_index: Which caption to use from caption_1..caption_5.
            sampling_rate: Optional output sampling rate for decoded audio.
        """
        if caption_index < 1 or caption_index > 5:
            raise ValueError("caption_index must be in [1, 5].")

        self.split = split
        self.dataset_id = dataset_id
        self.caption_index = caption_index
        self.sampling_rate = sampling_rate


        self._data = load_dataset(self.dataset_id, split=self.split)

        # Ensure audio is decoded to arrays and optionally resampled.
        if self.sampling_rate is None:
            self._data = self._data.cast_column("audio", Audio())
        else:
            self._data = self._data.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        sample: dict[str, Any] = self._data[idx]
        audio_feature = sample["audio"]

        audio_array = audio_feature["array"]
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        if audio_tensor.ndim > 1:
            # Keep the base interface simple by returning mono waveform.
            audio_tensor = audio_tensor.mean(dim=-1)

        description = sample["text"].split('.')[self.caption_index - 1]
        return audio_tensor, description
