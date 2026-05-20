from __future__ import annotations

import re
from typing import Any

import torch
from datasets import Audio, load_dataset

from src.data.base_audio_text_dataset import BaseAudioTextDataset

_CAPTION_SPLIT = re.compile(r"\.\s+")


def _captions_from_text(text: str) -> list[str]:
    """Parse Clotho's period-joined `text` field back into individual captions."""
    return [p.strip() for p in _CAPTION_SPLIT.split(text.rstrip(".")) if p.strip()]


class ClothoAudioTextDataset(BaseAudioTextDataset):
    """Hugging Face CLAPv2/Clotho dataset wrapper."""

    def __init__(
        self,
        split: str = "train",
        dataset_id: str = "CLAPv2/Clotho",
        caption_index: int = 1,
        sampling_rate: int | None = None,
    ) -> None:
        if caption_index < 1 or caption_index > 5:
            raise ValueError("caption_index must be in [1, 5].")

        self.split = split
        self.dataset_id = dataset_id
        self.caption_index = caption_index
        self.sampling_rate = sampling_rate

        raw = load_dataset(self.dataset_id, split=self.split)

        # raw_text is empty in this dataset upload, so captions must come from
        # the period-joined `text` field. Keep only samples that recover all 5
        # captions cleanly — this makes every caption_index in [1, 5] valid on
        # every retained sample, so the Phase C comparison is apples-to-apples.
        raw = raw.filter(
            lambda x: len(_captions_from_text(x["text"])) == 5,
            desc=f"Filtering {self.split} to 5-caption samples",
        )

        if self.sampling_rate is None:
            self._data = raw.cast_column("audio", Audio())
        else:
            self._data = raw.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        sample: dict[str, Any] = self._data[idx]
        audio_feature = sample["audio"]

        audio_tensor = torch.tensor(audio_feature["array"], dtype=torch.float32)
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.mean(dim=-1)

        captions = _captions_from_text(sample["text"])
        return audio_tensor, captions[self.caption_index - 1]
