from __future__ import annotations

from typing import Any

import torch
from datasets import Audio, load_dataset

from src.data.base_audio_text_dataset import BaseAudioTextDataset


class AudioCapsAudioTextDataset(BaseAudioTextDataset):
    """Hugging Face OpenSound/AudioCaps wrapper.

    AudioCaps is an audio *captioning* dataset: each row is a single
    (audio, natural-language caption) pair, so the caption is used directly as
    the target text — no per-sample parsing needed.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_id: str = "OpenSound/AudioCaps",
        sampling_rate: int | None = 16000,
    ) -> None:
        self.split = split
        self.dataset_id = dataset_id
        self.sampling_rate = sampling_rate

        raw = load_dataset(self.dataset_id, split=self.split)

        # Drop rows with an empty caption — nothing to score (defensive, same
        # spirit as the Clotho/AudioSet handlers).
        raw = raw.filter(
            lambda x: bool(str(x["caption"]).strip()),
            desc=f"Filtering {self.split} for non-empty captions",
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

        return audio_tensor, str(sample["caption"]).strip()
