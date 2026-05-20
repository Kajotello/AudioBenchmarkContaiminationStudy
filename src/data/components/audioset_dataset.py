from __future__ import annotations

from typing import Any

import torch
from datasets import Audio, load_dataset

from src.data.base_audio_text_dataset import BaseAudioTextDataset


def _labels_to_text(human_labels: list[str], template: str, sep: str) -> str:
    """Turn AudioSet's human-readable label list into a single description string."""
    cleaned = [str(label).strip() for label in human_labels if str(label).strip()]
    return template.format(labels=sep.join(cleaned))


class AudioSetAudioTextDataset(BaseAudioTextDataset):
    """Hugging Face agkphysics/AudioSet wrapper.

    AudioSet is an audio *classification* dataset: each 10 s clip carries one or
    more ontology labels rather than a natural-language caption. We synthesize a
    description string from `human_labels` so the same audio+text MIA machinery
    used for Clotho applies unchanged.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_id: str = "agkphysics/AudioSet",
        config_name: str = "balanced",
        sampling_rate: int | None = 16000,
        label_template: str = "{labels}",
        label_separator: str = ", ",
    ) -> None:
        self.split = split
        self.dataset_id = dataset_id
        self.config_name = config_name
        self.sampling_rate = sampling_rate
        self.label_template = label_template
        self.label_separator = label_separator

        raw = load_dataset(self.dataset_id, self.config_name, split=self.split)

        # Drop clips with no usable label text — without a target string there is
        # nothing to score (same defensive filtering as the Clotho handler).
        raw = raw.filter(
            lambda x: any(str(label).strip() for label in x["human_labels"]),
            desc=f"Filtering {self.config_name}/{self.split} for non-empty labels",
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

        description = _labels_to_text(
            sample["human_labels"], self.label_template, self.label_separator
        )
        return audio_tensor, description
        