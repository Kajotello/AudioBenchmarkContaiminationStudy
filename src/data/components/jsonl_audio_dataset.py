from __future__ import annotations

import json
from pathlib import Path

import librosa
import soundfile as sf
import torch

from src.data.base_audio_text_dataset import BaseAudioTextDataset


class JsonlAudioDataset(BaseAudioTextDataset):
    """Audio+text dataset backed by a JSONL index + local WAV files.

    Expected layout (produced by ``scripts/download_dataset.py``):
        <root>/audio/000000.wav
        <root>/metadata.jsonl   # each line: {"audio": "<path>", "text": "<gt>"}

    Audio paths in the JSONL are resolved relative to the parent directory of
    the JSONL file when they are not absolute.
    """

    def __init__(self, jsonl_path: str, sampling_rate: int | None = 16000) -> None:
        self.jsonl_path = Path(jsonl_path).resolve()
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")

        self.sampling_rate = sampling_rate
        self._base_dir = self.jsonl_path.parent

        self._records: list[dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self._records)

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        # Try cwd-relative first (paths in JSONL are repo-root-relative)
        cwd_rel = Path.cwd() / p
        if cwd_rel.exists():
            return cwd_rel
        # Fall back to JSONL-dir-relative
        return self._base_dir / p

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        rec = self._records[idx]
        audio_path = self._resolve(rec["audio"])

        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)

        if self.sampling_rate is not None and sr != self.sampling_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sampling_rate)

        audio_tensor = torch.from_numpy(data).float()

        text_field = rec["text"]
        if isinstance(text_field, list):
            text = text_field[0]
        else:
            text = str(text_field)

        return audio_tensor, text
