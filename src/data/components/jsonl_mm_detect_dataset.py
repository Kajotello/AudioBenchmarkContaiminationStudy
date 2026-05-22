from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import soundfile as sf
import torch

from src.data.base_mm_detect_dataset import BaseMMDetectDataset
from src.data.components.jsonl_audio_dataset import jsonl_path_for_split


class JsonlMMDetectDataset(BaseMMDetectDataset):
    """JSONL-backed dataset for MM-DETECT.

    Expects metadata produced by ``scripts/download_dataset.py`` with
    ``--back-translate`` (default). Each record must include parallel lists
    (one entry per caption, selected via ``caption_index``):

      - ``masked_original_captions``
      - ``masked_targets_original``
      - ``masked_back_captions``
      - ``masked_targets_back``

    Records where masking failed (``None`` in any of the four fields) are
    dropped, matching the non-empty filtering of ``JsonlAudioDataset``.
    """

    _MM_FIELDS = (
        "masked_original_captions",
        "masked_targets_original",
        "masked_back_captions",
        "masked_targets_back",
    )

    def __init__(
        self,
        dataset_name: str,
        split: str,
        data_dir: str | Path = "./data",
        sampling_rate: int | None = 16000,
        caption_index: int = 1,
        require_num_captions: int | None = None,
    ) -> None:
        if caption_index < 1:
            raise ValueError(f"caption_index must be >= 1, got {caption_index}")

        jsonl_path = jsonl_path_for_split(dataset_name, split, data_dir)
        self.jsonl_path = Path(jsonl_path).resolve()
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL not found: {self.jsonl_path}")

        self.sampling_rate = sampling_rate
        self.caption_index = caption_index
        self.require_num_captions = require_num_captions
        self._base_dir = self.jsonl_path.parent

        raw_records: list[dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        if raw_records and not all(k in raw_records[0] for k in self._MM_FIELDS):
            raise RuntimeError(
                f"{self.jsonl_path} is missing MM-DETECT fields "
                f"{self._MM_FIELDS}. Regenerate with "
                "scripts/download_dataset.py --back-translate."
            )

        if require_num_captions is not None and raw_records and "captions" not in raw_records[0]:
            raise RuntimeError(
                f"{self.jsonl_path} has no 'captions' field but "
                f"require_num_captions={require_num_captions} was requested."
            )

        cap_idx = caption_index - 1
        self._records: list[tuple[dict, str, str, str, str]] = []
        for rec in raw_records:
            if require_num_captions is not None:
                caps = rec.get("captions") or []
                if len(caps) != require_num_captions:
                    continue

            masked_lists = [rec[k] for k in self._MM_FIELDS]
            n_caps = min(len(lst) for lst in masked_lists)
            if cap_idx >= n_caps:
                continue

            masked_orig, target_orig, masked_back, target_back = (
                masked_lists[i][cap_idx] for i in range(4)
            )
            if not all(
                v is not None and str(v).strip()
                for v in (masked_orig, target_orig, masked_back, target_back)
            ):
                continue

            self._records.append(
                (
                    rec,
                    str(masked_orig).strip(),
                    str(target_orig).strip(),
                    str(masked_back).strip(),
                    str(target_back).strip(),
                )
            )

        if raw_records and not self._records:
            raise RuntimeError(
                f"No usable MM-DETECT records in {self.jsonl_path} "
                f"(caption_index={caption_index}, "
                f"require_num_captions={require_num_captions})."
            )

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        cwd_rel = Path.cwd() / p
        if cwd_rel.exists():
            return cwd_rel
        return self._base_dir / p

    def _load_audio(self, rec: dict) -> torch.Tensor:
        audio_path = self._resolve(rec["audio"])
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)
        if self.sampling_rate is not None and sr != self.sampling_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sampling_rate)
        return torch.from_numpy(data).float()

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec, masked_orig, target_orig, masked_back, target_back = self._records[idx]
        return {
            "audio": self._load_audio(rec),
            "target_original": target_orig,
            "masked_original": masked_orig,
            "target_back": target_back,
            "masked_back": masked_back,
        }
