from __future__ import annotations

import json
from pathlib import Path

import librosa
import soundfile as sf
import torch

from src.data.base_audio_text_dataset import BaseAudioTextDataset


def jsonl_path_for_split(
    dataset_name: str,
    split: str,
    data_dir: str | Path = "./data",
) -> Path:
    """``<data_dir>/<dataset_name>/<split>/metadata.jsonl``

    ``dataset_name`` may include subdirs (e.g. ``agkphysics__AudioSet/balanced``).
    """
    return Path(data_dir) / dataset_name / split / "metadata.jsonl"


class JsonlAudioDataset(BaseAudioTextDataset):
    """Audio+text dataset backed by a JSONL index + local WAV files.

    The single fetch-once reader for every dataset: HuggingFace is touched only
    by ``scripts/download_dataset.py``; every run reads from disk (like AF2).

    Expected layout (produced by ``scripts/download_dataset.py``):
        <root>/audio/000000.wav
        <root>/metadata.jsonl   # each line:
            {"index": N, "audio": "<path>", "text": "<full>",
             "captions": [...]}   # caption-style datasets (Clotho, AudioCaps)
            ... or ...
            {"index": N, "audio": "<path>", "text": "<full>",
             "labels": [...]}     # label-style datasets (AudioSet)

    Text construction mirrors the original per-dataset classes and stays here,
    at load time, so the knobs remain in config:

      caption-style (``captions`` present):
        - ``caption_index`` picks ``captions[caption_index - 1]`` (1-based).
        - ``require_num_captions`` keeps only records with exactly that many
          captions (set 5 for Clotho's apples-to-apples sample set).
      label-style (``labels`` present):
        - verbalised as ``label_template.format(labels=label_separator.join(...))``
          over the non-empty labels (matches AudioSetAudioTextDataset).

    Records whose resulting target text is empty are dropped, reproducing the
    non-empty filtering of the AudioCaps/AudioSet/Clotho handlers. JSONL order is
    preserved through all filtering, so the retained sample set and ordering are
    identical to the HF-direct classes.

    Audio paths resolve relative to the current working directory first (JSONL
    paths are repo-root-relative), then relative to the JSONL's own directory.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        data_dir: str | Path = "./data",
        sampling_rate: int | None = 16000,
        caption_index: int = 1,
        require_num_captions: int | None = None,
        label_template: str = "{labels}",
        label_separator: str = ", ",
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
        self.label_template = label_template
        self.label_separator = label_separator
        self._base_dir = self.jsonl_path.parent

        raw_records: list[dict] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_records.append(json.loads(line))

        if require_num_captions is not None and raw_records and "captions" not in raw_records[0]:
            raise RuntimeError(
                f"{self.jsonl_path} has no 'captions' field but "
                f"require_num_captions={require_num_captions} was requested. "
                "This is a stale or wrong-mode metadata.jsonl -- regenerate it "
                "with the current scripts/download_dataset.py."
            )

        # Pre-resolve target text once, applying caption-count + non-empty filters.
        self._records: list[tuple[dict, str]] = []
        for rec in raw_records:
            if require_num_captions is not None:
                caps = rec.get("captions") or []
                if len(caps) != require_num_captions:
                    continue
            text = self._build_text(rec)
            if text:
                self._records.append((rec, text))

        if raw_records and not self._records:
            raise RuntimeError(
                f"No usable records in {self.jsonl_path} "
                f"(caption_index={caption_index}, "
                f"require_num_captions={require_num_captions})."
            )

    # ------------------------------------------------------------------ helpers

    def _build_text(self, rec: dict) -> str:
        caps = rec.get("captions")
        if caps:
            idx = min(self.caption_index, len(caps))  # clamp; safe under the filter
            return str(caps[idx - 1]).strip()

        labels = rec.get("labels")
        if labels:
            cleaned = [str(x).strip() for x in labels if str(x).strip()]
            if not cleaned:
                return ""
            return self.label_template.format(labels=self.label_separator.join(cleaned))

        # Backward-compatible fallback for the original {"audio", "text"} format,
        # where `text` may be a single string or a list of captions.
        text = rec.get("text", "")
        if isinstance(text, list):
            return str(text[0]).strip() if text else ""
        return str(text).strip()

    def _resolve(self, rel_or_abs: str) -> Path:
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        cwd_rel = Path.cwd() / p
        if cwd_rel.exists():
            return cwd_rel
        return self._base_dir / p

    # ------------------------------------------------------------------ Dataset

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        rec, text = self._records[idx]
        audio_path = self._resolve(rec["audio"])

        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=-1)

        if self.sampling_rate is not None and sr != self.sampling_rate:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sampling_rate)

        return torch.from_numpy(data).float(), text
