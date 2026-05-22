#!/usr/bin/env python3
"""Download a HuggingFace audio-text dataset to local WAV files + JSONL index.

Fetch-once: every dataset is materialised to disk a single time, then the single
``JsonlAudioDataset`` wrapper (and therefore the models) reads purely from the
local filesystem -- no per-run HuggingFace access, exactly like the AF2 path.

Output layout:
    <output_dir>/<dataset_safe_id>/[<config_name>/]<split>/audio/000000.wav ...
    <output_dir>/<dataset_safe_id>/[<config_name>/]<split>/metadata.jsonl

Each JSONL record always has ``index``, ``audio``, ``text`` (best-effort human
readable) plus ONE structured field that lets the wrapper reproduce the original
dataset class's text logic without re-touching HuggingFace:

    clotho     -> "captions": [<5 captions>]   (period-joined `text` re-split)
    audiocaps  -> "captions": [<1 caption>]    (the `caption` column)
    audioset   -> "labels":   [<human labels>] (verbalised at load time)

When ``--back-translate`` is enabled (default), each record also has
``back_translated_caption``: en→zh→en via NLLB-200-3.3B. Clotho and AudioSet
translate every caption/label string separately; AudioCaps stores a single
string (one caption per clip).

Caption/label construction stays at *load* time in JsonlAudioDataset, so the
caption_index / require_num_captions / label_template knobs remain in config.

See ``--caption-mode``:
  - clotho    : split the period-joined text column on r"\\.\\s+"
  - audiocaps : use the caption column verbatim as a single caption
  - audioset  : store the human-label list (verbalised by the wrapper)
  - single    : treat the text column as a single caption (no splitting)
  - raw       : use a list-typed column (e.g. `raw_text`) directly
  - auto      : infer from dataset_id (clotho / audiocaps / audioset), else single
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from nllb_backtranslate import NllbRoundTripTranslator

# Identical to ClothoAudioTextDataset so member/non-member text matches exactly.
_CAPTION_SPLIT = re.compile(r"\.\s+")


def safe_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def _clotho_captions(text: str) -> list[str]:
    return [p.strip() for p in _CAPTION_SPLIT.split(text.rstrip(".")) if p.strip()]


def resolve_caption_mode(mode: str, dataset_id: str) -> str:
    if mode != "auto":
        return mode
    did = dataset_id.lower()
    if "clotho" in did:
        return "clotho"
    if "audiocaps" in did:
        return "audiocaps"
    if "audioset" in did:
        return "audioset"
    return "single"


def extract_text_fields(sample: dict, mode: str, cols: argparse.Namespace) -> dict:
    """Return the text-bearing JSONL fields for one sample (no audio)."""
    if mode == "clotho":
        text = str(sample[cols.text_col])
        return {"text": text, "captions": _clotho_captions(text)}

    if mode == "audiocaps":
        cap = str(sample[cols.caption_col]).strip()
        return {"text": cap, "captions": [cap]}

    if mode == "audioset":
        labels = [str(x).strip() for x in sample[cols.labels_col] if str(x).strip()]
        return {"text": ", ".join(labels), "labels": labels}

    if mode == "raw":
        raw = sample.get(cols.raw_text_col) or []
        caps = [str(c).strip() for c in raw if str(c).strip()]
        return {"text": caps[0] if caps else "", "captions": caps}

    # single
    val = sample[cols.text_col]
    if isinstance(val, list):
        caps = [str(c).strip() for c in val if str(c).strip()]
    else:
        caps = [str(val).strip()]
    return {"text": caps[0] if caps else "", "captions": caps}


def translatable_strings(fields: dict, caption_mode: str) -> list[str]:
    """Source strings for round-trip translation (one entry per caption/label)."""
    if caption_mode == "audioset":
        return list(fields.get("labels") or [])
    return list(fields.get("captions") or [])


def attach_back_translated(
    fields: dict, caption_mode: str, translated: list[str]
) -> None:
    """Write ``back_translated_caption`` parallel to captions/labels."""
    if caption_mode == "audiocaps" and len(translated) == 1:
        fields["back_translated_caption"] = translated[0]
    else:
        fields["back_translated_caption"] = translated


def add_back_translations(
    records: list[dict],
    caption_mode: str,
    translator: NllbRoundTripTranslator,
) -> None:
    """Batch round-trip all captions/labels in a split, then attach per record."""
    flat: list[str] = []
    spans: list[tuple[int, int]] = []

    for rec in records:
        units = translatable_strings(rec, caption_mode)
        start = len(flat)
        flat.extend(units)
        spans.append((start, len(flat)))

    if not flat:
        return

    print(f"  Back-translating {len(flat)} strings "
          f"(cache size {translator.cache_size} before this split)...")
    back = translator.round_trip(flat)

    for rec, (start, end) in zip(records, spans):
        attach_back_translated(rec, caption_mode, back[start:end])


def split_dir_for(output_dir: Path, dataset_id: str,
                  config_name: str | None, split: str) -> Path:
    base = output_dir / safe_id(dataset_id)
    if config_name:
        base = base / config_name
    return base / split


def process_split(
    dataset_id: str,
    config_name: str | None,
    split: str,
    audio_col: str,
    caption_mode: str,
    cols: argparse.Namespace,
    sample_rate: int,
    output_dir: Path,
    repo_root: Path,
    translator: NllbRoundTripTranslator | None,
) -> None:
    split_dir = split_dir_for(output_dir, dataset_id, config_name, split)
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = split_dir / "metadata.jsonl"

    # Idempotent: skip WAVs that already exist; always rebuild the JSONL fresh
    # (overwrite, never append) so re-runs can't duplicate records and can
    # refresh the caption/label fields in place.
    existing = {int(p.stem) for p in audio_dir.glob("*.wav")}

    ds = (load_dataset(dataset_id, config_name, split=split)
          if config_name else load_dataset(dataset_id, split=split))

    # Fast path: if every WAV already exists, decode=False makes the rebuild a
    # cheap text-only pass (no audio decoding). Otherwise decode for writing.
    n = len(ds)
    need_write = not all(i in existing for i in range(n))
    ds = ds.cast_column(audio_col, Audio(sampling_rate=sample_rate, decode=need_write))

    records: list[dict] = []
    written = 0
    for idx, sample in enumerate(tqdm(ds, desc=split, unit="sample")):
        wav_path = audio_dir / f"{idx:06d}.wav"

        if need_write and idx not in existing:
            arr = np.array(sample[audio_col]["array"], dtype=np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            sf.write(str(wav_path), arr, sample_rate, subtype="FLOAT")
            written += 1

        rec = {"index": idx, "audio": str(wav_path.relative_to(repo_root))}
        rec.update(extract_text_fields(sample, caption_mode, cols))
        records.append(rec)

    if translator is not None:
        add_back_translations(records, caption_mode, translator)

    with jsonl_path.open("w", encoding="utf-8") as jf:
        for rec in records:
            jf.write(json.dumps(rec) + "\n")

    bt = " + back_translated_caption" if translator else ""
    print(f"  [{split}] {written} new WAVs, {len(records)} records "
          f"(mode={caption_mode}{bt}) -> {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="CLAPv2/Clotho")
    parser.add_argument("--config-name", default=None,
                        help="HF dataset config (e.g. 'balanced' for AudioSet)")
    parser.add_argument("--split", default="train,validation,test",
                        help="Comma-separated splits")
    parser.add_argument("--audio-col", default="audio")
    parser.add_argument("--caption-mode", default="auto",
                        choices=["auto", "clotho", "audiocaps", "audioset",
                                 "single", "raw"])
    # column names per mode
    parser.add_argument("--text-col", default="text")            # clotho / single
    parser.add_argument("--caption-col", default="caption")      # audiocaps
    parser.add_argument("--labels-col", default="human_labels")  # audioset
    parser.add_argument("--raw-text-col", default="raw_text")    # raw
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output-dir", "-b", default="./data",
                        help="Base output directory (default: ./data)")
    parser.add_argument(
        "--back-translate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="en→zh→en NLLB round-trip into back_translated_caption (default: on)",
    )
    parser.add_argument(
        "--nllb-model",
        default="facebook/nllb-200-3.3B",
        help="HuggingFace model id for back-translation",
    )
    parser.add_argument("--translate-batch-size", type=int, default=32)
    parser.add_argument("--translate-device", default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--translate-num-beams", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    splits = [s.strip() for s in args.split.split(",")]
    caption_mode = resolve_caption_mode(args.caption_mode, args.dataset_id)

    translator: NllbRoundTripTranslator | None = None
    if args.back_translate:
        print(f"Loading NLLB: {args.nllb_model} ({args.translate_device})")
        translator = NllbRoundTripTranslator(
            model_name=args.nllb_model,
            device=args.translate_device,
            batch_size=args.translate_batch_size,
            num_beams=args.translate_num_beams,
        )

    print(f"Dataset      : {args.dataset_id}"
          + (f" ({args.config_name})" if args.config_name else ""))
    print(f"Splits       : {splits}")
    print(f"Caption mode : {caption_mode}")
    print(f"Back-translate: {args.back_translate}")
    print(f"Output       : {output_dir}")

    for split in splits:
        process_split(
            dataset_id=args.dataset_id,
            config_name=args.config_name,
            split=split,
            audio_col=args.audio_col,
            caption_mode=caption_mode,
            cols=args,
            sample_rate=args.sample_rate,
            output_dir=output_dir,
            repo_root=repo_root,
            translator=translator,
        )

    print("Done.")


if __name__ == "__main__":
    main()
