#!/usr/bin/env python3
"""Download a HuggingFace audio-text dataset to local WAV files + JSONL index.

Output layout:
    <output_dir>/<dataset_safe_id>/<split>/audio/000000.wav ...
    <output_dir>/<dataset_safe_id>/<split>/metadata.jsonl

Each JSONL record: {"index": N, "audio": "<rel_path>", "text": "<gt>"}
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm


def safe_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def get_text(sample: dict, text_col: str) -> str:
    val = sample[text_col]
    if isinstance(val, list):
        return val[0]
    # dot-joined captions (CLAPv2/Clotho `text` column) — take first
    return str(val).split(".")[0].strip()


def process_split(
    dataset_id: str,
    split: str,
    audio_col: str,
    text_col: str,
    sample_rate: int,
    output_dir: Path,
    repo_root: Path,
) -> None:
    split_dir = output_dir / safe_id(dataset_id) / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = split_dir / "metadata.jsonl"

    # Find already-written indices to support idempotent re-runs
    existing = {int(p.stem) for p in audio_dir.glob("*.wav")}

    ds = load_dataset(dataset_id, split=split)
    ds = ds.cast_column(audio_col, Audio(sampling_rate=sample_rate))

    written = 0
    with jsonl_path.open("a") as jf:
        for idx, sample in enumerate(tqdm(ds, desc=f"{split}", unit="sample")):
            wav_path = audio_dir / f"{idx:06d}.wav"

            if idx not in existing:
                audio = sample[audio_col]
                arr = np.array(audio["array"], dtype=np.float32)
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                sf.write(str(wav_path), arr, sample_rate, subtype="FLOAT")
                written += 1

            text = get_text(sample, text_col)
            rel_path = str(wav_path.relative_to(repo_root))
            jf.write(json.dumps({"index": idx, "audio": rel_path, "text": text}) + "\n")

    print(f"  [{split}] {written} new WAVs written → {split_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="CLAPv2/Clotho")
    parser.add_argument("--split", default="train,validation,test",
                        help="Comma-separated splits")
    parser.add_argument("--audio-col", default="audio")
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--output-dir", "-b", default="./data",
                        help="Base output directory (default: ./data)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()
    splits = [s.strip() for s in args.split.split(",")]

    print(f"Dataset : {args.dataset_id}")
    print(f"Splits  : {splits}")
    print(f"Output  : {output_dir}")

    for split in splits:
        process_split(
            dataset_id=args.dataset_id,
            split=split,
            audio_col=args.audio_col,
            text_col=args.text_col,
            sample_rate=args.sample_rate,
            output_dir=output_dir,
            repo_root=repo_root,
        )

    print("Done.")


if __name__ == "__main__":
    main()
