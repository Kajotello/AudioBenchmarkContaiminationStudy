#!/usr/bin/env python3
"""Download finished wandb runs for this project and emit per-method CSVs.

For each (method, model_version, member_dataset, non_member_dataset) setup we
keep ONLY the run with the most samples (so a 1000-sample grid run shadows any
5-sample smoke). Two rows are written per kept codec/mm_detect run — one for
the member-side score, one for the non-member-side score. MM-Detect runs on
benchmark datasets (mmau, clotho_aqa) have no non-member split and emit a
single member-only row. MIA runs emit one row per (method, model, dataset).

Usage:
    python scripts/download_wandb_results.py \\
        --entity nask-di --project audio-benchmark --output-dir results/

Add --include-smoke to also keep runs tagged "smoke" (useful for verification).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import wandb


# ---------------------------------------------------------------------------
# Mappings used to short-name model + dataset identifiers.
# ---------------------------------------------------------------------------

_DATASET_SHORT_NAME = {
    # Keys match what `dataset_label()` writes to wandb config (it
    # already replaces "/" with "_" before storing).
    "CLAPv2__Clotho": "clotho",
    "OpenSound__AudioCaps": "audiocaps",
    "agkphysics__AudioSet_balanced": "audioset",
    "TwinkStart__MMAU": "mmau",
    "lmms-lab__ClothoAQA_clotho_aqa": "clotho_aqa",
}

_CODEC_TARGET = "src.methods.codec_method.CoDeCMethod"
_MM_DETECT_TARGET = "src.methods.mm_detect.MMDetectMethod"

_MIA_METHODS = {
    "src.methods.mia_perplexity_method.MIAPerplexityMethod": "MIA_perplexity",
    "src.methods.mia_perplexity_method.YeomPerplexityMethod": "yeom_perplexity",
    "src.methods.min_k_method.MinKProbMethod": "min_k",
    "src.methods.min_k_plus_plus_method.MinKPlusPlusMethod": "min_k_pp",
    "src.methods.vl_mia_entropy_method.VLMIAEntropyMethod": "vl_mia_entropy",
}
_MIA_KINDS = frozenset(_MIA_METHODS.values())

_CODEC_REQUIRED = (
    "num_members", "num_non_members",
    "member_score_mean", "non_member_score_mean",
)
_MM_DETECT_MEMBER_REQUIRED = (
    "member_num_samples",
    "member_cr", "member_pcr", "member_delta", "member_phi",
)
_MM_DETECT_REQUIRED = _MM_DETECT_MEMBER_REQUIRED + (
    "non_member_num_samples",
    "non_member_cr", "non_member_pcr", "non_member_delta", "non_member_phi",
)
_MIA_REQUIRED = ("num_members", "num_non_members", "roc_auc")


# ---------------------------------------------------------------------------
# Field resolvers
# ---------------------------------------------------------------------------

def _model_version(model_cfg: dict[str, Any]) -> str | None:
    """Pick a readable model identifier from the model config.

    AF2 variants set `hf_repo_id` (e.g. `nvidia/audio-flamingo-2-1.5B`); the AF3
    wrapper sets `model_id` (e.g. `nvidia/audio-flamingo-3-hf`). We strip the
    `nvidia/` prefix and remap the size-less AF2 repo to `audio-flamingo-2-3B`.
    """
    repo = model_cfg.get("hf_repo_id") or model_cfg.get("model_id")
    if not repo:
        return None
    if "/" in repo:
        repo = repo.split("/", 1)[1]
    if repo == "audio-flamingo-2":
        repo = "audio-flamingo-2-3B"
    return repo


def _dataset_short(dataset_label: str | None) -> str | None:
    """`CLAPv2__Clotho:train` → `clotho`. Falls back to the raw name."""
    if not dataset_label:
        return None
    name = dataset_label.split(":", 1)[0]
    return _DATASET_SHORT_NAME.get(name, name)


def _method_kind(config: dict[str, Any]) -> str | None:
    target = (config.get("method_cfg") or {}).get("_target_")
    if target == _CODEC_TARGET:
        return "codec"
    if target == _MM_DETECT_TARGET:
        return "mm_detect"
    if target in _MIA_METHODS:
        return _MIA_METHODS[target]
    return None


def _summary_has_keys(summary: Any, keys: tuple[str, ...]) -> bool:
    return all(k in summary and summary[k] is not None for k in keys)


# ---------------------------------------------------------------------------
# Run → rows
# ---------------------------------------------------------------------------

def _codec_rows(run, model_version: str, member_ds: str, non_member_ds: str) -> list[dict[str, Any]]:
    s = run.summary
    base = {
        "run_id": run.id,
        "run_url": run.url,
        "created_at": run.created_at,
        "member_normalized_score": float(s["member_normalized_score"]) if "member_normalized_score" in s else None,
        "non_member_normalized_score": float(s["non_member_normalized_score"]) if "non_member_normalized_score" in s else None,
    }
    return [
        {
            "model_version": model_version,
            "dataset": member_ds,
            "is_member": 1,
            "contamination_score": float(s["member_score_mean"]),
            "num_samples": int(s["num_members"]),
            **base,
        },
        {
            "model_version": model_version,
            "dataset": non_member_ds,
            "is_member": 0,
            "contamination_score": float(s["non_member_score_mean"]),
            "num_samples": int(s["num_non_members"]),
            **base,
        },
    ]


def _mm_detect_rows(run, model_version: str, member_ds: str, non_member_ds: str | None, is_single_split: bool) -> list[dict[str, Any]]:
    s = run.summary
    base = {
        "run_id": run.id,
        "run_url": run.url,
        "created_at": run.created_at,
    }
    rows = [
        {
            "model_version": model_version,
            "dataset": member_ds,
            "is_member": 1,
            "cr": float(s["member_cr"]),
            "pcr": float(s["member_pcr"]),
            "delta": float(s["member_delta"]),
            "phi": float(s["member_phi"]),
            "num_samples": int(s["member_num_samples"]),
            **base,
        },
    ]
    if not is_single_split:
        rows.append({
            "model_version": model_version,
            "dataset": non_member_ds,
            "is_member": 0,
            "cr": float(s["non_member_cr"]),
            "pcr": float(s["non_member_pcr"]),
            "delta": float(s["non_member_delta"]),
            "phi": float(s["non_member_phi"]),
            "num_samples": int(s["non_member_num_samples"]),
            **base,
        })
    return rows


def _mia_row(run, method_kind: str, model_version: str, member_ds: str) -> dict[str, Any]:
    s = run.summary
    return {
        "method": method_kind,
        "model_version": model_version,
        "dataset": member_ds,
        "num_members": int(s["num_members"]),
        "num_non_members": int(s["num_non_members"]),
        "num_samples": int(s["num_members"]) + int(s["num_non_members"]),
        "roc_auc": float(s["roc_auc"]),
        "run_id": run.id,
        "run_url": run.url,
        "created_at": run.created_at,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def collect(entity: str, project: str, include_smoke: bool) -> tuple[list[dict], list[dict], list[dict], dict[str, int]]:
    api = wandb.Api()
    counts = defaultdict(int)
    # group_key -> (sample_total, created_at, kind, model_version, member_ds, non_member_ds, is_single_split, run)
    best: dict[tuple, tuple] = {}

    for run in api.runs(f"{entity}/{project}"):
        counts["seen"] += 1

        if run.state != "finished":
            counts["dropped_unfinished"] += 1
            continue
        tags = list(run.tags or [])
        if (not include_smoke) and "smoke" in tags:
            counts["dropped_smoke"] += 1
            continue

        cfg = dict(run.config)
        kind = _method_kind(cfg)
        if kind is None:
            counts["dropped_unknown_method"] += 1
            continue

        model_version = _model_version(cfg.get("model_cfg") or {})
        if not model_version:
            counts["dropped_no_model"] += 1
            continue

        member_ds = _dataset_short(cfg.get("dataset_member"))
        non_member_ds = _dataset_short(cfg.get("dataset_non_member"))
        if not member_ds:
            counts["dropped_no_dataset"] += 1
            continue

        is_single_split = False

        if kind in _MIA_KINDS:
            if not _summary_has_keys(run.summary, _MIA_REQUIRED):
                counts["dropped_missing_summary"] += 1
                continue
            sample_total = int(run.summary["num_members"]) + int(run.summary["num_non_members"])
            key = (kind, model_version, member_ds, None)

        elif kind == "mm_detect":
            is_single_split = "non_member_num_samples" not in run.summary or run.summary.get("non_member_num_samples") is None
            required = _MM_DETECT_MEMBER_REQUIRED if is_single_split else _MM_DETECT_REQUIRED
            if not _summary_has_keys(run.summary, required):
                counts["dropped_missing_summary"] += 1
                continue
            if is_single_split:
                sample_total = int(run.summary["member_num_samples"])
                key = (kind, model_version, member_ds, None)
            else:
                if not non_member_ds:
                    counts["dropped_no_dataset"] += 1
                    continue
                sample_total = int(run.summary["member_num_samples"]) + int(run.summary["non_member_num_samples"])
                key = (kind, model_version, member_ds, non_member_ds)

        else:  # codec
            if not non_member_ds:
                counts["dropped_no_dataset"] += 1
                continue
            if not _summary_has_keys(run.summary, _CODEC_REQUIRED):
                counts["dropped_missing_summary"] += 1
                continue
            sample_total = int(run.summary["num_members"]) + int(run.summary["num_non_members"])
            key = (kind, model_version, member_ds, non_member_ds)

        prev = best.get(key)
        # Pick max samples; latest created_at as tiebreaker.
        candidate = (sample_total, run.created_at, kind, model_version, member_ds, non_member_ds, is_single_split, run)
        if (prev is None) or (candidate[0], candidate[1]) > (prev[0], prev[1]):
            best[key] = candidate
            counts["kept_or_replaced"] += 1

    codec_rows: list[dict] = []
    mm_rows: list[dict] = []
    mia_rows: list[dict] = []
    for (_, _, kind, model_version, member_ds, non_member_ds, is_single_split, run) in best.values():
        if kind == "codec":
            codec_rows.extend(_codec_rows(run, model_version, member_ds, non_member_ds))
        elif kind == "mm_detect":
            mm_rows.extend(_mm_detect_rows(run, model_version, member_ds, non_member_ds, is_single_split))
        else:
            mia_rows.append(_mia_row(run, kind, model_version, member_ds))

    # Stable sort for diffable CSVs.
    codec_rows.sort(key=lambda r: (r["model_version"], r["dataset"], r["is_member"]))
    mm_rows.sort(key=lambda r: (r["model_version"], r["dataset"], r["is_member"]))
    mia_rows.sort(key=lambda r: (r["method"], r["model_version"], r["dataset"]))
    counts["final_codec_runs"] = len(codec_rows) // 2
    counts["final_mm_detect_runs"] = sum(1 for k in best if k[0] == "mm_detect")
    counts["final_mia_runs"] = len(mia_rows)
    return codec_rows, mm_rows, mia_rows, dict(counts)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entity", default="nask-di")
    p.add_argument("--project", default="audio-benchmark")
    p.add_argument("--output-dir", default="results", type=Path)
    p.add_argument("--include-smoke", action="store_true",
                   help="Keep runs tagged 'smoke' (default: drop them)")
    args = p.parse_args()

    print(f"Fetching runs from {args.entity}/{args.project} ...", file=sys.stderr)
    codec_rows, mm_rows, mia_rows, counts = collect(args.entity, args.project, args.include_smoke)

    codec_cols = [
        "model_version", "dataset", "is_member", "contamination_score",
        "member_normalized_score", "non_member_normalized_score",
        "num_samples", "run_id", "run_url", "created_at",
    ]
    mm_cols = [
        "model_version", "dataset", "is_member", "cr", "pcr", "delta", "phi",
        "num_samples", "run_id", "run_url", "created_at",
    ]
    mia_cols = [
        "method", "model_version", "dataset", "num_members", "num_non_members",
        "num_samples", "roc_auc", "run_id", "run_url", "created_at",
    ]

    codec_path = args.output_dir / "codec_results.csv"
    mm_path = args.output_dir / "mm_detect_results.csv"
    mia_path = args.output_dir / "mia_results.csv"
    _write_csv(codec_path, codec_rows, codec_cols)
    _write_csv(mm_path, mm_rows, mm_cols)
    _write_csv(mia_path, mia_rows, mia_cols)

    print(f"\nWrote {codec_path}  ({len(codec_rows)} rows, {counts.get('final_codec_runs', 0)} unique setups)",
          file=sys.stderr)
    print(f"Wrote {mm_path}  ({len(mm_rows)} rows, {counts.get('final_mm_detect_runs', 0)} unique setups)",
          file=sys.stderr)
    print(f"Wrote {mia_path}  ({len(mia_rows)} rows, {counts.get('final_mia_runs', 0)} unique setups)",
          file=sys.stderr)
    print("\nCounts:", file=sys.stderr)
    for k in ("seen", "dropped_unfinished", "dropped_smoke", "dropped_unknown_method",
              "dropped_no_model", "dropped_no_dataset", "dropped_missing_summary"):
        print(f"  {k:30s} {counts.get(k, 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()
