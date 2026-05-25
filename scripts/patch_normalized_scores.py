#!/usr/bin/env python3
"""Back-fill normalized contamination scores onto existing CoDeC runs.

For each run directory under --logs-dir that contains codec_per_sample_results.csv,
computes:

    Score(D) = (1/N) * sum_i 1[delta(x_i) < 0]

separately for member (label=1) and non-member (label=0) rows, then:
  - Appends member_normalized_score / non_member_normalized_score to codec_metrics.txt
    (idempotent: skips if keys already present).
  - Updates the matching wandb run summary via wandb.Api().

Usage:
    python scripts/patch_normalized_scores.py \\
        --entity nask-di --project audio-benchmark \\
        [--logs-dir logs/contamination/runs] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional


def _compute_normalized(csv_path: Path) -> tuple[float, float]:
    """Return (member_normalized_score, non_member_normalized_score) from CSV."""
    member_neg = member_total = 0
    non_member_neg = non_member_total = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            score = float(row["score"])
            label = int(row["label"])
            if label == 1:
                member_total += 1
                if score < 0:
                    member_neg += 1
            else:
                non_member_total += 1
                if score < 0:
                    non_member_neg += 1

    m = member_neg / member_total if member_total else float("nan")
    nm = non_member_neg / non_member_total if non_member_total else float("nan")
    return m, nm


def _patch_metrics_file(
    metrics_path: Path, member_val: float, non_member_val: float, dry_run: bool
) -> bool:
    """Append missing keys to codec_metrics.txt. Returns True if file was (or would be) changed."""
    text = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else ""
    existing_keys = {
        line.split(":")[0].strip() for line in text.splitlines() if ":" in line
    }

    additions: list[str] = []
    if "member_normalized_score" not in existing_keys:
        additions.append(f"member_normalized_score: {member_val}")
    if "non_member_normalized_score" not in existing_keys:
        additions.append(f"non_member_normalized_score: {non_member_val}")

    if not additions:
        return False

    if not dry_run:
        updated = text.rstrip("\n") + "\n" + "\n".join(additions) + "\n"
        metrics_path.write_text(updated, encoding="utf-8")
    return True


def _find_wandb_run_id(run_dir: Path) -> Optional[str]:
    """Extract wandb run ID from the wandb/ sub-directory (run-TIMESTAMP-ID pattern)."""
    wandb_dir = run_dir / "wandb"
    if not wandb_dir.is_dir():
        return None
    for entry in wandb_dir.iterdir():
        if entry.name.startswith("run-") and entry.is_dir():
            parts = entry.name.split("-")
            if len(parts) >= 3:
                return parts[-1]
    return None


_WANDB_UNAVAILABLE = "unavailable"
_WANDB_ALREADY_SET = "already_set"
_WANDB_UPDATED = "updated"


def _patch_wandb(
    entity: str,
    project: str,
    run_id: str,
    member_val: float,
    non_member_val: float,
    dry_run: bool,
) -> str:
    """Update wandb run summary. Returns one of _WANDB_* constants."""
    try:
        import wandb
    except ImportError:
        return _WANDB_UNAVAILABLE

    api = wandb.Api()
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception as exc:
        print(f"  [WARN] Could not fetch run {run_id}: {exc}", file=sys.stderr)
        return _WANDB_UNAVAILABLE

    updates: dict[str, float] = {}
    if (
        "member_normalized_score" not in run.summary
        or run.summary["member_normalized_score"] is None
    ):
        updates["member_normalized_score"] = member_val
    if (
        "non_member_normalized_score" not in run.summary
        or run.summary["non_member_normalized_score"] is None
    ):
        updates["non_member_normalized_score"] = non_member_val

    if not updates:
        return _WANDB_ALREADY_SET

    if not dry_run:
        run.summary.update(updates)
    return _WANDB_UPDATED


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/contamination/runs"),
        help="Root directory containing run sub-directories",
    )
    p.add_argument("--entity", default="nask-di")
    p.add_argument("--project", default="audio-benchmark")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing anything",
    )
    args = p.parse_args()

    csv_files = sorted(args.logs_dir.glob("*/codec_per_sample_results.csv"))
    if not csv_files:
        print(
            f"No codec_per_sample_results.csv found under {args.logs_dir}",
            file=sys.stderr,
        )
        sys.exit(0)

    print(
        f"Found {len(csv_files)} run(s) to process.{' [DRY-RUN]' if args.dry_run else ''}"
    )

    patched_local = patched_wandb = skipped = 0

    for csv_path in csv_files:
        run_dir = csv_path.parent
        member_val, non_member_val = _compute_normalized(csv_path)

        label = run_dir.name
        if math.isnan(member_val) or math.isnan(non_member_val):
            print(f"  {label}: no samples for one split, skipping")
            skipped += 1
            continue

        print(
            f"  {label}: member_norm={member_val:.4f}  non_member_norm={non_member_val:.4f}"
        )

        metrics_path = run_dir / "codec_metrics.txt"
        file_changed = _patch_metrics_file(
            metrics_path, member_val, non_member_val, args.dry_run
        )
        if file_changed:
            patched_local += 1
            print(f"    local file {'[would update]' if args.dry_run else 'updated'}")
        else:
            print(f"    local file already up-to-date")

        run_id = _find_wandb_run_id(run_dir)
        if run_id:
            wb_status = _patch_wandb(
                args.entity,
                args.project,
                run_id,
                member_val,
                non_member_val,
                args.dry_run,
            )
            if wb_status == _WANDB_UPDATED:
                patched_wandb += 1
                print(
                    f"    wandb run {run_id} {'[would update]' if args.dry_run else 'updated'}"
                )
            elif wb_status == _WANDB_ALREADY_SET:
                print(f"    wandb run {run_id} already up-to-date")
            else:
                print(f"    wandb unavailable, skipped run {run_id}")
        else:
            print(f"    no wandb run ID found, skipping wandb update")

    print(
        f"\nDone. local_patched={patched_local}  wandb_patched={patched_wandb}  skipped={skipped}"
    )


if __name__ == "__main__":
    main()
