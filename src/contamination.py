"""Contamination detection entrypoint (parallel to ``src/eval.py``).

Runs the CoDeC method (or any method that consumes a context pool) over a member /
non-member dataset pair, producing per-sample scores and aggregate metrics.

Re-uses scoring + metric helpers from ``src/eval_mia.py`` to avoid duplication.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.eval_mia import (
    _compute_best_threshold_accuracy,
    _compute_roc_auc,
    _score_dataset,
)
from src.utils import (
    RankedLogger,
    extras,
    init_wandb_run,
    log_wandb_metrics,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _build_context_pool(
    dataset: Any,
    pool_size: int,
    seed: int,
) -> list[tuple[torch.Tensor, str]]:
    """Sample (audio, text) pairs to serve as the in-context demonstration pool."""
    n = len(dataset)
    pool_size = min(pool_size, n)
    rng = random.Random(seed)
    indices = rng.sample(range(n), pool_size)
    return [tuple(dataset[i]) for i in indices]  # type: ignore[misc]


@task_wrapper
def detect_contamination(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "data_member" not in cfg or "data_non_member" not in cfg:
        raise ValueError(
            "Contamination detection requires both `data_member` and `data_non_member`."
        )

    init_wandb_run(cfg)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    log.info(f"Instantiating method <{cfg.method._target_}>")
    method = hydra.utils.instantiate(cfg.method)

    log.info(f"Instantiating member dataset <{cfg.data_member._target_}>")
    member_dataset = hydra.utils.instantiate(cfg.data_member)

    log.info(f"Instantiating non-member dataset <{cfg.data_non_member._target_}>")
    non_member_dataset = hydra.utils.instantiate(cfg.data_non_member)

    object_dict = {
        "cfg": cfg,
        "model": model,
        "method": method,
        "member_dataset": member_dataset,
        "non_member_dataset": non_member_dataset,
    }

    # Build context pool from the member dataset by default
    pool_size = int(cfg.get("context_pool_size", 50))
    seed = int(cfg.get("seed", 42))
    log.info(f"Building context pool of size {pool_size} from member dataset")
    pool = _build_context_pool(member_dataset, pool_size=pool_size, seed=seed)
    if hasattr(method, "set_context_pool"):
        method.set_context_pool(pool)
    else:
        log.warning(
            "Selected method has no set_context_pool(); context pool is unused."
        )

    max_member_samples = cfg.get("max_member_samples", None)
    max_non_member_samples = cfg.get("max_non_member_samples", None)
    batch_size = int(cfg.get("batch_size", 1))

    log.info("Scoring member dataset...")
    with torch.no_grad():
        member_results = _score_dataset(
            model=model,
            method=method,
            dataset=member_dataset,
            label=1,
            split_name="member",
            max_samples=max_member_samples,
            batch_size=batch_size,
            seed=seed,
        )

    log.info("Scoring non-member dataset...")
    with torch.no_grad():
        non_member_results = _score_dataset(
            model=model,
            method=method,
            dataset=non_member_dataset,
            label=0,
            split_name="non_member",
            max_samples=max_non_member_samples,
            batch_size=batch_size,
            seed=seed + 1,
        )

    all_results = member_results + non_member_results
    labels = [row["label"] for row in all_results]
    scores = [row["score"] for row in all_results]
    member_scores = [row["score"] for row in member_results]
    non_member_scores = [row["score"] for row in non_member_results]

    roc_auc = _compute_roc_auc(labels, scores)
    best_threshold, best_acc = _compute_best_threshold_accuracy(labels, scores)

    metric_dict = {
        "method_name": cfg.method._target_,
        "mode": getattr(method, "mode", None),
        "num_context_examples": getattr(method, "num_context_examples", None),
        "context_pool_size": pool_size,
        "num_members": len(member_results),
        "num_non_members": len(non_member_results),
        "member_score_mean": (
            float(sum(member_scores) / len(member_scores))
            if member_scores
            else float("nan")
        ),
        "non_member_score_mean": (
            float(sum(non_member_scores) / len(non_member_scores))
            if non_member_scores
            else float("nan")
        ),
        # Score(D) = (1/N) * sum_i 1[delta(x_i) < 0]
        "member_normalized_score": (
            float(sum(1 for s in member_scores if s < 0) / len(member_scores))
            if member_scores
            else float("nan")
        ),
        "non_member_normalized_score": (
            float(sum(1 for s in non_member_scores if s < 0) / len(non_member_scores))
            if non_member_scores
            else float("nan")
        ),
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
        "best_accuracy": best_acc,
    }

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = output_dir / "codec_per_sample_results.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "split", "label", "score", "text", "text_length_chars"],
        )
        writer.writeheader()
        writer.writerows(all_results)

    metrics_path = output_dir / "codec_metrics.txt"
    metrics_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in metric_dict.items()),
        encoding="utf-8",
    )

    log.info(f"Saved per-sample contamination results to {per_sample_path}")
    log.info(f"Saved contamination metrics to {metrics_path}")

    log_wandb_metrics(metric_dict)

    return metric_dict, object_dict


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="contamination.yaml"
)
def main(cfg: DictConfig) -> None:
    extras(cfg)
    detect_contamination(cfg)


if __name__ == "__main__":
    main()
