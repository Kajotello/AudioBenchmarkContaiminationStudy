from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _compute_roc_auc(labels: list[int], scores: list[float]) -> float:
    """
    Compute ROC AUC from scratch.

    Assumption:
    - lower score => more likely member
    - label 1 => member
    - label 0 => non-member

    We therefore negate scores so that higher value means more likely positive.
    """
    n = len(labels)
    if n == 0:
        return float("nan")

    pos_count = sum(labels)
    neg_count = n - pos_count
    if pos_count == 0 or neg_count == 0:
        return float("nan")

    transformed = [-s for s in scores]
    paired = list(zip(transformed, labels))

    # Rank-based AUC with average ranks for ties
    paired_sorted = sorted(enumerate(paired), key=lambda x: x[1][0])

    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and paired_sorted[j + 1][1][0] == paired_sorted[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            original_idx = paired_sorted[k][0]
            ranks[original_idx] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(r for r, y in zip(ranks, labels) if y == 1)
    auc = (sum_pos_ranks - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count)
    return float(auc)


def _compute_best_threshold_accuracy(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """
    Find the best threshold on scores.

    Prediction rule:
    - predict member (1) if score <= threshold
    - predict non-member (0) otherwise
    """
    if not scores:
        return float("nan"), float("nan")

    unique_thresholds = sorted(set(scores))
    best_acc = -1.0
    best_thr = unique_thresholds[0]

    for thr in unique_thresholds:
        preds = [1 if s <= thr else 0 for s in scores]
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return float(best_thr), float(best_acc)


def _score_dataset(
    model: Any,
    method: Any,
    dataset: Any,
    label: int,
    split_name: str,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    limit = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))

    for idx in range(limit):
        audio, text = dataset[idx]
        score = float(method.run(model=model, audio=audio, text=text))
        results.append(
            {
                "idx": idx,
                "split": split_name,
                "label": label,  # 1 = member, 0 = non-member
                "score": score,
                "text": text,
                "text_length_chars": len(text),
            }
        )

    return results


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run a real MIA test:
    - score member dataset
    - score non-member dataset
    - save per-sample results
    - compute separation metrics
    """

    if "data_member" not in cfg or "data_non_member" not in cfg:
        raise ValueError(
            "MIA evaluation requires both `data_member` and `data_non_member` configs."
        )

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

    max_member_samples = cfg.get("max_member_samples", None)
    max_non_member_samples = cfg.get("max_non_member_samples", None)

    log.info("Scoring member dataset...")
    with torch.no_grad():
        member_results = _score_dataset(
            model=model,
            method=method,
            dataset=member_dataset,
            label=1,
            split_name="member",
            max_samples=max_member_samples,
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
        "num_members": len(member_results),
        "num_non_members": len(non_member_results),
        "member_score_mean": float(sum(member_scores) / len(member_scores)) if member_scores else float("nan"),
        "non_member_score_mean": float(sum(non_member_scores) / len(non_member_scores)) if non_member_scores else float("nan"),
        "roc_auc": roc_auc,
        "best_threshold": best_threshold,
        "best_accuracy": best_acc,
    }

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_sample_path = output_dir / "mia_per_sample_results.csv"
    with per_sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["idx", "split", "label", "score", "text", "text_length_chars"],
        )
        writer.writeheader()
        writer.writerows(all_results)

    metrics_path = output_dir / "mia_metrics.txt"
    metrics_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in metric_dict.items()),
        encoding="utf-8",
    )

    log.info(f"Saved per-sample MIA results to {per_sample_path}")
    log.info(f"Saved MIA metrics to {metrics_path}")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":
    main()
