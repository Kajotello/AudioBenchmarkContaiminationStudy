"""MM-DETECT evaluation entrypoint (parallel to ``src/eval_mia.py`` / ``src/contamination.py``).

Collects samples from ``JsonlMMDetectDataset`` and delegates whole-split scoring to
``MMDetectMethod.run_on_dataset`` (CR, PCR, delta only — no per-sample outputs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.methods.mm_detect import MMDetectMethod
from src.utils import RankedLogger, extras, init_wandb_run, log_wandb_metrics, task_wrapper

log = RankedLogger(__name__, rank_zero_only=True)


def _collect_samples(dataset: Any, max_samples: int | None) -> list[dict[str, Any]]:
    limit = len(dataset) if max_samples is None else min(int(max_samples), len(dataset))
    return [dataset[i] for i in range(limit)]


def _run_split(
    model: Any,
    method: MMDetectMethod,
    dataset: Any,
    max_samples: int | None,
) -> dict[str, float]:
    samples = _collect_samples(dataset, max_samples)
    return method.run_on_dataset(model, samples)


@task_wrapper
def evaluate_mm_detect(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if "data_member" not in cfg or "data_non_member" not in cfg:
        raise ValueError(
            "MM-DETECT evaluation requires both `data_member` and `data_non_member`."
        )

    init_wandb_run(cfg)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    log.info(f"Instantiating method <{cfg.method._target_}>")
    method = hydra.utils.instantiate(cfg.method)
    if not isinstance(method, MMDetectMethod):
        raise TypeError(
            f"Expected MMDetectMethod, got {type(method).__name__}. "
            "Set method=mm_detect in the Hydra config."
        )

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
        member_split = _run_split(
            model=model,
            method=method,
            dataset=member_dataset,
            max_samples=max_member_samples,
        )

    log.info("Scoring non-member dataset...")
    with torch.no_grad():
        non_member_split = _run_split(
            model=model,
            method=method,
            dataset=non_member_dataset,
            max_samples=max_non_member_samples,
        )

    metric_dict = {
        "method_name": cfg.method._target_,
        "prompt_template": method.prompt_template,
        "member_num_samples": int(member_split["num_samples"]),
        "member_cr": member_split["cr"],
        "member_pcr": member_split["pcr"],
        "member_delta": member_split["delta"],
        "member_phi": member_split["phi"],
        "non_member_num_samples": int(non_member_split["num_samples"]),
        "non_member_cr": non_member_split["cr"],
        "non_member_pcr": non_member_split["pcr"],
        "non_member_delta": non_member_split["delta"],
        "non_member_phi": non_member_split["phi"],
        "delta_gap": float(member_split["delta"] - non_member_split["delta"]),
    }

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "mm_detect_metrics.txt"
    metrics_path.write_text(
        "\n".join(f"{k}: {v}" for k, v in metric_dict.items()),
        encoding="utf-8",
    )

    log.info(f"Saved MM-DETECT metrics to {metrics_path}")

    log_wandb_metrics(metric_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval_mm_detect.yaml")
def main(cfg: DictConfig) -> None:
    extras(cfg)
    evaluate_mm_detect(cfg)


if __name__ == "__main__":
    main()
