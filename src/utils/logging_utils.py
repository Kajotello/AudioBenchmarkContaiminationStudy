from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

_WANDB_METRIC_SKIP = frozenset({"method_name"})


def target_short_name(target: str) -> str:
    return target.rsplit(".", 1)[-1]


def dataset_label(data_cfg: Optional[DictConfig]) -> str:
    if data_cfg is None or not hasattr(data_cfg, "_target_"):
        return "unknown"
    dataset_name = data_cfg.get("dataset_name")
    split = data_cfg.get("split")
    if dataset_name and split:
        safe_name = str(dataset_name).replace("/", "_")
        return f"{safe_name}:{split}"
    if dataset_name:
        return str(dataset_name).replace("/", "_")
    return "unknown"


def dataset_short_name(data_cfg: Optional[DictConfig]) -> str:
    """Dataset id without split (for wandb group)."""
    label = dataset_label(data_cfg)
    return label.split(":", 1)[0]


def build_wandb_group(cfg: DictConfig) -> str:
    wb_group = cfg.get("logger", {}).get("wandb", {}).get("group") if cfg.get("logger") else None
    if wb_group:
        return str(wb_group)
    method = target_short_name(str(cfg.method._target_))
    dataset_member = dataset_short_name(cfg.get("data_member"))
    dataset_non_member = dataset_short_name(cfg.get("data_non_member"))
    return f"{method}-{dataset_member}-vs-{dataset_non_member}"


def build_wandb_run_config(cfg: DictConfig) -> Dict[str, Any]:
    """Static experiment setup logged to wandb.config."""
    run_config: Dict[str, Any] = {
        "task_name": cfg.get("task_name"),
        "method": target_short_name(str(cfg.method._target_)),
        "model": target_short_name(str(cfg.model._target_)),
        "dataset_member": dataset_label(cfg.get("data_member")),
        "dataset_non_member": dataset_label(cfg.get("data_non_member")),
        "method_cfg": OmegaConf.to_container(cfg.method, resolve=True),
        "model_cfg": OmegaConf.to_container(cfg.model, resolve=True),
        "max_member_samples": cfg.get("max_member_samples"),
        "max_non_member_samples": cfg.get("max_non_member_samples"),
        "batch_size": cfg.get("batch_size"),
        "seed": cfg.get("seed"),
        "context_pool_size": cfg.get("context_pool_size"),
    }
    return {k: v for k, v in run_config.items() if v is not None}


def build_wandb_run_name(cfg: DictConfig) -> str:
    method = target_short_name(str(cfg.method._target_))
    model = target_short_name(str(cfg.model._target_))
    member = dataset_label(cfg.get("data_member"))
    non_member = dataset_label(cfg.get("data_non_member"))
    task = cfg.get("task_name", "run")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{task}-{method}-{model}-{member}-vs-{non_member}-{timestamp}"


def wandb_is_enabled(cfg: DictConfig) -> bool:
    if not cfg.get("logger") or not cfg.logger.get("wandb"):
        return False
    return bool(cfg.logger.wandb.get("enabled", True))


def wandb_numeric_metrics(metric_dict: Mapping[str, Any]) -> Dict[str, float | int]:
    """Extract numeric results for wandb.log (skip identifiers and NaNs)."""
    metrics: Dict[str, float | int] = {}
    for key, value in metric_dict.items():
        if key in _WANDB_METRIC_SKIP:
            continue
        if isinstance(value, bool):
            metrics[key] = int(value)
        elif isinstance(value, int):
            metrics[key] = value
        elif isinstance(value, float) and not math.isnan(value):
            metrics[key] = value
    return metrics


@rank_zero_only
def init_wandb_run(cfg: DictConfig) -> bool:
    """Start a wandb run from Hydra config. Returns True if a run was created."""
    if not wandb_is_enabled(cfg):
        log.info("wandb disabled; skipping run init.")
        return False

    try:
        import wandb
    except ImportError:
        log.warning("wandb is not installed; skipping run init.")
        return False

    wb = cfg.logger.wandb
    run_name = wb.get("name") or build_wandb_run_name(cfg)
    tags = list(wb.get("tags") or [])
    for tag in cfg.get("tags") or []:
        if tag not in tags:
            tags.append(tag)

    init_kwargs: Dict[str, Any] = {
        "project": wb.get("project", "audio-benchmark"),
        "name": run_name,
        "config": build_wandb_run_config(cfg),
        "dir": str(wb.get("save_dir", cfg.paths.output_dir)),
        "mode": "offline" if wb.get("offline") else "online",
        "tags": tags,
        "group": build_wandb_group(cfg),
    }
    if wb.get("entity"):
        init_kwargs["entity"] = wb.entity

    wandb.init(**init_kwargs)
    log.info(f"Started wandb run: {run_name} (group={init_kwargs['group']})")
    return True


@rank_zero_only
def log_wandb_metrics(metric_dict: Mapping[str, Any]) -> None:
    """Log evaluation metrics to the active wandb run."""
    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    metrics = wandb_numeric_metrics(metric_dict)
    if metrics:
        wandb.log(metrics)
