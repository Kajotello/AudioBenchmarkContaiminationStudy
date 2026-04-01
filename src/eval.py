from pathlib import Path
from typing import Any, Dict, Tuple

import hydra
import rootutils
import torch
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    model.eval()

    log.info(f"Instantiating dataset <{cfg.data._target_}>")
    dataset = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating method <{cfg.method._target_}>")
    method = hydra.utils.instantiate(cfg.method)

    object_dict = {
        "cfg": cfg,
        "dataset": dataset,
        "model": model,
        "method": method,
    }

    max_samples = int(cfg.get("max_samples", len(dataset)))
    max_samples = min(max_samples, len(dataset))
    scores: list[float] = []

    log.info("Running method on dataset...")
    with torch.no_grad():
        for idx in range(max_samples):
            audio, text = dataset[idx]
            score = method.run(model=model, audio=audio, text=text)
            scores.append(float(score))

    metric_dict = method.aggregate(scores)
    metric_dict["method_name"] = cfg.method._target_

    output_path = Path(cfg.paths.output_dir) / "eval_metrics.txt"
    output_path.write_text("\n".join(f"{k}: {v}" for k, v in metric_dict.items()), encoding="utf-8")
    log.info(f"Saved metrics to {output_path}")

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
