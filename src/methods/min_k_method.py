"""
Shi et al. (2024), ICLR — arXiv:2310.16789.

Min-K% Prob: instead of averaging log-prob over ALL target tokens (Yeom),
average over the K% tokens with the LOWEST log-probability. The intuition is
that members rarely have "outlier" tokens with very low log-prob; non-members
tend to have a thicker low-prob tail.

Paper's raw score (higher = more likely member):
    s = mean_{t in min-K%} log p(x_t | x_<t)

We negate so that lower score => more likely member (project convention).
"""
from __future__ import annotations

from typing import Any

import torch

from src.methods.base_method import MethodBaseClass


class MinKProbMethod(MethodBaseClass):
    def __init__(self, k_pct: float = 20.0, prompt: str | None = None) -> None:
        if not (0.0 < k_pct <= 100.0):
            raise ValueError(f"k_pct must be in (0, 100], got {k_pct}")
        self.k_pct = k_pct
        self.prompt = prompt

    def _score_from_dict(self, sd: dict[str, Any]) -> float:
        log_probs: torch.Tensor = sd["token_log_probs"]
        n = log_probs.numel()
        k = max(1, int(round(n * self.k_pct / 100.0)))

        # k smallest log-probs (the "min-K%" set: most surprising tokens)
        bottom_k, _ = torch.topk(log_probs, k=k, largest=False)
        mean_log_prob = float(bottom_k.mean().item())
        return -mean_log_prob
