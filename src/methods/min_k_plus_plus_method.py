"""
Zhang et al. (2024) — arXiv:2404.02936.

Min-K%++: same min-K% aggregation as Shi et al., but on a calibrated per-token
score that normalizes for how concentrated the model's distribution is at each
position.

    z_t = (log p(x_t | x_<t) - μ_t) / σ_t
    μ_t = E_{y ~ p_t}[ log p_t(y) ] = Σ_v p_t(v) · log p_t(v) = -H(p_t)
    σ_t = std_{y ~ p_t}[ log p_t(y) ]

Tokens at the conditional mode satisfy z_t > 0; "surprising" tokens have
z_t < 0. Members tend to keep z_t close to or above zero even in their
bottom-K% tail, where non-members go strongly negative.

Paper's raw score (higher = more likely member):
    s = mean_{t in min-K% by z} z_t

We negate.
"""
from __future__ import annotations

from typing import Any

import torch

from src.methods.base_method import MethodBaseClass


class MinKPlusPlusMethod(MethodBaseClass):
    def __init__(self, k_pct: float = 20.0, prompt: str | None = None) -> None:
        if not (0.0 < k_pct <= 100.0):
            raise ValueError(f"k_pct must be in (0, 100], got {k_pct}")
        self.k_pct = k_pct
        self.prompt = prompt

    def _score_from_dict(self, sd: dict[str, Any]) -> float:
        log_probs: torch.Tensor = sd["token_log_probs"]
        mu:        torch.Tensor = sd["token_log_prob_mean"]
        sigma:     torch.Tensor = sd["token_log_prob_std"]

        z = (log_probs - mu) / sigma.clamp_min(1e-8)
        n = z.numel()
        k = max(1, int(round(n * self.k_pct / 100.0)))

        bottom_k, _ = torch.topk(z, k=k, largest=False)
        mean_z = float(bottom_k.mean().item())
        return -mean_z
