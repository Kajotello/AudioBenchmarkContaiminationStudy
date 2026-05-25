"""
Li et al. (2024), NeurIPS — arXiv:2411.02902.

VL-MIA: at HIGH-ENTROPY positions, the model has many plausible continuations
and so log p(x_t | x_<t) is generally low; for members, memorization still
pulls log p(x_t | x_<t) up above the random-pick floor, while for non-members
it stays near it. Restricting the score to high-entropy positions amplifies
the member/non-member gap.

Raw score (higher = more likely member):
    s = mean_{t in top-X% by H_t} log p(x_t | x_<t)

We negate.
"""

from __future__ import annotations

from typing import Any

import torch

from src.methods.base_method import MethodBaseClass


class VLMIAEntropyMethod(MethodBaseClass):
    def __init__(self, top_pct: float = 20.0, prompt: str | None = None) -> None:
        if not (0.0 < top_pct <= 100.0):
            raise ValueError(f"top_pct must be in (0, 100], got {top_pct}")
        self.top_pct = top_pct
        self.prompt = prompt

    def _score_from_dict(self, sd: dict[str, Any]) -> float:
        log_probs: torch.Tensor = sd["token_log_probs"]
        entropies: torch.Tensor = sd["token_entropies"]

        n = log_probs.numel()
        k = max(1, int(round(n * self.top_pct / 100.0)))

        # Top-k by entropy: most uncertain positions
        _, top_idx = torch.topk(entropies, k=k, largest=True)
        selected = log_probs[top_idx]
        mean_log_prob = float(selected.mean().item())
        return -mean_log_prob
