"""
Yeom et al. (2018), IEEE CSF — arXiv:1709.01604.

The original perplexity-threshold MIA baseline. Score = perplexity
(equivalently, mean NLL) of the target text under the model.

Convention: lower perplexity => more likely member.
"""

from __future__ import annotations

import math
from typing import Any

from src.methods.base_method import MethodBaseClass


class YeomPerplexityMethod(MethodBaseClass):
    def __init__(self, prompt: str | None = None, use_perplexity: bool = True) -> None:
        self.prompt = prompt
        self.use_perplexity = use_perplexity

    def _score_from_dict(self, sd: dict[str, Any]) -> float:
        mean_nll = float(sd["mean_nll"])
        return math.exp(mean_nll) if self.use_perplexity else mean_nll


# Backwards-compatible alias so existing configs (configs/method/MIA_perplexity.yaml)
# don't break.
MIAPerplexityMethod = YeomPerplexityMethod
