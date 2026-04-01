from abc import ABC, abstractmethod
from typing import Any

import torch

from src.models.base_AL_model import BaseAudioLanguageModel


class MethodBaseClass(ABC):
    """Base class for all evaluation methods."""

    @abstractmethod
    def run(
        self,
        model: BaseAudioLanguageModel,
        audio: torch.Tensor,
        text: str,
    ) -> float:
        """Run method for one (audio, text) example and return scalar score."""
        raise NotImplementedError

    def aggregate(self, scores: list[float]) -> dict[str, Any]:
        """Aggregate per-sample scores; can be overridden by subclasses."""
        if len(scores) == 0:
            return {
                "num_samples": 0,
                "score_mean": float("nan"),
                "score_min": float("nan"),
                "score_max": float("nan"),
            }
        return {
            "num_samples": len(scores),
            "score_mean": float(sum(scores) / len(scores)),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
        }
