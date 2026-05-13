from abc import ABC, abstractmethod
from typing import Any

import torch

from src.models.base_AL_model import BaseAudioLanguageModel


class MethodBaseClass(ABC):
    """
    Base class for all MIA methods.

    Convention: a method returns a scalar score per sample, where
        LOWER score => MORE LIKELY MEMBER.

    Subclasses implement `_score_from_dict`, a pure function over the dict
    produced by `BaseAudioLanguageModel.score_text_given_audio`. `run` and
    `run_batch` are provided by this base class.
    """

    prompt: str | None = None

    @abstractmethod
    def _score_from_dict(self, score_dict: dict[str, Any]) -> float:
        raise NotImplementedError

    def run(
        self,
        model: BaseAudioLanguageModel,
        audio: torch.Tensor,
        text: str,
    ) -> float:
        sd = model.score_text_given_audio(
            audio=audio, target_text=text, prompt=self.prompt,
        )
        return self._score_from_dict(sd)

    def run_batch(
        self,
        model: BaseAudioLanguageModel,
        audios: list[torch.Tensor],
        texts: list[str],
    ) -> list[float]:
        sds = model.score_text_given_audio_batch(
            audios=audios, target_texts=texts, prompt=self.prompt,
        )
        return [self._score_from_dict(sd) for sd in sds]

    def aggregate(self, scores: list[float]) -> dict[str, Any]:
        if not scores:
            return {"num_samples": 0,
                    "score_mean": float("nan"),
                    "score_min": float("nan"),
                    "score_max": float("nan")}
        return {"num_samples": len(scores),
                "score_mean": float(sum(scores) / len(scores)),
                "score_min": float(min(scores)),
                "score_max": float(max(scores))}
