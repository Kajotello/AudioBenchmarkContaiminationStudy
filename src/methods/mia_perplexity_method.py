from __future__ import annotations

import math

import torch

from src.methods.base_method import MethodBaseClass
from src.models.base_AL_model import BaseAudioLanguageModel


class MIAPerplexityMethod(MethodBaseClass):
    """
    Membership Inference Attack baseline based on per-token perplexity.

    Interpretation:
        - lower perplexity => sample is more likely to be a member
        - higher perplexity => sample is more likely to be a non-member
    """

    def __init__(self, prompt: str | None = None) -> None:
        self.prompt = prompt

    def run(
        self,
        model: BaseAudioLanguageModel,
        audio: torch.Tensor,
        text: str,
    ) -> float:
        score_dict = model.score_text_given_audio(
            audio=audio,
            target_text=text,
            prompt=self.prompt,
        )

        mean_nll = float(score_dict["mean_nll"])
        perplexity = float(math.exp(mean_nll))

        return perplexity
