from abc import abstractmethod

import torch
from typing import Any
from lightning import LightningModule


class BaseAudioLanguageModel(LightningModule):
    @abstractmethod
    def score_text_given_audio(
            self,
            audio: torch.Tensor,
            target_text: str,
            prompt: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        """Generate text based on audio input."""
        raise NotImplementedError
