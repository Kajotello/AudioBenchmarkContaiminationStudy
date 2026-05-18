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

    def score_text_given_audio_with_context(
            self,
            audio: torch.Tensor,
            target_text: str,
            context: list[tuple[torch.Tensor | None, str]],
            prompt: str | None = None,
            mode: str = "full",
    ) -> dict[str, Any]:
        """Score target_text given a target audio plus N in-context examples.

        Args:
            audio: target audio tensor.
            target_text: text to be scored token-by-token.
            context: list of (context_audio_or_None, context_answer) tuples. When
                an entry's audio is None or ``mode == "no_audio"``, the context
                example is treated as text-only (no audio for that demonstration).
            prompt: user instruction. Defaults to a generic captioning prompt.
            mode: "full" -> include context audio + caption; "no_audio" -> drop
                  the audio from context examples (text-only demonstration).

        Returns the same dict shape as ``score_text_given_audio``.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        """Generate text based on audio input."""
        raise NotImplementedError
