"""CoDeC — Contamination Detection via Context (Audio adaptation).

Paper: https://arxiv.org/abs/2510.27055
Reference (LLM) implementation:
https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/examples/notebooks/contamination_detection_demo.ipynb

Idea: compare the model's confidence in predicting the target text in two settings:
  (a) no_context: score target_text given target audio alone
  (b) with_context: score target_text given target audio + N (audio, answer) demonstrations

If the model has already seen this (audio, text) pair during training (i.e. the dataset is
contaminated), context shouldn't help. If novel, context provides distributional cues that
improve confidence.

Per the project convention (LOWER score => MORE LIKELY MEMBER), we return
    score = mean(log p_with_context) - mean(log p_no_context)
so that a *negative* score (context didn't help / hurt) means likely member.

Two modes are supported:
  - "full"     (default): context demonstration = (prompt + audio + answer)
  - "no_audio":           context demonstration = (prompt + answer), no context audio
"""
from __future__ import annotations

import random
from typing import Any

import torch

from src.methods.base_method import MethodBaseClass
from src.models.base_AL_model import BaseAudioLanguageModel


class CoDeCMethod(MethodBaseClass):
    def __init__(
        self,
        num_context_examples: int = 1,
        mode: str = "full",
        token_range: tuple[int, int] = (10, -1),
        seed: int = 42,
        prompt: str | None = None,
    ) -> None:
        if mode not in ("full", "no_audio"):
            raise ValueError(f"mode must be 'full' or 'no_audio', got {mode!r}")
        if num_context_examples < 1:
            raise ValueError(f"num_context_examples must be >= 1, got {num_context_examples}")

        self.num_context_examples = num_context_examples
        self.mode = mode
        self.token_range = tuple(token_range)
        self.seed = seed
        self.prompt = prompt

        self._rng = random.Random(seed)
        self._context_pool: list[tuple[torch.Tensor, str]] = []

    def set_context_pool(self, pool: list[tuple[torch.Tensor, str]]) -> None:
        """Inject the pool of (audio, text) tuples we sample context demos from."""
        self._context_pool = list(pool)

    def _score_from_dict(self, score_dict: dict[str, Any]) -> float:
        # Not used directly — we override ``run`` because CoDeC needs two scoring passes.
        # Provided to satisfy the abstract base class.
        return float("nan")

    def _trim(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Slice token_log_probs by the configured token_range, like the reference notebook.

        Falls back to the full sequence if the configured range would yield an
        empty or degenerate slice (e.g. trim_head=10 with only 11 tokens).
        """
        start, end = self.token_range
        n = log_probs.numel()
        if end <= 0:
            end = n + end if end < 0 else n
        end = min(end, n)
        if start >= n or end <= start:
            return log_probs
        return log_probs[start:end]

    def _sample_context(
        self, exclude_text: str | None,
    ) -> list[tuple[torch.Tensor, str]]:
        if not self._context_pool:
            raise RuntimeError(
                "CoDeC context pool is empty. Call set_context_pool(...) before run()."
            )
        candidates = [
            (a, t) for (a, t) in self._context_pool if t != exclude_text
        ] or list(self._context_pool)

        k = min(self.num_context_examples, len(candidates))
        return self._rng.sample(candidates, k)

    def run(
        self,
        model: BaseAudioLanguageModel,
        audio: torch.Tensor,
        text: str,
    ) -> float:
        context = self._sample_context(exclude_text=text)

        sd_no_ctx = model.score_text_given_audio(
            audio=audio, target_text=text, prompt=self.prompt,
        )
        sd_with_ctx = model.score_text_given_audio_with_context(
            audio=audio, target_text=text, context=context,
            prompt=self.prompt, mode=self.mode,
        )

        lp_no = self._trim(sd_no_ctx["token_log_probs"])
        lp_ctx = self._trim(sd_with_ctx["token_log_probs"])

        if lp_no.numel() == 0 or lp_ctx.numel() == 0:
            return float("nan")

        conf_no = float(lp_no.mean().item())
        conf_ctx = float(lp_ctx.mean().item())

        # Lower => more likely member. If context did NOT help, conf_ctx < conf_no
        # and the diff is negative.
        return conf_ctx - conf_no

    def run_batch(
        self,
        model: BaseAudioLanguageModel,
        audios: list[torch.Tensor],
        texts: list[str],
    ) -> list[float]:
        return [self.run(model=model, audio=a, text=t) for a, t in zip(audios, texts)]
