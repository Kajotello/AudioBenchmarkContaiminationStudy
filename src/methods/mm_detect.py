"""MM-DETECT — Slot Guessing for Perturbed Caption (Continuous Adaptation).

Adapted from: "Both Text and Images Leaked! A Systematic Analysis of Data
Contamination in Multimodal LLM" (Song et al., 2024).

Idea: Compare the model's confidence in predicting a masked keyword in the
original caption vs. a semantically-preserved, back-translated caption.

If the model has seen the original (audio, text) pair during training
(i.e., it is contaminated), it will exhibit unusually high confidence in
predicting the exact missing word in the original text, but significantly
lower confidence for the back-translated version.

Per the project convention (LOWER score => MORE LIKELY MEMBER), we return:
    score = mean(log p_back_translated) - mean(log p_original)
A *negative* score means the model was much more confident on the original
format, indicating likely dataset membership.
"""
from __future__ import annotations

from typing import Any

import torch

from src.methods.base_method import MethodBaseClass
from src.models.base_AL_model import BaseAudioLanguageModel


class MMDetectMethod(MethodBaseClass):
    def __init__(self, prompt_template: str | None = None) -> None:
        """
        Args:
            prompt_template: An optional string to wrap the masked text. 
                             e.g., "Fill in the [MASK] of the following sentence in one word: {caption}"
        """
        self.prompt_template = prompt_template or "{caption}"

    def _check_single_sample(
        self,
        model: BaseAudioLanguageModel,
        audio: torch.Tensor,
        orig_target: str,
        orig_masked: str,
        back_target: str,
        back_masked: str,
    ) -> tuple[bool, bool]:
        """
        Note: The signature differs slightly from MethodBaseClass to accommodate 
        the paired original/perturbed texts required by MM-DETECT.
        """
        # Format the prompts wrapping the masked texts
        prompt_orig = self.prompt_template.format(caption=orig_masked)
        prompt_back = self.prompt_template.format(caption=back_masked)

        # Pass 1: Score the original target word given the original masked context
        pred_orig = model.generate(
            audio=audio, prompt=prompt_orig,
        )
        is_orig_correct = (pred_orig.lower().strip().strip(',').strip('.') == orig_target.lower().strip().strip(',').strip('.'))
        
        # Pass 2: Score the perturbed target word given the back-translated masked context
        pred_back = model.generate(
            audio=audio, prompt=prompt_back,
        )
        is_back_correct = (pred_back.lower().strip().strip(',').strip('.') == back_target.lower().strip().strip(',').strip('.'))
        return is_orig_correct, is_back_correct


    def run_on_dataset(
        self,
        model: BaseAudioLanguageModel,
        dataset: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Evaluate all samples; return split-level CR, PCR, and delta (PCR - CR)."""
        correct_original = 0
        correct_perturbed = 0
        suspicious_instances = 0

        for elem in dataset:
            is_orig_correct, is_back_correct = self._check_single_sample(
                model,
                elem["audio"],
                elem["target_original"],
                elem["masked_original"],
                elem["target_back"],
                elem["masked_back"],
            )
            if is_orig_correct:
                correct_original += 1
            if is_back_correct:
                correct_perturbed += 1
            if is_orig_correct and not is_back_correct:
                suspicious_instances += 1

        n = len(dataset)
        if n == 0:
            cr = pcr = delta = phi = float("nan")
        else:
            cr = (correct_original / n) * 100.0
            pcr = (correct_perturbed / n) * 100.0
            delta = pcr - cr
            phi = (suspicious_instances / n) * 100.0

        return {
            "cr": float(cr),
            "pcr": float(pcr),
            "delta": float(delta),
            "num_samples": float(n),
            "phi": float(phi),
        }