import math

from src.methods.base_method import MethodBaseClass
from src.models.base_AL_model import BaseAudioLanguageModel


class MIAPerplexityMethod(MethodBaseClass):
    """Per-example perplexity-style score using model log-probabilities."""

    def run(self, model: BaseAudioLanguageModel, audio, text: str) -> float:
        seq_log_prob = model.get_log_probs(audio=audio, text=text)
        if hasattr(seq_log_prob, "item"):
            seq_log_prob = seq_log_prob.item()

        # Perplexity proxy from sequence log-probability.
        return float(math.exp(-float(seq_log_prob)))
