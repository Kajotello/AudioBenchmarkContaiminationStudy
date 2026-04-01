import torch
from lightning import LightningModule


class BaseAudioLanguageModel(LightningModule):
    def get_log_probs(self, audio, text):
        raise NotImplementedError

    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        """Generate text based on audio input."""
        raise NotImplementedError