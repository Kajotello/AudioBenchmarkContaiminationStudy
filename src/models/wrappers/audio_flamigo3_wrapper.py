import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from src.models.base_AL_model import BaseAudioLanguageModel

class AudioFlamingoWrapper(BaseAudioLanguageModel):
    def __init__(self, model_id: str):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id)

    def get_log_probs(self, audio, text):
        inputs = self.processor(text=text, audio=audio, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits

        labels = inputs["input_ids"].clone()
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        # Causal LM alignment: token at t is predicted from positions < t.
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        valid_mask = shift_labels != -100
        token_log_probs = token_log_probs.masked_fill(~valid_mask, 0.0)

        # Return sequence log-probability per sample.
        return token_log_probs.sum(dim=-1)

    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        raise NotImplementedError