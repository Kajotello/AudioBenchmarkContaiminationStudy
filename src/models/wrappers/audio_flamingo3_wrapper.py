import torch
from typing import Any
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from src.models.base_AL_model import BaseAudioLanguageModel

class AudioFlamingoWrapper(BaseAudioLanguageModel):
    def __init__(self, model_id: str):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id)

    @torch.no_grad()
    def score_text_given_audio(
            self,
            audio: torch.Tensor,
            target_text: str,
            prompt: str | None = None,
    ) -> dict[str, Any]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe the audio."},
                    {"type": "audio", "audio": audio},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
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
        valid_mask = shift_labels != -100

        safe_labels = shift_labels.clone()
        safe_labels[~valid_mask] = 0

        token_log_probs = torch.log_softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)

        valid_token_log_probs = token_log_probs[valid_mask]
        num_tokens = int(valid_token_log_probs.numel())

        if num_tokens == 0:
            raise ValueError("No valid target tokens found for scoring.")

        sequence_log_prob = float(valid_token_log_probs.sum().item())
        mean_log_prob = float(valid_token_log_probs.mean().item())
        mean_nll = -mean_log_prob

        return {
            "token_log_probs": valid_token_log_probs.detach().cpu(),
            "mean_log_prob": mean_log_prob,
            "mean_nll": mean_nll,
            "num_tokens": num_tokens,
            "sequence_log_prob": sequence_log_prob,
        }

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        inputs = self.processor(
            text=prompt,
            audio=audio,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generated = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]
