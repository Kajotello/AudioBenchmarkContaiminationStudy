import torch
from typing import Any
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from src.models.base_AL_model import BaseAudioLanguageModel

class AudioFlamingoWrapper(BaseAudioLanguageModel):
    def __init__(self, model_id: str):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )

    @torch.no_grad()
    def score_text_given_audio(
            self,
            audio: torch.Tensor,
            target_text: str,
            prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Score how likely target_text is as a response to the audio.

        We build a full conversation (user prompt + assistant response),
        tokenize it, then compute NLL only over the assistant (target_text) tokens.
        """
        # Build conversation WITH the target text as the assistant reply
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe the audio."},
                    {"type": "audio", "audio": audio.numpy()},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text},
                ],
            },
        ]

        # Tokenize the full conversation (prompt + response)
        full_inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # Tokenize only the prompt (without assistant response) to find where target starts
        prompt_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "Describe the audio."},
                    {"type": "audio", "audio": audio.numpy()},
                ],
            },
        ]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        prompt_len = prompt_inputs["input_ids"].shape[1]

        device = next(self.model.parameters()).device
        full_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in full_inputs.items()
        }
        full_inputs = {
            k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
            for k, v in full_inputs.items()
        }

        outputs = self.model(**full_inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Causal LM: logits[t] predicts token[t+1]
        # We want loss over target tokens only (positions prompt_len onwards)
        # shift_logits[prompt_len-1:] predicts tokens[prompt_len:]
        shift_logits = logits[:, :-1, :]
        shift_labels = full_inputs["input_ids"][:, 1:]

        # Create mask: only score positions where we're predicting target tokens
        target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        # prompt_len-1 because of the shift: logit at position prompt_len-1 predicts token at prompt_len
        target_mask[:, prompt_len - 1:] = True

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        target_log_probs = token_log_probs[target_mask]
        num_tokens = int(target_log_probs.numel())

        if num_tokens == 0:
            raise ValueError("No valid target tokens found for scoring.")

        sequence_log_prob = float(target_log_probs.sum().item())
        mean_log_prob = float(target_log_probs.mean().item())
        mean_nll = -mean_log_prob

        return {
            "token_log_probs": target_log_probs.detach().cpu(),
            "mean_log_prob": mean_log_prob,
            "mean_nll": mean_nll,
            "num_tokens": num_tokens,
            "sequence_log_prob": sequence_log_prob,
        }

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "audio": audio.numpy()},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        inputs = {
            k: v.half() if isinstance(v, torch.Tensor) and v.dtype == torch.float32 else v
            for k, v in inputs.items()
        }

        generated = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
