import logging

import torch

import numpy as np
from typing import Any
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

from src.models.base_AL_model import BaseAudioLanguageModel

log = logging.getLogger(__name__)


class AudioFlamingoWrapper(BaseAudioLanguageModel):
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        super().__init__()
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype
        ).to(device)
        self.model.eval()

        self._device = torch.device(device)
        self._dtype = torch_dtype

        # Diagnostics
        log.info("Loaded model: %s", model_id)
        log.info("Requested device: %s, dtype: %s", device, dtype)
        log.info("First param device: %s", next(self.model.parameters()).device)
        log.info("First param dtype:  %s", next(self.model.parameters()).dtype)

        param_devices = {p.device for p in self.model.parameters()}
        log.info("All parameter devices: %s", param_devices)
        if len(param_devices) > 1:
            log.warning("Model is split across multiple devices: %s", param_devices)

        if torch.cuda.is_available():
            log.info(
                "CUDA mem allocated: %.2f GB / reserved: %.2f GB",
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.memory_reserved() / 1e9,
            )

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if hf_device_map is not None:
            log.info("hf_device_map: %s", hf_device_map)

    @torch.no_grad()
    def score_text_given_audio(
        self,
        audio: torch.Tensor,
        target_text: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """
        Score target_text token-by-token, conditioned on audio.

        Returns per-token statistics that downstream MIA methods consume:
          - token_log_probs:       log p(x_t | x_<t)                     (N,)
          - token_entropies:       H(p_t)                                (N,)
          - token_log_prob_mean:   μ_t = E_{y~p_t}[log p_t(y)] = -H(p_t) (N,)
          - token_log_prob_std:    σ_t = std_{y~p_t}[log p_t(y)]         (N,)
          - mean_log_prob, mean_nll, num_tokens, sequence_log_prob
        """
        conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt or "Describe the audio."},
                {"type": "audio", "audio": audio.numpy()},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]
        full_inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )

        prompt_conversation = [
            {"role": "user", "content": [
                {"type": "text", "text": prompt or "Describe the audio."},
                {"type": "audio", "audio": audio.numpy()},
            ]},
        ]
        prompt_inputs = self.processor.apply_chat_template(
            prompt_conversation, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        assert torch.equal(
            full_inputs["input_ids"][:, :prompt_len],
            prompt_inputs["input_ids"],
        ), "Prompt token prefix mismatch — label boundary is wrong"

        device = next(self.model.parameters()).device
        full_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in full_inputs.items()}
        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                full_inputs[k] = v.to(self._dtype)

        outputs = self.model(**full_inputs)
        logits = outputs.logits                                       # (1, T, V)

        # Causal shift, fp32 for log_softmax numerical stability
        shift_logits = logits[:, :-1, :].float()                      # (1, T-1, V)
        shift_labels = full_inputs["input_ids"][:, 1:]                # (1, T-1)

        target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        target_mask[:, prompt_len - 1:] = True

        # Slice down to target positions first — saves memory on the big (N,V) ops
        target_logits = shift_logits[target_mask]                     # (N, V)
        target_labels = shift_labels[target_mask]                     # (N,)
        if target_labels.numel() == 0:
            raise ValueError("No valid target tokens found for scoring.")

        log_probs = torch.log_softmax(target_logits, dim=-1)          # (N, V)
        probs = log_probs.exp()                                       # (N, V)

        token_log_probs = log_probs.gather(
            dim=-1, index=target_labels.unsqueeze(-1)
        ).squeeze(-1)                                                 # (N,)

        # μ_t = Σ_v p_v · log p_v  (== -H_t)
        mu = (probs * log_probs).sum(dim=-1)                          # (N,)
        entropies = -mu                                               # (N,)

        # σ_t² = Σ_v p_v · (log p_v)² − μ_t²
        second_moment = (probs * log_probs.pow(2)).sum(dim=-1)
        var = (second_moment - mu.pow(2)).clamp_min(1e-8)
        sigma = var.sqrt()                                            # (N,)

        n = int(token_log_probs.numel())
        mean_log_prob = float(token_log_probs.mean().item())

        return {
            "token_log_probs":     token_log_probs.detach().cpu(),
            "token_entropies":     entropies.detach().cpu(),
            "token_log_prob_mean": mu.detach().cpu(),
            "token_log_prob_std":  sigma.detach().cpu(),
            "mean_log_prob":       mean_log_prob,
            "mean_nll":           -mean_log_prob,
            "num_tokens":          n,
            "sequence_log_prob":   float(token_log_probs.sum().item()),
        }

    @torch.no_grad()
    def score_text_given_audio_batch(
        self,
        audios: list[torch.Tensor],
        target_texts: list[str],
        prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        assert len(audios) == len(target_texts), \
            "audios and target_texts must be the same length"
        batch_size = len(audios)
        user_prompt = prompt or "Describe the audio."

        tokenizer = self.processor.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "right"           # required for our label-boundary math
        if tokenizer.pad_token_id is None:
            # decoder-only tokenizers often ship without an explicit pad token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        try:
            # --- Step 1: per-sample tokenization. The processor pads audio to 30 s
            # and builds a correct input_features_mask when given one conversation.
            per_full: list[dict[str, torch.Tensor]] = []
            per_prompt: list[dict[str, torch.Tensor]] = []

            for audio, target_text in zip(audios, target_texts):
                audio_np = (
                    audio.detach().cpu().numpy()
                    if isinstance(audio, torch.Tensor) else np.asarray(audio)
                )
                full_conv = [
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "audio", "audio": audio_np},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": target_text},
                    ]},
                ]
                prompt_conv = [
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "audio", "audio": audio_np},
                    ]},
                ]
                per_full.append(self.processor.apply_chat_template(
                    full_conv, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_tensors="pt",
                ))
                per_prompt.append(self.processor.apply_chat_template(
                    prompt_conv, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                ))

            # --- Step 2: collate text + concat audio
            prompt_lens = [p["input_ids"].shape[1] for p in per_prompt]
            full_lens   = [f["input_ids"].shape[1] for f in per_full]
            max_len     = max(full_lens)
            pad_id      = tokenizer.pad_token_id

            batch_input_ids = torch.full(
                (batch_size, max_len), pad_id, dtype=per_full[0]["input_ids"].dtype,
            )
            batch_attn = torch.zeros(
                (batch_size, max_len), dtype=per_full[0]["attention_mask"].dtype,
            )
            for b, fin in enumerate(per_full):
                L = full_lens[b]
                batch_input_ids[b, :L] = fin["input_ids"][0]
                batch_attn[b, :L]      = fin["attention_mask"][0]

            # Pass through every remaining tensor key the processor returned
            # (input_features, input_features_mask, anything else AF3 adds in future
            # versions). Each one is already (1, ...) and shape-uniform across the
            # batch because every audio was padded to 30 s, so cat on dim=0 works.
            batch: dict[str, Any] = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attn,
            }
            for key, val in per_full[0].items():
                if key in ("input_ids", "attention_mask"):
                    continue
                if isinstance(val, torch.Tensor):
                    batch[key] = torch.cat([s[key] for s in per_full], dim=0)

            # --- Step 3: sanity check the label boundary survived collation
            for b in range(batch_size):
                p_len = prompt_lens[b]
                if not torch.equal(
                    batch_input_ids[b, :p_len],
                    per_prompt[b]["input_ids"][0, :p_len],
                ):
                    raise RuntimeError(
                        f"Prompt prefix mismatch at batch index {b} — collation "
                        f"corrupted the label boundary."
                    )

            # --- Step 4: device + dtype
            batch = {
                k: (v.to(self._device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    batch[k] = v.to(self._dtype)

            # --- Step 5: forward
            outputs = self.model(**batch)
            logits = outputs.logits                              # (B, T, V)
            input_ids = batch["input_ids"]
            attn = batch["attention_mask"]

            shift_logits = logits[:, :-1, :].float()             # fp32 for log_softmax
            shift_labels = input_ids[:, 1:]
            shift_attn   = attn[:, 1:]

            results: list[dict[str, Any]] = []
            for b in range(batch_size):
                p_len = prompt_lens[b]
                mask = torch.zeros_like(shift_labels[b], dtype=torch.bool)
                mask[p_len - 1:] = True
                mask &= shift_attn[b].bool()

                target_logits_b = shift_logits[b][mask]          # (N_b, V)
                target_labels_b = shift_labels[b][mask]          # (N_b,)
                if target_labels_b.numel() == 0:
                    raise ValueError(f"No valid target tokens for batch item {b}.")

                log_probs = torch.log_softmax(target_logits_b, dim=-1)
                probs = log_probs.exp()

                token_log_probs = log_probs.gather(
                    dim=-1, index=target_labels_b.unsqueeze(-1)
                ).squeeze(-1)
                mu = (probs * log_probs).sum(dim=-1)
                entropies = -mu
                second_moment = (probs * log_probs.pow(2)).sum(dim=-1)
                var = (second_moment - mu.pow(2)).clamp_min(1e-8)
                sigma = var.sqrt()

                n = int(token_log_probs.numel())
                mean_log_prob = float(token_log_probs.mean().item())

                results.append({
                    "token_log_probs":     token_log_probs.detach().cpu(),
                    "token_entropies":     entropies.detach().cpu(),
                    "token_log_prob_mean": mu.detach().cpu(),
                    "token_log_prob_std":  sigma.detach().cpu(),
                    "mean_log_prob":       mean_log_prob,
                    "mean_nll":           -mean_log_prob,
                    "num_tokens":          n,
                    "sequence_log_prob":   float(token_log_probs.sum().item()),
                })

            return results
        finally:
            tokenizer.padding_side = old_padding_side

    @torch.no_grad()
    def score_text_given_audio_with_context(
        self,
        audio: torch.Tensor,
        target_text: str,
        context: list[tuple[torch.Tensor | None, str]],
        prompt: str | None = None,
        mode: str = "full",
    ) -> dict[str, Any]:
        """Score target_text given the target audio plus N in-context (audio, answer) examples.

        Identical return shape to ``score_text_given_audio``. Only the tokens of the
        FINAL assistant turn are scored.
        """
        if mode not in ("full", "no_audio"):
            raise ValueError(f"mode must be 'full' or 'no_audio', got {mode!r}")

        user_prompt = prompt or "Describe the audio."

        def _user_turn(ctx_audio: torch.Tensor | None, include_audio: bool) -> dict:
            content: list[dict] = [{"type": "text", "text": user_prompt}]
            if include_audio and ctx_audio is not None:
                content.append({"type": "audio", "audio": ctx_audio.numpy()})
            return {"role": "user", "content": content}

        # Build the in-context demonstrations
        full_conv: list[dict] = []
        for ctx_audio, ctx_answer in context:
            include_audio = (mode == "full") and (ctx_audio is not None)
            full_conv.append(_user_turn(ctx_audio, include_audio))
            full_conv.append({"role": "assistant",
                              "content": [{"type": "text", "text": ctx_answer}]})

        # Target turn — always with target audio
        full_conv.append({"role": "user", "content": [
            {"type": "text", "text": user_prompt},
            {"type": "audio", "audio": audio.numpy()},
        ]})
        full_conv.append({"role": "assistant",
                          "content": [{"type": "text", "text": target_text}]})

        # Prompt-only (no target answer) version, used to locate the label boundary
        prompt_conv = full_conv[:-1]   # drop the last assistant turn

        full_inputs = self.processor.apply_chat_template(
            full_conv, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_conv, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        assert torch.equal(
            full_inputs["input_ids"][:, :prompt_len],
            prompt_inputs["input_ids"],
        ), "Prompt token prefix mismatch — label boundary is wrong"

        device = next(self.model.parameters()).device
        full_inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in full_inputs.items()}
        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                full_inputs[k] = v.to(self._dtype)

        outputs = self.model(**full_inputs)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = full_inputs["input_ids"][:, 1:]

        target_mask = torch.zeros_like(shift_labels, dtype=torch.bool)
        target_mask[:, prompt_len - 1:] = True

        target_logits = shift_logits[target_mask]
        target_labels = shift_labels[target_mask]
        if target_labels.numel() == 0:
            raise ValueError("No valid target tokens found for scoring.")

        log_probs = torch.log_softmax(target_logits, dim=-1)
        probs = log_probs.exp()

        token_log_probs = log_probs.gather(
            dim=-1, index=target_labels.unsqueeze(-1)
        ).squeeze(-1)

        mu = (probs * log_probs).sum(dim=-1)
        entropies = -mu
        second_moment = (probs * log_probs.pow(2)).sum(dim=-1)
        var = (second_moment - mu.pow(2)).clamp_min(1e-8)
        sigma = var.sqrt()

        n = int(token_log_probs.numel())
        mean_log_prob = float(token_log_probs.mean().item())

        return {
            "token_log_probs":     token_log_probs.detach().cpu(),
            "token_entropies":     entropies.detach().cpu(),
            "token_log_prob_mean": mu.detach().cpu(),
            "token_log_prob_std":  sigma.detach().cpu(),
            "mean_log_prob":       mean_log_prob,
            "mean_nll":           -mean_log_prob,
            "num_tokens":          n,
            "sequence_log_prob":   float(token_log_probs.sum().item()),
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
        full_inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        full_inputs = {k: (v.to(self._device) if isinstance(v, torch.Tensor) else v) for k, v in full_inputs.items()}
        for k, v in full_inputs.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                full_inputs[k] = v.to(self._dtype)

        generated = self.model.generate(**full_inputs, max_new_tokens=128)
        return self.processor.batch_decode(
            generated[:, full_inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
