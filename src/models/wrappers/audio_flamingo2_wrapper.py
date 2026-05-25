"""Audio Flamingo 2 wrapper conforming to BaseAudioLanguageModel.

This wrapper loads the HF-pretrained AF2 model defined in
``src/audio_flamingo_2/inference_HF_pretrained`` and exposes:
- ``score_text_given_audio``: per-token log-probs given (audio, target_text)
- ``score_text_given_audio_with_context``: same, with N (audio, answer) demonstrations
- ``generate``: free-form generation

Because the AF2 inference package uses relative-style imports (``from src.factory ...``)
that collide with the project's top-level ``src/`` package, we load it by temporarily
chdir'ing into the AF2 directory + sys.path-prepending, then restore cwd afterwards.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import yaml

from src.models.base_AL_model import BaseAudioLanguageModel

log = logging.getLogger(__name__)


_SHADOWED_TOP_LEVEL = ("src", "utils")


@contextmanager
def _pushd_and_syspath(path: Path):
    """Temporarily chdir into ``path``, prepend it to ``sys.path``, and shadow
    the project's top-level packages whose names collide with AF2's
    (notably ``src`` and ``utils``) so AF2's own modules can be imported."""
    old_cwd = os.getcwd()
    path_str = str(path)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    os.chdir(path_str)

    saved_modules: dict[str, Any] = {}
    for name in list(sys.modules):
        if name in _SHADOWED_TOP_LEVEL or any(
            name.startswith(p + ".") for p in _SHADOWED_TOP_LEVEL
        ):
            saved_modules[name] = sys.modules.pop(name)

    try:
        yield
    finally:
        for name in list(sys.modules):
            if name in _SHADOWED_TOP_LEVEL or any(
                name.startswith(p + ".") for p in _SHADOWED_TOP_LEVEL
            ):
                del sys.modules[name]
        sys.modules.update(saved_modules)

        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass


class AudioFlamingo2Wrapper(BaseAudioLanguageModel):
    def __init__(
        self,
        hf_repo_id: str = "nvidia/audio-flamingo-2-0.5B",
        af2_dir: str = "src/audio_flamingo_2/inference_HF_pretrained",
        config_path: str = "configs/inference.yaml",
        device: str = "cuda",
        dtype: str = "float16",
        default_prompt: str = "describe audio",
        lang_encoder_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        super().__init__()

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")

        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                       "float32": torch.float32}[dtype]

        repo_root = Path(__file__).resolve().parents[3]
        af2_path = (repo_root / af2_dir).resolve()
        if not af2_path.exists():
            raise FileNotFoundError(f"AF2 directory not found: {af2_path}")

        self.af2_dir = af2_path
        self.default_prompt = default_prompt
        self._device = torch.device(device)
        self._dtype = torch_dtype

        with _pushd_and_syspath(af2_path):
            from huggingface_hub import snapshot_download
            from safetensors.torch import load_file
            # The AF2 inference package
            from src.factory import create_model_and_transforms  # type: ignore
            from utils import Dict2Class, get_cast_dtype          # type: ignore

            # configs/inference.yaml is committed alongside the AF2 source code
            # (it's NOT in the HF snapshot), so load it from af2_path.
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            model_config = dict(config["model_config"])
            # The committed inference.yaml encodes the 0.5B architecture
            # (Qwen2.5-0.5B). For larger AF2 sizes the language encoder must
            # match the checkpoint's hidden size (1.5B → Qwen2.5-1.5B,
            # 3B → Qwen2.5-3B), otherwise state_dict load fails on size
            # mismatch in lang_encoder.model.embed_tokens.weight.
            if lang_encoder_path:
                model_config["lang_encoder_path"] = lang_encoder_path
                model_config["tokenizer_path"] = tokenizer_path or lang_encoder_path
            elif tokenizer_path:
                model_config["tokenizer_path"] = tokenizer_path

            # inference.yaml has cache_dir: .cache (relative) — when cwd is
            # af2_path that resolves into the source tree, which is $HOME and
            # gets blown out quickly by Qwen2.5 weights. Redirect to the HF
            # hub cache so downloads land on scratch ($HF_HOME / $HF_HUB_CACHE).
            import os as _os
            hf_cache_dir = _os.environ.get("HF_HUB_CACHE") or _os.environ.get(
                "TRANSFORMERS_CACHE"
            )
            if hf_cache_dir:
                model_config["cache_dir"] = hf_cache_dir
            self.clap_config = dict(config["clap_config"])
            train_args = Dict2Class(config["train_config"])

            # Snapshot contains the model-specific safe_ckpt/ and clap_ckpt/
            # under HF cache ($HF_HOME / $HF_HUB_CACHE → scratch). Returns the
            # cached snapshot's absolute path; every AF2 size lives under its
            # own models--<safe_repo_id>/ dir so parallel jobs never collide.
            ckpt_root = Path(snapshot_download(repo_id=hf_repo_id))

            # inference.yaml references clap_ckpt/<file> relative to the model
            # snapshot. Rewrite to absolute so CLAP loads from the cache, not
            # from cwd.
            cp = self.clap_config.get("checkpoint")
            if cp and not Path(cp).is_absolute():
                self.clap_config["checkpoint"] = str(ckpt_root / cp)

            # PyTorch 2.6 flipped the torch.load default to weights_only=True,
            # which trips the AF2 / CLAP pickle-based checkpoints (they store
            # numpy scalars). Restore the legacy default just for this load.
            _orig_torch_load = torch.load
            def _torch_load_legacy(*args, **kwargs):
                kwargs.setdefault("weights_only", False)
                return _orig_torch_load(*args, **kwargs)
            torch.load = _torch_load_legacy
            try:
                model, tokenizer = create_model_and_transforms(
                    **model_config,
                    clap_config=self.clap_config,
                    use_local_files=train_args.offline,
                    gradient_checkpointing=train_args.gradient_checkpointing,
                    freeze_lm_embeddings=train_args.freeze_lm_embeddings,
                )
            finally:
                torch.load = _orig_torch_load

            # Keep model in float32: AF2's CLAP submodule casts audio to fp32
            # internally (int16_to_float32_torch), so casting weights to fp16
            # breaks batchnorm dtype matching. Upstream inference.py also keeps
            # fp32 weights + autocast. dtype is honoured via autocast in forward.
            model = model.to(self._device)
            model.eval()

            with open(ckpt_root / "safe_ckpt/metadata.json", "r") as f:
                metadata = json.load(f)
            state_dict: dict[str, torch.Tensor] = {}
            for chunk_name in metadata:
                state_dict.update(load_file(str(ckpt_root / f"safe_ckpt/{chunk_name}.safetensors")))
            missing, unexpected = model.load_state_dict(state_dict, False)
            if missing:
                log.warning("AF2 missing keys: %d", len(missing))
            if unexpected:
                log.warning("AF2 unexpected keys: %d", len(unexpected))

            self.model = model
            self.tokenizer = tokenizer
            self._cast_dtype = get_cast_dtype(train_args.precision)

        # Cache token ids
        self._sep_token_id = tokenizer.sep_token_id
        self._eos_token_id = tokenizer.eos_token_id
        self._eoc_token_id = tokenizer.encode("<|endofchunk|>")[-1]
        self._audio_token_id = tokenizer.encode("<audio>")[-1]

        log.info("Loaded AudioFlamingo2 from %s onto %s (%s)", hf_repo_id, device, dtype)

    # ---------------------------------------------------------------- helpers

    def _tensor_to_clips(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a 1-D audio tensor to (audio_clips, audio_embed_mask).

        Re-uses ``inference.load_audio`` by writing the tensor to a temporary WAV.
        Keeps windowing/padding logic identical to upstream AF2 inference.
        """
        with _pushd_and_syspath(self.af2_dir):
            from inference import load_audio  # type: ignore

            arr = audio.detach().cpu().numpy().astype(np.float32)
            if arr.ndim > 1:
                arr = arr.mean(axis=-1 if arr.shape[-1] < arr.shape[0] else 0)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            try:
                sf.write(tmp_path, arr, 16000, subtype="FLOAT")
                clips, mask = load_audio(tmp_path, self.clap_config)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        return clips, mask

    def _tokenize_with_boundary(self, prompt_text: str, full_text: str) -> tuple[torch.Tensor, int]:
        """Tokenize the full prompt+target text and return (input_ids, prompt_len)."""
        full = self.tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
        prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)

        full_ids = full["input_ids"]
        prompt_ids = prompt["input_ids"]
        prompt_len = prompt_ids.shape[1]

        if not torch.equal(full_ids[:, :prompt_len], prompt_ids):
            raise RuntimeError(
                "Prompt prefix mismatch — tokenization boundary inconsistent. "
                f"prompt_text={prompt_text!r}, full_text={full_text!r}"
            )
        return full_ids, prompt_len

    def _forward_score(
        self,
        audio_clips: torch.Tensor,
        audio_mask: torch.Tensor,
        full_ids: torch.Tensor,
        prompt_len: int,
    ) -> dict[str, Any]:
        """Run the forward pass and compute the per-token scoring dict."""
        device = self._device
        # Keep audio_x in fp32 to match the always-fp32 CLAP audio pipeline.
        audio_x = audio_clips.unsqueeze(0).to(device, dtype=torch.float32)
        audio_x_mask = audio_mask.unsqueeze(0).to(device)
        lang_x = full_ids.to(device)

        autocast_enabled = self._dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=self._dtype, enabled=autocast_enabled,
        ):
            outputs = self.model(
                audio_x=audio_x,
                audio_x_mask=audio_x_mask,
                lang_x=lang_x,
                clear_conditioned_layers=True,
            )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()
        shift_labels = lang_x[:, 1:]

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

    # ---------------------------------------------------------------- API

    @torch.no_grad()
    def score_text_given_audio(
        self,
        audio: torch.Tensor,
        target_text: str,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        user_prompt = (prompt or self.default_prompt).strip().lower()
        sep = self.tokenizer.sep_token

        prompt_text = f"<audio>{user_prompt}{sep}"
        full_text = f"{prompt_text}{target_text.strip()}"

        clips, mask = self._tensor_to_clips(audio)
        full_ids, prompt_len = self._tokenize_with_boundary(prompt_text, full_text)
        return self._forward_score(clips, mask, full_ids, prompt_len)

    @torch.no_grad()
    def score_text_given_audio_with_context(
        self,
        audio: torch.Tensor,
        target_text: str,
        context: list[tuple[torch.Tensor | None, str]],
        prompt: str | None = None,
        mode: str = "full",
    ) -> dict[str, Any]:
        if mode not in ("full", "no_audio"):
            raise ValueError(f"mode must be 'full' or 'no_audio', got {mode!r}")

        user_prompt = (prompt or self.default_prompt).strip().lower()
        sep = self.tokenizer.sep_token

        # Build context segments + collect context audio clips when applicable
        ctx_segments: list[str] = []
        ctx_clip_list: list[torch.Tensor] = []
        ctx_mask_list: list[torch.Tensor] = []
        for ctx_audio, ctx_answer in context:
            ctx_answer = str(ctx_answer).strip()
            if mode == "full" and ctx_audio is not None:
                ctx_segments.append(
                    f"<audio>{user_prompt}{sep}{ctx_answer}<|endofchunk|>"
                )
                cc, cm = self._tensor_to_clips(ctx_audio)
                ctx_clip_list.append(cc)
                ctx_mask_list.append(cm)
            else:
                # text-only demonstration
                ctx_segments.append(
                    f"{user_prompt}{sep}{ctx_answer}<|endofchunk|>"
                )

        target_clips, target_mask = self._tensor_to_clips(audio)

        if ctx_clip_list:
            all_clips = torch.cat(ctx_clip_list + [target_clips], dim=0)
            all_mask = torch.cat(ctx_mask_list + [target_mask], dim=0)
        else:
            all_clips = target_clips
            all_mask = target_mask

        prompt_text = "".join(ctx_segments) + f"<audio>{user_prompt}{sep}"
        full_text = f"{prompt_text}{target_text.strip()}"

        full_ids, prompt_len = self._tokenize_with_boundary(prompt_text, full_text)
        return self._forward_score(all_clips, all_mask, full_ids, prompt_len)

    @torch.no_grad()
    def generate(self, audio: torch.Tensor, prompt: str) -> str:
        user_prompt = (prompt or self.default_prompt).strip().lower()
        sep = self.tokenizer.sep_token
        prompt_text = f"<audio>{user_prompt}{sep}"

        clips, mask = self._tensor_to_clips(audio)
        ids = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

        device = self._device
        audio_x = clips.unsqueeze(0).to(device, dtype=torch.float32)
        audio_x_mask = mask.unsqueeze(0).to(device)
        lang_x = ids.to(device)

        autocast_enabled = self._dtype in (torch.float16, torch.bfloat16)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=self._dtype, enabled=autocast_enabled,
        ):
            output = self.model.generate(
                audio_x=audio_x,
                audio_x_mask=audio_x_mask,
                lang_x=lang_x,
                eos_token_id=self._eos_token_id,
                max_new_tokens=128,
                do_sample=False,
            )[0]
        decoded = self.tokenizer.decode(output)
        return (
            decoded.split(sep)[-1]
            .replace(self.tokenizer.eos_token, "")
            .replace(self.tokenizer.pad_token, "")
            .replace("<|endofchunk|>", "")
            .strip()
        )
