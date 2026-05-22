"""English → Chinese → English round-trip via NLLB-200-3.3B."""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_EN = "eng_Latn"
NLLB_ZH = "zho_Hans"
DEFAULT_MODEL = "facebook/nllb-200-3.3B"


class NllbRoundTripTranslator:
    """Batched en→zh→en back-translation with per-string deduplication."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cuda",
        batch_size: int = 32,
        max_new_tokens: int = 128,
        num_beams: int = 1,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for NLLB but no GPU is available.")

        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model = self.model.to(device)
        self.model.eval()

        self._cache: dict[str, str] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _translate_batch(self, texts: list[str], src_lang: str, tgt_lang: str) -> list[str]:
        if not texts:
            return []

        self.tokenizer.src_lang = src_lang
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_lang)

        outputs: list[str] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
            outputs.extend(
                self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        return outputs

    def round_trip(self, texts: list[str]) -> list[str]:
        """Back-translate each string independently (en→zh→en)."""
        if not texts:
            return []

        results: list[str | None] = [None] * len(texts)
        pending_idx: list[int] = []
        pending_text: list[str] = []

        for i, raw in enumerate(texts):
            s = str(raw).strip()
            if not s:
                results[i] = ""
                continue
            if s in self._cache:
                results[i] = self._cache[s]
                continue
            pending_idx.append(i)
            pending_text.append(s)

        if pending_text:
            zh = self._translate_batch(pending_text, NLLB_EN, NLLB_ZH)
            en = self._translate_batch(zh, NLLB_ZH, NLLB_EN)
            for i, src, back in zip(pending_idx, pending_text, en):
                back = back.strip()
                results[i] = back
                self._cache[src] = back

        return [r if r is not None else "" for r in results]
