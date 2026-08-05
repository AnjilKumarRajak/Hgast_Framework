"""
backbones/seamless_m4t.py
=========================
Wraps Meta SeamlessM4T (v2) for EN->HI text translation, and optionally
speech input via its ASR/S2TT path. Swap `model_id` to point at your
fine-tuned checkpoint (SeamlessM4T-FT row in your table) without touching
anything else in the framework.
"""

import logging
import torch
from ..config import DEVICE
from .base import TranslationBackbone

log = logging.getLogger(__name__)


class SeamlessM4TBackbone(TranslationBackbone):
    name = "SeamlessM4T"

    def __init__(self, model_id: str = "facebook/seamless-m4t-v2-large"):
        # REPLACE HERE: point at your fine-tuned checkpoint dir for
        # the "SeamlessM4T-FT" row, e.g. "/path/to/checkpoints/seamless_ft"
        self.model_id = model_id
        self._processor = None
        self._model = None

    def load(self):
        if self._model is not None:
            return
        from transformers import AutoProcessor, SeamlessM4Tv2Model
        log.info(f"[{self.name}] loading {self.model_id} ...")
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = SeamlessM4Tv2Model.from_pretrained(self.model_id).to(DEVICE)
        self._model.eval()
        log.info(f"[{self.name}] loaded on {DEVICE}.")

    def unload(self):
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def translate_en_to_hi(self, text: str) -> str:
        self.load()
        inputs = self._processor(text=text, src_lang="eng", return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            out_tokens = self._model.generate(
                **inputs, tgt_lang="hin", generate_speech=False
            )
        
        if hasattr(out_tokens, "sequences"):
            seqs = out_tokens.sequences
        elif isinstance(out_tokens, tuple):
            seqs = out_tokens[0]
        else:
            seqs = out_tokens
            
        text_out = self._processor.decode(seqs[0].tolist(), skip_special_tokens=True)
        return text_out.strip()

    def transcribe(self, audio_array, sampling_rate: int):
        self.load()
        inputs = self._processor(
            audios=audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).to(DEVICE)
        with torch.inference_mode():
            out_tokens = self._model.generate(
                **inputs, tgt_lang="eng", generate_speech=False
            )
        return self._processor.decode(out_tokens[0].tolist(), skip_special_tokens=True)
