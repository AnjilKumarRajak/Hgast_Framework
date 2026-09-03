"""Cascaded Speech-to-Text translation backbones (ASR + MT)."""

import logging
from typing import Optional, Union
from .base import TranslationBackbone

log = logging.getLogger(__name__)


class CascadedBackbone(TranslationBackbone):
    """
    Combines an upstream ASR model (e.g., IndicConformer or Whisper)
    with a downstream translation model (e.g., IndicTrans2 or NLLB-200).
    """

    def __init__(
        self,
        name: str,
        asr_module,
        mt_backbone: TranslationBackbone,
    ):
        self.name = name
        self.asr = asr_module
        self.mt = mt_backbone

    def load(self):
        if hasattr(self.asr, "load"):
            self.asr.load()
        if hasattr(self.mt, "load"):
            self.mt.load()

    def unload(self):
        if hasattr(self.asr, "unload"):
            self.asr.unload()
        if hasattr(self.mt, "unload"):
            self.mt.unload()

    def transcribe(self, audio_array, sampling_rate: int = 16000) -> str:
        """Transcribe source speech audio to English text."""
        if hasattr(self.asr, "transcribe"):
            return self.asr.transcribe(audio_array, sampling_rate=sampling_rate)
        raise NotImplementedError(f"{self.name} ASR does not implement transcribe().")

    def translate_en_to_hi(self, text: str) -> str:
        """Translate English text to Hindi."""
        return self.mt.translate_en_to_hi(text)

    def translate_speech(self, audio_array, sampling_rate: int = 16000) -> str:
        """End-to-end cascaded speech translation: audio -> English -> Hindi."""
        en_text = self.transcribe(audio_array, sampling_rate=sampling_rate)
        if not en_text:
            return ""
        return self.translate_en_to_hi(en_text)
