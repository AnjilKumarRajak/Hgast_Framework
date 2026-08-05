"""
backbones/base.py
==================
Every SOTA model (IndicSeamless, SeamlessM4T, IndicTrans2, Whisper+MT
cascade, NLLB, etc.) is wrapped behind this ONE interface.

To add a new model: subclass `TranslationBackbone`, implement
`translate_en_to_hi()` (and `transcribe()` if it also does ASR), and
register it in `backbones/registry.py`. Nothing else in the framework
needs to change — the gender-correction pipeline calls only these
methods, never a model-specific API directly.
"""

from abc import ABC, abstractmethod
from typing import Optional


class TranslationBackbone(ABC):
    """Common interface every backbone must implement."""

    #: human-readable name used in result tables / logs
    name: str = "unnamed_backbone"

    @abstractmethod
    def translate_en_to_hi(self, text: str) -> str:
        """Raw EN->HI translation, no gender correction applied."""
        raise NotImplementedError

    def transcribe(self, audio_array, sampling_rate: int) -> Optional[str]:
        """
        Optional ASR step (audio -> English text). Only needed for
        cascaded S2ST backbones (e.g. Whisper -> MT). Text-only
        backbones can leave this unimplemented.
        """
        raise NotImplementedError(
            f"{self.name} does not implement transcribe(); "
            f"pass English text directly instead."
        )

    def load(self):
        """Optional lazy-load hook. Called once before first use."""
        pass

    def unload(self):
        """Optional cleanup hook (free GPU memory between backbone swaps)."""
        pass

    def __repr__(self):
        return f"<TranslationBackbone: {self.name}>"
