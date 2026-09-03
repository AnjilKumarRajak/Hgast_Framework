"""Whisper ASR adapter for speech transcription."""

import logging
import os
from typing import Optional, Union

log = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3"


class WhisperASR:
    """ASR wrapper for OpenAI Whisper models."""

    def __init__(self, model_id: str = DEFAULT_WHISPER_MODEL, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline
            dev = self.device or (0 if torch.cuda.is_available() else -1)
            log.info(f"[Whisper] Loading {self.model_id} on device={dev}...")
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=dev,
                chunk_length_s=30,
            )
            log.info("[Whisper] Loaded successfully.")
        except Exception as exc:
            log.warning(f"Whisper ASR load failed: {exc}")

    def transcribe(self, audio_input: Union[str, any], sampling_rate: int = 16000) -> str:
        self.load()
        if self._pipe is None:
            log.warning("Whisper pipeline unavailable. Returning empty transcription.")
            return ""
        try:
            result = self._pipe(audio_input, generate_kwargs={"task": "transcribe", "language": "en", "num_beams": 4})
            if isinstance(result, dict) and "text" in result:
                return result["text"].strip()
            return str(result).strip()
        except Exception as exc:
            log.warning(f"Whisper transcription failed: {exc}")
            return ""
