"""IndicConformer ASR adapter for speech-to-text transcription."""

import logging
import os
from typing import Optional, Union

log = logging.getLogger(__name__)

INDIC_CONFORMER_EN = "ai4bharat/indicconformer_stt_en_hybrid_ctc_rnnt_large"


class IndicConformerASR:
    """ASR wrapper for AI4Bharat IndicConformer models."""

    def __init__(self, model_id: str = INDIC_CONFORMER_EN, device: Optional[str] = None):
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._backend = None

    def load(self):
        if self._model is not None:
            return

        # Attempt 1: NeMo ASR model
        try:
            import nemo.collections.asr as nemo_asr
            import torch
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"[IndicConformer] Loading {self.model_id} via NeMo on {dev}...")
            try:
                self._model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(
                    model_name=self.model_id
                ).to(dev)
            except Exception:
                self._model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
                    model_name=self.model_id
                ).to(dev)
            self._model.eval()
            self._backend = "nemo"
            log.info("[IndicConformer] Loaded successfully via NeMo.")
            return
        except ImportError:
            log.debug("NeMo not installed, trying HuggingFace transformers...")
        except Exception as exc:
            log.warning(f"NeMo load failed: {exc}, trying HuggingFace...")

        # Attempt 2: Transformers pipeline / AutoModelForCTC
        try:
            from transformers import AutoModelForCTC, AutoProcessor
            import torch
            dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            log.info(f"[IndicConformer] Loading {self.model_id} via transformers on {dev}...")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForCTC.from_pretrained(self.model_id).to(dev)
            self._model.eval()
            self._backend = "transformers"
            log.info("[IndicConformer] Loaded successfully via transformers.")
            return
        except Exception as exc:
            log.warning(f"HuggingFace IndicConformer load failed: {exc}")

        log.warning(
            f"IndicConformer could not be loaded. Please install nemo_toolkit['asr'] "
            f"or ensure {self.model_id} is accessible."
        )

    def transcribe(self, audio_input: Union[str, any], sampling_rate: int = 16000) -> str:
        """Transcribe audio array or file path to English text."""
        self.load()
        if self._model is None:
            log.warning("IndicConformer model is not loaded. Returning empty transcription.")
            return ""

        # NeMo inference
        if self._backend == "nemo":
            try:
                import tempfile
                import soundfile as sf
                if isinstance(audio_input, str) and os.path.exists(audio_input):
                    audio_path = audio_input
                    tmp_file = None
                else:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    sf.write(tmp.name, audio_input, sampling_rate)
                    audio_path = tmp.name
                    tmp_file = tmp.name

                transcriptions = self._model.transcribe([audio_path])
                if tmp_file and os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass

                if transcriptions and isinstance(transcriptions, list):
                    # Handle hybrid ctc/rnnt output format
                    first = transcriptions[0]
                    return first[0] if isinstance(first, tuple) else str(first).strip()
                return ""
            except Exception as exc:
                log.warning(f"NeMo IndicConformer transcription failed: {exc}")
                return ""

        # Transformers inference
        if self._backend == "transformers":
            try:
                import torch
                import librosa
                if isinstance(audio_input, str):
                    audio_array, sr = librosa.load(audio_input, sr=16000)
                else:
                    audio_array = audio_input
                    if sampling_rate != 16000:
                        audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

                inputs = self._processor(audio_array, sampling_rate=16000, return_tensors="pt")
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                with torch.inference_mode():
                    logits = self._model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self._processor.batch_decode(predicted_ids)[0]
                return transcription.strip()
            except Exception as exc:
                log.warning(f"Transformers IndicConformer transcription failed: {exc}")
                return ""

        return ""
