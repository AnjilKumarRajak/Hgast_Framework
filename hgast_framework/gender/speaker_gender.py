
import logging
from ..config import DEVICE

log = logging.getLogger(__name__)

_gender_model = None
_gender_feature_extractor = None
_FAILED = object()


def _load():
    global _gender_model, _gender_feature_extractor
    if _gender_model is not None:
        return
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        model_id = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
        _gender_feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        _gender_model = AutoModelForAudioClassification.from_pretrained(model_id).to(DEVICE)
        _gender_model.eval()
        log.info("Speaker gender model loaded.")
    except Exception as exc:
        log.warning(f"Speaker gender model load failed: {exc}")
        _gender_model = _FAILED


def detect_gender_from_array(audio_array, sampling_rate: int) -> tuple:
    """Returns (gender_int, confidence, backend_name). -1 = unknown."""
    _load()
    if _gender_model is _FAILED or _gender_model is None:
        return (-1, 0.0, "unavailable")
    try:
        import torch
        import librosa

        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)

        inputs = _gender_feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = _gender_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)

        pred_id = int(probs.argmax(dim=-1))
        score = float(probs.max().item())
        label = _gender_model.config.id2label[pred_id].lower()

        # Section 2.2: calibrate conservative confidence threshold tau_ac = 0.65
        # Triggering abstention (-1) rather than forced assignment when acoustic evidence is weak
        if score < 0.65:
            return (-1, score, "wav2vec2_gender_abstain")

        if "female" in label:
            return (1, score, "wav2vec2_gender")
        if "male" in label:
            return (0, score, "wav2vec2_gender")
        return (-1, 0.0, "unknown_label")
    except Exception as exc:
        log.warning(f"Gender detection failed: {exc}")
        return (-1, 0.0, "failed")
