"""Model registry for translation and speech translation backbones."""

from .indictrans2 import IndicTrans2Backbone
from .seamless_m4t import SeamlessM4TBackbone
from .generic_hf import GenericHFBackbone
from .indic_conformer import IndicConformerASR
from .whisper_asr import WhisperASR
from .cascaded import CascadedBackbone

_REGISTRY = {
    # 1. IndicConformer + IndicTrans2 (Specialized cascaded pipeline)
    "indicconformer_indictrans2": lambda: CascadedBackbone(
        name="IndicConformer + IndicTrans2",
        asr_module=IndicConformerASR(),
        mt_backbone=IndicTrans2Backbone(),
    ),

    # 2. SeamlessM4T-v2-Large (End-to-end direct speech-to-text model)
    "seamless_m4t": lambda: SeamlessM4TBackbone(),
    "seamless_m4t_ft": lambda: SeamlessM4TBackbone(
        model_id="checkpoints/seamless_m4t_ft"
    ),

    # 3. Whisper + NLLB-200 (General-purpose cascaded pipeline)
    "whisper_nllb": lambda: CascadedBackbone(
        name="Whisper + NLLB-200",
        asr_module=WhisperASR(),
        mt_backbone=GenericHFBackbone(
            model_id="facebook/nllb-200-distilled-600M",
            name="NLLB-200",
            prompt_template="{text}",
        ),
    ),

    # Text-only MT backbones
    "indictrans2": lambda: IndicTrans2Backbone(),
    "indictrans2_ft": lambda: IndicTrans2Backbone(
        model_id="checkpoints/indictrans2_ft"
    ),

    # Standalone ASR modules
    "indic_conformer": lambda: IndicConformerASR(),
    "whisper": lambda: WhisperASR(),
}


def get_backbone(key: str):
    """
    Returns an instantiated backbone. Accepts either a registered key,
    or "generic:<hf_model_id>" for any HF model not yet given its own class.
    """
    if key.startswith("generic:"):
        model_id = key.split("generic:", 1)[1]
        return GenericHFBackbone(model_id=model_id, name=model_id)

    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone key '{key}'. "
            f"Available: {list(_REGISTRY.keys())} or 'generic:<hf_model_id>'"
        )
    return _REGISTRY[key]()


def list_backbones():
    return list(_REGISTRY.keys())
