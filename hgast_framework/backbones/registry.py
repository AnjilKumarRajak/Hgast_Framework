

"""Model registry for translation backbones."""

from .indictrans2 import IndicTrans2Backbone
from .seamless_m4t import SeamlessM4TBackbone
from .generic_hf import GenericHFBackbone

_REGISTRY = {
    "indictrans2": lambda: IndicTrans2Backbone(),
    "indictrans2_ft": lambda: IndicTrans2Backbone(
        model_id="checkpoints/indictrans2_ft"
    ),
    "seamless_m4t": lambda: SeamlessM4TBackbone(),
    "seamless_m4t_ft": lambda: SeamlessM4TBackbone(
        model_id="checkpoints/seamless_m4t_ft"
    ),
}


def get_backbone(key: str):
   
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
