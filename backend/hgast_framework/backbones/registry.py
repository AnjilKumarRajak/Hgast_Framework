"""
Deployment registry.
Only SeamlessM4T is included in the Hugging Face demo.
"""

from .seamless_m4t import SeamlessM4TBackbone
from ..hf_inference_adapter import HFInferenceBackbone

_REGISTRY = {
    "seamless_m4t": lambda: SeamlessM4TBackbone(),
    "hf_inference": lambda: HFInferenceBackbone(),
}

def get_backbone(key: str):
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown backbone '{key}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[key]()

def list_backbones():
    return list(_REGISTRY.keys())

