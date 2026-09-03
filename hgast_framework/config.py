
import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sentence quality prefilter thresholds
MIN_SENTENCE_WORDS = 4
MAX_SENTENCE_WORDS = 60

LENGTH_RATIO_MIN = 0.55

FUSION_EARLY_LAYERS = (0, 4)
FUSION_MIDDLE_LAYERS = (4, 8)
FUSION_LATE_LAYERS = (8, 12)

FUSION_MODE_SEMANTIC_ONLY = "semantic_only"
FUSION_MODE_SPEAKER_STYLE = "speaker_style"
FUSION_MODE_POLITENESS_STYLE = "politeness_style"
FUSION_MODE_SUBJECT_STYLE = "subject_style"
FUSION_MODE_MORPH_REFINE = "morph_refine"

# Trigger classifier cache path
TRIGGER_CLF_PATH = "trigger_clf.joblib"

_CUDA_CACHE_EVERY = 50

OLLAMA_AVAILABLE = False   # flipped to True by llm_refine.configure_llm()
SEMANTIC_MODEL = None      # set by llm_refine.configure_llm()
