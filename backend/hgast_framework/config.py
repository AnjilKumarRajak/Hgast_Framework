"""
config.py
=========
Single source of truth for every constant/threshold used across the framework.
The original script referenced many globals (DEVICE, LENGTH_RATIO_MIN,
FUSION_EARLY_LAYERS, etc.) that were never defined in-file. They are all
defined here, once, so nothing silently NameErrors at runtime.
"""

import torch

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Sentence quality prefilter thresholds
# ---------------------------------------------------------------------------
MIN_SENTENCE_WORDS = 4
MAX_SENTENCE_WORDS = 60

# ---------------------------------------------------------------------------
# Length-ratio sanity check (Hindi words / English words)
# ---------------------------------------------------------------------------
LENGTH_RATIO_MIN = 0.55

# ---------------------------------------------------------------------------
# FiLM fusion-stage layer ranges (used only if you actually condition a
# decoder with FiLM layers; kept as config rather than magic numbers).
# ---------------------------------------------------------------------------
FUSION_EARLY_LAYERS = (0, 4)
FUSION_MIDDLE_LAYERS = (4, 8)
FUSION_LATE_LAYERS = (8, 12)

FUSION_MODE_SEMANTIC_ONLY = "semantic_only"
FUSION_MODE_SPEAKER_STYLE = "speaker_style"
FUSION_MODE_POLITENESS_STYLE = "politeness_style"
FUSION_MODE_SUBJECT_STYLE = "subject_style"
FUSION_MODE_MORPH_REFINE = "morph_refine"

# ---------------------------------------------------------------------------
# Trigger classifier cache path
# ---------------------------------------------------------------------------
TRIGGER_CLF_PATH = "trigger_clf.joblib"

# ---------------------------------------------------------------------------
# CUDA cache flush frequency
# ---------------------------------------------------------------------------
_CUDA_CACHE_EVERY = 50

# ---------------------------------------------------------------------------
# LLM backend switch (rule-based-only fallback if no LLM configured)
# ---------------------------------------------------------------------------
OLLAMA_AVAILABLE = False   # flipped to True by llm_refine.configure_llm()
SEMANTIC_MODEL = None      # set by llm_refine.configure_llm()
