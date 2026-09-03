# HGAST Framework: Hierarchical Gender Arbitration for Speech Translation

A framework for gender-agreement correction in speech and text translation for Indo-Aryan languages (Hindi, Marathi, Gujarati). 

The framework operates on top of any translation backbone (e.g., IndicTrans2, SeamlessM4T) without requiring retraining. It orchestrates linguistic analysis, person-conditioned dual-control routing, rule-based morphological inflection, and LLM-based fluency refinement with safety verification gates.

## Compared Backbones

As evaluated in the paper, HGAST supports both cascaded and end-to-end speech translation architectures:

1. **IndicConformer + IndicTrans2**: A specialized cascaded pipeline combining `ai4bharat/indicconformer_stt_en_hybrid_ctc_rnnt_large` (ASR) and `ai4bharat/indictrans2-en-indic-1B` (MT).
2. **SeamlessM4T-v2-Large**: A direct end-to-end multilingual speech-to-text model (`facebook/seamless-m4t-v2-large`).
3. **Whisper + NLLB-200**: A general-purpose cascaded pipeline combining `openai/whisper-large-v3` (ASR) and `facebook/nllb-200-distilled-600M` (MT).

## Architecture

1. **Linguistic Analysis**: Extracts subject identity, coreference chains, grammatical person (1st, 2nd, 3rd), and entity gender cues.
2. **Dual-Control Gender Routing**: 
   - *1st Person*: Authoritative speaker gender (e.g., detected from audio or user input).
   - *2nd Person*: Politeness register and speaker gender fallback.
   - *3rd Person*: Grammatical subject gender (with confidence gating) or speaker fallback.
3. **Rule-Based Morphology Correction**: Deterministic surface inflection mapping for verbs, participles, and adjectives across target genders (Hindi, Marathi, Gujarati).
4. **LLM Fluency Refinement**: Model-agnostic rewrite stage with safety validation gates to prevent lexical hallucination, morphological corruption, or punctuation distortion.

## Project Structure

```
.
├── hgast_framework/
│   ├── config.py                     # Configuration constants and thresholds
│   ├── pipeline.py                   # Main orchestrator (HGASTFramework)
│   ├── run_experiment.py             # End-to-end execution and evaluation script
│   ├── backbones/                    # Translation backbone adapters
│   │   ├── base.py                   # TranslationBackbone interface
│   │   ├── indictrans2.py            # AI4Bharat IndicTrans2 adapter
│   │   ├── seamless_m4t.py           # Meta SeamlessM4T (v2) adapter
│   │   ├── generic_hf.py             # Generic HuggingFace seq2seq/causal adapter
│   │   └── registry.py               # Model registry
│   ├── gender/                       # Gender analysis and correction modules
│   │   ├── linguistic_analysis.py    # Subject and coreference parsing
│   │   ├── dual_control.py           # Person-conditioned routing
│   │   ├── morphology_rules.py       # Hindi morphology tables and token scoring
│   │   ├── morphology_rules_mr.py    # Marathi morphology rules
│   │   ├── morphology_rules_gu.py    # Gujarati morphology rules
│   │   ├── marathi_prompt_examples.py# Marathi few-shot prompt definitions
│   │   ├── gujarati_prompt_examples.py# Gujarati few-shot prompt definitions
│   │   ├── speaker_gender.py         # Audio-based wav2vec2 gender detector
│   │   ├── llm_refine.py             # LLM refiner with safety validation gates
│   │   └── qwen_adapter.py           # Qwen adapter for refinement
│   └── evaluation/                   # Evaluation suite
│       ├── metrics.py                # BLEU, chrF++, COMET, BERTScore, WER, TER
│       ├── gender_metrics.py         # Male, female, and macro gender accuracy
│       ├── significance.py           # Bootstrap CI/compare, McNemar's test
│       ├── error_analysis.py         # Failure taxonomy and classification
│       └── ablation.py               # Progressive ablation evaluation
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Quick Start

Run an end-to-end evaluation with the default pipeline:

```bash
python -m hgast_framework.run_experiment
```

Or use the framework in Python:

```python
from hgast_framework.backbones.registry import get_backbone
from hgast_framework.gender.llm_refine import LLMGenderRefiner
from hgast_framework.pipeline import HGASTFramework

# Initialize backbone and refiner
backbone = get_backbone("indictrans2")
refiner = LLMGenderRefiner(chat_fn=None)  # or provide a custom chat_fn callable

framework = HGASTFramework(backbone=backbone, llm_refiner=refiner)

result = framework.translate(
    en_text="I am going home.",
    speaker_gender=1,  # 0=male, 1=female, -1=unknown
    speaker_confidence=0.92,
)

print("Translated Hindi:", result.hindi)
```

## Evaluation Suite

The framework includes comprehensive evaluation tools:
- **Translation Quality**: BLEU, chrF++, METEOR, BERTScore, COMET, WER, TER.
- **Gender Accuracy**: Male, female, and macro-average accuracy based on morphological token validation.
- **Statistical Significance**: Paired McNemar's test on binary agreement decisions and bootstrap resampling for MT metrics.
- **Error Analysis**: Automated diagnostic bucketing (coreference error, subject parsing error, rule coverage gap, threshold miss, etc.).
- **Ablation Studies**: Component-wise ablations evaluating the incremental contribution of each module.
