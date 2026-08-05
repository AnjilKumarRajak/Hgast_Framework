# 🎙️ HGAST Framework: Hybrid Gender-Aware Speech Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnjilKumarRajak/Hgast_Framework/blob/main/HGAST_Colab_Launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**HGAST (Hierarchical Gender Arbitration for Speech Translation)** is a post-hoc arbitration framework designed to correct grammatical gender bias in English-to-Indic speech translation without requiring retraining of the underlying backbone models. 

It combines speaker & subject linguistic analysis, dual-controller gender routing, and morphological alignment to produce grammatically accurate, gender-faithful translations directly from speech audio.

---

## ⚡ 1-Click Live Demo on Google Colab

Click the Colab badge at the top of this page to launch the **HGAST Framework** directly in Google Colab (Free GPU/CPU). Running the notebook generates a live public Gradio URL (`https://xxxx.gradio.live`) that anyone can open on mobile or desktop to test real-time microphone speech translation!

---

## 🚀 Pipeline Architecture

1. **🎤 Speech ASR**: Real-time `openai/whisper-small` / `whisper-large-v3` speech-to-text.
2. **👤 Speaker Gender Recognition**: `wav2vec2-large-xlsr` acoustic speaker gender detection.
3. **🌐 Baseline Translation**: Supports Meta `SeamlessM4T-v2-Large`, `IndicTrans2`, and `NLLB`.
4. **🧠 HGAST Gender Arbitration**: Dual controller gender-faithful morphological alignment.

---

## 💻 Local Quickstart (Frontend Interface)

To easily test the translation pipeline locally, you can run the web-based frontend interface:

```bash
# Clone the repository
git clone https://github.com/AnjilKumarRajak/Hgast_Framework.git
cd Hgast_Framework

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio frontend interface
cd frontend
python app.py
```

---

## 🛠️ Programmatic Usage

To integrate HGAST into your own evaluation pipeline programmatically:

```python
from backbones.registry import get_backbone
from gender.llm_refine import LLMGenderRefiner
from pipeline import HGASTFramework

# Initialize components
backbone = get_backbone("indictrans2")
refiner = LLMGenderRefiner(chat_fn=my_llm_chat_fn)
framework = HGASTFramework(backbone=backbone, llm_refiner=refiner)

# Translate and apply gender correction
result = framework.translate(
    en_text="I am going home.",
    speaker_gender=1,            # 1 = female, 0 = male, -1 = unknown
    speaker_confidence=0.92
)

print(result.hindi)
```
