# 🎙️ HGAST Framework
### Hierarchical Gender Arbitration for Speech Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnjilKumarRajak/Hgast_Framework/blob/main/HGAST_Colab_Launcher.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**HGAST** (Hierarchical Gender Arbitration for Speech Translation) is a framework that combines speaker & subject linguistic analysis, dual-controller gender routing, and morphological alignment to produce grammatically accurate gender-faithful Hindi translations directly from speech audio.

---

## ⚡ 1-Click Live Demo on Google Colab

Click the badge above or below to launch the **HGAST Framework** directly in Google Colab (Free GPU/CPU):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnjilKumarRajak/Hgast_Framework/blob/main/HGAST_Colab_Launcher.ipynb)

Running the notebook generates a live public Gradio URL (`https://xxxx.gradio.live`) that anyone can open on mobile or desktop to test real-time microphone speech translation!

---

## 🚀 Pipeline Architecture

1. **🎤 Speech ASR**: Real-time `openai/whisper-small` / `whisper-large-v3` speech-to-text.
2. **👤 Speaker Gender Recognition**: `wav2vec2-large-xlsr` acoustic speaker gender detection.
3. **🌐 Baseline Translation**: Meta `SeamlessM4T-v2-Large` neural translation.
4. **🧠 HGAST Gender Arbitration**: Dual controller gender-faithful morphological alignment.

---

## 💻 Local Quickstart

```bash
# Clone the repository
git clone https://github.com/AnjilKumarRajak/Hgast_Framework.git
cd Hgast_Framework

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio web application
python app.py
```
