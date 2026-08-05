---
title: HGAST++ Framework (HF Inference API - Option 3)
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
header: mini
short_description: Hybrid Gender-Aware Speech Translation using HF Serverless Inference API
---

# 🎙️ HGAST++ Framework (Option 3: HF Inference API Hybrid Approach)

**HGAST++** (Hierarchical Gender-Aligned Speech Translation Framework) combines speaker & subject linguistic analysis, dual-controller gender routing, and LLM refinement to produce grammatically accurate gender-aligned Hindi translations.

## 🚀 Option 3: Serverless HF Inference API Architecture

This Space runs on **Hugging Face's Free CPU Infrastructure** by offloading heavy ML model math to **Hugging Face's Serverless Inference API**:

1. **ASR (Speech-to-Text)**: Offloaded to `openai/whisper-small` / `whisper-large-v3` via HF Inference API.
2. **Speaker Gender Recognition**: Offloaded to `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech` via HF Inference API.
3. **Translation Backbone**: Offloaded to `facebook/seamless-m4t-v2-large` via HF Inference API.
4. **Linguistic Dual Controller & Morphology Engine**: Fast, lightweight local Python execution.
5. **LLM Gender Refiner**: Offloaded to `Qwen/Qwen2.5-7B-Instruct` / `zephyr-7b-beta` via HF Serverless Chat Completions API.
6. **TTS Voice Synthesis**: `Edge-TTS` neural voice generation (`hi-IN-SwaraNeural` / `hi-IN-MadhurNeural`).

---

## 🎨 Features & Static Space Integration

- **Zero GPU / Paid Hardware Requirement**: Deploys on standard free HF Spaces.
- **Fast Execution**: Zero model warm-up wait time.
- **Interactive Gradio UI & Static Space Template**: Accessible via both Gradio App and Static Space POST requests.
- **HF Token Support**: Optional `HF_TOKEN` configuration for higher rate limits.
