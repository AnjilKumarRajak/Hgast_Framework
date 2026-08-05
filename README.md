---
title: HGAST++ Framework (Option 3 HF Inference API)
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# 🎙️ HGAST++ Framework (Option 3: HF Inference API Hybrid Approach)

**HGAST++** (Hierarchical Gender-Aligned Speech Translation Framework) combines speaker & subject linguistic analysis, dual-controller gender routing, and LLM refinement to produce grammatically accurate gender-aligned Hindi translations.

## 🚀 Option 3: Serverless HF Inference API Architecture

This Static Space runs on **Hugging Face Serverless Infrastructure** with **zero CPU compute quota usage**:

1. **ASR (Speech-to-Text)**: Offloaded to `openai/whisper-small` / `whisper-large-v3` via HF Serverless Inference API.
2. **Speaker Gender Recognition**: Offloaded to `alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech` via HF Inference API.
3. **Translation Backbone**: Offloaded to `facebook/seamless-m4t-v2-large` via HF Inference API.
4. **LLM Gender Refiner**: Offloaded to `Qwen/Qwen2.5-7B-Instruct` / `zephyr-7b-beta` via HF Serverless Chat Completions API.
5. **Static Frontend**: Pure HTML5/CSS3/JS UI served statically via `index.html`.
