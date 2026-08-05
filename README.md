---
title: HGAST Framework
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
header: mini
short_description: Hierarchical Gender Arbitration for Speech Translation
---

# 🎙️ HGAST Framework

**HGAST Framework** (Hierarchical Gender Arbitration for Speech Translation) combines speaker & subject linguistic analysis, dual-controller gender routing, and morphological refinement to produce grammatically accurate gender-aligned Hindi translations.

## 🚀 Dynamic Speech Translation Pipeline

1. **🎤 Speech ASR**: Real-time Whisper ASR speech-to-text.
2. **👤 Speaker Gender Detection**: Wav2Vec2 acoustic speaker gender recognition.
3. **🌐 Baseline Output**: Meta SeamlessM4T-v2-Large neural translation.
4. **🧠 HGAST Output**: Dual controller gender-faithful morphological correction.
