"""
hf_inference_adapter.py
=======================
Hugging Face Inference API Integration for Option 3 (Hybrid Approach).
Offloads heavy model math (ASR, Wav2Vec2 gender classification, SeamlessM4T/LLM translation & refinement)
to Hugging Face's free serverless infrastructure.
"""

import os
import json
import logging
import requests
import tempfile
from typing import Optional, Tuple, Dict, Any
from huggingface_hub import InferenceClient
from .backbones.base import TranslationBackbone

log = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "openai/whisper-small"
DEFAULT_GENDER_MODEL = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
DEFAULT_TRANSLATION_MODEL = "facebook/seamless-m4t-v2-large"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

class HFInferenceManager:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("HF_TOKEN")
        self._init_client()

    def set_api_key(self, api_key: str):
        if api_key and api_key.strip():
            self.api_key = api_key.strip()
        self._init_client()

    def _init_client(self):
        try:
            if self.api_key:
                self.client = InferenceClient(api_key=self.api_key)
            else:
                self.client = InferenceClient()
        except Exception as e:
            log.warning(f"Could not initialize InferenceClient: {e}")
            self.client = None

    def transcribe_audio(self, audio_filepath: str) -> str:
        """Call HF Serverless ASR API for Speech-to-Text."""
        if not audio_filepath or not os.path.exists(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

        # Method 1: Try huggingface_hub InferenceClient
        if self.client:
            try:
                with open(audio_filepath, "rb") as f:
                    audio_bytes = f.read()
                res = self.client.automatic_speech_recognition(audio_bytes, model=DEFAULT_ASR_MODEL)
                if isinstance(res, dict) and "text" in res:
                    return res["text"].strip()
                if hasattr(res, "text"):
                    return res.text.strip()
                if isinstance(res, str):
                    return res.strip()
            except Exception as e:
                log.warning(f"InferenceClient ASR failed ({e}). Attempting direct HTTP request...")

        # Method 2: Direct HTTP POST request to HF Inference API
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"https://api-inference.huggingface.co/models/{DEFAULT_ASR_MODEL}"

        with open(audio_filepath, "rb") as f:
            data = f.read()

        resp = requests.post(url, headers=headers, data=data, timeout=30)
        if resp.status_code == 200:
            result = resp.json()
            if isinstance(result, dict) and "text" in result:
                return result["text"].strip()

        raise RuntimeError(f"HF Inference ASR failed (HTTP {resp.status_code}): {resp.text}")

    def detect_speaker_gender(self, audio_filepath: str) -> Tuple[str, float]:
        """Call HF Serverless Audio Classification API for Wav2Vec2 gender recognition."""
        if not audio_filepath or not os.path.exists(audio_filepath):
            return "female", 0.5

        if self.client:
            try:
                with open(audio_filepath, "rb") as f:
                    audio_bytes = f.read()
                preds = self.client.audio_classification(audio_bytes, model=DEFAULT_GENDER_MODEL)
                if isinstance(preds, list) and len(preds) > 0:
                    top = preds[0]
                    label = top.get("label", "female").lower()
                    score = top.get("score", 0.8)
                    return label, score
            except Exception as e:
                log.warning(f"InferenceClient Gender Classifier failed ({e}). Trying HTTP...")

        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            url = f"https://api-inference.huggingface.co/models/{DEFAULT_GENDER_MODEL}"
            with open(audio_filepath, "rb") as f:
                data = f.read()
            resp = requests.post(url, headers=headers, data=data, timeout=30)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    label = result[0].get("label", "female").lower()
                    score = result[0].get("score", 0.8)
                    return label, score
        except Exception as e:
            log.warning(f"HTTP Gender API failed: {e}")

        return "female", 0.5

    def translate_en_to_hi(self, text: str) -> str:
        """Call HF Serverless API / LLM for EN->HI translation."""
        if not text or not text.strip():
            return ""

        # Attempt 1: Chat model on HF Inference API
        if self.client:
            try:
                prompt = (
                    "Translate the following English sentence to Hindi accurately. "
                    "Return ONLY the Hindi translation string, no explanations, no quotes.\n"
                    f"English: {text}\nHindi:"
                )
                res = self.client.chat.completions.create(
                    model=DEFAULT_LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.1
                )
                out = res.choices[0].message.content.strip()
                out = out.replace('"', '').replace("'", "").strip()
                if out:
                    return out
            except Exception as e:
                log.warning(f"HF Chat Translation failed ({e}). Trying translation endpoint...")

        # Attempt 2: Direct HTTP request to translation model endpoint
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            url = f"https://api-inference.huggingface.co/models/{DEFAULT_TRANSLATION_MODEL}"
            resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=30)
            if resp.status_code == 200:
                res_json = resp.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    return res_json[0].get("translation_text", "").strip()
        except Exception as e:
            log.warning(f"HTTP translation request failed: {e}")

        return text

    def llm_refine_chat_fn(self, prompt: str) -> str:
        """chat_fn adapter for LLMGenderRefiner using HF Serverless Inference API."""
        if self.client:
            try:
                res = self.client.chat.completions.create(
                    model=DEFAULT_LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                return res.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"HF LLM Refine chat failed ({e}). Trying fallback model Zephyr...")

            try:
                res = self.client.chat.completions.create(
                    model="HuggingFaceH4/zephyr-7b-beta",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.1
                )
                return res.choices[0].message.content.strip()
            except Exception as e:
                log.warning(f"Secondary model chat failed: {e}")

        # HTTP Fallback
        try:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            url = f"https://api-inference.huggingface.co/models/{DEFAULT_LLM_MODEL}"
            resp = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=30)
            if resp.status_code == 200:
                res_json = resp.json()
                if isinstance(res_json, list) and len(res_json) > 0:
                    return res_json[0].get("generated_text", "").strip()
        except Exception as e:
            log.error(f"HTTP LLM refine failed: {e}")

        raise RuntimeError("LLM Inference API unavailable")


class HFInferenceBackbone(TranslationBackbone):
    """Translation Backbone that delegates translation math to Hugging Face Inference API."""
    name = "HF_Inference_API_SeamlessM4T"

    def __init__(self, manager: Optional[HFInferenceManager] = None):
        self.manager = manager or HFInferenceManager()

    def translate_en_to_hi(self, text: str) -> str:
        return self.manager.translate_en_to_hi(text)
