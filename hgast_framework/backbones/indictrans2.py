

import re
import logging
import torch
from ..config import DEVICE
from .base import TranslationBackbone

log = logging.getLogger(__name__)


def _clean_hindi(text: str) -> str:
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ ]+([?.!,।])", r"\1", text)
    return text.strip()


class IndicTrans2Backbone(TranslationBackbone):
    name = "IndicTrans2"

    def __init__(self, model_id: str = "ai4bharat/indictrans2-en-indic-1B"):
        self.model_id = model_id
        self._tokenizer = None
        self._model = None

    def load(self):
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        log.info(f"[{self.name}] loading {self.model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(DEVICE)
        self._model.eval()
        log.info(f"[{self.name}] loaded on {DEVICE}.")

    def unload(self):
        self._model = None
        self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def translate_en_to_hi(self, text: str) -> str:
        self.load()
        tagged = f"eng_Latn hin_Deva {text}"
        inputs = self._tokenizer(
            tagged, return_tensors="pt", truncation=True, max_length=256, padding=True
        ).to(DEVICE)
        inputs.pop("token_type_ids", None)

        with torch.inference_mode():
            token_ids = self._model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                length_penalty=0.9,
                early_stopping=True,
            )
        result = _clean_hindi(
            self._tokenizer.batch_decode(token_ids, skip_special_tokens=True)[0]
        )
        return result if len(result.split()) >= 2 else ""
