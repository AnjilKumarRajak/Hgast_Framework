

import logging
import torch
from ..config import DEVICE
from .base import TranslationBackbone

log = logging.getLogger(__name__)


class GenericHFBackbone(TranslationBackbone):
    def __init__(
        self,
        model_id: str,
        name: str = None,
        prompt_template: str = "{text}",
        is_causal: bool = False,
        max_new_tokens: int = 256,
    ):
        self.model_id = model_id
        self.name = name or model_id
        self.prompt_template = prompt_template
        self.is_causal = is_causal
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def load(self):
        if self._model is not None:
            return
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
        )
        log.info(f"[{self.name}] loading {self.model_id} ...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        model_cls = AutoModelForCausalLM if self.is_causal else AutoModelForSeq2SeqLM
        self._model = model_cls.from_pretrained(
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
        prompt = self.prompt_template.format(text=text)
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
        inputs.pop("token_type_ids", None)

        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
            )
        if self.is_causal:
            out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        result = self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
        return result.strip()
