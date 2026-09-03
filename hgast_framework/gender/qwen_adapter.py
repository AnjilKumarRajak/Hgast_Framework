
import logging

log = logging.getLogger(__name__)

_qwen_model = None
_qwen_tokenizer = None


def _load_qwen():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is not None:
        return _qwen_model, _qwen_tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            log.info("CUDA not available. Qwen 4-bit generative refiner will be bypassed.")
            return None, None

        log.info("Loading Qwen2.5-7B-Instruct in 4-bit...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
        model_id = "Qwen/Qwen2.5-7B-Instruct"
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quantization_config, device_map={"": "cuda"}
        )
        log.info("Qwen loaded successfully.")
        return _qwen_model, _qwen_tokenizer
    except Exception as exc:
        log.warning(f"Qwen refiner unavailable ({exc}). Generative refiner bypassed.")
        return None, None


def qwen_chat_fn(prompt: str) -> str:
    """Drop-in chat_fn for LLMGenderRefiner(chat_fn=qwen_chat_fn)."""
    model, tokenizer = _load_qwen()
    if model is None or tokenizer is None:
        return ""

    try:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs, max_new_tokens=512, temperature=0.01, do_sample=False
            )
        generated_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as exc:
        log.warning(f"Qwen generation failed: {exc}")
        return ""
