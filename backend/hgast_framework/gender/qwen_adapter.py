"""
gender/qwen_adapter.py
========================
Your original Qwen2.5-7B-Instruct 4-bit setup, wrapped as a chat_fn so it
plugs straight into LLMGenderRefiner(chat_fn=...). This is "yours" — the
exact loading/generation logic you had — just returning a plain string
instead of an Ollama-style dict, and with the lazy-singleton pattern so
importing this file doesn't immediately try to load a 7B model.
"""

import logging

log = logging.getLogger(__name__)

_qwen_model = None
_qwen_tokenizer = None


def _load_qwen():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is not None:
        return _qwen_model, _qwen_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    log.info("Loading Qwen2.5-7B-Instruct in 4-bit...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    _qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
    _qwen_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=quantization_config, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Fix rotary embeddings device mismatch in older transformers/Qwen versions
    for name, module in _qwen_model.named_modules():
        if "rotary_emb" in name:
            if hasattr(module, "inv_freq"):
                module.inv_freq = module.inv_freq.to(_qwen_model.device)
                
    log.info("Qwen loaded successfully.")
    return _qwen_model, _qwen_tokenizer


def qwen_chat_fn(prompt: str) -> str:
    """Drop-in chat_fn for LLMGenderRefiner(chat_fn=qwen_chat_fn)."""
    import torch

    model, tokenizer = _load_qwen()
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        torch.cuda.empty_cache()
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=512, temperature=0.01, do_sample=False
        )
    generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
