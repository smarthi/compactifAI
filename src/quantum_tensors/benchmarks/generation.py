from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from quantum_tensors.modeling.checkpoint import load_tensorized_adapter, read_adapter_config
from quantum_tensors.utils import parse_torch_dtype


@dataclass
class GenerationConfig:
    """Runtime generation settings shared by benchmark runners."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    max_input_tokens: int = 120_000


def load_hf_model(
    model_id: str,
    checkpoint_dir: str | Path | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
):
    """Load a base causal LM and, if given, swap in a tensorized adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = parse_torch_dtype(torch_dtype)
    if checkpoint_dir is not None:
        adapter = read_adapter_config(checkpoint_dir)
        model_id = adapter.get("base_model_id") or model_id

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    if checkpoint_dir is not None:
        load_tensorized_adapter(model, checkpoint_dir)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def truncate_text_to_budget(tokenizer, text: str, max_tokens: int) -> str:
    """Trim text to ``max_tokens`` preserving head + tail with a middle marker."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    head_budget = max_tokens // 2
    tail_budget = max_tokens - head_budget
    head = tokenizer.decode(token_ids[:head_budget], skip_special_tokens=True)
    tail = tokenizer.decode(token_ids[-tail_budget:], skip_special_tokens=True)
    return f"{head}\n\n[... middle truncated to fit context budget ...]\n\n{tail}"


def render_chat(tokenizer, messages: list[dict[str, str]], max_input_tokens: int):
    """Render chat messages into tokenizer tensors, truncating to ``max_input_tokens``."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() > max_input_tokens:
            input_ids = input_ids[-max_input_tokens:]
            encoded["input_ids"] = input_ids.unsqueeze(0)
            if "attention_mask" in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        return encoded

    prompt = "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)
    prompt += "\n\nASSISTANT:"
    return tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens)


@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    config: GenerationConfig,
) -> str:
    """Generate one decoded assistant response for a chat-style prompt."""
    encoded = render_chat(tokenizer, messages, max_input_tokens=config.max_input_tokens)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generate_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.temperature > 0,
        "top_p": config.top_p,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if config.temperature > 0:
        generate_kwargs["temperature"] = config.temperature
    output = model.generate(**encoded, **generate_kwargs)
    prompt_length = encoded["input_ids"].shape[-1]
    generated = output[0][prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
