from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from quantum_tensors.benchmarks.data import load_qmsum
from quantum_tensors.benchmarks.generation import load_hf_model
from quantum_tensors.modeling.checkpoint import read_adapter_config, save_tensorized_adapter
from quantum_tensors.modeling.tensorize import trainable_tensorized_parameters_only
from quantum_tensors.utils import read_jsonl, write_json

LABEL_IGNORE_INDEX = -100


@dataclass
class HealingConfig:
    """Configuration for post-compression healing fine-tuning."""

    output_dir: str
    dataset_jsonl: str | None = None
    qmsum_path: str | None = None
    qmsum_split: str = "train"
    max_seq_length: int = 8192
    learning_rate: float = 1e-5
    max_steps: int = 200
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    train_tensorized_only: bool = True


def _record_to_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    """Convert a healing record into a chat message list, or None for raw text."""
    if isinstance(record.get("messages"), list):
        return list(record["messages"])
    if record.get("prompt") is not None and record.get("completion") is not None:
        return [
            {"role": "user", "content": str(record["prompt"])},
            {"role": "assistant", "content": str(record["completion"])},
        ]
    if record.get("text") is not None:
        return None
    raise ValueError("Healing records need messages, prompt/completion, or text fields.")


def _build_sft_example(
    tokenizer,
    record: dict[str, Any],
    max_seq_length: int,
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` with prompt tokens masked to ``-100``.

    For chat-style records we render the full conversation and the prompt-only
    prefix with the tokenizer's chat template and mask the shared prefix. For raw
    ``text`` records we fall back to full-sequence causal LM with no masking.
    """
    messages = _record_to_messages(record)
    if messages is None:
        ids = tokenizer(
            str(record["text"]),
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,
        )["input_ids"]
        return list(ids), list(ids)

    has_template = getattr(tokenizer, "chat_template", None) and hasattr(tokenizer, "apply_chat_template")
    has_assistant_end = bool(messages) and messages[-1].get("role") == "assistant"

    if not (has_template and has_assistant_end):
        text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        ids = tokenizer(text, truncation=True, max_length=max_seq_length, add_special_tokens=True)["input_ids"]
        return list(ids), list(ids)

    full_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_seq_length,
        )
    )
    prompt_ids = list(
        tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            max_length=max_seq_length,
        )
    )
    # Mask the longest common prefix. Apply the common-prefix scan instead of
    # trusting len(prompt_ids) directly because chat templates sometimes append
    # role-closing tokens to the assistant turn that differ between the two
    # renderings.
    prefix = 0
    for pid, fid in zip(prompt_ids, full_ids):
        if pid != fid:
            break
        prefix += 1
    labels = [LABEL_IGNORE_INDEX] * prefix + list(full_ids[prefix:])
    if prefix == len(full_ids):
        # No completion tokens survived truncation; skip via empty labels.
        labels = [LABEL_IGNORE_INDEX] * len(full_ids)
    return full_ids, labels


def _qmsum_records(qmsum_path: str, split: str) -> list[dict[str, Any]]:
    """Convert QMSum examples into chat-style healing records."""
    return [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise meeting summarization assistant. Use only the transcript.",
                },
                {
                    "role": "user",
                    "content": (
                        "Summarize the meeting content requested by the query.\n\n"
                        f"Query: {example.query}\n\n"
                        f"Transcript:\n{example.transcript}\n\n"
                        "Answer with a concise, faithful summary."
                    ),
                },
                {"role": "assistant", "content": example.reference},
            ]
        }
        for example in load_qmsum(qmsum_path, split=split)
    ]


class SFTDataset(Dataset):
    """Lazy chat-style SFT dataset with prompt-masked labels."""

    def __init__(self, records: list[dict[str, Any]], tokenizer, max_seq_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = _build_sft_example(self.tokenizer, self.records[index], self.max_seq_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SFTCollator:
    """Pad input_ids with the tokenizer pad token and labels with ``-100``."""

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=pad_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=LABEL_IGNORE_INDEX,
        )
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def load_healing_records(config: HealingConfig) -> list[dict[str, Any]]:
    """Load JSONL and QMSum-derived healing records into one list."""
    records: list[dict[str, Any]] = []
    if config.dataset_jsonl:
        records.extend(read_jsonl(config.dataset_jsonl))
    if config.qmsum_path:
        records.extend(_qmsum_records(config.qmsum_path, split=config.qmsum_split))
    if not records:
        raise ValueError("Provide --dataset-jsonl or --qmsum-path for healing.")
    return records


def _trainer_precision_flags(model, torch_dtype: str) -> tuple[bool, bool]:
    """Decide ``(bf16, fp16)`` from the requested dtype string and the loaded model."""
    normalized = torch_dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return True, False
    if normalized in {"fp16", "float16", "half"}:
        return False, True
    if normalized in {"fp32", "float32", "float"}:
        return False, False
    # "auto" or unrecognized — inspect the actual model dtype.
    try:
        loaded = next(model.parameters()).dtype
    except StopIteration:
        return False, False
    if loaded == torch.bfloat16:
        return True, False
    if loaded == torch.float16:
        return False, True
    return False, False


def run_healing(
    checkpoint_dir: str | Path,
    config: HealingConfig,
    model_id: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> dict[str, Any]:
    """Fine-tune a tensorized adapter and save the healed adapter."""
    from transformers import Trainer, TrainingArguments

    adapter = read_adapter_config(checkpoint_dir)
    base_model_id = model_id or adapter["base_model_id"]
    model, tokenizer = load_hf_model(
        model_id=base_model_id,
        checkpoint_dir=checkpoint_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if config.train_tensorized_only:
        trainable_tensorized_parameters_only(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        # Required when the base model (including input embeddings) is frozen:
        # otherwise checkpointed segments break the gradient path to MPO cores.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    records = load_healing_records(config)
    dataset = SFTDataset(records, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
    bf16, fp16 = _trainer_precision_flags(model, torch_dtype)
    args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        bf16=bf16,
        fp16=fp16,
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=SFTCollator(tokenizer),
    )
    trainer.train()
    output_dir = Path(config.output_dir)
    saved_config = save_tensorized_adapter(
        model,
        output_dir,
        base_model_id=base_model_id,
        extra_config={"healing": config.__dict__},
    )
    tokenizer.save_pretrained(output_dir)
    summary = {
        "base_model_id": base_model_id,
        "checkpoint_dir": str(checkpoint_dir),
        "output_dir": str(output_dir),
        "num_training_records": len(records),
        "train_tensorized_only": config.train_tensorized_only,
        "adapter": saved_config,
    }
    write_json(output_dir / "healing_summary.json", summary)
    return summary
