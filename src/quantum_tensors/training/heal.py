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


@dataclass
class HealingConfig:
    """Configuration for post-compression healing fine-tuning.

    The paper's method recovers accuracy after local SVD truncation by briefly
    retraining the tensorized model. Use this dataclass to pass dataset, schedule,
    and trainable-parameter choices into ``run_healing``.
    """

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


def _messages_to_text(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render chat messages into training text using the tokenizer template.

    Healing data can be stored as message lists, and gpt-oss-style models need
    their tokenizer's chat format. Use this before tokenizing supervised
    fine-tuning examples.
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def _record_to_text(tokenizer, record: dict[str, Any]) -> str:
    """Normalize one healing record into a text sequence.

    Local datasets may use ``messages``, ``prompt``/``completion``, or raw
    ``text`` fields. Use this to support those common formats without changing
    the training dataset implementation.
    """
    if isinstance(record.get("messages"), list):
        return _messages_to_text(tokenizer, record["messages"])
    if record.get("prompt") is not None and record.get("completion") is not None:
        messages = [
            {"role": "user", "content": str(record["prompt"])},
            {"role": "assistant", "content": str(record["completion"])},
        ]
        return _messages_to_text(tokenizer, messages)
    if record.get("text") is not None:
        return str(record["text"])
    raise ValueError("Healing JSONL records need messages, prompt/completion, or text fields.")


def _qmsum_records(qmsum_path: str, split: str) -> list[dict[str, Any]]:
    """Convert QMSum examples into chat-style healing records.

    QMSum can double as task-specific healing data for meeting summarization.
    Use this when ``--qmsum-path`` is provided instead of a custom instruction
    JSONL file.
    """
    records = []
    for example in load_qmsum(qmsum_path, split=split):
        records.append(
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
        )
    return records


class TextSFTDataset(Dataset):
    """Tokenized supervised fine-tuning dataset backed by text examples.

    Healing uses causal language modeling over fully rendered prompt/answer text.
    Use this small dataset wrapper with a Hugging Face ``Trainer`` and the
    matching collator below.
    """

    def __init__(self, texts: list[str], tokenizer, max_seq_length: int) -> None:
        """Store raw training texts and tokenization settings.

        The dataset tokenizes lazily so startup remains cheap for local JSONL and
        QMSum-derived training data. Instantiate it inside ``run_healing`` after
        loading the tokenizer.
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        """Return the number of supervised training examples.

        The Hugging Face trainer uses this to size epochs and progress reporting.
        Call it implicitly through standard dataset protocols.
        """
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize and return one causal-LM training example.

        Tokenization applies truncation to the configured sequence length so
        oversized meeting examples cannot break batching. The trainer calls this
        automatically while forming batches.
        """
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
            add_special_tokens=True,
        )
        return {"input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long)}


class CausalLMCollator:
    """Pad text SFT examples and create causal-LM labels.

    The trainer needs consistent input ids, attention masks, and labels where pad
    tokens are ignored. Use this collator with ``TextSFTDataset`` for healing.
    """

    def __init__(self, tokenizer) -> None:
        """Store the tokenizer whose pad token defines batch padding.

        Padding must match the model/tokenizer pair being healed. Instantiate the
        collator after setting a pad token on the tokenizer when necessary.
        """
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad a batch and mask padded label positions with ``-100``.

        Hugging Face causal-LM losses ignore labels equal to ``-100``. The
        trainer invokes this method for each batch.
        """
        input_ids = [feature["input_ids"] for feature in features]
        padded = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = (padded != self.tokenizer.pad_token_id).long()
        labels = padded.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": padded, "attention_mask": attention_mask, "labels": labels}


def load_healing_texts(tokenizer, config: HealingConfig) -> list[str]:
    """Load and render all configured healing examples as text.

    Healing can combine custom JSONL examples with QMSum-derived examples. Use
    this before constructing ``TextSFTDataset`` so the training loop sees one
    uniform list of sequences.
    """
    records: list[dict[str, Any]] = []
    if config.dataset_jsonl:
        records.extend(read_jsonl(config.dataset_jsonl))
    if config.qmsum_path:
        records.extend(_qmsum_records(config.qmsum_path, split=config.qmsum_split))
    if not records:
        raise ValueError("Provide --dataset-jsonl or --qmsum-path for healing.")
    return [_record_to_text(tokenizer, record) for record in records]


def run_healing(
    checkpoint_dir: str | Path,
    config: HealingConfig,
    model_id: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> dict[str, Any]:
    """Fine-tune a tensorized adapter and save the healed adapter.

    Local SVD compression is not globally optimal, so a short supervised training
    pass helps recover benchmark quality. Use this after ``compress`` and before
    running QMSum or ELITR-Bench comparisons.
    """
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

    texts = load_healing_texts(tokenizer, config)
    dataset = TextSFTDataset(texts, tokenizer=tokenizer, max_seq_length=config.max_seq_length)
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
        bf16=torch_dtype.lower() in {"bf16", "bfloat16"},
        fp16=torch_dtype.lower() in {"fp16", "float16", "half"},
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
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
        "num_training_records": len(texts),
        "train_tensorized_only": config.train_tensorized_only,
        "adapter": saved_config,
    }
    write_json(output_dir / "healing_summary.json", summary)
    return summary
