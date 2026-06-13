from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from quantum_tensors.benchmarks.data import ELITRExample, load_elitr
from quantum_tensors.benchmarks.generation import (
    GenerationConfig,
    generate_response,
    load_hf_model,
    truncate_text_to_budget,
)
from quantum_tensors.benchmarks.judge import score_with_openai_judge
from quantum_tensors.metrics import aggregate_metric_rows, exact_match, rouge_scores, token_f1
from quantum_tensors.utils import ensure_dir, write_json, write_jsonl


def elitr_base_messages(example: ELITRExample, tokenizer, max_input_tokens: int) -> list[dict[str, str]]:
    """Build the single-turn ELITR-Bench QA prompt for one example."""
    transcript = truncate_text_to_budget(tokenizer, example.transcript, max_input_tokens=max_input_tokens)
    return [
        {
            "role": "system",
            "content": (
                "You are a meeting assistant. Answer questions using only the meeting transcript. "
                "If the transcript does not contain the answer, say that it is not available."
            ),
        },
        {
            "role": "user",
            "content": f"Meeting transcript:\n{transcript}\n\nQuestion: {example.question}",
        },
    ]


def _score_row(
    example: ELITRExample,
    prediction: str,
    latency_seconds: float,
    judge_model: str | None,
) -> dict[str, object]:
    """Build a scored row from one prediction (ROUGE, token F1, EM, optional judge)."""
    scores = rouge_scores(prediction, example.reference)
    row: dict[str, object] = {
        **example.to_dict(),
        "prediction": prediction,
        "latency_seconds": latency_seconds,
        "token_f1": token_f1(prediction, example.reference),
        "exact_match": exact_match(prediction, example.reference),
        **scores,
    }
    if judge_model:
        row.update(score_with_openai_judge(example.question, example.reference, prediction, judge_model))
    return row


def _run_single_turn(
    model,
    tokenizer,
    examples: list[ELITRExample],
    generation_config: GenerationConfig,
    judge_model: str | None,
) -> list[dict[str, object]]:
    """Score each example independently (no conversational state)."""
    rows: list[dict[str, object]] = []
    for example in tqdm(examples, desc="ELITR single-turn"):
        start = perf_counter()
        prediction = generate_response(
            model,
            tokenizer,
            elitr_base_messages(example, tokenizer, generation_config.max_input_tokens),
            generation_config,
        )
        rows.append(_score_row(example, prediction, perf_counter() - start, judge_model))
    return rows


def _run_multi_turn(
    model,
    tokenizer,
    examples: list[ELITRExample],
    generation_config: GenerationConfig,
    judge_model: str | None,
) -> list[dict[str, object]]:
    """Score each meeting as a single conversation that accumulates prior turns."""
    grouped: dict[str, list[ELITRExample]] = defaultdict(list)
    for example in examples:
        grouped[example.meeting_id].append(example)

    rows: list[dict[str, object]] = []
    for meeting_id, meeting_examples in tqdm(grouped.items(), desc="ELITR multi-turn"):
        meeting_examples = sorted(meeting_examples, key=lambda item: item.question_index)
        first = meeting_examples[0]
        transcript = truncate_text_to_budget(
            tokenizer,
            first.transcript,
            max_input_tokens=generation_config.max_input_tokens,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a meeting assistant. Answer questions using only the meeting transcript. "
                    "Use previous turns only as conversational context."
                ),
            },
            {"role": "user", "content": f"Meeting transcript:\n{transcript}"},
            {"role": "assistant", "content": "Ready."},
        ]
        for example in meeting_examples:
            messages.append({"role": "user", "content": example.question})
            start = perf_counter()
            prediction = generate_response(model, tokenizer, messages, generation_config)
            elapsed = perf_counter() - start
            messages.append({"role": "assistant", "content": prediction})
            row = _score_row(example, prediction, elapsed, judge_model)
            row["conversation_meeting_id"] = meeting_id
            rows.append(row)
    return rows


def run_elitr_benchmark(
    model_id: str,
    elitr_path: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    split: str = "test",
    mode: str = "single-turn-qa",
    max_samples: int | None = None,
    generation_config: GenerationConfig | None = None,
    judge_model: str | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> dict[str, object]:
    """Run ELITR-Bench generation in the chosen mode and write predictions + summary."""
    output_path = ensure_dir(output_dir)
    generation_config = generation_config or GenerationConfig()
    setting = "conv" if "conv" in mode else "qa"
    examples = load_elitr(elitr_path, split=split, setting=setting)
    if max_samples is not None:
        examples = examples[:max_samples]

    model, tokenizer = load_hf_model(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    if mode.startswith("single"):
        rows = _run_single_turn(model, tokenizer, examples, generation_config, judge_model)
    else:
        rows = _run_multi_turn(model, tokenizer, examples, generation_config, judge_model)

    summary = {
        "dataset": "elitr-bench",
        "split": split,
        "mode": mode,
        "model_id": model_id,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "num_examples": len(rows),
        "metrics": aggregate_metric_rows(rows),
    }
    write_jsonl(output_path / "predictions.jsonl", rows)
    write_json(output_path / "summary.json", summary)
    return summary
