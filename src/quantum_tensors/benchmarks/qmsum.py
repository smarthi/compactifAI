from __future__ import annotations

from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from quantum_tensors.benchmarks.data import QMSumExample, load_qmsum
from quantum_tensors.benchmarks.generation import (
    GenerationConfig,
    generate_response,
    load_hf_model,
    truncate_text_to_budget,
)
from quantum_tensors.metrics import aggregate_metric_rows, rouge_scores, token_f1
from quantum_tensors.utils import ensure_dir, write_json, write_jsonl


def qmsum_messages(example: QMSumExample, tokenizer, max_input_tokens: int) -> list[dict[str, str]]:
    """Build the chat prompt for one QMSum query-focused summary."""
    transcript = truncate_text_to_budget(tokenizer, example.transcript, max_input_tokens=max_input_tokens)
    return [
        {
            "role": "system",
            "content": "You are a precise meeting summarization assistant. Use only the transcript.",
        },
        {
            "role": "user",
            "content": (
                "Summarize the meeting content requested by the query.\n\n"
                f"Query: {example.query}\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Answer with a concise, faithful summary."
            ),
        },
    ]


def run_qmsum_benchmark(
    model_id: str,
    qmsum_path: str | Path,
    output_dir: str | Path,
    checkpoint_dir: str | Path | None = None,
    split: str = "test",
    domain: str = "ALL",
    max_samples: int | None = None,
    generation_config: GenerationConfig | None = None,
    torch_dtype: str = "auto",
    device_map: str = "auto",
) -> dict[str, object]:
    """Run QMSum generation, score with ROUGE + token F1, and write outputs."""
    output_path = ensure_dir(output_dir)
    generation_config = generation_config or GenerationConfig()
    model, tokenizer = load_hf_model(
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    examples = load_qmsum(qmsum_path, split=split, domain=domain)
    if max_samples is not None:
        examples = examples[:max_samples]

    rows: list[dict[str, object]] = []
    for example in tqdm(examples, desc="QMSum"):
        start = perf_counter()
        prediction = generate_response(
            model,
            tokenizer,
            qmsum_messages(example, tokenizer, generation_config.max_input_tokens),
            generation_config,
        )
        elapsed = perf_counter() - start
        scores = rouge_scores(prediction, example.reference)
        row = {
            **example.to_dict(),
            "prediction": prediction,
            "latency_seconds": elapsed,
            "token_f1": token_f1(prediction, example.reference),
            **scores,
        }
        rows.append(row)

    summary = {
        "dataset": "qmsum",
        "split": split,
        "model_id": model_id,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        "num_examples": len(rows),
        "metrics": aggregate_metric_rows(rows),
    }
    write_jsonl(output_path / "predictions.jsonl", rows)
    write_json(output_path / "summary.json", summary)
    return summary
