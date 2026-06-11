from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from quantum_tensors.benchmarks.elitr import run_elitr_benchmark
from quantum_tensors.benchmarks.generation import GenerationConfig, load_hf_model
from quantum_tensors.benchmarks.qmsum import run_qmsum_benchmark
from quantum_tensors.modeling.checkpoint import save_tensorized_adapter
from quantum_tensors.modeling.tensorize import (
    TensorizationConfig,
    count_parameters,
    should_tensorize_module,
    tensorization_summary,
    tensorize_model,
)
from quantum_tensors.training.heal import HealingConfig, run_healing
from quantum_tensors.utils import ensure_dir, human_int, write_json

app = typer.Typer(help="MPO tensorization and meeting benchmarks for gpt-oss models.")


@app.command()
def compress(
    model_id: str = typer.Option("openai/gpt-oss-20b", help="Base Hugging Face model id."),
    output_dir: Path = typer.Option(..., help="Directory for the tensorized adapter."),
    max_rank: int = typer.Option(16, help="Maximum MPO bond dimension."),
    order: int = typer.Option(4, help="Number of MPO cores per matrix."),
    target_regex: str = typer.Option(TensorizationConfig.target_regex, help="Regex for target module names."),
    exclude_regex: str = typer.Option(TensorizationConfig.exclude_regex, help="Regex for excluded module names."),
    min_linear_size: int = typer.Option(4096, help="Minimum dense weight size to tensorize."),
    layer_start: Optional[int] = typer.Option(None, help="First transformer layer index to tensorize."),
    layer_end: Optional[int] = typer.Option(None, help="Last transformer layer index to tensorize."),
    skip_mlp_output: bool = typer.Option(False, help="Skip down/out projection modules."),
    relative_tolerance: float = typer.Option(0.0, help="Optional SVD relative energy discard tolerance."),
    torch_dtype: str = typer.Option("auto", help="Model dtype: auto, bf16, fp16, fp32."),
    device_map: str = typer.Option("auto", help="Transformers device_map."),
) -> None:
    """Convert a base model into a tensorized adapter checkpoint.

    This command implements the CompactifAI-style compression step by replacing
    selected dense linear layers with MPO modules and saving only the tensorized
    adapter weights. Use it first in an experiment, before healing or benchmark
    evaluation.
    """
    model, tokenizer = load_hf_model(model_id=model_id, checkpoint_dir=None, torch_dtype=torch_dtype, device_map=device_map)
    before = count_parameters(model)
    config = TensorizationConfig(
        max_rank=max_rank,
        order=order,
        target_regex=target_regex,
        exclude_regex=exclude_regex,
        min_linear_size=min_linear_size,
        layer_start=layer_start,
        layer_end=layer_end,
        skip_mlp_output=skip_mlp_output,
        relative_tolerance=relative_tolerance,
    )
    reports = tensorize_model(model, config)
    after = count_parameters(model)
    adapter = save_tensorized_adapter(model, output_dir, base_model_id=model_id, extra_config={"tensorization": config.__dict__})
    tokenizer.save_pretrained(output_dir)
    summary = tensorization_summary(reports)
    summary.update(
        {
            "base_model_id": model_id,
            "total_parameters_before": before,
            "total_parameters_after_loaded_model": after,
            "adapter_format": adapter["format_version"],
        }
    )
    write_json(Path(output_dir) / "conversion_report.json", summary)
    typer.echo(
        f"Tensorized {summary['modules_tensorized']} modules: "
        f"{human_int(summary['dense_parameters_replaced'])} dense params -> "
        f"{human_int(summary['tensorized_parameters'])} MPO params."
    )


@app.command("benchmark-qmsum")
def benchmark_qmsum(
    qmsum_path: Path = typer.Option(..., help="Path to a QMSum checkout."),
    output_dir: Path = typer.Option(..., help="Directory for benchmark outputs."),
    model_id: str = typer.Option("openai/gpt-oss-20b", help="Base model id."),
    checkpoint_dir: Optional[Path] = typer.Option(None, help="Optional tensorized adapter checkpoint."),
    split: str = typer.Option("test", help="QMSum split."),
    domain: str = typer.Option("ALL", help="QMSum domain folder."),
    max_samples: Optional[int] = typer.Option(None, help="Limit examples for smoke tests."),
    max_input_tokens: int = typer.Option(120000, help="Maximum prompt tokens."),
    max_new_tokens: int = typer.Option(512, help="Maximum generated tokens."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Sampling top-p."),
    torch_dtype: str = typer.Option("auto", help="Model dtype."),
    device_map: str = typer.Option("auto", help="Transformers device_map."),
) -> None:
    """Evaluate a base or tensorized model on QMSum summarization.

    This command loads QMSum, generates query-focused meeting summaries, computes
    ROUGE and token metrics, and writes prediction/summary files. Use it to
    compare the base gpt-oss-20b model against compressed or healed adapters.
    """
    summary = run_qmsum_benchmark(
        model_id=model_id,
        qmsum_path=qmsum_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        split=split,
        domain=domain,
        max_samples=max_samples,
        generation_config=GenerationConfig(max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    typer.echo(summary)


@app.command("benchmark-elitr")
def benchmark_elitr(
    elitr_path: Path = typer.Option(..., help="Path to ELITR-Bench."),
    output_dir: Path = typer.Option(..., help="Directory for benchmark outputs."),
    model_id: str = typer.Option("openai/gpt-oss-20b", help="Base model id."),
    checkpoint_dir: Optional[Path] = typer.Option(None, help="Optional tensorized adapter checkpoint."),
    split: str = typer.Option("test", help="ELITR split."),
    mode: str = typer.Option("single-turn-qa", help="single-turn-qa, multi-turn-qa, or multi-turn-conv."),
    max_samples: Optional[int] = typer.Option(None, help="Limit examples for smoke tests."),
    max_input_tokens: int = typer.Option(120000, help="Maximum prompt tokens."),
    max_new_tokens: int = typer.Option(256, help="Maximum generated tokens."),
    temperature: float = typer.Option(0.0, help="Sampling temperature."),
    top_p: float = typer.Option(1.0, help="Sampling top-p."),
    judge_model: Optional[str] = typer.Option(None, help="Optional OpenAI judge model."),
    torch_dtype: str = typer.Option("auto", help="Model dtype."),
    device_map: str = typer.Option("auto", help="Transformers device_map."),
) -> None:
    """Evaluate a base or tensorized model on ELITR-Bench meeting QA.

    This command runs single-turn or multi-turn meeting assistant prompts,
    computes lexical proxy metrics, and optionally calls an OpenAI judge. Use it
    to measure whether tensorization preserves meeting-question answering
    behavior.
    """
    summary = run_elitr_benchmark(
        model_id=model_id,
        elitr_path=elitr_path,
        output_dir=output_dir,
        checkpoint_dir=checkpoint_dir,
        split=split,
        mode=mode,
        max_samples=max_samples,
        generation_config=GenerationConfig(max_input_tokens=max_input_tokens, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p),
        judge_model=judge_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    typer.echo(summary)


@app.command()
def heal(
    checkpoint_dir: Path = typer.Option(..., help="Tensorized adapter checkpoint to heal."),
    output_dir: Path = typer.Option(..., help="Output directory for healed adapter."),
    model_id: Optional[str] = typer.Option(None, help="Override base model id."),
    dataset_jsonl: Optional[Path] = typer.Option(None, help="Instruction JSONL for healing."),
    qmsum_path: Optional[Path] = typer.Option(None, help="Optional QMSum path for healing data."),
    qmsum_split: str = typer.Option("train", help="QMSum split for healing."),
    max_seq_length: int = typer.Option(8192, help="Training sequence length."),
    learning_rate: float = typer.Option(1e-5, help="Learning rate."),
    max_steps: int = typer.Option(200, help="Max training steps."),
    num_train_epochs: float = typer.Option(1.0, help="Number of train epochs if max_steps is not reached."),
    per_device_train_batch_size: int = typer.Option(1, help="Per-device batch size."),
    gradient_accumulation_steps: int = typer.Option(8, help="Gradient accumulation steps."),
    train_all_parameters: bool = typer.Option(False, help="Train all model parameters instead of MPO params only."),
    torch_dtype: str = typer.Option("auto", help="Model dtype."),
    device_map: str = typer.Option("auto", help="Transformers device_map."),
) -> None:
    """Run post-compression supervised healing on an MPO adapter.

    Tensorization truncates weights layer by layer, so a short fine-tuning phase
    helps recover task quality. Use this command after ``compress`` with either
    custom JSONL instruction data or QMSum-derived examples.
    """
    config = HealingConfig(
        output_dir=str(output_dir),
        dataset_jsonl=str(dataset_jsonl) if dataset_jsonl else None,
        qmsum_path=str(qmsum_path) if qmsum_path else None,
        qmsum_split=qmsum_split,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        max_steps=max_steps,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_tensorized_only=not train_all_parameters,
    )
    summary = run_healing(checkpoint_dir, config, model_id=model_id, torch_dtype=torch_dtype, device_map=device_map)
    typer.echo(summary)


@app.command()
def profile(
    model_id: str = typer.Option("openai/gpt-oss-20b", help="Base model id."),
    output_file: Path = typer.Option(..., help="JSON output file."),
    max_rank: int = typer.Option(16, help="MPO rank for reconstruction profile."),
    order: int = typer.Option(4, help="MPO order."),
    max_modules: int = typer.Option(64, help="Maximum modules to profile."),
    target_regex: str = typer.Option(TensorizationConfig.target_regex, help="Regex for target module names."),
    exclude_regex: str = typer.Option(TensorizationConfig.exclude_regex, help="Regex for excluded module names."),
    layer_start: Optional[int] = typer.Option(None, help="First layer index."),
    layer_end: Optional[int] = typer.Option(None, help="Last layer index."),
    torch_dtype: str = typer.Option("auto", help="Model dtype."),
    device_map: str = typer.Option("auto", help="Transformers device_map."),
) -> None:
    """Profile reconstruction error for candidate tensorized layers.

    This lightweight diagnostic estimates how much each selected layer changes at
    a given MPO rank and order before running expensive task benchmarks. Use it
    to pick layer ranges or ranks for a full compression run.
    """
    import torch
    from torch import nn

    from quantum_tensors.mpo import MPOLinear

    model, _ = load_hf_model(model_id=model_id, checkpoint_dir=None, torch_dtype=torch_dtype, device_map=device_map)
    config = TensorizationConfig(
        max_rank=max_rank,
        order=order,
        target_regex=target_regex,
        exclude_regex=exclude_regex,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or not should_tensorize_module(name, module, config):
            continue
        with torch.no_grad():
            mpo = MPOLinear.from_linear(module, max_rank=max_rank, order=order)
            reconstructed = mpo.dense_weight().to(device=module.weight.device, dtype=module.weight.dtype)
            error = torch.linalg.norm(module.weight - reconstructed) / torch.linalg.norm(module.weight)
            info = mpo.mpo_info()
        rows.append(
            {
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "relative_reconstruction_error": float(error.detach().cpu()),
                "dense_parameters": info.dense_parameters,
                "tensorized_parameters": info.tensorized_parameters,
                "compression_ratio": info.compression_ratio,
                "ranks": list(info.ranks),
            }
        )
        if len(rows) >= max_modules:
            break
    ensure_dir(output_file.parent)
    write_json(output_file, {"model_id": model_id, "rank": max_rank, "order": order, "modules": rows})
    typer.echo(f"Wrote {len(rows)} module profiles to {output_file}")
