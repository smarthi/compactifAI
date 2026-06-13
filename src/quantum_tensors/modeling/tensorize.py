from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import nn

from quantum_tensors.mpo import MPOLinear
from quantum_tensors.utils import compile_optional_regex, get_submodule_parent, module_layer_index


@dataclass
class TensorizationConfig:
    """Selects which dense linear layers get replaced by MPO layers."""

    max_rank: int = 16
    order: int = 4
    target_regex: str = (
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|gate_up_proj|"
        r"c_attn|c_proj|mlp|expert)"
    )
    exclude_regex: str = r"(lm_head|embed|embedding|router|score)"
    min_dense_parameters: int = 1 << 18  # 262144; skip layers smaller than this
    layer_start: int | None = None
    layer_end: int | None = None
    skip_mlp_output: bool = False
    relative_tolerance: float = 0.0
    svd_dtype: str = "float32"


@dataclass
class TensorizedModuleReport:
    name: str
    in_features: int
    out_features: int
    dense_parameters: int
    tensorized_parameters: int
    compression_ratio: float
    ranks: tuple[int, ...]
    in_dims: tuple[int, ...]
    out_dims: tuple[int, ...]


def _iter_named_linears(model: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def _is_mlp_output_projection(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ["down_proj", "out_proj", "output_proj", "c_proj"])


def should_tensorize_module(name: str, module: nn.Linear, config: TensorizationConfig) -> bool:
    """Apply size, layer-range, regex, and sensitivity filters to one Linear."""
    if module.in_features * module.out_features < config.min_dense_parameters:
        return False

    layer_index = module_layer_index(name)
    if config.layer_start is not None and layer_index is not None and layer_index < config.layer_start:
        return False
    if config.layer_end is not None and layer_index is not None and layer_index > config.layer_end:
        return False

    if config.skip_mlp_output and _is_mlp_output_projection(name):
        return False

    target_regex = compile_optional_regex(config.target_regex)
    exclude_regex = compile_optional_regex(config.exclude_regex)
    if exclude_regex and exclude_regex.search(name):
        return False
    if target_regex and not target_regex.search(name):
        return False
    return True


def _parse_svd_dtype(dtype: str) -> torch.dtype:
    normalized = dtype.lower()
    if normalized in {"float32", "fp32"}:
        return torch.float32
    if normalized in {"float64", "fp64", "double"}:
        return torch.float64
    raise ValueError("SVD dtype must be float32 or float64.")


def tensorize_model(model: nn.Module, config: TensorizationConfig) -> list[TensorizedModuleReport]:
    """Replace selected ``nn.Linear`` modules with ``MPOLinear`` modules in-place."""
    svd_dtype = _parse_svd_dtype(config.svd_dtype)
    candidates = [
        (name, module)
        for name, module in _iter_named_linears(model)
        if should_tensorize_module(name, module, config)
    ]
    reports: list[TensorizedModuleReport] = []
    for name, module in candidates:
        tensorized = MPOLinear.from_linear(
            module,
            max_rank=config.max_rank,
            order=config.order,
            relative_tolerance=config.relative_tolerance,
            svd_dtype=svd_dtype,
        )
        tensorized = tensorized.to(device=module.weight.device)
        parent, child_name = get_submodule_parent(model, name)
        setattr(parent, child_name, tensorized)
        info = tensorized.mpo_info()
        reports.append(
            TensorizedModuleReport(
                name=name,
                in_features=info.in_features,
                out_features=info.out_features,
                dense_parameters=info.dense_parameters,
                tensorized_parameters=info.tensorized_parameters,
                compression_ratio=info.compression_ratio,
                ranks=info.ranks,
                in_dims=info.in_dims,
                out_dims=info.out_dims,
            )
        )
    return reports


def tensorization_summary(reports: list[TensorizedModuleReport]) -> dict[str, object]:
    """Aggregate per-module reports into a JSON-ready model-level summary."""
    dense = sum(report.dense_parameters for report in reports)
    tensorized = sum(report.tensorized_parameters for report in reports)
    return {
        "modules_tensorized": len(reports),
        "dense_parameters_replaced": dense,
        "tensorized_parameters": tensorized,
        "compression_ratio": tensorized / dense if dense else 1.0,
        "modules": [asdict(report) for report in reports],
    }


def trainable_tensorized_parameters_only(model: nn.Module) -> None:
    """Freeze everything except parameters inside ``MPOLinear`` modules."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, MPOLinear):
            for parameter in module.parameters():
                parameter.requires_grad_(True)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return sum(parameter.numel() for parameter in model.parameters())
