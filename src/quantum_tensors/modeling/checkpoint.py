from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from safetensors.torch import load_file, save_file
from torch import nn

from quantum_tensors.mpo import MPOLinear
from quantum_tensors.utils import ensure_dir, get_submodule_parent, read_json, write_json

ADAPTER_CONFIG = "tensorized_config.json"
ADAPTER_WEIGHTS = "tensorized_model.safetensors"


def _module_entries(model: nn.Module) -> list[tuple[str, MPOLinear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, MPOLinear)]


def save_tensorized_adapter(
    model: nn.Module,
    output_dir: str | Path,
    base_model_id: str,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist MPO cores, biases, and metadata as a lightweight adapter directory."""
    output_path = ensure_dir(output_dir)
    state = {}
    modules = []
    for name, module in _module_entries(model):
        module_state = {
            "name": name,
            "in_features": module.in_features,
            "out_features": module.out_features,
            "in_dims": list(module.in_dims),
            "out_dims": list(module.out_dims),
            "has_bias": module.bias is not None,
            "num_cores": len(module.cores),
        }
        info = module.mpo_info()
        module_state["mpo_info"] = asdict(info)
        modules.append(module_state)
        for index, core in enumerate(module.cores):
            state[f"{name}.cores.{index}"] = core.detach().cpu()
        if module.bias is not None:
            state[f"{name}.bias"] = module.bias.detach().cpu()

    config = {
        "format_version": 1,
        "base_model_id": base_model_id,
        "modules": modules,
        "extra_config": extra_config or {},
    }
    save_file(state, str(output_path / ADAPTER_WEIGHTS))
    write_json(output_path / ADAPTER_CONFIG, config)
    return config


def load_tensorized_adapter(
    model: nn.Module,
    checkpoint_dir: str | Path,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a tensorized adapter into a base model in-place.

    With ``strict=True`` any missing core or bias raises ``KeyError`` before any
    modules are swapped. With ``strict=False`` modules with missing tensors are
    skipped and their names returned under ``"skipped"`` in the config dict.
    """
    checkpoint_path = Path(checkpoint_dir)
    config = read_json(checkpoint_path / ADAPTER_CONFIG)
    state = load_file(str(checkpoint_path / ADAPTER_WEIGHTS))

    plans: list[tuple[dict[str, Any], list, Any]] = []
    all_missing: dict[str, list[str]] = {}
    for module_config in config["modules"]:
        name = module_config["name"]
        module_missing: list[str] = []
        cores = []
        for index in range(module_config["num_cores"]):
            key = f"{name}.cores.{index}"
            tensor = state.get(key)
            if tensor is None:
                module_missing.append(key)
            else:
                cores.append(tensor)
        bias = None
        if module_config.get("has_bias"):
            bias_key = f"{name}.bias"
            bias = state.get(bias_key)
            if bias is None:
                module_missing.append(bias_key)
        if module_missing:
            all_missing[name] = module_missing
        else:
            plans.append((module_config, cores, bias))

    if all_missing and strict:
        raise KeyError(f"Missing tensorized weights: {all_missing}")

    for module_config, cores, bias in plans:
        name = module_config["name"]
        parent, child_name = get_submodule_parent(model, name)
        old_module = getattr(parent, child_name)
        device = getattr(getattr(old_module, "weight", None), "device", None)
        dtype = getattr(getattr(old_module, "weight", None), "dtype", None)
        tensorized = MPOLinear.from_state(
            cores=cores,
            bias=bias,
            in_features=int(module_config["in_features"]),
            out_features=int(module_config["out_features"]),
        )
        if device is not None:
            tensorized = tensorized.to(device=device)
        if dtype is not None:
            tensorized = tensorized.to(dtype=dtype)
        setattr(parent, child_name, tensorized)

    if all_missing:
        config = {**config, "skipped": all_missing}
    return config


def read_adapter_config(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Read adapter metadata without loading the weights."""
    return read_json(Path(checkpoint_dir) / ADAPTER_CONFIG)
