from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from torch import nn


def ensure_dir(path: str | Path) -> Path:
    """Create the directory if needed and return it as a ``Path``."""
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compile_optional_regex(pattern: str | None) -> re.Pattern[str] | None:
    if pattern is None or pattern == "":
        return None
    return re.compile(pattern)


def parse_torch_dtype(dtype: str):
    """Return a ``torch.dtype`` or the sentinel string ``"auto"``."""
    import torch

    normalized = dtype.lower()
    if normalized in {"auto", "none"}:
        return "auto"
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[normalized]


def get_submodule_parent(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    """Return ``(parent_module, child_attr_name)`` for ``qualified_name``."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def module_layer_index(name: str) -> int | None:
    """Extract a transformer layer index from a module's qualified name, or ``None``."""
    match = re.search(r"(?:layers|blocks|h|decoder\.layers)\.(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def human_int(value: int | float) -> str:
    """Render large parameter counts compactly (e.g. ``1.23B``)."""
    value = float(value)
    for suffix in ["", "K", "M", "B", "T"]:
        if abs(value) < 1000.0:
            if suffix:
                return f"{value:.2f}{suffix}"
            return str(int(value))
        value /= 1000.0
    return f"{value:.2f}P"
