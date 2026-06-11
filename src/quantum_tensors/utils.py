from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``.

    The runners and conversion commands need this small guard before writing
    reports, adapters, and predictions. Call it with any output location before
    writing files whose parent tree may not exist yet.
    """
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file and return the decoded payload.

    Centralizing JSON reads keeps config, adapter metadata, and benchmark
    summaries using the same encoding assumptions. Use it for structured files
    that should be loaded fully into memory.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON payload with stable formatting.

    Stable indentation and sorted keys make benchmark outputs and adapter
    metadata easier to diff in git. Use it whenever a command emits a summary,
    report, or reusable configuration file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write benchmark-style row records as UTF-8 JSON Lines.

    Prediction files can be large, so one JSON object per line is friendlier for
    streaming and post-processing than a single JSON array. Use this for records
    where each row is an independent example result.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file into a list of dictionaries.

    Healing accepts instruction examples in JSONL form, so this helper provides
    the matching reader for ``write_jsonl``. Use it for small-to-medium local
    training or evaluation files that fit comfortably in memory.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compile_optional_regex(pattern: str | None) -> re.Pattern[str] | None:
    """Compile a regex only when a pattern is provided.

    Tensorization filters support optional include/exclude expressions, and
    callers should not need to branch around empty settings. Use this before
    matching module names when ``None`` should mean "no filter".
    """
    if pattern is None or pattern == "":
        return None
    return re.compile(pattern)


def parse_torch_dtype(dtype: str):
    """Convert a user-facing dtype string into a Torch dtype or ``"auto"``.

    CLI commands accept compact values like ``bf16`` and ``fp16`` while
    Transformers expects Torch dtype objects. Use this at model-loading
    boundaries before passing dtype values to Hugging Face APIs.
    """
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


def module_layer_index(name: str) -> int | None:
    """Extract a transformer layer index from a module's qualified name.

    Layer-aware tensorization needs to skip early or late blocks without knowing
    every model family's exact attribute layout. Use this on module names such
    as ``model.layers.12.mlp.down_proj`` when applying layer range filters.
    """
    match = re.search(r"(?:layers|blocks|h|decoder\.layers)\.(\d+)", name)
    if match:
        return int(match.group(1))
    return None


def human_int(value: int | float) -> str:
    """Render large parameter counts in a compact human-readable form.

    Conversion reports often involve millions or billions of parameters, and the
    CLI needs readable progress messages. Use this only for display text; keep
    raw numeric values in JSON reports.
    """
    value = float(value)
    for suffix in ["", "K", "M", "B", "T"]:
        if abs(value) < 1000.0:
            if suffix:
                return f"{value:.2f}{suffix}"
            return str(int(value))
        value /= 1000.0
    return f"{value:.2f}P"
