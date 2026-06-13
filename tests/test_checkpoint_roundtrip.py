from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from quantum_tensors.modeling.checkpoint import load_tensorized_adapter, save_tensorized_adapter
from quantum_tensors.modeling.tensorize import TensorizationConfig, tensorize_model


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(64, 64),
                    nn.Linear(64, 64),
                )
                for _ in range(2)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            x = block(x)
        return x


def test_tensorize_save_load_roundtrip_preserves_forward(tmp_path: Path) -> None:
    """Tensorizing, saving, then loading must yield the same forward outputs."""
    torch.manual_seed(0)
    model = _TinyMLP()
    x = torch.randn(2, 64)
    config = TensorizationConfig(
        max_rank=128,
        order=2,
        target_regex=r"layers\.\d+\.\d+",
        exclude_regex="",
        min_dense_parameters=0,
    )
    tensorize_model(model, config)
    expected = model(x)
    save_tensorized_adapter(model, tmp_path, base_model_id="tiny-mlp")

    fresh = _TinyMLP()
    load_tensorized_adapter(fresh, tmp_path)
    actual = fresh(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
