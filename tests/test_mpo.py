from __future__ import annotations

import torch
from torch import nn

from quantum_tensors.mpo import MPOLinear, decompose_matrix_to_mpo, infer_mpo_factors, mpo_to_matrix


def test_mpo_reconstructs_small_matrix_at_full_rank() -> None:
    """Verify full-rank MPO decomposition can reconstruct a small dense matrix."""
    torch.manual_seed(7)
    weight = torch.randn(12, 10)
    out_dims, in_dims = infer_mpo_factors(12, 10, order=3)
    cores = decompose_matrix_to_mpo(weight, out_dims, in_dims, max_rank=128)
    reconstructed = mpo_to_matrix(cores)
    torch.testing.assert_close(reconstructed, weight, rtol=1e-4, atol=1e-4)


def test_mpo_linear_matches_dense_forward_at_full_rank() -> None:
    """Verify an MPO-wrapped linear layer matches the original dense forward pass."""
    torch.manual_seed(9)
    linear = nn.Linear(10, 12)
    mpo = MPOLinear.from_linear(linear, max_rank=128, order=3)
    x = torch.randn(4, 5, 10)
    torch.testing.assert_close(mpo(x), linear(x), rtol=1e-4, atol=1e-4)


def test_mpo_linear_compresses_parameter_count() -> None:
    """Verify low-rank MPO layers reduce parameter count versus dense layers."""
    linear = nn.Linear(64, 64)
    mpo = MPOLinear.from_linear(linear, max_rank=4, order=4)
    assert sum(parameter.numel() for parameter in mpo.parameters()) < sum(
        parameter.numel() for parameter in linear.parameters()
    )
