"""Quantum-inspired tensorization utilities for large language models."""

from quantum_tensors.mpo import MPOLinear, decompose_matrix_to_mpo, mpo_to_matrix

__all__ = ["MPOLinear", "decompose_matrix_to_mpo", "mpo_to_matrix"]

