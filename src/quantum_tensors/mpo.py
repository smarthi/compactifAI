from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import prod
from operator import mul

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _prime_factors(n: int) -> list[int]:
    if n < 1:
        raise ValueError("Dimension must be positive.")
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= n:
        while n % divisor == 0:
            factors.append(divisor)
            n //= divisor
        divisor += 1 if divisor == 2 else 2
    if n > 1:
        factors.append(n)
    return factors


def balanced_factors(n: int, order: int) -> tuple[int, ...]:
    """Split a dimension into a near-balanced product with a fixed order."""
    if order < 1:
        raise ValueError("order must be >= 1")
    buckets = [1] * order
    for factor in sorted(_prime_factors(n), reverse=True):
        index = min(range(order), key=lambda i: buckets[i])
        buckets[index] *= factor
    return tuple(sorted(buckets))


def validate_factors(name: str, factors: tuple[int, ...], expected: int) -> None:
    if reduce(mul, factors, 1) != expected:
        raise ValueError(f"{name} factors {factors} do not multiply to {expected}.")
    if any(dim < 1 for dim in factors):
        raise ValueError(f"{name} factors must all be positive: {factors}.")


def infer_mpo_factors(
    out_features: int,
    in_features: int,
    order: int,
    out_factors: tuple[int, ...] | None = None,
    in_factors: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    out_dims = out_factors or balanced_factors(out_features, order)
    in_dims = in_factors or balanced_factors(in_features, order)
    if len(out_dims) != len(in_dims):
        raise ValueError("out_factors and in_factors must have the same length.")
    validate_factors("out", tuple(out_dims), out_features)
    validate_factors("in", tuple(in_dims), in_features)
    return tuple(out_dims), tuple(in_dims)


def _select_rank(
    singular_values: Tensor,
    max_rank: int,
    relative_tolerance: float,
    min_rank: int,
) -> int:
    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")
    hard_limit = min(max_rank, singular_values.numel())
    if relative_tolerance <= 0:
        return max(min_rank, hard_limit)

    total_energy = torch.sum(singular_values.square())
    if total_energy <= 0:
        return max(min_rank, 1)
    cumulative = torch.cumsum(singular_values.square(), dim=0)
    retained = cumulative / total_energy
    target = 1.0 - relative_tolerance
    rank = int(torch.searchsorted(retained, target).item()) + 1
    return max(min_rank, min(rank, hard_limit))


def _interleave_matrix_tensor(weight: Tensor, out_dims: tuple[int, ...], in_dims: tuple[int, ...]) -> Tensor:
    order = len(out_dims)
    matrix_tensor = weight.reshape(*out_dims, *in_dims)
    permutation: list[int] = []
    for index in range(order):
        permutation.extend([index, order + index])
    return matrix_tensor.permute(*permutation).contiguous()


def decompose_matrix_to_mpo(
    weight: Tensor,
    out_dims: tuple[int, ...],
    in_dims: tuple[int, ...],
    max_rank: int,
    relative_tolerance: float = 0.0,
    min_rank: int = 1,
    svd_dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """Decompose a linear weight matrix into MPO cores with sequential SVD.

    The input weight is expected in PyTorch linear layout: [out_features, in_features].
    Returned cores have shape [rank_left, out_dim, in_dim, rank_right].
    """
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(weight.shape)}.")
    out_features, in_features = weight.shape
    validate_factors("out", out_dims, out_features)
    validate_factors("in", in_dims, in_features)
    if len(out_dims) != len(in_dims):
        raise ValueError("out_dims and in_dims must have equal length.")

    original_dtype = weight.dtype
    work = _interleave_matrix_tensor(weight.detach().to(dtype=svd_dtype), out_dims, in_dims)
    order = len(out_dims)
    current = work.reshape(1, *[dim for pair in zip(out_dims, in_dims) for dim in pair])
    rank_left = 1
    cores: list[Tensor] = []

    for index in range(order - 1):
        row_dim = rank_left * out_dims[index] * in_dims[index]
        matrix = current.reshape(row_dim, -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = _select_rank(s, max_rank=max_rank, relative_tolerance=relative_tolerance, min_rank=min_rank)
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]
        core = u.reshape(rank_left, out_dims[index], in_dims[index], rank)
        cores.append(core.to(dtype=original_dtype))
        current = s.unsqueeze(1) * vh
        rank_left = rank

    final_core = current.reshape(rank_left, out_dims[-1], in_dims[-1], 1)
    cores.append(final_core.to(dtype=original_dtype))
    return cores


def mpo_to_matrix(cores: list[Tensor] | tuple[Tensor, ...]) -> Tensor:
    """Reconstruct a dense matrix from MPO cores."""
    if not cores:
        raise ValueError("At least one MPO core is required.")
    tensor = cores[0]
    for core in cores[1:]:
        tensor = torch.tensordot(tensor, core, dims=([-1], [0]))
    tensor = tensor.squeeze(0).squeeze(-1)
    order = len(cores)
    out_dims = [core.shape[1] for core in cores]
    in_dims = [core.shape[2] for core in cores]
    permutation = list(range(0, 2 * order, 2)) + list(range(1, 2 * order, 2))
    tensor = tensor.permute(*permutation).contiguous()
    return tensor.reshape(prod(out_dims), prod(in_dims))


@dataclass(frozen=True)
class MPOInfo:
    in_features: int
    out_features: int
    in_dims: tuple[int, ...]
    out_dims: tuple[int, ...]
    ranks: tuple[int, ...]
    dense_parameters: int
    tensorized_parameters: int

    @property
    def compression_ratio(self) -> float:
        if self.dense_parameters == 0:
            return 1.0
        return self.tensorized_parameters / self.dense_parameters


class MPOLinear(nn.Module):
    """A Linear replacement parameterized as a Matrix Product Operator."""

    def __init__(
        self,
        cores: list[Tensor] | tuple[Tensor, ...],
        bias: Tensor | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        super().__init__()
        if not cores:
            raise ValueError("MPOLinear requires at least one core.")
        self.cores = nn.ParameterList([nn.Parameter(core.contiguous()) for core in cores])
        self.out_dims = tuple(int(core.shape[1]) for core in cores)
        self.in_dims = tuple(int(core.shape[2]) for core in cores)
        self.in_features = in_features or prod(self.in_dims)
        self.out_features = out_features or prod(self.out_dims)
        validate_factors("out", self.out_dims, self.out_features)
        validate_factors("in", self.in_dims, self.in_features)
        if bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(bias.detach().clone())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        max_rank: int,
        order: int = 4,
        out_factors: tuple[int, ...] | None = None,
        in_factors: tuple[int, ...] | None = None,
        relative_tolerance: float = 0.0,
        svd_dtype: torch.dtype = torch.float32,
    ) -> "MPOLinear":
        out_dims, in_dims = infer_mpo_factors(
            linear.out_features,
            linear.in_features,
            order=order,
            out_factors=out_factors,
            in_factors=in_factors,
        )
        cores = decompose_matrix_to_mpo(
            linear.weight.data,
            out_dims=out_dims,
            in_dims=in_dims,
            max_rank=max_rank,
            relative_tolerance=relative_tolerance,
            svd_dtype=svd_dtype,
        )
        bias = linear.bias.data if linear.bias is not None else None
        return cls(cores=cores, bias=bias, in_features=linear.in_features, out_features=linear.out_features)

    @classmethod
    def from_state(
        cls,
        cores: list[Tensor],
        bias: Tensor | None,
        in_features: int,
        out_features: int,
    ) -> "MPOLinear":
        return cls(cores=cores, bias=bias, in_features=in_features, out_features=out_features)

    def forward(self, input: Tensor) -> Tensor:
        if input.shape[-1] != self.in_features:
            raise ValueError(f"Expected last dim {self.in_features}, got {input.shape[-1]}.")
        original_batch_shape = input.shape[:-1]
        result = input.reshape(-1, *self.in_dims)

        for index, core in enumerate(self.cores):
            if index == 0:
                first_core = core[0]  # [out_dim, in_dim, rank_right]
                result = torch.tensordot(result, first_core, dims=([1], [1]))
            else:
                result = torch.tensordot(result, core, dims=([1, result.ndim - 1], [2, 0]))

        result = result.squeeze(-1).reshape(*original_batch_shape, self.out_features)
        if self.bias is not None:
            result = result + self.bias
        return result

    def dense_weight(self) -> Tensor:
        return mpo_to_matrix(list(self.cores))

    def to_linear(self) -> nn.Linear:
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data.copy_(self.dense_weight().to(dtype=linear.weight.dtype))
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data.to(dtype=linear.bias.dtype))
        return linear

    def mpo_info(self) -> MPOInfo:
        ranks = [int(self.cores[0].shape[0])]
        ranks.extend(int(core.shape[-1]) for core in self.cores)
        tensorized = sum(parameter.numel() for parameter in self.parameters())
        dense = self.in_features * self.out_features
        if self.bias is not None:
            dense += self.out_features
        return MPOInfo(
            in_features=self.in_features,
            out_features=self.out_features,
            in_dims=self.in_dims,
            out_dims=self.out_dims,
            ranks=tuple(ranks),
            dense_parameters=dense,
            tensorized_parameters=tensorized,
        )

    def extra_repr(self) -> str:
        info = self.mpo_info()
        rank_text = "x".join(str(rank) for rank in info.ranks)
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"order={len(self.cores)}, ranks={rank_text}, bias={self.bias is not None}, "
            f"param_ratio={info.compression_ratio:.4f}"
        )


def dense_linear_equivalent(module: nn.Module) -> nn.Linear:
    if isinstance(module, MPOLinear):
        return module.to_linear()
    if isinstance(module, nn.Linear):
        return module
    raise TypeError(f"Expected Linear or MPOLinear, got {type(module)!r}.")


def linear_forward_dense(module: nn.Module, input: Tensor) -> Tensor:
    linear = dense_linear_equivalent(module)
    return F.linear(input, linear.weight, linear.bias)

