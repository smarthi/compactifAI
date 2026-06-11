from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import prod
from operator import mul

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _prime_factors(n: int) -> list[int]:
    """Return the prime factors of a positive integer.

    MPO tensorization needs matrix dimensions split into smaller tensor legs,
    and prime factors are the raw material for that split. Use this only through
    ``balanced_factors`` unless a caller needs the exact factor list.
    """
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
    """Split a dimension into a near-balanced product with a fixed order.

    Balanced tensor legs avoid extremely skinny MPO cores, which can make SVD
    truncation less useful and forward contractions less regular. Use this when
    no hand-tuned input/output factorization is supplied for a linear layer.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    buckets = [1] * order
    for factor in sorted(_prime_factors(n), reverse=True):
        index = min(range(order), key=lambda i: buckets[i])
        buckets[index] *= factor
    return tuple(sorted(buckets))


def validate_factors(name: str, factors: tuple[int, ...], expected: int) -> None:
    """Validate that tensor leg factors multiply to the original dimension.

    A wrong factorization silently corrupts reshape and contraction semantics, so
    every conversion and load path checks dimensions up front. Use it whenever
    user-provided or serialized MPO factors are accepted.
    """
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
    """Infer or validate the input and output MPO factors for a linear weight.

    The CompactifAI-style decomposition reshapes a matrix into paired output and
    input tensor legs before sequential SVD. Use this helper from conversion code
    to combine explicit factors with automatic balanced defaults.
    """
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
    """Choose the retained SVD rank for one MPO split.

    This implements the compression knob: cap rank at ``max_rank`` and, when a
    tolerance is set, keep enough singular-value energy to respect it. Use it
    inside sequential SVD decomposition rather than applying ad hoc truncation.
    """
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
    """Reshape a dense linear weight into alternating output/input tensor legs.

    Sequential SVD expects each MPO site to contain one output and one input
    factor. Use this just before decomposition to turn ``[out, in]`` into
    ``[out_0, in_0, out_1, in_1, ...]``.
    """
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
    This is the main compression primitive used by model conversion: it trades a
    dense matrix for trainable tensor cores controlled by ``max_rank``. Use it
    directly for experiments or through ``MPOLinear.from_linear`` for model
    surgery.
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
    """Reconstruct a dense matrix from MPO cores.

    Dense reconstruction is useful for tests, diagnostics, export checks, and
    comparison against the original layer. Use it sparingly for large models
    because it materializes the full uncompressed matrix.
    """
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
    """Metadata describing one tensorized linear layer.

    Conversion reports and adapter metadata need the original shape, tensor leg
    shapes, ranks, and parameter counts in a serializable form. Use this object
    when emitting summaries or comparing compression settings across layers.
    """

    in_features: int
    out_features: int
    in_dims: tuple[int, ...]
    out_dims: tuple[int, ...]
    ranks: tuple[int, ...]
    dense_parameters: int
    tensorized_parameters: int

    @property
    def compression_ratio(self) -> float:
        """Return tensorized parameters divided by dense parameters.

        The ratio is the headline compression indicator for a layer, and a value
        below one means the MPO representation is smaller. Use it in reports and
        selection logic when comparing candidate ranks or modules.
        """
        if self.dense_parameters == 0:
            return 1.0
        return self.tensorized_parameters / self.dense_parameters


class MPOLinear(nn.Module):
    """A ``torch.nn.Linear`` replacement parameterized as an MPO.

    The module lets a tensorized layer participate in normal PyTorch forward and
    training flows while storing trainable MPO cores instead of a dense weight.
    Use it to replace selected attention and MLP projections during compression,
    healing, and benchmark inference.
    """

    def __init__(
        self,
        cores: list[Tensor] | tuple[Tensor, ...],
        bias: Tensor | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
    ) -> None:
        """Create an MPO linear layer from already-decomposed cores.

        Adapter loading and tests need to rebuild modules without rerunning SVD.
        Pass cores in ``[rank_left, out_dim, in_dim, rank_right]`` form plus an
        optional bias and, when useful, explicit dense feature counts.
        """
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
        """Tensorize an existing dense ``nn.Linear`` module.

        This is the standard model-surgery entry point: it infers tensor legs,
        decomposes the dense weight by sequential SVD, and carries over the bias.
        Use it inside conversion tools whenever a selected dense layer should be
        replaced by an MPO layer.
        """
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
        """Rebuild an MPO layer from serialized tensors.

        Saved adapters store raw cores and bias tensors rather than Python module
        objects. Use this in checkpoint loading to reconstruct the exact layer
        structure before inserting it into the base model.
        """
        return cls(cores=cores, bias=bias, in_features=in_features, out_features=out_features)

    def forward(self, input: Tensor) -> Tensor:
        """Apply the MPO linear projection to an input tensor.

        The method contracts input legs with MPO cores without materializing the
        dense weight, which is what makes the compressed representation usable at
        inference and during healing. Call it implicitly by running the model just
        like any other PyTorch module.
        """
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
        """Materialize the equivalent dense weight matrix.

        This is needed for reconstruction tests, debugging, and optional export
        paths that expect a conventional linear layer. Use it carefully on large
        layers because it allocates the full dense matrix.
        """
        return mpo_to_matrix(list(self.cores))

    def to_linear(self) -> nn.Linear:
        """Convert this MPO layer back into a dense ``nn.Linear`` module.

        Dense conversion is useful for compatibility checks or tooling that does
        not understand MPO modules. Use it only when memory permits, since it
        discards the runtime storage advantage of tensorization.
        """
        linear = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        linear.weight.data.copy_(self.dense_weight().to(dtype=linear.weight.dtype))
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.data.to(dtype=linear.bias.dtype))
        return linear

    def mpo_info(self) -> MPOInfo:
        """Return shape, rank, and parameter-count metadata for this layer.

        Reports and adapters need a compact description of what was compressed
        and how much was saved. Use this after conversion or loading to audit a
        tensorized model.
        """
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
        """Return a readable module summary for ``print(model)``.

        PyTorch calls this method when displaying module trees, so including rank
        and compression information makes model inspection useful. Use the
        printed output as a quick sanity check after replacement.
        """
        info = self.mpo_info()
        rank_text = "x".join(str(rank) for rank in info.ranks)
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"order={len(self.cores)}, ranks={rank_text}, bias={self.bias is not None}, "
            f"param_ratio={info.compression_ratio:.4f}"
        )


def dense_linear_equivalent(module: nn.Module) -> nn.Linear:
    """Return a dense ``nn.Linear`` view of a dense or MPO linear module.

    Diagnostics sometimes need to compare modules through a common dense API.
    Use this helper when code accepts either representation and must fall back to
    ordinary linear algebra.
    """
    if isinstance(module, MPOLinear):
        return module.to_linear()
    if isinstance(module, nn.Linear):
        return module
    raise TypeError(f"Expected Linear or MPOLinear, got {type(module)!r}.")


def linear_forward_dense(module: nn.Module, input: Tensor) -> Tensor:
    """Run a dense linear forward pass for either dense or MPO modules.

    This helper gives tests and debugging code a shared reference path by first
    converting MPO layers to dense form. Use it for correctness checks, not for
    large-scale benchmark inference.
    """
    linear = dense_linear_equivalent(module)
    return F.linear(input, linear.weight, linear.bias)
