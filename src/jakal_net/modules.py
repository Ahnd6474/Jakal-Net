from __future__ import annotations

import os

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _env_block_size(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


_HADAMARD_PAIRWISE_BLOCK_SIZE = _env_block_size("JAKAL_HADAMARD_PAIRWISE_BLOCK_SIZE", 8)
_HADAMARD_ROUTE_BLOCK_SIZE = _env_block_size("JAKAL_HADAMARD_ROUTE_BLOCK_SIZE", 8)
_ADDITIVE_PAIRWISE_BLOCK_SIZE = _env_block_size("JAKAL_ADDITIVE_PAIRWISE_BLOCK_SIZE", 128)
_ADDITIVE_ROUTE_BLOCK_SIZE = _env_block_size("JAKAL_ADDITIVE_ROUTE_BLOCK_SIZE", 128)


class ScalarAffine(nn.Module):
    def __init__(self, weight: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(weight))
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.weight + self.bias


class DiagonalBilinearPairwise(nn.Module):
    def __init__(self, dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        weighted_source = source_val * self.weight
        scores = torch.einsum("...id,...jd->...ij", target_val, weighted_source)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class BilinearPairwise(nn.Module):
    def __init__(self, dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        scores = torch.einsum("...id,df,...jf->...ij", target_val, self.weight, source_val)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class LowRankBilinearPairwise(nn.Module):
    def __init__(self, dim: int, rank: int, *, bias: bool = True) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.source_proj = nn.Linear(dim, rank, bias=False)
        self.target_proj = nn.Linear(dim, rank, bias=False)
        self.weight = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def effective_weight(self) -> Tensor:
        weighted_target = self.target_proj.weight.transpose(0, 1) * self.weight.view(1, -1)
        return weighted_target @ self.source_proj.weight

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        projected_target = self.target_proj(target_val)
        projected_source = self.source_proj(source_val) * self.weight
        scores = torch.einsum("...ir,...jr->...ij", projected_target, projected_source)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class ScaledCosinePairwise(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6, scale: float | None = None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.eps = float(eps)
        self.scale = float(dim**0.5 if scale is None else scale)
        self.register_buffer("eps_buffer", torch.tensor(self.eps))
        self.register_buffer("scale_buffer", torch.tensor(self.scale))

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        normalized_target = F.normalize(target_val, dim=-1, eps=self.eps)
        normalized_source = F.normalize(source_val, dim=-1, eps=self.eps)
        return torch.einsum("...id,...jd->...ij", normalized_target, normalized_source) * self.scale


class HadamardMLPPairwise(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = dim if hidden_dim is None else hidden_dim
        self.proj_in = nn.Linear(dim, width)
        self.proj_out = nn.Linear(width, 1)
        self.activation = nn.SiLU()

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        target_nodes = target_val.shape[-2]
        source_nodes = source_val.shape[-2]
        row_blocks: list[Tensor] = []
        for target_start in range(0, target_nodes, _HADAMARD_PAIRWISE_BLOCK_SIZE):
            target_end = min(target_start + _HADAMARD_PAIRWISE_BLOCK_SIZE, target_nodes)
            target_chunk = target_val[..., target_start:target_end, :]
            col_blocks: list[Tensor] = []
            for source_start in range(0, source_nodes, _HADAMARD_PAIRWISE_BLOCK_SIZE):
                source_end = min(source_start + _HADAMARD_PAIRWISE_BLOCK_SIZE, source_nodes)
                source_chunk = source_val[..., source_start:source_end, :]
                hidden = torch.einsum(
                    "...id,hd,...jd->...ijh",
                    target_chunk,
                    self.proj_in.weight,
                    source_chunk,
                )
                if self.proj_in.bias is not None:
                    hidden = hidden + self.proj_in.bias.view(
                        *((1,) * (hidden.ndim - 1)),
                        -1,
                    )
                hidden = self.activation(hidden)
                col_blocks.append(self.proj_out(hidden).squeeze(-1))
            row_blocks.append(torch.cat(col_blocks, dim=-1))
        return torch.cat(row_blocks, dim=-2)


class AdditiveLowRankPairwise(nn.Module):
    def __init__(self, dim: int, rank: int | None = None, *, bias: bool = True) -> None:
        super().__init__()
        width = dim if rank is None else rank
        if width <= 0:
            raise ValueError("rank must be positive.")
        self.target_proj = nn.Linear(dim, width, bias=False)
        self.source_proj = nn.Linear(dim, width, bias=False)
        self.target_out = nn.Linear(width, 1, bias=False)
        self.source_out = nn.Linear(width, 1, bias=False)
        self.interaction_weight = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.interaction_weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.zeros(())) if bias else None
        self.activation = nn.SiLU()

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        projected_target = self.target_proj(target_val)
        projected_source = self.source_proj(source_val)
        target_linear = self.target_out(projected_target).squeeze(-1)
        source_linear = self.source_out(projected_source).squeeze(-1)
        target_nodes = projected_target.shape[-2]
        source_nodes = projected_source.shape[-2]
        row_blocks: list[Tensor] = []
        for target_start in range(0, target_nodes, _ADDITIVE_PAIRWISE_BLOCK_SIZE):
            target_end = min(target_start + _ADDITIVE_PAIRWISE_BLOCK_SIZE, target_nodes)
            target_chunk = projected_target[..., target_start:target_end, :]
            target_term = target_linear[..., target_start:target_end].unsqueeze(-1)
            col_blocks: list[Tensor] = []
            for source_start in range(0, source_nodes, _ADDITIVE_PAIRWISE_BLOCK_SIZE):
                source_end = min(source_start + _ADDITIVE_PAIRWISE_BLOCK_SIZE, source_nodes)
                source_chunk = projected_source[..., source_start:source_end, :]
                source_term = source_linear[..., source_start:source_end].unsqueeze(-2)
                interaction = self.activation(
                    target_chunk.unsqueeze(-2) * source_chunk.unsqueeze(-3)
                )
                scores = torch.einsum("...ijr,r->...ij", interaction, self.interaction_weight)
                scores = scores + target_term + source_term
                if self.bias is not None:
                    scores = scores + self.bias
                col_blocks.append(scores)
            row_blocks.append(torch.cat(col_blocks, dim=-1))
        return torch.cat(row_blocks, dim=-2)


class DiagonalBilinearRoute(nn.Module):
    expects_pairwise_inputs = True

    def __init__(
        self,
        src_dim: int,
        dst_dim: int | None = None,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        target_dim = src_dim if dst_dim is None else dst_dim
        if target_dim != src_dim:
            raise ValueError("DiagonalBilinearRoute requires matching src/dst dimensions.")
        self.weight = nn.Parameter(torch.ones(src_dim))
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def forward(self, source_val: Tensor, target_val: Tensor) -> Tensor:
        projected_source = source_val * self.weight
        scores = torch.einsum("...id,...kd->...ik", projected_source, target_val)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class LowRankBilinearRoute(nn.Module):
    expects_pairwise_inputs = True

    def __init__(
        self,
        src_dim: int,
        dst_dim: int | None = None,
        *,
        rank: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        target_dim = src_dim if dst_dim is None else dst_dim
        self.source_proj = nn.Linear(src_dim, rank, bias=False)
        self.target_proj = nn.Linear(target_dim, rank, bias=False)
        self.weight = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def forward(self, source_val: Tensor, target_val: Tensor) -> Tensor:
        projected_source = self.source_proj(source_val) * self.weight
        projected_target = self.target_proj(target_val)
        scores = torch.einsum("...ir,...kr->...ik", projected_source, projected_target)
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class QueryNormalizedDotRoute(nn.Module):
    expects_pairwise_inputs = True

    def __init__(
        self,
        src_dim: int,
        dst_dim: int | None = None,
        *,
        eps: float = 1e-6,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        target_dim = src_dim if dst_dim is None else dst_dim
        if target_dim != src_dim:
            raise ValueError("QueryNormalizedDotRoute requires matching src/dst dimensions.")
        self.eps = float(eps)
        self.scale = float(scale)
        self.register_buffer("eps_buffer", torch.tensor(self.eps))
        self.register_buffer("scale_buffer", torch.tensor(self.scale))

    def forward(self, source_val: Tensor, target_val: Tensor) -> Tensor:
        numerators = torch.einsum("...id,...kd->...ik", source_val, target_val)
        denominators = source_val.square().sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return numerators / denominators * self.scale


class BilinearPairwiseRoute(nn.Module):
    expects_pairwise_inputs = True

    def __init__(
        self,
        src_dim: int,
        dst_dim: int | None = None,
        *,
        route_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        target_dim = src_dim if dst_dim is None else dst_dim
        width = max(src_dim, target_dim) if route_dim is None else route_dim
        self.source_proj = nn.Linear(src_dim, width, bias=False)
        self.target_proj = nn.Linear(target_dim, width, bias=False)
        self.weight = nn.Parameter(torch.empty(width, width))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def forward(self, source_val: Tensor, target_val: Tensor) -> Tensor:
        projected_source = self.source_proj(source_val)
        projected_target = self.target_proj(target_val)
        scores = torch.einsum(
            "...id,df,...kf->...ik",
            projected_source,
            self.weight,
            projected_target,
        )
        if self.bias is not None:
            scores = scores + self.bias
        return scores


class SourceTargetHadamardMLPRoute(nn.Module):
    expects_pairwise_inputs = True

    def __init__(
        self,
        src_dim: int,
        dst_dim: int | None = None,
        *,
        route_dim: int | None = None,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        target_dim = src_dim if dst_dim is None else dst_dim
        width = max(src_dim, target_dim) if route_dim is None else route_dim
        hidden = width if hidden_dim is None else hidden_dim
        self.source_proj = nn.Linear(src_dim, width)
        self.target_proj = nn.Linear(target_dim, width)
        self.proj_in = nn.Linear(width * 3, hidden)
        self.proj_out = nn.Linear(hidden, 1)
        self.activation = nn.SiLU()

    def forward(self, source_val: Tensor, target_val: Tensor) -> Tensor:
        projected_source = self.source_proj(source_val)
        projected_target = self.target_proj(target_val)
        width = projected_source.shape[-1]
        source_weight, target_weight, hadamard_weight = torch.split(
            self.proj_in.weight,
            width,
            dim=-1,
        )
        source_linear = F.linear(projected_source, source_weight)
        target_linear = F.linear(projected_target, target_weight, self.proj_in.bias)
        source_nodes = projected_source.shape[-2]
        target_nodes = projected_target.shape[-2]
        row_blocks: list[Tensor] = []
        for source_start in range(0, source_nodes, _HADAMARD_ROUTE_BLOCK_SIZE):
            source_end = min(source_start + _HADAMARD_ROUTE_BLOCK_SIZE, source_nodes)
            source_chunk = projected_source[..., source_start:source_end, :]
            source_linear_chunk = source_linear[..., source_start:source_end, :].unsqueeze(-2)
            col_blocks: list[Tensor] = []
            for target_start in range(0, target_nodes, _HADAMARD_ROUTE_BLOCK_SIZE):
                target_end = min(target_start + _HADAMARD_ROUTE_BLOCK_SIZE, target_nodes)
                target_chunk = projected_target[..., target_start:target_end, :]
                target_linear_chunk = target_linear[..., target_start:target_end, :].unsqueeze(-3)
                hidden = torch.einsum(
                    "...id,hd,...kd->...ikh",
                    source_chunk,
                    hadamard_weight,
                    target_chunk,
                )
                hidden = hidden + source_linear_chunk
                hidden = hidden + target_linear_chunk
                hidden = self.activation(hidden)
                col_blocks.append(self.proj_out(hidden).squeeze(-1))
            row_blocks.append(torch.cat(col_blocks, dim=-1))
        return torch.cat(row_blocks, dim=-2)


class AdditiveLowRankRoute(nn.Module):
    expects_pairwise_inputs = True

    def __init__(
        self,
        src_dim: int,
        dst_dim: int | None = None,
        *,
        route_dim: int | None = None,
        hidden_dim: int | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        target_dim = src_dim if dst_dim is None else dst_dim
        width = (
            hidden_dim
            if hidden_dim is not None
            else (route_dim if route_dim is not None else max(src_dim, target_dim))
        )
        if width <= 0:
            raise ValueError("hidden_dim/rank must be positive.")
        self.source_proj = nn.Linear(src_dim, width, bias=False)
        self.target_proj = nn.Linear(target_dim, width, bias=False)
        self.source_out = nn.Linear(width, 1, bias=False)
        self.target_out = nn.Linear(width, 1, bias=False)
        self.interaction_weight = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.interaction_weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.zeros(())) if bias else None
        self.activation = nn.SiLU()

    def forward(self, source_val: Tensor, target_val: Tensor) -> Tensor:
        projected_source = self.source_proj(source_val)
        projected_target = self.target_proj(target_val)
        source_linear = self.source_out(projected_source).squeeze(-1)
        target_linear = self.target_out(projected_target).squeeze(-1)
        source_nodes = projected_source.shape[-2]
        target_nodes = projected_target.shape[-2]
        row_blocks: list[Tensor] = []
        for source_start in range(0, source_nodes, _ADDITIVE_ROUTE_BLOCK_SIZE):
            source_end = min(source_start + _ADDITIVE_ROUTE_BLOCK_SIZE, source_nodes)
            source_chunk = projected_source[..., source_start:source_end, :]
            source_term = source_linear[..., source_start:source_end].unsqueeze(-1)
            col_blocks: list[Tensor] = []
            for target_start in range(0, target_nodes, _ADDITIVE_ROUTE_BLOCK_SIZE):
                target_end = min(target_start + _ADDITIVE_ROUTE_BLOCK_SIZE, target_nodes)
                target_chunk = projected_target[..., target_start:target_end, :]
                target_term = target_linear[..., target_start:target_end].unsqueeze(-2)
                interaction = self.activation(
                    source_chunk.unsqueeze(-2) * target_chunk.unsqueeze(-3)
                )
                scores = torch.einsum("...ikr,r->...ik", interaction, self.interaction_weight)
                scores = scores + source_term + target_term
                if self.bias is not None:
                    scores = scores + self.bias
                col_blocks.append(scores)
            row_blocks.append(torch.cat(col_blocks, dim=-1))
        return torch.cat(row_blocks, dim=-2)


class LinearRoute(nn.Module):
    def __init__(self, src_dim: int, dst_nodes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(src_dim, dst_nodes)

    def forward(self, src_val: Tensor) -> Tensor:
        return self.linear(src_val)


class MLPRoute(nn.Module):
    def __init__(self, src_dim: int, dst_nodes: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(src_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dst_nodes),
        )

    def forward(self, src_val: Tensor) -> Tensor:
        return self.net(src_val)


class LearnedPositionEncoding(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = max(dim, 8) if hidden_dim is None else hidden_dim
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, dim),
        )

    def forward(
        self,
        num_nodes: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        base_dtype = self.net[0].weight.dtype
        if num_nodes == 1:
            positions = torch.zeros(1, 1, device=device, dtype=base_dtype)
        else:
            positions = torch.linspace(
                0.0,
                1.0,
                steps=num_nodes,
                device=device,
                dtype=base_dtype,
            ).unsqueeze(-1)
        encoding = self.net(positions)
        if dtype is not None:
            encoding = encoding.to(dtype=dtype)
        return encoding
