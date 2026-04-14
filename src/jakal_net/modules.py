from __future__ import annotations

import torch
from torch import Tensor, nn


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


class HadamardMLPPairwise(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        width = dim if hidden_dim is None else hidden_dim
        self.proj_in = nn.Linear(dim, width)
        self.proj_out = nn.Linear(width, 1)
        self.activation = nn.SiLU()

    def forward(self, target_val: Tensor, source_val: Tensor) -> Tensor:
        interaction = target_val.unsqueeze(-2) * source_val.unsqueeze(-3)
        hidden = self.activation(self.proj_in(interaction))
        return self.proj_out(hidden).squeeze(-1)


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
        projected_source = self.source_proj(source_val).unsqueeze(-2)
        projected_target = self.target_proj(target_val).unsqueeze(-3)
        interaction = projected_source * projected_target
        source_features = projected_source.expand_as(interaction)
        target_features = projected_target.expand_as(interaction)
        features = torch.cat(
            (source_features, target_features, interaction),
            dim=-1,
        )
        hidden = self.activation(self.proj_in(features))
        return self.proj_out(hidden).squeeze(-1)


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
