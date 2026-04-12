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
