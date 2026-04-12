from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net.core import (
    Layer,
    LayerDelta,
    SparsePropagationType,
    apply_optional_layer_fn,
    validate_pairwise_scores,
    validate_projected_state,
    validate_projected_val,
)


def _causal_window_mask(
    num_nodes: int, window: int, *, device: torch.device | None = None
) -> Tensor:
    if window < 0:
        raise ValueError("window must be non-negative.")
    target_idx = torch.arange(num_nodes, device=device).unsqueeze(-1)
    source_idx = torch.arange(num_nodes, device=device).unsqueeze(0)
    return (source_idx <= target_idx) & (source_idx >= target_idx - window)


def _topk_mask(scores: Tensor, topk: int) -> Tensor:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    k = min(topk, scores.shape[-1])
    if k == scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    indices = scores.topk(k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


class Propagation(nn.Module):
    def __init__(
        self,
        pairwise_fn: Callable[[Tensor, Tensor], Tensor] | nn.Module,
        *,
        edge_compress_fn: Callable[[Tensor], Tensor] = F.softsign,
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        norm_fn: Callable[[Layer], Layer] | None = None,
        residual: bool = True,
        return_delta: bool = True,
    ) -> None:
        super().__init__()
        self.pairwise_fn = pairwise_fn
        self.edge_compress_fn = edge_compress_fn
        self.val_proj_fn = nn.Identity() if val_proj_fn is None else val_proj_fn
        self.state_proj_fn = nn.Identity() if state_proj_fn is None else state_proj_fn
        self.norm_fn = norm_fn
        self.residual = residual
        self.return_delta = return_delta

    def compute_scores(self, layer: Layer) -> Tensor:
        scores = self.pairwise_fn(layer.val, layer.val)
        validate_pairwise_scores(scores, layer)
        return scores

    def compute_edges(self, layer: Layer) -> Tensor:
        return self.edge_compress_fn(self.compute_scores(layer))

    def compute_delta(self, layer: Layer) -> LayerDelta:
        edges = self.compute_edges(layer)
        projected_state = self.state_proj_fn(layer.state)
        projected_val = self.val_proj_fn(layer.val)
        validate_projected_state(projected_state, layer)
        validate_projected_val(projected_val, layer)

        delta_state = torch.einsum("...ij,...j->...i", edges, projected_state)
        delta_val = torch.einsum("...ij,...jd->...id", edges, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def forward(self, layer: Layer) -> LayerDelta | Layer:
        delta = self.compute_delta(layer)
        if self.return_delta:
            return delta

        merge_mode = "add" if self.residual else "replace"
        updated = layer.apply_delta(delta, merge_mode=merge_mode)
        return apply_optional_layer_fn(updated, self.norm_fn)


class SparsePropagation(Propagation):
    def __init__(
        self,
        pairwise_fn: Callable[[Tensor, Tensor], Tensor] | nn.Module,
        *,
        sparse_type: SparsePropagationType,
        edge_compress_fn: Callable[[Tensor], Tensor] = F.softsign,
        topk: int | None = None,
        window: int | None = None,
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        norm_fn: Callable[[Layer], Layer] | None = None,
        residual: bool = True,
        return_delta: bool = True,
    ) -> None:
        super().__init__(
            pairwise_fn=pairwise_fn,
            edge_compress_fn=edge_compress_fn,
            val_proj_fn=val_proj_fn,
            state_proj_fn=state_proj_fn,
            norm_fn=norm_fn,
            residual=residual,
            return_delta=return_delta,
        )
        if sparse_type not in {"window", "topk"}:
            raise ValueError(f"Unsupported sparse_type: {sparse_type!r}.")
        if sparse_type == "window" and window is None:
            raise ValueError("window sparse propagation requires window.")
        if sparse_type == "topk" and topk is None:
            raise ValueError("topk sparse propagation requires topk.")
        self.sparse_type = sparse_type
        self.topk = topk
        self.window = window

    def compute_edges(self, layer: Layer) -> Tensor:
        scores = self.compute_scores(layer)
        edges = self.edge_compress_fn(scores)

        if self.sparse_type == "window":
            mask_2d = _causal_window_mask(
                layer.num_nodes, self.window or 0, device=scores.device
            )
            view_shape = (1,) * (scores.ndim - 2) + mask_2d.shape
            mask = mask_2d.view(view_shape)
        else:
            mask = _topk_mask(scores, self.topk or layer.num_nodes)

        return edges * mask.to(dtype=edges.dtype)
