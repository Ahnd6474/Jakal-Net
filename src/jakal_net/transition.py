from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net.core import (
    Layer,
    LayerDelta,
    MergeMode,
    apply_optional_layer_fn,
    validate_merge_mode,
    validate_projected_state,
    validate_route_logits,
)


def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
    masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked_logits = torch.where(mask, logits, masked_logits)
    return torch.softmax(masked_logits, dim=dim)


class Transition(nn.Module):
    def __init__(
        self,
        route_fn: Callable[[Tensor], Tensor] | nn.Module,
        *,
        norm_fn: Callable[[Layer], Layer] | None = None,
        state_activation_fn: Callable[[Tensor], Tensor] = F.softplus,
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        merge_mode: MergeMode = "add",
    ) -> None:
        super().__init__()
        validate_merge_mode(merge_mode)
        self.route_fn = route_fn
        self.norm_fn = norm_fn
        self.state_activation_fn = state_activation_fn
        self.val_proj_fn = nn.Identity() if val_proj_fn is None else val_proj_fn
        self.state_proj_fn = nn.Identity() if state_proj_fn is None else state_proj_fn
        self.merge_mode = merge_mode

    def compute_route_logits(self, src_layer: Layer, dst_layer: Layer) -> Tensor:
        logits = self.route_fn(src_layer.val)
        validate_route_logits(logits, src_layer, dst_layer)
        return logits

    def compute_routes(self, src_layer: Layer, dst_layer: Layer) -> Tensor:
        logits = self.compute_route_logits(src_layer, dst_layer)
        return torch.softmax(logits, dim=-1)

    def compute_delta(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        routes = self.compute_routes(src_layer, dst_layer)
        projected_val = self.val_proj_fn(src_layer.val)
        projected_state = self.state_proj_fn(src_layer.state)

        if tuple(projected_val.shape) != (*src_layer.state.shape, dst_layer.dim):
            raise ValueError(
                "val_proj_fn must return [..., src_nodes, dst_dim], "
                f"expected {(*src_layer.state.shape, dst_layer.dim)}, "
                f"got {tuple(projected_val.shape)}."
            )
        validate_projected_state(projected_state, src_layer)

        sender_strength = self.state_activation_fn(src_layer.state)
        if sender_strength.shape != src_layer.state.shape:
            raise ValueError(
                "state_activation_fn must preserve the src_layer.state shape."
            )
        weighted_routes = routes * sender_strength.unsqueeze(-1)

        delta_state = torch.einsum("...jk,...j->...k", weighted_routes, projected_state)
        delta_val = torch.einsum("...jk,...jd->...kd", weighted_routes, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def forward(self, src_layer: Layer, dst_layer: Layer) -> Layer:
        delta = self.compute_delta(src_layer, dst_layer)
        updated = dst_layer.apply_delta(delta, merge_mode=self.merge_mode)
        return apply_optional_layer_fn(updated, self.norm_fn)


class SparseTransition(Transition):
    def __init__(
        self,
        route_fn: Callable[[Tensor], Tensor] | nn.Module,
        *,
        topk: int,
        norm_fn: Callable[[Layer], Layer] | None = None,
        state_activation_fn: Callable[[Tensor], Tensor] = F.softplus,
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        merge_mode: MergeMode = "add",
    ) -> None:
        super().__init__(
            route_fn=route_fn,
            norm_fn=norm_fn,
            state_activation_fn=state_activation_fn,
            val_proj_fn=val_proj_fn,
            state_proj_fn=state_proj_fn,
            merge_mode=merge_mode,
        )
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.topk = topk

    def compute_routes(self, src_layer: Layer, dst_layer: Layer) -> Tensor:
        logits = self.compute_route_logits(src_layer, dst_layer)
        k = min(self.topk, dst_layer.num_nodes)
        if k == dst_layer.num_nodes:
            return torch.softmax(logits, dim=-1)

        indices = logits.topk(k=k, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(-1, indices, True)
        return _masked_softmax(logits, mask, dim=-1)
