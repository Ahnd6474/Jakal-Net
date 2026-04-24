from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net._architectural_common import (
    LOW_RANK_SCALE_INIT,
    PARAM_INIT_STD,
    apply_delta,
    init_linear,
    init_pairwise_or_route_scales,
    layer_with_val_norm,
    make_pairwise,
    make_route,
    signed_abs_softmax_edges,
    signed_softmax_state,
)
from jakal_net.core import Layer, LayerDelta
from jakal_net.propagation import SparsePropagation
from jakal_net.transition import SparseTransition


@dataclass(frozen=True, slots=True)
class KModuleOutput:
    routed_k_layer: Layer
    propagated_k_layer: Layer
    b_delta: LayerDelta
    updated_b_layer: Layer | None = None

    @property
    def k_layer(self) -> Layer:
        return self.propagated_k_layer


class KModule(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_nodes: int,
        route_kind: str = "low_rank_bilinear",
        pairwise_kind: str = "low_rank_bilinear",
        route_rank: int = 64,
        pairwise_rank: int = 64,
        route_heads: int = 1,
        pairwise_heads: int = 1,
        route_frozen_heads: int = 0,
        pairwise_frozen_heads: int = 0,
        route_anchor_heads: int = 0,
        pairwise_anchor_heads: int = 0,
        route_anchor_kind: str = "fixed_projection",
        pairwise_anchor_kind: str = "scaled_cosine",
        route_topk: int = 16,
        propagation_topk: int = 16,
        propagation_layers: int = 1,
        route_compress_name: str = "signed_abs_softmax",
        propagation_residual: bool = True,
        implementation: str = "streaming",
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive.")
        if propagation_layers <= 0:
            raise ValueError("propagation_layers must be positive.")
        if route_topk <= 0:
            raise ValueError("route_topk must be positive.")
        if propagation_topk <= 0:
            raise ValueError("propagation_topk must be positive.")

        self.dim = dim
        self.num_nodes = num_nodes
        self.route_kind = route_kind
        self.pairwise_kind = pairwise_kind
        self.route_rank = route_rank
        self.pairwise_rank = pairwise_rank
        self.route_heads = route_heads
        self.pairwise_heads = pairwise_heads
        self.route_topk = route_topk
        self.propagation_topk = propagation_topk
        self.propagation_layers_count = propagation_layers
        self.route_compress_name = route_compress_name
        self.propagation_residual = propagation_residual
        self.implementation = implementation

        self.init_state = nn.Parameter(torch.empty(num_nodes))
        self.init_val = nn.Parameter(torch.empty(num_nodes, dim))

        self.b_to_k = SparseTransition(
            route_fn=make_route(
                route_kind,
                dim=dim,
                rank=route_rank,
                heads=route_heads,
                frozen_heads=route_frozen_heads,
                anchor_heads=route_anchor_heads,
                anchor_kind=route_anchor_kind,
            ),
            topk=min(route_topk, num_nodes),
            state_activation_fn=F.softplus,
            route_compress_name=route_compress_name,
            implementation=implementation,
            merge_mode="add",
            use_direction_only=True,
        )
        self.k_to_b = SparseTransition(
            route_fn=make_route(
                route_kind,
                dim=dim,
                rank=route_rank,
                heads=route_heads,
                frozen_heads=route_frozen_heads,
                anchor_heads=route_anchor_heads,
                anchor_kind=route_anchor_kind,
            ),
            topk=max(1, route_topk),
            state_activation_fn=F.softplus,
            route_compress_name=route_compress_name,
            implementation=implementation,
            merge_mode="add",
            use_direction_only=True,
        )
        self.propagation_layers = nn.ModuleList(
            SparsePropagation(
                pairwise_fn=make_pairwise(
                pairwise_kind,
                    dim=dim,
                    rank=pairwise_rank,
                    heads=pairwise_heads,
                    frozen_heads=pairwise_frozen_heads,
                    anchor_heads=pairwise_anchor_heads,
                    anchor_kind=pairwise_anchor_kind,
                ),
                sparse_type="topk",
                topk=min(propagation_topk, num_nodes),
                edge_compress_fn=signed_abs_softmax_edges,
                state_weight_edges=False,
                implementation=implementation,
                residual=propagation_residual,
                use_direction_only=True,
            )
            for _ in range(propagation_layers)
        )

        self.k_val_norm = nn.LayerNorm(dim)
        self.b_read_norm = nn.LayerNorm(dim)
        self.propagation_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(propagation_layers))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.init_state, mean=0.0, std=PARAM_INIT_STD)
        nn.init.normal_(self.init_val, mean=0.0, std=PARAM_INIT_STD)
        for module in self.modules():
            init_pairwise_or_route_scales(module)

    def initialize_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Layer:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        state = self.init_state.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        val = self.init_val.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        return Layer(
            dim=self.dim,
            num_nodes=self.num_nodes,
            state=signed_softmax_state(state.clone()),
            val=self.k_val_norm(val.clone()),
        )

    def reset_state(
        self,
        k_layer: Layer,
        *,
        reset_mask: Tensor | None,
    ) -> Layer:
        if reset_mask is None:
            return k_layer
        if reset_mask.ndim != 1:
            raise ValueError("reset_mask must have shape [batch].")
        if reset_mask.shape[0] != k_layer.state.shape[0]:
            raise ValueError("reset_mask batch size must match k_layer batch size.")
        if not torch.any(reset_mask):
            return k_layer
        fresh = self.initialize_state(
            reset_mask.shape[0],
            device=k_layer.val.device,
            dtype=k_layer.val.dtype,
        )
        mask = reset_mask.to(device=k_layer.val.device, dtype=torch.bool).view(-1, 1)
        state = torch.where(mask, fresh.state, k_layer.state)
        val = torch.where(mask.unsqueeze(-1), fresh.val, k_layer.val)
        return k_layer.with_tensors(state=state, val=val)

    def route_from_b(
        self,
        b_layer: Layer,
        k_layer: Layer,
    ) -> Layer:
        if b_layer.dim != self.dim or k_layer.dim != self.dim:
            raise ValueError("B and K layers must match KModule.dim.")
        normalized_k = layer_with_val_norm(k_layer, self.k_val_norm)
        delta = self.b_to_k.compute_delta(b_layer, normalized_k)
        return apply_delta(k_layer, delta, residual=True, val_norm=self.k_val_norm)

    def propagate(
        self,
        k_layer: Layer,
        *,
        steps: int | None = None,
    ) -> Layer:
        step_count = self.propagation_layers_count if steps is None else steps
        if step_count < 0:
            raise ValueError("steps must be non-negative.")
        layer = k_layer
        for step_index in range(step_count):
            propagation = self.propagation_layers[step_index % len(self.propagation_layers)]
            norm = self.propagation_norms[step_index % len(self.propagation_norms)]
            delta = propagation.compute_delta(layer_with_val_norm(layer, norm))
            layer = apply_delta(
                layer,
                delta,
                residual=self.propagation_residual,
                val_norm=self.k_val_norm,
            )
        return layer

    def transition_to_b(
        self,
        k_layer: Layer,
        b_layer: Layer,
        *,
        update_b: bool = False,
    ) -> LayerDelta | Layer:
        normalized_k = layer_with_val_norm(k_layer, self.k_val_norm)
        normalized_b = layer_with_val_norm(b_layer, self.b_read_norm)
        delta = self.k_to_b.compute_delta(normalized_k, normalized_b)
        if not update_b:
            return delta
        return apply_delta(b_layer, delta, residual=True, val_norm=self.b_read_norm)

    def forward(
        self,
        b_layer: Layer,
        *,
        k_layer: Layer | None = None,
        reset_mask: Tensor | None = None,
        propagation_steps: int | None = None,
        update_b: bool = False,
    ) -> KModuleOutput:
        if k_layer is None:
            k_layer = self.initialize_state(
                b_layer.state.shape[0],
                device=b_layer.val.device,
                dtype=b_layer.val.dtype,
            )
        k_layer = self.reset_state(k_layer, reset_mask=reset_mask)
        routed_k_layer = self.route_from_b(b_layer, k_layer)
        propagated_k_layer = self.propagate(routed_k_layer, steps=propagation_steps)
        b_delta = self.transition_to_b(propagated_k_layer, b_layer, update_b=False)
        updated_b_layer = None
        if update_b:
            updated_b_layer = apply_delta(
                b_layer,
                b_delta,
                residual=True,
                val_norm=self.b_read_norm,
            )
        return KModuleOutput(
            routed_k_layer=routed_k_layer,
            propagated_k_layer=propagated_k_layer,
            b_delta=b_delta,
            updated_b_layer=updated_b_layer,
        )
