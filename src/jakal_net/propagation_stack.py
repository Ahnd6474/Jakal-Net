from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from jakal_net._architectural_common import apply_delta, signed_softmax_state, softsign_state
from jakal_net.core import Layer, LayerDelta
from jakal_net.modules import ResidualFeedForward, StateValueFeedForward
from jakal_net.propagation import SparsePropagation


def make_propagation_ffn(
    *,
    dim: int,
    enabled: bool,
    kind: str,
    hidden_mult: float,
    residual_scale: float,
    learnable_residual_scale: bool,
    zero_init_output: bool,
    activation: str,
) -> nn.Module:
    if not enabled:
        return nn.Identity()
    if kind == "state_val":
        return StateValueFeedForward(
            dim,
            hidden_mult=hidden_mult,
            residual_scale=residual_scale,
            learnable_residual_scale=learnable_residual_scale,
            zero_init_output=zero_init_output,
            activation=activation,
        )
    return ResidualFeedForward(
        dim,
        hidden_mult=hidden_mult,
        residual_scale=residual_scale,
        learnable_residual_scale=learnable_residual_scale,
        activation=activation,
    )


def can_use_dense_apply_fastpath(layer: Layer, propagation: SparsePropagation) -> bool:
    if propagation.sparse_type != "window":
        return False
    return int(propagation.window or 0) + 1 >= int(layer.num_nodes)


def apply_dense_delta_fastpath(
    layer: Layer,
    delta_state: Tensor,
    delta_val: Tensor,
    norm: nn.Module,
    *,
    unit_norm_values: bool,
) -> Layer:
    updated_val = layer.val + delta_val
    val = norm(updated_val)
    touched = delta_val.detach().abs().amax(dim=-1) > 0
    val = torch.where(touched.unsqueeze(-1), val, updated_val)
    return layer.with_tensors(
        state=softsign_state(layer.state + delta_state)
        if unit_norm_values
        else signed_softmax_state(layer.state + delta_state),
        val=val,
    )


def apply_propagation_ffn(
    layer: Layer,
    ffn: nn.Module,
    *,
    unit_norm_values: bool,
) -> Layer:
    if isinstance(ffn, StateValueFeedForward):
        state, val = ffn(layer.state, layer.val)
        if unit_norm_values:
            state = softsign_state(state)
        return layer.with_tensors(state=state, val=val)
    val = ffn(layer.val)
    return layer.with_tensors(val=val)


class PropagationLayer(nn.Module):
    def __init__(
        self,
        *,
        propagation: SparsePropagation,
        norm: nn.Module,
        ffn: nn.Module,
        unit_norm_values: bool,
        residual_gate_init: float,
    ) -> None:
        super().__init__()
        self.propagation = propagation
        self.norm = norm
        self.ffn = ffn
        self.unit_norm_values = bool(unit_norm_values)
        self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))

    def _run_layer(self, layer: Layer) -> Layer:
        delta = self.propagation.compute_delta(layer)
        scaled_delta = LayerDelta(
            delta_state=delta.delta_state
            * self.residual_gate.to(device=delta.delta_state.device, dtype=delta.delta_state.dtype),
            delta_val=delta.delta_val
            * self.residual_gate.to(device=delta.delta_val.device, dtype=delta.delta_val.dtype),
        )
        if can_use_dense_apply_fastpath(layer, self.propagation):
            next_layer = apply_dense_delta_fastpath(
                layer,
                scaled_delta.delta_state,
                scaled_delta.delta_val,
                self.norm,
                unit_norm_values=self.unit_norm_values,
            )
        else:
            next_layer = apply_delta(
                layer,
                scaled_delta,
                residual=True,
                val_norm=self.norm,
                unit_norm_values=self.unit_norm_values,
            )
        return apply_propagation_ffn(
            next_layer,
            self.ffn,
            unit_norm_values=self.unit_norm_values,
        )

    def forward(self, layer: Layer, *, checkpoint_layer: bool = False) -> Layer:
        if checkpoint_layer and torch.is_grad_enabled():
            num_nodes = layer.num_nodes

            def _run(state: Tensor, val: Tensor) -> tuple[Tensor, Tensor]:
                current_layer = Layer(dim=layer.dim, num_nodes=num_nodes, state=state, val=val)
                next_layer = self._run_layer(current_layer)
                return next_layer.state, next_layer.val

            next_state, next_val = torch_checkpoint(
                _run,
                layer.state,
                layer.val,
                use_reentrant=False,
            )
            return Layer(dim=layer.dim, num_nodes=num_nodes, state=next_state, val=next_val)
        return self._run_layer(layer)


class PropagationStack(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        propagation_factory: Callable[[], SparsePropagation],
        norm_factory: Callable[[], nn.Module],
        ffn_factory: Callable[[], nn.Module],
        unit_norm_values: bool,
        residual_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive.")
        self.blocks = nn.ModuleList(
            PropagationLayer(
                propagation=propagation_factory(),
                norm=norm_factory(),
                ffn=ffn_factory(),
                unit_norm_values=unit_norm_values,
                residual_gate_init=residual_gate_init,
            )
            for _ in range(depth)
        )

    @property
    def propagations(self) -> tuple[SparsePropagation, ...]:
        return tuple(block.propagation for block in self.blocks)

    @property
    def norms(self) -> tuple[nn.Module, ...]:
        return tuple(block.norm for block in self.blocks)

    @property
    def ffns(self) -> tuple[nn.Module, ...]:
        return tuple(block.ffn for block in self.blocks)

    def forward_range(
        self,
        layer: Layer,
        *,
        start: int = 0,
        end: int | None = None,
        checkpoint_layers: bool = False,
    ) -> Layer:
        blocks = self.blocks[start:end]
        for block in blocks:
            layer = block(layer, checkpoint_layer=checkpoint_layers)
        return layer

    def forward(self, layer: Layer, *, checkpoint_layers: bool = False) -> Layer:
        return self.forward_range(layer, checkpoint_layers=checkpoint_layers)
