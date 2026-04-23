from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net._architectural_common import (
    apply_delta,
    identity_state_activation,
    init_linear,
    make_pairwise,
    make_route,
    signed_abs_softmax_edges,
    signed_softmax_state,
    unit_normalize_values,
)
from jakal_net.core import Layer, LayerDelta
from jakal_net.propagation import SparsePropagation
from jakal_net.transition import SparseTransition


@dataclass(frozen=True, slots=True)
class BScanOutput:
    query_layer: Layer
    memory_state: tuple[Layer, ...]
    bridge_layer: Layer | None = None
    knowledge_state: Layer | None = None
    knowledge_output: Any | None = None


class _MemoryLevel(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_slots: int,
        pairwise_kind: str,
        route_kind: str,
        pairwise_rank: int,
        route_rank: int,
        pairwise_heads: int,
        route_heads: int,
        pairwise_frozen_heads: int,
        route_frozen_heads: int,
        pairwise_anchor_heads: int,
        route_anchor_heads: int,
        pairwise_anchor_kind: str,
        route_anchor_kind: str,
        memory_topk: int,
        implementation: str,
        unit_norm_values: bool,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.unit_norm_values = unit_norm_values
        self.init_state = nn.Parameter(torch.empty(num_slots))
        self.init_val = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.init_state, mean=0.0, std=0.02)
        nn.init.normal_(self.init_val, mean=0.0, std=0.02)
        self.write = SparseTransition(
            route_fn=make_route(
                route_kind,
                dim=dim,
                rank=route_rank,
                heads=route_heads,
                frozen_heads=route_frozen_heads,
                anchor_heads=route_anchor_heads,
                anchor_kind=route_anchor_kind,
            ),
            topk=min(memory_topk, num_slots),
            state_activation_fn=F.softplus,
            route_compress_name="signed_abs_softmax",
            implementation=implementation,
            merge_mode="add",
        )
        self.propagation = SparsePropagation(
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
            topk=min(memory_topk, num_slots),
            edge_compress_fn=signed_abs_softmax_edges,
            state_weight_edges=True,
            implementation=implementation,
            residual=True,
        )
        self.val_norm = nn.LayerNorm(dim)

    def initialize(
        self,
        *,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> Layer:
        state = self.init_state.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        val = self.init_val.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        val = self.val_norm(val.clone())
        if self.unit_norm_values:
            val = unit_normalize_values(val)
        return Layer(
            dim=self.dim,
            num_nodes=self.num_slots,
            state=signed_softmax_state(state.clone()),
            val=val,
        )


class BModule(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        memory_slots: Sequence[int],
        memory_update_intervals: Sequence[int] | None = None,
        memory_topk: int,
        pairwise_kind: str,
        route_kind: str,
        pairwise_rank: int,
        route_rank: int,
        pairwise_heads: int = 1,
        route_heads: int = 1,
        pairwise_frozen_heads: int = 0,
        route_frozen_heads: int = 0,
        pairwise_anchor_heads: int = 0,
        route_anchor_heads: int = 0,
        pairwise_anchor_kind: str = "scaled_cosine",
        route_anchor_kind: str = "fixed_projection",
        implementation: str,
        unit_norm_values: bool = False,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if not memory_slots:
            raise ValueError("memory_slots must contain at least one level.")
        if any(slots <= 0 for slots in memory_slots):
            raise ValueError("memory_slots must be positive.")
        if memory_update_intervals is None:
            memory_update_intervals = tuple(2**level_index for level_index in range(len(memory_slots)))
        if len(memory_update_intervals) != len(memory_slots):
            raise ValueError("memory_update_intervals must match memory_slots length.")
        if any(interval <= 0 for interval in memory_update_intervals):
            raise ValueError("memory_update_intervals must be positive.")
        if memory_topk <= 0:
            raise ValueError("memory_topk must be positive.")

        self.dim = dim
        self.memory_slots = tuple(int(slots) for slots in memory_slots)
        self.memory_update_intervals = tuple(int(interval) for interval in memory_update_intervals)
        self.num_memory_levels = len(self.memory_slots)
        self.unit_norm_values = unit_norm_values

        self.memory_levels = nn.ModuleList(
            _MemoryLevel(
                dim=dim,
                num_slots=slots,
                pairwise_kind=pairwise_kind,
                route_kind=route_kind,
                pairwise_rank=pairwise_rank,
                route_rank=route_rank,
                pairwise_heads=pairwise_heads,
                route_heads=route_heads,
                pairwise_frozen_heads=pairwise_frozen_heads,
                route_frozen_heads=route_frozen_heads,
                pairwise_anchor_heads=pairwise_anchor_heads,
                route_anchor_heads=route_anchor_heads,
                pairwise_anchor_kind=pairwise_anchor_kind,
                route_anchor_kind=route_anchor_kind,
                memory_topk=memory_topk,
                implementation=implementation,
                unit_norm_values=unit_norm_values,
            )
            for slots in self.memory_slots
        )
        self.level_transitions = nn.ModuleList(
            SparseTransition(
                route_fn=make_route(
                    route_kind,
                    dim=dim,
                    rank=route_rank,
                    heads=route_heads,
                    frozen_heads=route_frozen_heads,
                    anchor_heads=route_anchor_heads,
                    anchor_kind=route_anchor_kind,
                ),
                topk=min(memory_topk, self.memory_slots[index + 1]),
                state_activation_fn=F.softplus,
                route_compress_name="signed_abs_softmax",
                implementation=implementation,
                merge_mode="add",
            )
            for index in range(len(self.memory_slots) - 1)
        )
        self.level_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in self.memory_slots)

        self.skip_transitions = nn.ModuleDict()
        self.skip_gates = nn.ParameterDict()
        if len(self.memory_slots) >= 2:
            self.skip_transitions["token_to_1"] = SparseTransition(
                route_fn=make_route(
                    route_kind,
                    dim=dim,
                    rank=route_rank,
                    heads=route_heads,
                    frozen_heads=route_frozen_heads,
                    anchor_heads=route_anchor_heads,
                    anchor_kind=route_anchor_kind,
                ),
                topk=min(memory_topk, self.memory_slots[1]),
                state_activation_fn=F.softplus,
                route_compress_name="signed_abs_softmax",
                implementation=implementation,
                merge_mode="add",
            )
            self.skip_gates["token_to_1"] = nn.Parameter(torch.tensor(-1.5))
        for level_index in range(2, len(self.memory_slots)):
            key = f"{level_index - 2}_to_{level_index}"
            self.skip_transitions[key] = SparseTransition(
                route_fn=make_route(
                    route_kind,
                    dim=dim,
                    rank=route_rank,
                    heads=route_heads,
                    frozen_heads=route_frozen_heads,
                    anchor_heads=route_anchor_heads,
                    anchor_kind=route_anchor_kind,
                ),
                topk=min(memory_topk, self.memory_slots[level_index]),
                state_activation_fn=F.softplus,
                route_compress_name="signed_abs_softmax",
                implementation=implementation,
                merge_mode="add",
            )
            self.skip_gates[key] = nn.Parameter(torch.tensor(-1.5))

        self.read_template_val = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.read_template_val, mean=0.0, std=0.02)
        self.read_projections = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in self.memory_slots)
        self.read_gates = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in self.memory_slots)
        self.bridge_input_norm = nn.LayerNorm(dim)
        self.bridge_to_levels = nn.ModuleList(
            SparseTransition(
                route_fn=make_route(
                    route_kind,
                    dim=dim,
                    rank=route_rank,
                    heads=route_heads,
                    frozen_heads=route_frozen_heads,
                    anchor_heads=route_anchor_heads,
                    anchor_kind=route_anchor_kind,
                ),
                topk=min(memory_topk, slots),
                state_activation_fn=F.softplus,
                route_compress_name="signed_abs_softmax",
                implementation=implementation,
                merge_mode="add",
            )
            for slots in self.memory_slots
        )

    def initialize_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[Layer, ...]:
        return tuple(
            level.initialize(batch_size=batch_size, device=device, dtype=dtype)
            for level in self.memory_levels
        )

    def constrain_layer(self, level_index: int, layer: Layer) -> Layer:
        level_module = self.memory_levels[level_index]
        state = signed_softmax_state(layer.state)
        val = level_module.val_norm(layer.val)
        if self.unit_norm_values:
            val = unit_normalize_values(val)
        return layer.with_tensors(state=state, val=val)

    def constrain_memory_state(self, memory_state: Sequence[Layer]) -> tuple[Layer, ...]:
        return tuple(
            self.constrain_layer(level_index, layer)
            for level_index, layer in enumerate(memory_state)
        )

    def reset_state(
        self,
        memory_state: Sequence[Layer],
        *,
        reset_mask: Tensor | None,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> tuple[Layer, ...]:
        if reset_mask is None:
            return tuple(memory_state)
        if reset_mask.ndim != 1:
            raise ValueError("reset_mask must have shape [batch].")
        batch_size = reset_mask.shape[0]
        if not torch.any(reset_mask):
            return tuple(memory_state)
        fresh_state = self.initialize_state(batch_size, device=device, dtype=dtype)
        mask = reset_mask.to(device=device, dtype=torch.bool).view(batch_size, 1)
        reset_layers: list[Layer] = []
        for current, fresh in zip(memory_state, fresh_state):
            state = torch.where(mask, fresh.state, current.state)
            val = torch.where(mask.unsqueeze(-1), fresh.val, current.val)
            reset_layers.append(current.with_tensors(state=state, val=val))
        return self.constrain_memory_state(tuple(reset_layers))

    def update(
        self,
        token_layer: Layer,
        memory_state: Sequence[Layer],
        *,
        time_index: int,
    ) -> tuple[Layer, ...]:
        next_levels: list[Layer] = []
        first_level_module = self.memory_levels[0]
        level = memory_state[0]
        level = apply_delta(
            level,
            first_level_module.write.compute_delta(token_layer, level),
            residual=True,
            val_norm=first_level_module.val_norm,
            unit_norm_values=self.unit_norm_values,
        )
        level = apply_delta(
            level,
            first_level_module.propagation.compute_delta(level),
            residual=True,
            val_norm=first_level_module.val_norm,
            unit_norm_values=self.unit_norm_values,
        )
        next_levels.append(level)

        for level_index in range(1, self.num_memory_levels):
            level_module = self.memory_levels[level_index]
            level = memory_state[level_index]
            if (time_index + 1) % self.memory_update_intervals[level_index] != 0:
                next_levels.append(level)
                continue
            parent = next_levels[level_index - 1]
            level = apply_delta(
                level,
                self.level_transitions[level_index - 1].compute_delta(
                    parent,
                    level,
                ),
                residual=True,
                val_norm=level_module.val_norm,
                unit_norm_values=self.unit_norm_values,
            )
            if level_index == 1 and "token_to_1" in self.skip_transitions:
                gate = torch.sigmoid(self.skip_gates["token_to_1"])
                skip_delta = self.skip_transitions["token_to_1"].compute_delta(
                    token_layer,
                    level,
                )
                level = apply_delta(
                    level,
                    LayerDelta(
                        delta_state=skip_delta.delta_state * gate,
                        delta_val=skip_delta.delta_val * gate,
                    ),
                    residual=True,
                    val_norm=level_module.val_norm,
                    unit_norm_values=self.unit_norm_values,
                )
            key = f"{level_index - 2}_to_{level_index}"
            if key in self.skip_transitions:
                gate = torch.sigmoid(self.skip_gates[key])
                skip_delta = self.skip_transitions[key].compute_delta(
                    next_levels[level_index - 2],
                    level,
                )
                level = apply_delta(
                    level,
                    LayerDelta(
                        delta_state=skip_delta.delta_state * gate,
                        delta_val=skip_delta.delta_val * gate,
                    ),
                    residual=True,
                    val_norm=level_module.val_norm,
                    unit_norm_values=self.unit_norm_values,
                )
            level = apply_delta(
                level,
                level_module.propagation.compute_delta(level),
                residual=True,
                val_norm=level_module.val_norm,
                unit_norm_values=self.unit_norm_values,
            )
            next_levels.append(level)
        return tuple(next_levels)

    def read(self, memory_state: Sequence[Layer]) -> Tensor:
        read_terms: list[Tensor] = []
        for level, level_module, projection, gate in zip(
            memory_state,
            self.memory_levels,
            self.read_projections,
            self.read_gates,
        ):
            sender_strength = F.softplus(level.state).unsqueeze(-1)
            read_summary = (sender_strength * level.val).sum(dim=-2)
            read_summary = read_summary + self.read_template_val.to(
                device=level.val.device,
                dtype=level.val.dtype,
            ).unsqueeze(0)
            read_terms.append(torch.sigmoid(gate) * projection(read_summary))
        return torch.stack(read_terms, dim=0).sum(dim=0)

    def flatten_memory_state(self, memory_state: Sequence[Layer]) -> tuple[Tensor, ...]:
        flat: list[Tensor] = []
        for layer in memory_state:
            flat.extend((layer.state, layer.val))
        return tuple(flat)

    def unflatten_memory_state(self, flat_tensors: Sequence[Tensor]) -> tuple[Layer, ...]:
        if len(flat_tensors) != self.num_memory_levels * 2:
            raise ValueError("flat_tensors does not match the configured memory hierarchy.")
        levels: list[Layer] = []
        for level_index, num_slots in enumerate(self.memory_slots):
            state = flat_tensors[level_index * 2]
            val = flat_tensors[level_index * 2 + 1]
            levels.append(Layer(dim=self.dim, num_nodes=num_slots, state=state, val=val))
        return tuple(levels)

    def build_bridge_layer(
        self,
        memory_state: Sequence[Layer],
        *,
        state_projection: nn.Module,
        bridge_val: Tensor | None = None,
    ) -> Layer:
        summary = self.read(memory_state) if bridge_val is None else bridge_val
        bridge_val_tensor = self.bridge_input_norm(summary).unsqueeze(-2)
        if self.unit_norm_values:
            bridge_val_tensor = unit_normalize_values(bridge_val_tensor)
        bridge_state = state_projection(bridge_val_tensor).squeeze(-1)
        return Layer(dim=self.dim, num_nodes=1, state=bridge_state, val=bridge_val_tensor)

    def inject_bridge(
        self,
        bridge_layer: Layer,
        memory_state: Sequence[Layer],
    ) -> tuple[Layer, ...]:
        next_levels: list[Layer] = []
        for level, level_module, transition in zip(
            memory_state,
            self.memory_levels,
            self.bridge_to_levels,
        ):
            delta = transition.compute_delta(bridge_layer, level)
            next_levels.append(
                apply_delta(
                    level,
                    delta,
                    residual=True,
                    val_norm=level_module.val_norm,
                    unit_norm_values=self.unit_norm_values,
                )
            )
        return tuple(next_levels)

    def scan(
        self,
        aligned_s: Tensor,
        memory_state: Sequence[Layer],
        *,
        state_projection: nn.Module,
        query_projection: nn.Module,
        query_input_norm: nn.Module,
        knowledge_module: nn.Module | None = None,
        knowledge_state: Layer | None = None,
        knowledge_reset_mask: Tensor | None = None,
    ) -> BScanOutput:
        batch_size, seq_len, _ = aligned_s.shape
        current_memory = tuple(memory_state)
        projected_s = query_projection(aligned_s)
        query_steps: list[Tensor] = []
        latest_bridge_layer: Layer | None = None
        latest_knowledge_output: Any | None = None

        for time_index in range(seq_len):
            token_val = aligned_s.narrow(1, time_index, 1)
            token_layer = Layer(
                dim=self.dim,
                num_nodes=1,
                state=state_projection(token_val).squeeze(-1),
                val=token_val,
            )
            current_memory = self.update(token_layer, current_memory, time_index=time_index)
            bridge_layer = self.build_bridge_layer(
                current_memory,
                state_projection=state_projection,
            )
            if knowledge_module is not None:
                knowledge_output = knowledge_module(
                    bridge_layer,
                    k_layer=knowledge_state,
                    reset_mask=knowledge_reset_mask if time_index == 0 else None,
                    update_b=True,
                )
                knowledge_state = knowledge_output.k_layer
                latest_knowledge_output = knowledge_output
                if knowledge_output.updated_b_layer is None:
                    raise ValueError(
                        "knowledge_module must return updated_b_layer when update_b=True."
                    )
                current_memory = self.inject_bridge(
                    knowledge_output.updated_b_layer,
                    current_memory,
                )
                bridge_layer = self.build_bridge_layer(
                    current_memory,
                    state_projection=state_projection,
                )
            latest_bridge_layer = bridge_layer
            read_vector = self.read(current_memory)
            query_step = query_input_norm(projected_s[:, time_index, :] + read_vector)
            if self.unit_norm_values:
                query_step = unit_normalize_values(query_step)
            query_steps.append(query_step)

        query_val = torch.stack(query_steps, dim=1)
        query_state = state_projection(query_val).squeeze(-1)
        return BScanOutput(
            query_layer=Layer(dim=self.dim, num_nodes=seq_len, state=query_state, val=query_val),
            memory_state=current_memory,
            bridge_layer=latest_bridge_layer,
            knowledge_state=knowledge_state,
            knowledge_output=latest_knowledge_output,
        )

    def reset_projection_parameters(self) -> None:
        nn.init.normal_(self.read_template_val, mean=0.0, std=0.02)
        for projection in self.read_projections:
            init_linear(projection)
