from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from jakal_net.core import Layer, LayerDelta
from jakal_net.modules import (
    AdditiveLowRankPairwise,
    AdditiveLowRankRoute,
    BilinearPairwise,
    BilinearPairwiseRoute,
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    LearnedPositionEncoding,
    LowRankBilinearPairwise,
    LowRankBilinearRoute,
)
from jakal_net.propagation import SparsePropagation
from jakal_net.transition import SparseTransition

_STATE_MASS = 4.0
_PARAM_INIT_STD = 0.02
_LOW_RANK_SCALE_INIT = 0.1


def _make_pairwise(
    kind: str,
    *,
    dim: int,
    rank: int,
) -> nn.Module:
    if kind == "low_rank_bilinear":
        return LowRankBilinearPairwise(dim=dim, rank=rank)
    if kind == "diagonal_bilinear":
        return DiagonalBilinearPairwise(dim=dim)
    if kind == "bilinear":
        return BilinearPairwise(dim=dim)
    if kind == "additive_low_rank":
        return AdditiveLowRankPairwise(dim=dim, rank=rank)
    raise ValueError(f"Unsupported pairwise kind: {kind!r}.")


def _make_route(
    kind: str,
    *,
    dim: int,
    rank: int,
) -> nn.Module:
    if kind == "low_rank_bilinear":
        return LowRankBilinearRoute(src_dim=dim, dst_dim=dim, rank=rank)
    if kind == "diagonal_bilinear":
        return DiagonalBilinearRoute(src_dim=dim, dst_dim=dim)
    if kind == "bilinear":
        return BilinearPairwiseRoute(src_dim=dim, dst_dim=dim, route_dim=dim)
    if kind == "additive_low_rank":
        return AdditiveLowRankRoute(src_dim=dim, dst_dim=dim, route_dim=rank)
    raise ValueError(f"Unsupported route kind: {kind!r}.")


def _layer_with_val_norm(layer: Layer, norm: nn.LayerNorm) -> Layer:
    return layer.with_tensors(val=norm(layer.val))


def _signed_softmax_state(state: Tensor) -> Tensor:
    clean_state = torch.nan_to_num(state)
    magnitude = torch.softmax(clean_state.abs(), dim=-1)
    return torch.sign(clean_state) * magnitude * _STATE_MASS


def _signed_abs_softmax_edges(scores: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    return torch.sign(clean_scores) * torch.softmax(clean_scores.abs(), dim=-1)


def _identity_state_activation(state: Tensor) -> Tensor:
    return state


def _apply_delta(
    layer: Layer,
    delta: LayerDelta,
    *,
    residual: bool = True,
    val_norm: nn.LayerNorm | None = None,
) -> Layer:
    updated = layer.apply_delta(delta, merge_mode="add" if residual else "replace")
    state = _signed_softmax_state(updated.state)
    val = updated.val if val_norm is None else val_norm(updated.val)
    return updated.with_tensors(state=state, val=val)


def _clone_layer(layer: Layer) -> Layer:
    return Layer(
        dim=layer.dim,
        num_nodes=layer.num_nodes,
        state=layer.state.clone(),
        val=layer.val.clone(),
    )


def _init_linear(linear: nn.Linear, *, std: float = _PARAM_INIT_STD) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_pairwise_or_route_scales(module: nn.Module) -> None:
    if isinstance(module, (LowRankBilinearPairwise, LowRankBilinearRoute)):
        module.weight.data.fill_(_LOW_RANK_SCALE_INIT)
        _init_linear(module.source_proj)
        _init_linear(module.target_proj)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, (DiagonalBilinearPairwise, DiagonalBilinearRoute)):
        module.weight.data.fill_(_LOW_RANK_SCALE_INIT)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, BilinearPairwise):
        nn.init.normal_(module.weight, mean=0.0, std=_PARAM_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, BilinearPairwiseRoute):
        _init_linear(module.source_proj)
        _init_linear(module.target_proj)
        nn.init.normal_(module.weight, mean=0.0, std=_PARAM_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, AdditiveLowRankPairwise):
        _init_linear(module.target_proj)
        _init_linear(module.source_proj)
        _init_linear(module.target_out)
        _init_linear(module.source_out)
        nn.init.normal_(module.interaction_weight, mean=0.0, std=_PARAM_INIT_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return
    if isinstance(module, AdditiveLowRankRoute):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                _init_linear(layer)


@dataclass(frozen=True, slots=True)
class MemoryScanOutput:
    logits: Tensor
    memory_state: tuple[Layer, ...]
    sequence_layer: Layer | None = None
    query_layer: Layer | None = None


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
        memory_topk: int,
        implementation: str,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.init_state = nn.Parameter(torch.empty(num_slots))
        self.init_val = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.normal_(self.init_state, mean=0.0, std=0.02)
        nn.init.normal_(self.init_val, mean=0.0, std=0.02)
        self.write = SparseTransition(
            route_fn=_make_route(route_kind, dim=dim, rank=route_rank),
            topk=min(memory_topk, num_slots),
            state_activation_fn=_identity_state_activation,
            route_compress_name="signed_abs_softmax",
            implementation=implementation,
            merge_mode="add",
        )
        self.propagation = SparsePropagation(
            pairwise_fn=_make_pairwise(pairwise_kind, dim=dim, rank=pairwise_rank),
            sparse_type="topk",
            topk=min(memory_topk, num_slots),
            edge_compress_fn=_signed_abs_softmax_edges,
            state_weight_edges=True,
            implementation=implementation,
            residual=True,
        )
        self.val_norm = nn.LayerNorm(dim)

    def initialize(
        self,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Layer:
        state = self.init_state.unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        val = self.init_val.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
        return Layer(
            dim=self.dim,
            num_nodes=self.num_slots,
            state=_signed_softmax_state(state.clone()),
            val=self.val_norm(val.clone()),
        )


class CausalHierarchicalMemoryLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int = 512,
        max_seq_len: int = 2048,
        s_layers: int = 2,
        memory_slots: Sequence[int] = (256, 64, 16),
        prediction_layers: int = 2,
        s_window: int | None = None,
        s_microbatch_size: int | None = None,
        prediction_window: int = 64,
        memory_topk: int = 16,
        pairwise_kind: str = "low_rank_bilinear",
        route_kind: str = "low_rank_bilinear",
        pairwise_rank: int = 64,
        route_rank: int = 64,
        implementation: str = "streaming",
        tie_embedding_head: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if s_layers <= 0:
            raise ValueError("s_layers must be positive.")
        if prediction_layers <= 0:
            raise ValueError("prediction_layers must be positive.")
        if s_microbatch_size is not None and s_microbatch_size <= 0:
            raise ValueError("s_microbatch_size must be positive when provided.")
        if not memory_slots:
            raise ValueError("memory_slots must contain at least one level.")
        if any(slots <= 0 for slots in memory_slots):
            raise ValueError("memory_slots must be positive.")
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.num_memory_levels = len(tuple(memory_slots))
        self.memory_slots = tuple(int(slots) for slots in memory_slots)
        self.prediction_window = prediction_window
        self.s_microbatch_size = s_microbatch_size

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_encoding = LearnedPositionEncoding(dim)
        self.anchor_state = nn.Parameter(torch.zeros(()))
        self.anchor_val = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.anchor_val, mean=0.0, std=0.02)

        self.value_to_state = nn.Linear(dim, 1)
        self.sequence_input_norm = nn.LayerNorm(dim)
        self.sequence_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(s_layers))
        self.sequence_layers = nn.ModuleList(
            SparsePropagation(
                pairwise_fn=_make_pairwise(pairwise_kind, dim=dim, rank=pairwise_rank),
                sparse_type="window",
                window=max_seq_len if s_window is None else max(1, s_window),
                edge_compress_fn=_signed_abs_softmax_edges,
                state_weight_edges=True,
                implementation=implementation,
                residual=True,
            )
            for _ in range(s_layers)
        )

        self.memory_levels = nn.ModuleList(
            _MemoryLevel(
                dim=dim,
                num_slots=slots,
                pairwise_kind=pairwise_kind,
                route_kind=route_kind,
                pairwise_rank=pairwise_rank,
                route_rank=route_rank,
                memory_topk=memory_topk,
                implementation=implementation,
            )
            for slots in self.memory_slots
        )
        self.level_transitions = nn.ModuleList(
            SparseTransition(
                route_fn=_make_route(route_kind, dim=dim, rank=route_rank),
                topk=min(memory_topk, self.memory_slots[index + 1]),
                state_activation_fn=_identity_state_activation,
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
                route_fn=_make_route(route_kind, dim=dim, rank=route_rank),
                topk=min(memory_topk, self.memory_slots[1]),
                state_activation_fn=_identity_state_activation,
                route_compress_name="signed_abs_softmax",
                implementation=implementation,
                merge_mode="add",
            )
            self.skip_gates["token_to_1"] = nn.Parameter(torch.tensor(-1.5))
        for level_index in range(2, len(self.memory_slots)):
            key = f"{level_index - 2}_to_{level_index}"
            self.skip_transitions[key] = SparseTransition(
                route_fn=_make_route(route_kind, dim=dim, rank=route_rank),
                topk=min(memory_topk, self.memory_slots[level_index]),
                state_activation_fn=_identity_state_activation,
                route_compress_name="signed_abs_softmax",
                implementation=implementation,
                merge_mode="add",
            )
            self.skip_gates[key] = nn.Parameter(torch.tensor(-1.5))

        self.read_template_val = nn.Parameter(torch.empty(dim))
        nn.init.normal_(self.read_template_val, mean=0.0, std=0.02)
        self.read_projections = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in self.memory_slots)
        self.read_gates = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in self.memory_slots)

        self.s_prediction_proj = nn.Linear(dim, dim, bias=False)
        self.prediction_input_norm = nn.LayerNorm(dim)
        self.prediction_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(prediction_layers))
        self.prediction_layers = nn.ModuleList(
            SparsePropagation(
                pairwise_fn=_make_pairwise(pairwise_kind, dim=dim, rank=pairwise_rank),
                sparse_type="window",
                window=max(1, prediction_window),
                edge_compress_fn=_signed_abs_softmax_edges,
                state_weight_edges=True,
                implementation=implementation,
                residual=True,
            )
            for _ in range(prediction_layers)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embedding_head:
            self.lm_head.weight = self.token_embedding.weight
        self._reset_parameters(tie_embedding_head=tie_embedding_head)

    def _reset_parameters(self, *, tie_embedding_head: bool) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=_PARAM_INIT_STD)
        nn.init.normal_(self.anchor_val, mean=0.0, std=_PARAM_INIT_STD)
        nn.init.normal_(self.read_template_val, mean=0.0, std=_PARAM_INIT_STD)
        nn.init.zeros_(self.anchor_state)
        _init_linear(self.value_to_state)
        _init_linear(self.s_prediction_proj)
        for projection in self.read_projections:
            _init_linear(projection)
        if not tie_embedding_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=_PARAM_INIT_STD)
        for module in self.modules():
            _init_pairwise_or_route_scales(module)

    def initialize_memory_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Layer, ...]:
        return tuple(
            level.initialize(batch_size=batch_size, device=device, dtype=dtype)
            for level in self.memory_levels
        )

    def _reset_memory_items(
        self,
        memory_state: Sequence[Layer],
        *,
        reset_mask: Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Layer, ...]:
        if reset_mask is None:
            return tuple(memory_state)
        if reset_mask.ndim != 1:
            raise ValueError("reset_mask must have shape [batch].")
        batch_size = reset_mask.shape[0]
        if not torch.any(reset_mask):
            return tuple(memory_state)
        fresh_state = self.initialize_memory_state(batch_size, device=device, dtype=dtype)
        mask = reset_mask.to(device=device, dtype=torch.bool).view(batch_size, 1)
        reset_layers: list[Layer] = []
        for current, fresh in zip(memory_state, fresh_state):
            state = torch.where(mask, fresh.state, current.state)
            val = torch.where(mask.unsqueeze(-1), fresh.val, current.val)
            reset_layers.append(current.with_tensors(state=state, val=val))
        return tuple(reset_layers)

    def _encode_sequence(self, input_ids: Tensor) -> Layer:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len].")
        batch_size, seq_len = input_ids.shape
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len={seq_len} exceeds configured max_seq_len={self.max_seq_len}."
            )
        token_val = self.token_embedding(input_ids)
        token_val = token_val + self.position_encoding(
            seq_len,
            device=token_val.device,
            dtype=token_val.dtype,
        ).unsqueeze(0)
        token_val = self.sequence_input_norm(token_val)

        anchor_val = self.anchor_val.expand(batch_size, 1, -1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        anchor_state = self.anchor_state.expand(batch_size, 1).to(
            device=token_val.device,
            dtype=token_val.dtype,
        )
        seq_val = torch.cat((anchor_val, token_val), dim=1)
        seq_state = torch.cat(
            (
                anchor_state,
                self.value_to_state(token_val).squeeze(-1),
            ),
            dim=1,
        )
        layer = Layer(dim=self.dim, num_nodes=seq_len + 1, state=seq_state, val=seq_val)
        for propagation, norm in zip(self.sequence_layers, self.sequence_norms):
            layer = _apply_delta(
                layer,
                propagation.compute_delta(_layer_with_val_norm(layer, norm)),
                residual=True,
                val_norm=norm,
            )
        return layer

    def _encode_sequence_microbatched(self, input_ids: Tensor) -> Layer:
        if self.s_microbatch_size is None or input_ids.shape[0] <= self.s_microbatch_size:
            return self._encode_sequence(input_ids)

        chunks: list[Layer] = []
        for start in range(0, input_ids.shape[0], self.s_microbatch_size):
            end = min(start + self.s_microbatch_size, input_ids.shape[0])
            chunks.append(self._encode_sequence(input_ids[start:end]))
        return Layer(
            dim=self.dim,
            num_nodes=chunks[0].num_nodes,
            state=torch.cat([chunk.state for chunk in chunks], dim=0),
            val=torch.cat([chunk.val for chunk in chunks], dim=0),
        )

    def _update_memory(
        self,
        token_layer: Layer,
        memory_state: Sequence[Layer],
    ) -> tuple[Layer, ...]:
        next_levels: list[Layer] = []
        first_level_module = self.memory_levels[0]
        level = memory_state[0]
        level = _apply_delta(
            level,
            first_level_module.write.compute_delta(
                token_layer,
                _layer_with_val_norm(level, first_level_module.val_norm),
            ),
            residual=True,
            val_norm=first_level_module.val_norm,
        )
        level = _apply_delta(
            level,
            first_level_module.propagation.compute_delta(
                _layer_with_val_norm(level, first_level_module.val_norm)
            ),
            residual=True,
            val_norm=first_level_module.val_norm,
        )
        next_levels.append(level)

        for level_index in range(1, self.num_memory_levels):
            level_module = self.memory_levels[level_index]
            level = memory_state[level_index]
            parent = next_levels[level_index - 1]
            normalized_level = _layer_with_val_norm(level, level_module.val_norm)
            normalized_parent = _layer_with_val_norm(parent, self.level_norms[level_index - 1])
            level = _apply_delta(
                level,
                self.level_transitions[level_index - 1].compute_delta(
                    normalized_parent,
                    normalized_level,
                ),
                residual=True,
                val_norm=level_module.val_norm,
            )
            if level_index == 1 and "token_to_1" in self.skip_transitions:
                gate = torch.sigmoid(self.skip_gates["token_to_1"])
                skip_delta = self.skip_transitions["token_to_1"].compute_delta(
                    token_layer,
                    normalized_level,
                )
                level = _apply_delta(
                    level,
                    LayerDelta(
                        delta_state=skip_delta.delta_state * gate,
                        delta_val=skip_delta.delta_val * gate,
                    ),
                    residual=True,
                    val_norm=level_module.val_norm,
                )
            key = f"{level_index - 2}_to_{level_index}"
            if key in self.skip_transitions:
                gate = torch.sigmoid(self.skip_gates[key])
                normalized_skip_source = _layer_with_val_norm(
                    next_levels[level_index - 2],
                    self.level_norms[level_index - 2],
                )
                skip_delta = self.skip_transitions[key].compute_delta(
                    normalized_skip_source,
                    normalized_level,
                )
                level = _apply_delta(
                    level,
                    LayerDelta(
                        delta_state=skip_delta.delta_state * gate,
                        delta_val=skip_delta.delta_val * gate,
                    ),
                    residual=True,
                    val_norm=level_module.val_norm,
                )
            level = _apply_delta(
                level,
                level_module.propagation.compute_delta(
                    _layer_with_val_norm(level, level_module.val_norm)
                ),
                residual=True,
                val_norm=level_module.val_norm,
            )
            next_levels.append(level)
        return tuple(next_levels)

    def _read_memory(self, memory_state: Sequence[Layer]) -> Tensor:
        read_terms: list[Tensor] = []
        for level, level_module, projection, gate in zip(
            memory_state,
            self.memory_levels,
            self.read_projections,
            self.read_gates,
        ):
            read_layer = _layer_with_val_norm(level, level_module.val_norm)
            sender_strength = read_layer.state.unsqueeze(-1)
            read_summary = (sender_strength * read_layer.val).sum(dim=-2)
            read_summary = read_summary + self.read_template_val.to(
                device=level.val.device,
                dtype=level.val.dtype,
            ).unsqueeze(0)
            read_terms.append(torch.sigmoid(gate) * projection(read_summary))
        return torch.stack(read_terms, dim=0).sum(dim=0)

    def _scan_memory_batch(
        self,
        aligned_s: Tensor,
        memory_state: Sequence[Layer],
    ) -> tuple[Layer, tuple[Layer, ...]]:
        batch_size, seq_len, _ = aligned_s.shape
        current_memory = tuple(memory_state)
        projected_s = self.s_prediction_proj(aligned_s)
        query_val = aligned_s.new_empty(batch_size, seq_len, self.dim)

        # The time axis remains recurrent, but every step updates every document stream
        # in the batch together.
        for time_index in range(seq_len):
            token_val = aligned_s.narrow(1, time_index, 1)
            token_layer = Layer(
                dim=self.dim,
                num_nodes=1,
                state=self.value_to_state(token_val).squeeze(-1),
                val=token_val,
            )
            current_memory = self._update_memory(token_layer, current_memory)
            read_vector = self._read_memory(current_memory)
            query_val[:, time_index, :] = self.prediction_input_norm(
                projected_s[:, time_index, :] + read_vector
            )

        query_state = self.value_to_state(query_val).squeeze(-1)
        return (
            Layer(dim=self.dim, num_nodes=seq_len, state=query_state, val=query_val),
            current_memory,
        )

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory_state: Sequence[Layer] | None = None,
        reset_mask: Tensor | None = None,
        return_memory_state: bool = False,
        return_layers: bool = False,
    ) -> Tensor | MemoryScanOutput:
        sequence_layer = self._encode_sequence_microbatched(input_ids)
        aligned_s = sequence_layer.val[:, 1:, :]
        batch_size, seq_len, _ = aligned_s.shape
        device = aligned_s.device
        dtype = aligned_s.dtype

        if memory_state is None:
            memory_state = self.initialize_memory_state(batch_size, device=device, dtype=dtype)
        if len(tuple(memory_state)) != self.num_memory_levels:
            raise ValueError("memory_state does not match the configured memory hierarchy.")
        current_memory = self._reset_memory_items(
            tuple(memory_state),
            reset_mask=reset_mask,
            device=device,
            dtype=dtype,
        )
        query_layer, current_memory = self._scan_memory_batch(aligned_s, current_memory)
        for propagation, norm in zip(self.prediction_layers, self.prediction_norms):
            query_layer = _apply_delta(
                query_layer,
                propagation.compute_delta(_layer_with_val_norm(query_layer, norm)),
                residual=True,
                val_norm=norm,
            )

        logits = self.lm_head(self.output_norm(query_layer.val))
        if not (return_memory_state or return_layers):
            return logits
        return MemoryScanOutput(
            logits=logits,
            memory_state=tuple(_clone_layer(layer) for layer in current_memory),
            sequence_layer=sequence_layer if return_layers else None,
            query_layer=query_layer if return_layers else None,
        )
