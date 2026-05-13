from __future__ import annotations

from typing import TypeVar

import torch
from torch import Tensor, nn

from jakal_net._architectural_common import (
    make_pairwise,
    signed_abs_softmax_edges,
    softsign_state,
)
from jakal_net.core import Layer
from jakal_net.modules import LearnedPositionEncoding, ResidualFeedForward, StateValueFeedForward
from jakal_net.propagation import SparsePropagation
from jakal_net.propagation_stack import (
    PropagationStack,
    apply_dense_delta_fastpath,
    apply_propagation_ffn,
    can_use_dense_apply_fastpath,
    make_propagation_ffn,
)

TModule = TypeVar("TModule", bound=nn.Module)


class SModule(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        max_seq_len: int,
        s_layers: int,
        additional_layers: int = 0,
        pairwise_kind: str,
        pairwise_rank: int,
        pairwise_heads: int = 1,
        pairwise_frozen_heads: int = 0,
        pairwise_anchor_heads: int = 0,
        pairwise_anchor_kind: str = "scaled_cosine",
        pairwise_head_aggregate: str = "signed_smoothmax",
        sequence_anchor: bool = True,
        implementation: str,
        s_window: int | None = None,
        s_microbatch_size: int | None = None,
        checkpoint_sequence_layers: bool = False,
        unit_norm_values: bool = False,
        propagation_residual_gate_init: float = 0.1,
        feed_forward_layers: bool = True,
        feed_forward_hidden_mult: float = 2.0,
        feed_forward_kind: str = "value",
        feed_forward_residual_scale: float = 1.0,
        feed_forward_learnable_residual_scale: bool = False,
        feed_forward_zero_init_output: bool = True,
        feed_forward_activation: str = "gelu",
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
        if additional_layers < 0:
            raise ValueError("additional_layers must be non-negative.")
        if s_microbatch_size is not None and s_microbatch_size <= 0:
            raise ValueError("s_microbatch_size must be positive when provided.")
        if feed_forward_hidden_mult <= 0.0:
            raise ValueError("feed_forward_hidden_mult must be positive.")
        if feed_forward_kind not in {"value", "state_val"}:
            raise ValueError(f"Unsupported feed_forward_kind: {feed_forward_kind!r}.")
        if feed_forward_residual_scale < 0.0:
            raise ValueError("feed_forward_residual_scale must be non-negative.")
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.s_layers = int(s_layers)
        self.additional_layers = int(additional_layers)
        self.s_microbatch_size = s_microbatch_size
        self.checkpoint_sequence_layers = checkpoint_sequence_layers
        self.unit_norm_values = unit_norm_values
        self.propagation_residual_gate_init = float(propagation_residual_gate_init)
        self.feed_forward_layers = bool(feed_forward_layers)
        self.feed_forward_hidden_mult = float(feed_forward_hidden_mult)
        self.feed_forward_kind = feed_forward_kind
        self.feed_forward_residual_scale = float(feed_forward_residual_scale)
        self.feed_forward_learnable_residual_scale = bool(feed_forward_learnable_residual_scale)
        self.feed_forward_zero_init_output = bool(feed_forward_zero_init_output)
        self.feed_forward_activation = feed_forward_activation
        self.sequence_anchor = bool(sequence_anchor)

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_encoding = LearnedPositionEncoding(dim)
        if self.sequence_anchor:
            self.anchor_state = nn.Parameter(torch.zeros(()))
            self.anchor_val = nn.Parameter(torch.empty(dim))
            nn.init.normal_(self.anchor_val, mean=0.0, std=0.02)
        else:
            self.register_parameter("anchor_state", None)
            self.register_parameter("anchor_val", None)

        if unit_norm_values:
            self.sequence_input_norm = nn.Identity()
        else:
            self.sequence_input_norm = nn.LayerNorm(dim)
        sequence_norm_factory = nn.Identity if unit_norm_values else lambda: nn.LayerNorm(dim)
        full_dense_causal = s_window is None or int(s_window) <= 0
        sequence_window = max_seq_len if full_dense_causal else max(1, int(s_window))
        sequence_nodes = max_seq_len + (1 if self.sequence_anchor else 0)
        full_window_block_size = sequence_nodes if sequence_window + 1 >= sequence_nodes else None
        self.sequence_stack = PropagationStack(
            depth=self.total_layers,
            propagation_factory=lambda: SparsePropagation(
                pairwise_fn=make_pairwise(
                    pairwise_kind,
                    dim=dim,
                    rank=pairwise_rank,
                    heads=pairwise_heads,
                    frozen_heads=pairwise_frozen_heads,
                    anchor_heads=pairwise_anchor_heads,
                    anchor_kind=pairwise_anchor_kind,
                    aggregate=pairwise_head_aggregate,
                ),
                sparse_type="window",
                window=sequence_window,
                edge_compress_fn=signed_abs_softmax_edges,
                state_weight_edges=True,
                implementation=implementation,
                residual=True,
                target_block_size=full_window_block_size,
                source_block_size=full_window_block_size,
                use_direction_only=unit_norm_values,
            ),
            norm_factory=sequence_norm_factory,
            ffn_factory=lambda: make_propagation_ffn(
                dim=self.dim,
                enabled=self.feed_forward_layers,
                kind=self.feed_forward_kind,
                hidden_mult=self.feed_forward_hidden_mult,
                residual_scale=self.feed_forward_residual_scale,
                learnable_residual_scale=self.feed_forward_learnable_residual_scale,
                zero_init_output=self.feed_forward_zero_init_output,
                activation=self.feed_forward_activation,
            ),
            unit_norm_values=self.unit_norm_values,
            residual_gate_init=self.propagation_residual_gate_init,
        )
        self.full_dense_causal = bool(full_dense_causal)

    def _can_use_dense_apply_fastpath(self, layer: Layer, propagation: SparsePropagation) -> bool:
        return can_use_dense_apply_fastpath(layer, propagation)

    def _apply_dense_delta_fastpath(
        self,
        layer: Layer,
        delta_state: Tensor,
        delta_val: Tensor,
        norm: nn.Module,
    ) -> Layer:
        return apply_dense_delta_fastpath(
            layer,
            delta_state,
            delta_val,
            norm,
            unit_norm_values=self.unit_norm_values,
        )

    def _apply_ffn(self, layer: Layer, ffn: nn.Module) -> Layer:
        return apply_propagation_ffn(layer, ffn, unit_norm_values=self.unit_norm_values)

    @property
    def total_layers(self) -> int:
        return self.s_layers + self.additional_layers

    def _layer_slice(
        self,
        modules: tuple[TModule, ...],
        *,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[TModule, ...]:
        return modules[start:end]

    @property
    def propagation_layers(self) -> tuple[SparsePropagation, ...]:
        return self.sequence_stack.propagations

    @property
    def propagation_norms(self) -> tuple[nn.Module, ...]:
        return self.sequence_stack.norms

    @property
    def propagation_ffns(self) -> tuple[nn.Module, ...]:
        return self.sequence_stack.ffns

    def set_track_stats(self, enabled: bool) -> None:
        for propagation in self.propagation_layers:
            propagation.track_stats = bool(enabled)
            if not enabled:
                propagation.last_stats = None

    def collect_propagation_stats(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        for index, block in enumerate(self.sequence_stack.blocks):
            prefix = f"layer_{index:02d}"
            stats[f"{prefix}/residual_gate"] = float(block.residual_gate.detach().float().item())
            if getattr(block.propagation, "last_stats", None):
                assert block.propagation.last_stats is not None
                for key, value in block.propagation.last_stats.items():
                    stats[f"{prefix}/{key}"] = float(value)
        return stats

    @property
    def sequence_layers(self) -> tuple[SparsePropagation, ...]:
        return self._layer_slice(self.propagation_layers, end=self.s_layers)

    @property
    def sequence_norms(self) -> tuple[nn.Module, ...]:
        return self._layer_slice(self.propagation_norms, end=self.s_layers)

    @property
    def sequence_ffns(self) -> tuple[nn.Module, ...]:
        return self._layer_slice(self.propagation_ffns, end=self.s_layers)

    def run_propagation(
        self,
        layer: Layer,
        *,
        start: int = 0,
        end: int | None = None,
        checkpoint_layers: bool | None = None,
    ) -> Layer:
        if checkpoint_layers is None:
            checkpoint_layers = self.checkpoint_sequence_layers
        return self.sequence_stack.forward_range(
            layer,
            start=start,
            end=end,
            checkpoint_layers=checkpoint_layers,
        )

    def encode(
        self,
        input_ids: Tensor,
        *,
        state_projection: nn.Module,
        start: int = 0,
        end: int | None = None,
    ) -> Layer:
        if self.s_microbatch_size is None or input_ids.shape[0] <= self.s_microbatch_size:
            return self._encode_single(
                input_ids,
                state_projection=state_projection,
                start=start,
                end=end,
            )

        chunks: list[Layer] = []
        for chunk_start in range(0, input_ids.shape[0], self.s_microbatch_size):
            chunk_end = min(chunk_start + self.s_microbatch_size, input_ids.shape[0])
            chunks.append(
                self._encode_single(
                    input_ids[chunk_start:chunk_end],
                    state_projection=state_projection,
                    start=start,
                    end=end,
                )
            )
        return Layer(
            dim=self.dim,
            num_nodes=chunks[0].num_nodes,
            state=torch.cat([chunk.state for chunk in chunks], dim=0),
            val=torch.cat([chunk.val for chunk in chunks], dim=0),
        )

    def make_token_layer(
        self,
        token_val: Tensor,
        *,
        state_projection: nn.Module,
    ) -> Layer:
        token_state_source = token_val
        token_state = state_projection(token_state_source).squeeze(-1)
        if self.unit_norm_values:
            token_state = softsign_state(token_state)
        return Layer(
            dim=self.dim,
            num_nodes=token_val.shape[-2],
            state=token_state,
            val=token_val,
        )

    def _encode_single(
        self,
        input_ids: Tensor,
        *,
        state_projection: nn.Module,
        start: int = 0,
        end: int | None = None,
    ) -> Layer:
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
        token_state_source = token_val

        token_state = state_projection(token_state_source).squeeze(-1)
        if self.unit_norm_values:
            token_state = softsign_state(token_state)
        if self.sequence_anchor:
            if self.anchor_val is None or self.anchor_state is None:
                raise RuntimeError("sequence_anchor is enabled but anchor parameters are missing.")
            anchor_val = self.anchor_val.expand(batch_size, 1, -1).to(
                device=token_val.device,
                dtype=token_val.dtype,
            )
            anchor_state = self.anchor_state.expand(batch_size, 1).to(
                device=token_val.device,
                dtype=token_val.dtype,
            )
            if self.unit_norm_values:
                anchor_state = softsign_state(anchor_state)
            seq_val = torch.cat((anchor_val, token_val), dim=1)
            seq_state = torch.cat((anchor_state, token_state), dim=1)
        else:
            seq_val = token_val
            seq_state = token_state
        num_nodes = int(seq_val.shape[1])
        layer = Layer(dim=self.dim, num_nodes=num_nodes, state=seq_state, val=seq_val)
        return self.run_propagation(
            layer,
            start=start,
            end=end,
            checkpoint_layers=self.checkpoint_sequence_layers,
        )
