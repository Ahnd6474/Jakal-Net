from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from jakal_net._architectural_common import (
    PARAM_INIT_STD,
    clone_layer,
    init_pairwise_or_route_scales,
    softsign_state,
)
from jakal_net.core import Layer
from jakal_net.propagation import SparsePropagation
from jakal_net.propagation_stack import (
    apply_dense_delta_fastpath,
    can_use_dense_apply_fastpath,
)
from jakal_net.sequence_module import SModule


@dataclass(frozen=True, slots=True)
class ModelRecurrentState:
    memory_state: tuple[Layer, ...]
    knowledge_state: Layer | None = None


@dataclass(frozen=True, slots=True)
class MemoryScanOutput:
    logits: Tensor
    memory_state: tuple[Layer, ...]
    knowledge_state: Layer | None = None
    sequence_layer: Layer | None = None
    query_layer: Layer | None = None

    @property
    def recurrent_state(self) -> ModelRecurrentState:
        return ModelRecurrentState(
            memory_state=self.memory_state,
            knowledge_state=self.knowledge_state,
        )


class ValueNormStateProjection(nn.Module):
    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, val: Tensor) -> Tensor:
        return torch.linalg.vector_norm(val, ord=2, dim=-1, keepdim=True).clamp_min(self.eps)


class CausalMemoryLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int = 512,
        max_seq_len: int = 2048,
        propagation_layers: int | None = None,
        s_layers: int = 2,
        memory_slots: Sequence[int] = (256, 64, 16),
        memory_update_intervals: Sequence[int] | None = None,
        prediction_layers: int = 2,
        s_window: int | None = None,
        s_microbatch_size: int | None = None,
        prediction_window: int = 64,
        checkpoint_sequence_layers: bool = False,
        checkpoint_prediction_layers: bool = False,
        memory_topk: int = 16,
        memory_train_mode: str = "dense",
        memory_eval_mode: str = "dense",
        eval_topk: int | None = None,
        pairwise_kind: str = "low_rank_bilinear",
        route_kind: str = "low_rank_bilinear",
        pairwise_rank: int = 64,
        route_rank: int = 64,
        pairwise_heads: int = 1,
        route_heads: int = 1,
        pairwise_frozen_heads: int = 0,
        route_frozen_heads: int = 0,
        pairwise_anchor_heads: int = 0,
        route_anchor_heads: int = 0,
        pairwise_anchor_kind: str = "scaled_cosine",
        route_anchor_kind: str = "fixed_projection",
        pairwise_head_aggregate: str = "signed_smoothmax",
        sequence_anchor: bool = True,
        scan_backend: str = "auto",
        scan_checkpoint_chunk_size: int | None = None,
        implementation: str = "streaming",
        unit_norm_values: bool = False,
        propagation_residual_gate_init: float = 0.1,
        feed_forward_layers: bool = True,
        memory_feed_forward_layers: bool | None = None,
        disable_memory: bool = False,
        disable_memory_read: bool = False,
        disable_memory_propagation: bool = False,
        feed_forward_hidden_mult: float = 2.0,
        feed_forward_kind: str = "value",
        feed_forward_residual_scale: float = 1.0,
        feed_forward_learnable_residual_scale: bool = False,
        feed_forward_zero_init_output: bool = True,
        feed_forward_activation: str = "gelu",
        tie_embedding_head: bool = True,
        knowledge_nodes: int = 0,
        knowledge_route_topk: int | None = None,
        knowledge_propagation_topk: int | None = None,
        knowledge_propagation_layers: int = 1,
        knowledge_module: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if scan_backend not in {"auto", "python", "native"}:
            raise ValueError(f"Unsupported scan_backend: {scan_backend!r}.")
        if scan_checkpoint_chunk_size is not None and scan_checkpoint_chunk_size <= 0:
            raise ValueError("scan_checkpoint_chunk_size must be positive when provided.")
        if feed_forward_hidden_mult <= 0.0:
            raise ValueError("feed_forward_hidden_mult must be positive.")
        if feed_forward_kind not in {"value", "state_val"}:
            raise ValueError(f"Unsupported feed_forward_kind: {feed_forward_kind!r}.")
        if feed_forward_residual_scale < 0.0:
            raise ValueError("feed_forward_residual_scale must be non-negative.")

        total_layers = int(propagation_layers) if propagation_layers is not None else int(s_layers) + int(prediction_layers)
        if total_layers <= 0:
            raise ValueError("propagation_layers must be positive.")
        if s_layers < 0 or prediction_layers < 0:
            raise ValueError("legacy s/prediction layer counts must be non-negative.")

        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.max_seq_len = int(max_seq_len)
        self.propagation_layers_count = total_layers
        if propagation_layers is None:
            self.legacy_sequence_layers_count = max(0, min(int(s_layers), total_layers))
            self.legacy_prediction_layers_count = max(0, total_layers - self.legacy_sequence_layers_count)
            self.legacy_prediction_layers_count = int(prediction_layers)
        else:
            # The current no-memory path is a single propagation stack.
            self.legacy_sequence_layers_count = total_layers
            self.legacy_prediction_layers_count = 0
        self.prediction_window = int(prediction_window)
        self.scan_backend = scan_backend
        self.scan_checkpoint_chunk_size = scan_checkpoint_chunk_size
        self.checkpoint_prediction_layers = bool(checkpoint_prediction_layers)
        self.unit_norm_values = bool(unit_norm_values)
        self.propagation_residual_gate_init = float(propagation_residual_gate_init)
        self.feed_forward_layers = bool(feed_forward_layers)
        self.memory_feed_forward_layers = (
            self.feed_forward_layers if memory_feed_forward_layers is None else bool(memory_feed_forward_layers)
        )
        self.disable_memory = True
        self.disable_memory_read = True
        self.disable_memory_propagation = True
        self.feed_forward_hidden_mult = float(feed_forward_hidden_mult)
        self.feed_forward_kind = feed_forward_kind
        self.feed_forward_residual_scale = float(feed_forward_residual_scale)
        self.feed_forward_learnable_residual_scale = bool(feed_forward_learnable_residual_scale)
        self.feed_forward_zero_init_output = bool(feed_forward_zero_init_output)
        self.feed_forward_activation = feed_forward_activation
        self.sequence_anchor = bool(sequence_anchor)
        self.memory_slots = ()
        self.num_memory_levels = 0
        self.memory_update_intervals = tuple(memory_update_intervals or ())
        self.memory_topk = int(memory_topk)
        self.memory_train_mode = memory_train_mode
        self.memory_eval_mode = memory_eval_mode
        self.eval_topk = eval_topk
        self.route_kind = route_kind
        self.route_rank = int(route_rank)
        self.route_heads = int(route_heads)
        self.route_frozen_heads = int(route_frozen_heads)
        self.route_anchor_heads = int(route_anchor_heads)
        self.route_anchor_kind = route_anchor_kind
        self.knowledge_nodes = int(knowledge_nodes)
        self.knowledge_route_topk = knowledge_route_topk
        self.knowledge_propagation_topk = knowledge_propagation_topk
        self.knowledge_propagation_layers = int(knowledge_propagation_layers)
        self.knowledge_module = knowledge_module
        self.legacy_disable_memory_flag = bool(disable_memory)
        self.legacy_disable_memory_read_flag = bool(disable_memory_read)
        self.legacy_disable_memory_propagation_flag = bool(disable_memory_propagation)

        self.value_to_state = ValueNormStateProjection()
        self.s_module = SModule(
            vocab_size=vocab_size,
            dim=dim,
            max_seq_len=max_seq_len,
            s_layers=total_layers,
            additional_layers=0,
            pairwise_kind=pairwise_kind,
            pairwise_rank=pairwise_rank,
            pairwise_heads=pairwise_heads,
            pairwise_frozen_heads=pairwise_frozen_heads,
            pairwise_anchor_heads=pairwise_anchor_heads,
            pairwise_anchor_kind=pairwise_anchor_kind,
            pairwise_head_aggregate=pairwise_head_aggregate,
            sequence_anchor=self.sequence_anchor,
            implementation=implementation,
            s_window=s_window,
            s_microbatch_size=s_microbatch_size,
            checkpoint_sequence_layers=checkpoint_sequence_layers or checkpoint_prediction_layers,
            unit_norm_values=unit_norm_values,
            propagation_residual_gate_init=self.propagation_residual_gate_init,
            feed_forward_layers=self.feed_forward_layers,
            feed_forward_hidden_mult=self.feed_forward_hidden_mult,
            feed_forward_kind=self.feed_forward_kind,
            feed_forward_residual_scale=self.feed_forward_residual_scale,
            feed_forward_learnable_residual_scale=self.feed_forward_learnable_residual_scale,
            feed_forward_zero_init_output=self.feed_forward_zero_init_output,
            feed_forward_activation=self.feed_forward_activation,
        )
        self.output_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embedding_head:
            self.lm_head.weight = self.s_module.token_embedding.weight
        self._reset_parameters(tie_embedding_head=tie_embedding_head)

    @property
    def s_layers(self) -> int:
        return self.legacy_sequence_layers_count

    @property
    def prediction_layers_count(self) -> int:
        return self.legacy_prediction_layers_count

    @property
    def propagation_layers(self) -> tuple[SparsePropagation, ...]:
        return self.s_module.propagation_layers

    @property
    def propagation_norms(self) -> tuple[nn.Module, ...]:
        return self.s_module.propagation_norms

    @property
    def propagation_ffns(self) -> tuple[nn.Module, ...]:
        return self.s_module.propagation_ffns

    @property
    def sequence_layers(self) -> tuple[SparsePropagation, ...]:
        return self.propagation_layers[: self.legacy_sequence_layers_count]

    @property
    def sequence_norms(self) -> tuple[nn.Module, ...]:
        return self.propagation_norms[: self.legacy_sequence_layers_count]

    @property
    def sequence_ffns(self) -> tuple[nn.Module, ...]:
        return self.propagation_ffns[: self.legacy_sequence_layers_count]

    @property
    def prediction_layers(self) -> tuple[SparsePropagation, ...]:
        return self.propagation_layers[self.legacy_sequence_layers_count :]

    @property
    def prediction_norms(self) -> tuple[nn.Module, ...]:
        return self.propagation_norms[self.legacy_sequence_layers_count :]

    @property
    def prediction_ffns(self) -> tuple[nn.Module, ...]:
        return self.propagation_ffns[self.legacy_sequence_layers_count :]

    def set_track_stats(self, enabled: bool) -> None:
        self.s_module.set_track_stats(enabled)

    def collect_internal_stats(self) -> dict[str, float]:
        return self.s_module.collect_propagation_stats()

    @property
    def memory_levels(self) -> nn.ModuleList:
        return nn.ModuleList()

    @property
    def read_projections(self) -> nn.ModuleList:
        return nn.ModuleList()

    @property
    def skip_gates(self) -> nn.ParameterDict:
        return nn.ParameterDict()

    def _reset_parameters(self, *, tie_embedding_head: bool) -> None:
        nn.init.normal_(self.s_module.token_embedding.weight, mean=0.0, std=PARAM_INIT_STD)
        if self.s_module.anchor_val is not None:
            nn.init.normal_(self.s_module.anchor_val, mean=0.0, std=PARAM_INIT_STD)
        if self.s_module.anchor_state is not None:
            nn.init.zeros_(self.s_module.anchor_state)
        if not tie_embedding_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=PARAM_INIT_STD)
        for module in self.modules():
            init_pairwise_or_route_scales(module)

    def _strip_sequence_anchor(self, layer: Layer) -> Layer:
        if not self.sequence_anchor:
            return layer
        return Layer(
            dim=layer.dim,
            num_nodes=layer.num_nodes - 1,
            state=layer.state[..., 1:],
            val=layer.val[..., 1:, :],
        )

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

    def _native_scan_supported_config(self) -> bool:
        return False

    def initialize_memory_state(
        self,
        batch_size: int,
        *,
        device,
        dtype,
    ) -> tuple[Layer, ...]:
        del batch_size, device, dtype
        return ()

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory_state: Sequence[Layer] | ModelRecurrentState | None = None,
        knowledge_state: Layer | None = None,
        reset_mask: Tensor | None = None,
        return_memory_state: bool = False,
        return_layers: bool = False,
        return_logits: bool = True,
    ) -> Tensor | MemoryScanOutput:
        del knowledge_state, reset_mask
        if isinstance(memory_state, ModelRecurrentState):
            memory_state = memory_state.memory_state
        if memory_state not in (None, (), tuple()):
            raise ValueError("This model does not support legacy hierarchical memory_state inputs.")

        sequence_layer = self.s_module.encode(input_ids, state_projection=self.value_to_state)
        query_layer = self._strip_sequence_anchor(sequence_layer)
        output_state_source = self.output_norm(query_layer.val)
        output_val = output_state_source
        output_state = self.value_to_state(output_state_source).squeeze(-1)
        if self.unit_norm_values:
            output_state = softsign_state(output_state)
        query_layer = query_layer.with_tensors(state=output_state, val=output_val)
        logits = self.lm_head(output_val) if return_logits else output_val.new_empty((0,))
        if not (return_memory_state or return_layers):
            return logits
        return MemoryScanOutput(
            logits=logits,
            memory_state=(),
            knowledge_state=None,
            sequence_layer=clone_layer(sequence_layer) if return_layers else None,
            query_layer=clone_layer(query_layer) if return_layers else None,
        )


CausalHierarchicalMemoryLM = CausalMemoryLM
