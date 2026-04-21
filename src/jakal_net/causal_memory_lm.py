from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from torch import Tensor, nn

from jakal_net._architectural_common import (
    PARAM_INIT_STD,
    apply_delta,
    clone_layer,
    init_linear,
    init_pairwise_or_route_scales,
    layer_with_val_norm,
    make_pairwise,
    signed_abs_softmax_edges,
)
from jakal_net.core import Layer
from jakal_net.hierarchical_memory import BModule, BScanOutput
from jakal_net.latent_graph import KModule
from jakal_net.modules import LowRankBilinearPairwise, LowRankBilinearRoute
from jakal_net.native_backend import (
    causal_memory_scan_fused_native,
    native_supports,
    native_supports_device,
)
from jakal_net.propagation import SparsePropagation
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
        scan_backend: str = "auto",
        scan_checkpoint_chunk_size: int | None = None,
        implementation: str = "streaming",
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
        if prediction_layers <= 0:
            raise ValueError("prediction_layers must be positive.")
        if not memory_slots:
            raise ValueError("memory_slots must contain at least one level.")
        if any(slots <= 0 for slots in memory_slots):
            raise ValueError("memory_slots must be positive.")
        if scan_backend not in {"auto", "python", "native"}:
            raise ValueError(f"Unsupported scan_backend: {scan_backend!r}.")
        if scan_checkpoint_chunk_size is not None and scan_checkpoint_chunk_size <= 0:
            raise ValueError("scan_checkpoint_chunk_size must be positive when provided.")
        if knowledge_nodes < 0:
            raise ValueError("knowledge_nodes must be non-negative.")
        if knowledge_propagation_layers <= 0:
            raise ValueError("knowledge_propagation_layers must be positive.")

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.memory_slots = tuple(int(slots) for slots in memory_slots)
        self.num_memory_levels = len(self.memory_slots)
        self.prediction_window = prediction_window
        self.scan_backend = scan_backend
        self.scan_checkpoint_chunk_size = scan_checkpoint_chunk_size
        if knowledge_module is None and knowledge_nodes > 0:
            knowledge_module = KModule(
                dim=dim,
                num_nodes=knowledge_nodes,
                route_kind=route_kind,
                pairwise_kind=pairwise_kind,
                route_rank=route_rank,
                pairwise_rank=pairwise_rank,
                route_topk=knowledge_route_topk or memory_topk,
                propagation_topk=knowledge_propagation_topk or memory_topk,
                propagation_layers=knowledge_propagation_layers,
                implementation=implementation,
            )
        self.knowledge_module = knowledge_module

        self.value_to_state = nn.Linear(dim, 1)
        self.s_module = SModule(
            vocab_size=vocab_size,
            dim=dim,
            max_seq_len=max_seq_len,
            s_layers=s_layers,
            pairwise_kind=pairwise_kind,
            pairwise_rank=pairwise_rank,
            implementation=implementation,
            s_window=s_window,
            s_microbatch_size=s_microbatch_size,
        )
        self.b_module = BModule(
            dim=dim,
            memory_slots=self.memory_slots,
            memory_topk=memory_topk,
            pairwise_kind=pairwise_kind,
            route_kind=route_kind,
            pairwise_rank=pairwise_rank,
            route_rank=route_rank,
            implementation=implementation,
        )

        self.s_prediction_proj = nn.Linear(dim, dim, bias=False)
        self.prediction_input_norm = nn.LayerNorm(dim)
        self.prediction_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(prediction_layers))
        self.prediction_layers = nn.ModuleList(
            SparsePropagation(
                pairwise_fn=make_pairwise(pairwise_kind, dim=dim, rank=pairwise_rank),
                sparse_type="window",
                window=max(1, prediction_window),
                edge_compress_fn=signed_abs_softmax_edges,
                state_weight_edges=True,
                implementation=implementation,
                residual=True,
            )
            for _ in range(prediction_layers)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        if tie_embedding_head:
            self.lm_head.weight = self.s_module.token_embedding.weight
        self._reset_parameters(tie_embedding_head=tie_embedding_head)

    def _reset_parameters(self, *, tie_embedding_head: bool) -> None:
        nn.init.normal_(self.s_module.token_embedding.weight, mean=0.0, std=PARAM_INIT_STD)
        nn.init.normal_(self.s_module.anchor_val, mean=0.0, std=PARAM_INIT_STD)
        nn.init.zeros_(self.s_module.anchor_state)
        init_linear(self.value_to_state)
        init_linear(self.s_prediction_proj)
        self.b_module.reset_projection_parameters()
        if not tie_embedding_head:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=PARAM_INIT_STD)
        for module in self.modules():
            init_pairwise_or_route_scales(module)

    def initialize_memory_state(
        self,
        batch_size: int,
        *,
        device,
        dtype,
    ) -> tuple[Layer, ...]:
        return self.b_module.initialize_state(batch_size, device=device, dtype=dtype)

    def forward(
        self,
        input_ids: Tensor,
        *,
        memory_state: Sequence[Layer] | None = None,
        knowledge_state: Layer | None = None,
        reset_mask: Tensor | None = None,
        return_memory_state: bool = False,
        return_layers: bool = False,
    ) -> Tensor | MemoryScanOutput:
        if isinstance(memory_state, ModelRecurrentState):
            if knowledge_state is None:
                knowledge_state = memory_state.knowledge_state
            memory_state = memory_state.memory_state
        sequence_layer = self.s_module.encode(input_ids, state_projection=self.value_to_state)
        aligned_s = sequence_layer.val[:, 1:, :]
        batch_size, _, _ = aligned_s.shape
        device = aligned_s.device
        dtype = aligned_s.dtype

        if memory_state is None:
            memory_state = self.initialize_memory_state(batch_size, device=device, dtype=dtype)
        if len(tuple(memory_state)) != self.num_memory_levels:
            raise ValueError("memory_state does not match the configured memory hierarchy.")
        current_memory = self.b_module.reset_state(
            tuple(memory_state),
            reset_mask=reset_mask,
            device=device,
            dtype=dtype,
        )
        scan_output = self._scan_memory(
            aligned_s,
            current_memory,
            knowledge_state=knowledge_state,
            reset_mask=reset_mask,
        )
        query_layer = scan_output.query_layer
        current_memory = scan_output.memory_state
        for propagation, norm in zip(self.prediction_layers, self.prediction_norms):
            query_layer = apply_delta(
                query_layer,
                propagation.compute_delta(layer_with_val_norm(query_layer, norm)),
                residual=True,
                val_norm=norm,
            )

        logits = self.lm_head(self.output_norm(query_layer.val))
        if not (return_memory_state or return_layers):
            return logits
        return MemoryScanOutput(
            logits=logits,
            memory_state=tuple(clone_layer(layer) for layer in current_memory),
            knowledge_state=None if scan_output.knowledge_state is None else clone_layer(scan_output.knowledge_state),
            sequence_layer=sequence_layer if return_layers else None,
            query_layer=query_layer if return_layers else None,
        )

    @staticmethod
    def _native_edge_compress_name(edge_compress_fn: object) -> str | None:
        name = getattr(edge_compress_fn, "__name__", "")
        if name in {"signed_abs_softmax", "signed_abs_softmax_edges", "_signed_abs_softmax_edges"}:
            return "signed_abs_softmax"
        if name in {"signed_entmax15", "_signed_entmax15_edges"}:
            return "signed_entmax15"
        return None

    @staticmethod
    def _tensor_or_empty(tensor: Tensor | None, reference: Tensor) -> Tensor:
        if tensor is None:
            return reference.new_empty((0,))
        return tensor

    def _native_scan_supported_config(self) -> bool:
        supported_route_compress = {"softmax", "signed_abs_softmax", "signed_entmax15"}
        supported_edge_compress = {"softsign", "signed_abs_softmax", "signed_entmax15"}

        def _is_low_rank_route(transition) -> bool:
            return (
                isinstance(transition.route_fn, LowRankBilinearRoute)
                and transition.route_compress_name in supported_route_compress
            )

        def _is_low_rank_topk_propagation(propagation) -> bool:
            return (
                isinstance(propagation.pairwise_fn, LowRankBilinearPairwise)
                and propagation.sparse_type == "topk"
                and propagation.state_weight_edges
                and self._native_edge_compress_name(propagation.edge_compress_fn) in supported_edge_compress
            )

        if not isinstance(self.value_to_state, nn.Linear):
            return False
        if not all(_is_low_rank_route(level.write) for level in self.b_module.memory_levels):
            return False
        if not all(_is_low_rank_topk_propagation(level.propagation) for level in self.b_module.memory_levels):
            return False
        if not all(_is_low_rank_route(transition) for transition in self.b_module.level_transitions):
            return False
        if not all(_is_low_rank_route(transition) for transition in self.b_module.skip_transitions.values()):
            return False
        route_names = {level.write.route_compress_name for level in self.b_module.memory_levels}
        route_names.update(transition.route_compress_name for transition in self.b_module.level_transitions)
        route_names.update(transition.route_compress_name for transition in self.b_module.skip_transitions.values())
        if len(route_names) != 1:
            return False
        propagation_names = {
            self._native_edge_compress_name(level.propagation.edge_compress_fn)
            for level in self.b_module.memory_levels
        }
        if None in propagation_names or len(propagation_names) != 1:
            return False
        return True

    def _pack_native_scan_inputs(
        self,
        aligned_s: Tensor,
        memory_state: Sequence[Layer],
    ) -> dict[str, object]:
        if not self._native_scan_supported_config():
            raise RuntimeError(
                "causal_memory_scan_fused currently supports only uniform low_rank_bilinear "
                "causal-memory configs with a shared route compress and propagation edge compress."
            )

        transition_compress_name = self.b_module.memory_levels[0].write.route_compress_name
        propagation_compress_name = self._native_edge_compress_name(
            self.b_module.memory_levels[0].propagation.edge_compress_fn
        )
        if propagation_compress_name is None:
            raise RuntimeError("Unsupported propagation edge_compress_fn for native scan.")

        skip_source_weights: list[Tensor] = []
        skip_target_weights: list[Tensor] = []
        skip_core_weights: list[Tensor] = []
        skip_biases: list[Tensor] = []
        skip_gates: list[Tensor] = []
        skip_topks: list[int] = []
        if self.num_memory_levels >= 2:
            transition = self.b_module.skip_transitions["token_to_1"]
            route = transition.route_fn
            assert isinstance(route, LowRankBilinearRoute)
            skip_source_weights.append(route.source_proj.weight)
            skip_target_weights.append(route.target_proj.weight)
            skip_core_weights.append(route.weight)
            skip_biases.append(self._tensor_or_empty(route.bias, aligned_s))
            skip_gates.append(self.b_module.skip_gates["token_to_1"])
            skip_topks.append(int(transition.topk))
        for level_index in range(2, self.num_memory_levels):
            key = f"{level_index - 2}_to_{level_index}"
            transition = self.b_module.skip_transitions[key]
            route = transition.route_fn
            assert isinstance(route, LowRankBilinearRoute)
            skip_source_weights.append(route.source_proj.weight)
            skip_target_weights.append(route.target_proj.weight)
            skip_core_weights.append(route.weight)
            skip_biases.append(self._tensor_or_empty(route.bias, aligned_s))
            skip_gates.append(self.b_module.skip_gates[key])
            skip_topks.append(int(transition.topk))

        return {
            "aligned_s": aligned_s,
            "flat_memory": self.b_module.flatten_memory_state(tuple(memory_state)),
            "value_to_state_weight": self.value_to_state.weight,
            "value_to_state_bias": self.value_to_state.bias,
            "s_prediction_weight": self.s_prediction_proj.weight,
            "prediction_input_norm_weight": self.prediction_input_norm.weight,
            "prediction_input_norm_bias": self.prediction_input_norm.bias,
            "read_template_val": self.b_module.read_template_val,
            "read_projection_weights": tuple(projection.weight for projection in self.b_module.read_projections),
            "read_gates": tuple(self.b_module.read_gates),
            "write_source_weights": tuple(level.write.route_fn.source_proj.weight for level in self.b_module.memory_levels),
            "write_target_weights": tuple(level.write.route_fn.target_proj.weight for level in self.b_module.memory_levels),
            "write_core_weights": tuple(level.write.route_fn.weight for level in self.b_module.memory_levels),
            "write_biases": tuple(
                self._tensor_or_empty(level.write.route_fn.bias, aligned_s)
                for level in self.b_module.memory_levels
            ),
            "write_topks": tuple(int(level.write.topk) for level in self.b_module.memory_levels),
            "transition_compress_name": transition_compress_name,
            "propagation_source_weights": tuple(level.propagation.pairwise_fn.source_proj.weight for level in self.b_module.memory_levels),
            "propagation_target_weights": tuple(level.propagation.pairwise_fn.target_proj.weight for level in self.b_module.memory_levels),
            "propagation_core_weights": tuple(level.propagation.pairwise_fn.weight for level in self.b_module.memory_levels),
            "propagation_biases": tuple(
                self._tensor_or_empty(level.propagation.pairwise_fn.bias, aligned_s)
                for level in self.b_module.memory_levels
            ),
            "propagation_topks": tuple(int(level.propagation.topk or level.num_slots) for level in self.b_module.memory_levels),
            "propagation_compress_name": propagation_compress_name,
            "val_norm_weights": tuple(level.val_norm.weight for level in self.b_module.memory_levels),
            "val_norm_biases": tuple(level.val_norm.bias for level in self.b_module.memory_levels),
            "level_transition_source_weights": tuple(transition.route_fn.source_proj.weight for transition in self.b_module.level_transitions),
            "level_transition_target_weights": tuple(transition.route_fn.target_proj.weight for transition in self.b_module.level_transitions),
            "level_transition_core_weights": tuple(transition.route_fn.weight for transition in self.b_module.level_transitions),
            "level_transition_biases": tuple(
                self._tensor_or_empty(transition.route_fn.bias, aligned_s)
                for transition in self.b_module.level_transitions
            ),
            "level_transition_topks": tuple(int(transition.topk) for transition in self.b_module.level_transitions),
            "level_norm_weights": tuple(norm.weight for norm in self.b_module.level_norms),
            "level_norm_biases": tuple(norm.bias for norm in self.b_module.level_norms),
            "skip_source_weights": tuple(skip_source_weights),
            "skip_target_weights": tuple(skip_target_weights),
            "skip_core_weights": tuple(skip_core_weights),
            "skip_biases": tuple(skip_biases),
            "skip_gates": tuple(skip_gates),
            "skip_topks": tuple(skip_topks),
        }

    def _scan_memory_native_fused(
        self,
        aligned_s: Tensor,
        memory_state: Sequence[Layer],
    ) -> BScanOutput:
        packed = self._pack_native_scan_inputs(aligned_s, memory_state)
        query_val, flat_memory = causal_memory_scan_fused_native(**packed)
        query_state = self.value_to_state(query_val).squeeze(-1)
        return BScanOutput(
            query_layer=Layer(dim=self.dim, num_nodes=aligned_s.shape[1], state=query_state, val=query_val),
            memory_state=self.b_module.unflatten_memory_state(flat_memory),
            bridge_layer=None,
            knowledge_state=None,
            knowledge_output=None,
        )

    def _scan_memory(
        self,
        aligned_s: Tensor,
        memory_state: Sequence[Layer],
        *,
        knowledge_state: Layer | None,
        reset_mask: Tensor | None,
    ) -> Any:
        use_native_scan = (
            self.knowledge_module is None
            and self.scan_backend != "python"
            and native_supports("causal_memory_scan_fused")
            and native_supports_device(aligned_s.device.type)
        )
        if use_native_scan:
            try:
                return self._scan_memory_native_fused(aligned_s, memory_state)
            except RuntimeError:
                if self.scan_backend == "native":
                    raise
        if self.knowledge_module is not None:
            return self.b_module.scan(
                aligned_s,
                memory_state,
                state_projection=self.value_to_state,
                query_projection=self.s_prediction_proj,
                query_input_norm=self.prediction_input_norm,
                knowledge_module=self.knowledge_module,
                knowledge_state=knowledge_state,
                knowledge_reset_mask=reset_mask,
            )
        return self.b_module.scan(
            aligned_s,
            memory_state,
            state_projection=self.value_to_state,
            query_projection=self.s_prediction_proj,
            query_input_norm=self.prediction_input_norm,
        )
