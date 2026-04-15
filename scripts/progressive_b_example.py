from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from jakal_net import (
    BilinearPairwiseRoute,
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    HadamardMLPPairwise,
    Layer,
    LayerDelta,
    LearnedPositionEncoding,
    LowRankBilinearPairwise,
    LowRankBilinearRoute,
    Propagation,
    SparsePropagation,
    SparseTransition,
    SourceTargetHadamardMLPRoute,
    Transition,
)
from jakal_net.kernel_common import (
    apply_slot_mask_to_state,
    apply_slot_mask_to_val,
    gather_state_by_indices,
    gather_val_by_indices,
    select_topk,
    supports_pairwise_kernel,
)
from jakal_net.native_backend import (
    native_supports,
    native_supports_device,
    propagation_query_topk_native,
)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class TemperatureScaledRoute(nn.Module):
    def __init__(self, route_fn: nn.Module, temperature: float) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.route_fn = route_fn
        self.temperature = temperature
        self.expects_pairwise_inputs = getattr(route_fn, "expects_pairwise_inputs", False)

    def forward(self, *args: Tensor) -> Tensor:
        return self.route_fn(*args) / self.temperature


def _scale_delta(
    delta: LayerDelta,
    *,
    state_scale: float = 1.0,
    val_scale: float = 1.0,
) -> LayerDelta:
    if state_scale == 1.0 and val_scale == 1.0:
        return delta
    return LayerDelta(
        delta_state=delta.delta_state * state_scale,
        delta_val=delta.delta_val * val_scale,
    )


def _apply_scaled_delta(
    layer: Layer,
    delta: LayerDelta,
    *,
    state_scale: float = 1.0,
    val_scale: float = 1.0,
) -> Layer:
    if state_scale == 0.0 and val_scale == 0.0:
        return layer
    return layer.apply_delta(
        _scale_delta(delta, state_scale=state_scale, val_scale=val_scale)
    )


def _make_value_norm(dim: int, norm_kind: str) -> nn.Module:
    if norm_kind == "identity":
        return nn.Identity()
    if norm_kind == "layernorm":
        return nn.LayerNorm(dim)
    if norm_kind == "rmsnorm":
        return RMSNorm(dim)
    raise ValueError(f"Unsupported norm kind: {norm_kind!r}.")


def _normalize_layer_values(layer: Layer, val_norm: nn.Module) -> Layer:
    return layer.with_tensors(val=val_norm(layer.val))


def _stabilize_layer(
    layer: Layer,
    val_norm: nn.Module,
    *,
    apply_value_norm: bool,
) -> Layer:
    val = val_norm(layer.val) if apply_value_norm else layer.val
    return layer.with_tensors(
        state=torch.tanh(layer.state),
        val=val,
    )


def _apply_layer_slot_mask(layer: Layer, slot_mask: Tensor) -> Layer:
    return layer.with_tensors(
        state=apply_slot_mask_to_state(layer.state, slot_mask),
        val=apply_slot_mask_to_val(layer.val, slot_mask),
    )


def _expand_position_encoding(
    position_encoding: Tensor,
    batch_shape: tuple[int, ...],
) -> Tensor:
    view_shape = (1,) * len(batch_shape) + tuple(position_encoding.shape)
    return position_encoding.view(view_shape).expand(*batch_shape, *position_encoding.shape)


def _make_sparse_or_dense_propagation(
    *,
    dim: int,
    sparse_type: str,
    implementation: str,
    window: int | None = None,
    topk: int | None = None,
    pairwise_fn: nn.Module | None = None,
    residual: bool = True,
) -> Propagation | SparsePropagation:
    pairwise = DiagonalBilinearPairwise(dim=dim) if pairwise_fn is None else pairwise_fn
    if sparse_type == "dense":
        return Propagation(
            pairwise_fn=pairwise,
            implementation=implementation,
            residual=residual,
        )
    if sparse_type == "window":
        if window is None:
            raise ValueError("window propagation requires window.")
        return SparsePropagation(
            pairwise_fn=pairwise,
            sparse_type="window",
            window=window,
            implementation=implementation,
            residual=residual,
        )
    if sparse_type == "topk":
        if topk is None:
            raise ValueError("topk propagation requires topk.")
        return SparsePropagation(
            pairwise_fn=pairwise,
            sparse_type="topk",
            topk=topk,
            implementation=implementation,
            residual=residual,
        )
    raise ValueError(f"Unsupported sparse_type: {sparse_type!r}.")


class QueryTopKPropagation(nn.Module):
    def __init__(
        self,
        pairwise_fn: nn.Module,
        *,
        topk: int,
        implementation: str,
        query_block_size: int | None = 128,
        source_block_size: int | None = 128,
        edge_compress_fn: Callable[[Tensor], Tensor] = F.softsign,
    ) -> None:
        super().__init__()
        if topk <= 0:
            raise ValueError("query topk must be positive.")
        self.pairwise_fn = pairwise_fn
        self.topk = topk
        self.implementation = implementation
        self.query_block_size = query_block_size
        self.source_block_size = source_block_size
        self.edge_compress_fn = edge_compress_fn

    def _native_edge_compress_name(self) -> str | None:
        name = getattr(self.edge_compress_fn, "__name__", "")
        if self.edge_compress_fn is F.softsign or name == "softsign":
            return "softsign"
        return None

    def _compute_delta_streaming(self, query_layer: Layer, source_layer: Layer) -> LayerDelta:
        projected_state = source_layer.state
        projected_val = source_layer.val
        query_nodes = query_layer.num_nodes
        source_nodes = source_layer.num_nodes
        k = min(self.topk, source_nodes)
        delta_state = torch.zeros(
            *query_layer.batch_shape,
            query_nodes,
            device=query_layer.state.device,
            dtype=projected_state.dtype,
        )
        delta_val = torch.zeros(
            *query_layer.batch_shape,
            query_nodes,
            query_layer.dim,
            device=query_layer.val.device,
            dtype=projected_val.dtype,
        )

        for query_start in range(0, query_nodes, self.query_block_size or query_nodes):
            query_end = min(query_nodes, query_start + (self.query_block_size or query_nodes))
            query_val = query_layer.val[..., query_start:query_end, :]
            best_scores: Tensor | None = None
            best_indices: Tensor | None = None
            for source_start in range(0, source_nodes, self.source_block_size or source_nodes):
                source_end = min(source_nodes, source_start + (self.source_block_size or source_nodes))
                source_val = source_layer.val[..., source_start:source_end, :]
                scores = self.pairwise_fn(query_val, source_val)
                if best_scores is None or best_indices is None:
                    best_scores = torch.full(
                        (*scores.shape[:-1], k),
                        fill_value=torch.finfo(scores.dtype).min,
                        device=scores.device,
                        dtype=scores.dtype,
                    )
                    best_indices = torch.zeros(
                        (*scores.shape[:-1], k),
                        device=scores.device,
                        dtype=torch.long,
                    )
                source_indices = torch.arange(
                    source_start,
                    source_end,
                    device=scores.device,
                    dtype=torch.long,
                )
                source_indices = source_indices.view(
                    (1,) * (scores.ndim - 1) + (source_end - source_start,)
                ).expand_as(scores)
                candidate_scores = torch.cat((best_scores, scores), dim=-1)
                candidate_indices = torch.cat((best_indices, source_indices), dim=-1)
                selected = select_topk(candidate_scores, k, dim=-1)
                best_scores = selected.values
                best_indices = torch.take_along_dim(
                    candidate_indices,
                    selected.indices,
                    dim=-1,
                )

            if best_scores is None or best_indices is None:
                continue
            edges = self.edge_compress_fn(best_scores)
            selected_state = gather_state_by_indices(projected_state, best_indices)
            selected_val = gather_val_by_indices(projected_val, best_indices)
            delta_state[..., query_start:query_end] = (
                edges.to(projected_state.dtype) * selected_state.to(projected_state.dtype)
            ).sum(dim=-1)
            delta_val[..., query_start:query_end, :] = (
                edges.to(projected_val.dtype).unsqueeze(-1)
                * selected_val.to(projected_val.dtype)
            ).sum(dim=-2)

        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def compute_delta(self, query_layer: Layer, source_layer: Layer) -> LayerDelta:
        edge_compress_name = self._native_edge_compress_name()
        if (
            self.implementation == "native"
            and edge_compress_name is not None
            and supports_pairwise_kernel(self.pairwise_fn)
            and native_supports("propagation_query_topk")
            and native_supports_device(query_layer.val.device.type)
        ):
            return propagation_query_topk_native(
                pairwise_fn=self.pairwise_fn,
                edge_compress_name=edge_compress_name,
                query_val=query_layer.val,
                source_val=source_layer.val,
                projected_state=source_layer.state,
                projected_val=source_layer.val,
                topk=min(self.topk, source_layer.num_nodes),
                query_block_size=self.query_block_size or query_layer.num_nodes,
                source_block_size=self.source_block_size or source_layer.num_nodes,
            )
        return self._compute_delta_streaming(query_layer, source_layer)


def _make_transition(
    *,
    dim: int,
    route_topk: int,
    route_mode: str,
    route_kind: str,
    route_hidden_dim: int | None,
    route_temperature: float,
    implementation: str,
    merge_mode: str = "add",
    edge_dropout_p: float = 0.0,
    usage_dropout_base: float = 0.0,
    usage_dropout_scale: float = 0.0,
    usage_dropout_max: float = 0.0,
    usage_ema_decay: float = 0.99,
) -> Transition | SparseTransition:
    if route_kind == "diagonal_bilinear":
        route_fn: nn.Module = DiagonalBilinearRoute(src_dim=dim, dst_dim=dim)
    elif route_kind == "low_rank_bilinear":
        route_fn = LowRankBilinearRoute(
            src_dim=dim,
            dst_dim=dim,
            rank=route_hidden_dim or max(1, dim // 2),
        )
    elif route_kind == "hadamard_mlp":
        route_fn = SourceTargetHadamardMLPRoute(
            src_dim=dim,
            dst_dim=dim,
            hidden_dim=route_hidden_dim,
        )
    else:
        raise ValueError(f"Unsupported route_kind: {route_kind!r}.")
    if route_temperature != 1.0:
        route_fn = TemperatureScaledRoute(route_fn, route_temperature)
    if route_mode == "dense" or route_topk <= 0:
        return Transition(
            route_fn=route_fn,
            implementation=implementation,
            merge_mode=merge_mode,
        )
    return SparseTransition(
        route_fn=route_fn,
        topk=route_topk,
        implementation=implementation,
        merge_mode=merge_mode,
        edge_dropout_p=edge_dropout_p,
        usage_dropout_base=usage_dropout_base,
        usage_dropout_scale=usage_dropout_scale,
        usage_dropout_max=usage_dropout_max,
        usage_ema_decay=usage_ema_decay,
    )


@dataclass(frozen=True, slots=True)
class ProgressiveBStageSpec:
    num_layers: int
    expanded_nodes: int
    compressed_nodes: int
    alpha_b: float
    beta_s_to_b: float
    beta_b_to_s: float

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.expanded_nodes <= 0:
            raise ValueError("expanded_nodes must be positive.")
        if self.compressed_nodes <= 0:
            raise ValueError("compressed_nodes must be positive.")
        if self.compressed_nodes > self.expanded_nodes:
            raise ValueError("compressed_nodes must be less than or equal to expanded_nodes.")
        for name, value in (
            ("alpha_b", self.alpha_b),
            ("beta_s_to_b", self.beta_s_to_b),
            ("beta_b_to_s", self.beta_b_to_s),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")


def build_progressive_b_stage_specs(
    seq_nodes: int,
    *,
    lite_layers: int = 2,
    mid_layers: int = 2,
    full_layers: int = 1,
    lite_expand_ratio: float = 1.05,
    lite_compress_ratio: float = 0.90,
    lite_alpha_b: float = 0.3,
    lite_beta_s_to_b: float = 0.25,
    lite_beta_b_to_s: float = 0.15,
    mid_expand_ratio: float = 1.10,
    mid_compress_ratio: float = 0.80,
    mid_alpha_b: float = 0.65,
    mid_beta_s_to_b: float = 0.55,
    mid_beta_b_to_s: float = 0.35,
    full_expand_ratio: float = 1.20,
    full_compress_ratio: float = 0.70,
    full_alpha_b: float = 1.0,
    full_beta_s_to_b: float = 0.9,
    full_beta_b_to_s: float = 0.8,
) -> list[ProgressiveBStageSpec]:
    if seq_nodes <= 0:
        raise ValueError("seq_nodes must be positive.")
    specs: list[ProgressiveBStageSpec] = []
    if lite_layers > 0:
        specs.append(
            ProgressiveBStageSpec(
            num_layers=lite_layers,
            expanded_nodes=max(1, math.ceil(seq_nodes * lite_expand_ratio)),
            compressed_nodes=max(1, math.ceil(seq_nodes * lite_compress_ratio)),
            alpha_b=lite_alpha_b,
            beta_s_to_b=lite_beta_s_to_b,
            beta_b_to_s=lite_beta_b_to_s,
        )
        )
    if mid_layers > 0:
        specs.append(
            ProgressiveBStageSpec(
            num_layers=mid_layers,
            expanded_nodes=max(1, math.ceil(seq_nodes * mid_expand_ratio)),
            compressed_nodes=max(1, math.ceil(seq_nodes * mid_compress_ratio)),
            alpha_b=mid_alpha_b,
            beta_s_to_b=mid_beta_s_to_b,
            beta_b_to_s=mid_beta_b_to_s,
        )
        )
    if full_layers > 0:
        specs.append(
            ProgressiveBStageSpec(
            num_layers=full_layers,
            expanded_nodes=max(1, math.ceil(seq_nodes * full_expand_ratio)),
            compressed_nodes=max(1, math.ceil(seq_nodes * full_compress_ratio)),
            alpha_b=full_alpha_b,
            beta_s_to_b=full_beta_s_to_b,
            beta_b_to_s=full_beta_b_to_s,
        )
        )
    if not specs:
        raise ValueError("At least one stage must have a positive layer count.")
    return specs


class ProgressiveBJointBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        seq_nodes: int,
        expanded_nodes: int,
        compressed_nodes: int,
        alpha_b: float,
        beta_s_to_b: float,
        beta_b_to_s: float,
        s_window: int = 4,
        route_topk: int = 4,
        expanded_topk: int = 4,
        compressed_topk: int = 4,
        sequence_sparse_type: str = "window",
        expanded_sparse_type: str = "topk",
        compressed_sparse_type: str = "topk",
        route_mode: str = "topk",
        value_norm_kind: str = "layernorm",
        norm_position: str = "post",
        expanded_window: int | None = None,
        compressed_window: int | None = None,
        implementation: str = "streaming",
        propagation_residual: bool = True,
        value_residual_scale: float = 1.0,
        state_residual_scale: float = 1.0,
        alpha_scale: float = 1.0,
        beta_s_to_b_scale: float = 1.0,
        beta_b_to_s_scale: float = 1.0,
        s_delta_scale: float = 0.25,
        b_delta_scale: float = 0.20,
        cross_delta_scale: float = 0.15,
        route_temperature: float = 1.0,
        route_kind: str = "diagonal_bilinear",
        route_hidden_dim: int | None = None,
        edge_dropout_p: float = 0.0,
        usage_dropout_base: float = 0.0,
        usage_dropout_scale: float = 0.0,
        usage_dropout_max: float = 0.0,
        usage_ema_decay: float = 0.99,
        s_pairwise_fn: nn.Module | None = None,
        expanded_pairwise_fn: nn.Module | None = None,
        compressed_pairwise_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.seq_nodes = seq_nodes
        self.expanded_nodes = expanded_nodes
        self.compressed_nodes = compressed_nodes
        self.alpha_b = alpha_b
        self.beta_s_to_b = beta_s_to_b
        self.beta_b_to_s = beta_b_to_s
        self.expanded_ratio = expanded_nodes / seq_nodes
        self.compressed_ratio = compressed_nodes / seq_nodes
        self.route_mode = route_mode
        self.value_norm_kind = value_norm_kind
        self.norm_position = norm_position
        self.alpha_scale = alpha_scale
        self.beta_s_to_b_scale = beta_s_to_b_scale
        self.beta_b_to_s_scale = beta_b_to_s_scale
        self.runtime_b_schedule_scale = 1.0
        self.value_residual_scale = value_residual_scale
        self.state_residual_scale = state_residual_scale
        self.s_delta_scale = s_delta_scale
        self.b_delta_scale = b_delta_scale
        self.cross_delta_scale = cross_delta_scale
        self.propagation_residual = propagation_residual
        self.route_temperature = route_temperature
        self.route_kind = route_kind
        self.route_hidden_dim = route_hidden_dim
        self.edge_dropout_p = edge_dropout_p
        self.usage_dropout_base = usage_dropout_base
        self.usage_dropout_scale = usage_dropout_scale
        self.usage_dropout_max = usage_dropout_max
        self.usage_ema_decay = usage_ema_decay
        self.track_stats = False
        self.last_runtime_stats: dict[str, float] | None = None
        self.s_val_norm = _make_value_norm(dim, value_norm_kind)
        self.expanded_val_norm = _make_value_norm(dim, value_norm_kind)
        self.compressed_val_norm = _make_value_norm(dim, value_norm_kind)
        self.expanded_position_encoding = LearnedPositionEncoding(dim)
        self.compressed_position_encoding = LearnedPositionEncoding(dim)

        self.s_propagation = _make_sparse_or_dense_propagation(
            dim=dim,
            sparse_type=sequence_sparse_type,
            window=s_window,
            implementation=implementation,
            pairwise_fn=s_pairwise_fn,
            residual=propagation_residual,
        )
        self.expand_transition = _make_transition(
            dim=dim,
            route_topk=route_topk,
            route_mode=route_mode,
            route_kind=route_kind,
            route_hidden_dim=route_hidden_dim,
            route_temperature=route_temperature,
            implementation=implementation,
            edge_dropout_p=edge_dropout_p,
            usage_dropout_base=usage_dropout_base,
            usage_dropout_scale=usage_dropout_scale,
            usage_dropout_max=usage_dropout_max,
            usage_ema_decay=usage_ema_decay,
        )
        self.expanded_propagation = _make_sparse_or_dense_propagation(
            dim=dim,
            sparse_type=expanded_sparse_type,
            window=expanded_window,
            topk=min(expanded_topk, expanded_nodes),
            implementation=implementation,
            pairwise_fn=expanded_pairwise_fn,
            residual=propagation_residual,
        )
        self.b_to_s = _make_transition(
            dim=dim,
            route_topk=route_topk,
            route_mode=route_mode,
            route_kind=route_kind,
            route_hidden_dim=route_hidden_dim,
            route_temperature=route_temperature,
            implementation=implementation,
            edge_dropout_p=edge_dropout_p,
            usage_dropout_base=usage_dropout_base,
            usage_dropout_scale=usage_dropout_scale,
            usage_dropout_max=usage_dropout_max,
            usage_ema_decay=usage_ema_decay,
        )
        self.compress_transition = _make_transition(
            dim=dim,
            route_topk=route_topk,
            route_mode=route_mode,
            route_kind=route_kind,
            route_hidden_dim=route_hidden_dim,
            route_temperature=route_temperature,
            implementation=implementation,
            edge_dropout_p=edge_dropout_p,
            usage_dropout_base=usage_dropout_base,
            usage_dropout_scale=usage_dropout_scale,
            usage_dropout_max=usage_dropout_max,
            usage_ema_decay=usage_ema_decay,
        )
        self.s_to_b = _make_transition(
            dim=dim,
            route_topk=route_topk,
            route_mode=route_mode,
            route_kind=route_kind,
            route_hidden_dim=route_hidden_dim,
            route_temperature=route_temperature,
            implementation=implementation,
            edge_dropout_p=edge_dropout_p,
            usage_dropout_base=usage_dropout_base,
            usage_dropout_scale=usage_dropout_scale,
            usage_dropout_max=usage_dropout_max,
            usage_ema_decay=usage_ema_decay,
        )
        self.compressed_propagation = _make_sparse_or_dense_propagation(
            dim=dim,
            sparse_type=compressed_sparse_type,
            window=compressed_window,
            topk=min(compressed_topk, compressed_nodes),
            implementation=implementation,
            pairwise_fn=compressed_pairwise_fn,
            residual=propagation_residual,
        )
        self.compressed_adapter = Transition(
            route_fn=BilinearPairwiseRoute(src_dim=dim, dst_dim=dim),
            merge_mode="add",
            implementation=implementation,
        )

    def set_track_stats(self, enabled: bool) -> None:
        self.track_stats = enabled
        for transition in (
            self.expand_transition,
            self.b_to_s,
            self.compress_transition,
            self.s_to_b,
        ):
            if hasattr(transition, "track_stats"):
                transition.track_stats = enabled
                if not enabled:
                    transition.last_stats = None

    def _prepare_operator_input(self, layer: Layer, val_norm: nn.Module) -> Layer:
        if self.norm_position != "pre":
            return layer
        return _normalize_layer_values(layer, val_norm)

    def _finalize_layer(self, layer: Layer, val_norm: nn.Module) -> Layer:
        return _stabilize_layer(
            layer,
            val_norm,
            apply_value_norm=self.norm_position == "post",
        )

    def _scaled_apply(
        self,
        layer: Layer,
        delta: LayerDelta,
        *,
        path_scale: float,
    ) -> Layer:
        return _apply_scaled_delta(
            layer,
            delta,
            state_scale=path_scale * self.state_residual_scale,
            val_scale=path_scale * self.value_residual_scale,
        )

    @staticmethod
    def _delta_norm(delta: LayerDelta) -> tuple[float, float]:
        state_norm = float(delta.delta_state.norm(dim=-1).mean().item())
        val_norm = float(delta.delta_val.norm(dim=-1).mean().item())
        return state_norm, val_norm

    def set_b_schedule_scale(self, scale: float) -> None:
        if scale < 0.0:
            raise ValueError("B schedule scale must be non-negative.")
        self.runtime_b_schedule_scale = scale

    def _resolve_b_nodes(self, seq_nodes: int) -> tuple[int, int]:
        expanded_nodes = max(1, math.ceil(seq_nodes * self.expanded_ratio))
        compressed_nodes = max(1, math.ceil(seq_nodes * self.compressed_ratio))
        return expanded_nodes, compressed_nodes

    def _make_b_layer(
        self,
        reference: Layer,
        num_nodes: int,
        position_encoding: LearnedPositionEncoding,
    ) -> Layer:
        batch_shape = tuple(reference.batch_shape)
        state = torch.zeros(
            *batch_shape,
            num_nodes,
            device=reference.state.device,
            dtype=reference.state.dtype,
        )
        base_position = position_encoding(
            num_nodes,
            device=reference.val.device,
            dtype=reference.val.dtype,
        )
        val = _expand_position_encoding(base_position, batch_shape)
        return Layer(
            dim=self.dim,
            num_nodes=num_nodes,
            state=state,
            val=val,
        )

    def _prepare_compressed_layer(
        self,
        s_layer: Layer,
        compressed_b: Layer | None,
        compressed_nodes: int,
    ) -> Layer:
        if compressed_b is None:
            return self._make_b_layer(
                s_layer,
                compressed_nodes,
                self.compressed_position_encoding,
            )
        if compressed_b.num_nodes == compressed_nodes:
            return compressed_b
        adapted = self._make_b_layer(
            s_layer,
            compressed_nodes,
            self.compressed_position_encoding,
        )
        return self.compressed_adapter(compressed_b, adapted)

    def forward_sequence_only(self, s_layer: Layer) -> Layer:
        delta = self.s_propagation.compute_delta(
            self._prepare_operator_input(s_layer, self.s_val_norm)
        )
        s_layer = self._scaled_apply(
            s_layer,
            delta,
            path_scale=self.s_delta_scale,
        )
        return self._finalize_layer(s_layer, self.s_val_norm)

    def forward(self, s_layer: Layer, compressed_b: Layer | None = None) -> tuple[Layer, Layer]:
        expanded_nodes, compressed_nodes = self._resolve_b_nodes(s_layer.num_nodes)
        alpha_b = self.alpha_b * self.alpha_scale * self.runtime_b_schedule_scale
        beta_s_to_b = self.beta_s_to_b * self.beta_s_to_b_scale
        beta_b_to_s = self.beta_b_to_s * self.beta_b_to_s_scale
        stats: dict[str, float] = {}
        s_delta = self.s_propagation.compute_delta(
            self._prepare_operator_input(s_layer, self.s_val_norm)
        )
        s_layer = self._scaled_apply(
            s_layer,
            s_delta,
            path_scale=self.s_delta_scale,
        )
        if self.track_stats:
            s_state_norm, s_val_norm = self._delta_norm(s_delta)
            stats["sequence_delta_state_norm"] = s_state_norm
            stats["sequence_delta_val_norm"] = s_val_norm
        s_layer = self._finalize_layer(s_layer, self.s_val_norm)
        compressed_b = self._prepare_compressed_layer(
            s_layer,
            compressed_b,
            compressed_nodes,
        )
        compressed_b = self._finalize_layer(compressed_b, self.compressed_val_norm)

        expanded_b = self._make_b_layer(
            s_layer,
            expanded_nodes,
            self.expanded_position_encoding,
        )
        expand_transition_delta = self.expand_transition.compute_delta(
            self._prepare_operator_input(compressed_b, self.compressed_val_norm),
            self._prepare_operator_input(expanded_b, self.expanded_val_norm),
        )
        expanded_b = self._scaled_apply(
            expanded_b,
            expand_transition_delta,
            path_scale=alpha_b * self.b_delta_scale,
        )
        expanded_prop_delta = self.expanded_propagation.compute_delta(
            self._prepare_operator_input(expanded_b, self.expanded_val_norm)
        )
        expanded_b = self._scaled_apply(
            expanded_b,
            expanded_prop_delta,
            path_scale=alpha_b * self.b_delta_scale,
        )
        expanded_b = self._finalize_layer(expanded_b, self.expanded_val_norm)

        b_to_s_delta = self.b_to_s.compute_delta(
            self._prepare_operator_input(expanded_b, self.expanded_val_norm),
            self._prepare_operator_input(s_layer, self.s_val_norm),
        )
        s_layer = self._scaled_apply(
            s_layer,
            b_to_s_delta,
            path_scale=alpha_b * beta_b_to_s * self.cross_delta_scale,
        )
        s_layer = self._finalize_layer(s_layer, self.s_val_norm)

        next_compressed = compressed_b.clone()
        compress_delta = self.compress_transition.compute_delta(
            self._prepare_operator_input(expanded_b, self.expanded_val_norm),
            self._prepare_operator_input(next_compressed, self.compressed_val_norm),
        )
        next_compressed = self._scaled_apply(
            next_compressed,
            compress_delta,
            path_scale=alpha_b * self.b_delta_scale,
        )
        s_to_b_delta = self.s_to_b.compute_delta(
            self._prepare_operator_input(s_layer, self.s_val_norm),
            self._prepare_operator_input(next_compressed, self.compressed_val_norm),
        )
        next_compressed = self._scaled_apply(
            next_compressed,
            s_to_b_delta,
            path_scale=alpha_b * beta_s_to_b * self.cross_delta_scale,
        )
        compressed_prop_delta = self.compressed_propagation.compute_delta(
            self._prepare_operator_input(next_compressed, self.compressed_val_norm)
        )
        next_compressed = self._scaled_apply(
            next_compressed,
            compressed_prop_delta,
            path_scale=alpha_b * self.b_delta_scale,
        )
        next_compressed = self._finalize_layer(next_compressed, self.compressed_val_norm)
        if self.track_stats:
            for prefix, transition in (
                ("expand", self.expand_transition),
                ("b_to_s", self.b_to_s),
                ("compress", self.compress_transition),
                ("s_to_b", self.s_to_b),
            ):
                if getattr(transition, "last_stats", None):
                    for key, value in transition.last_stats.items():
                        stats[f"{prefix}_{key}"] = value
            for prefix, delta in (
                ("expand_transition", expand_transition_delta),
                ("expanded_propagation", expanded_prop_delta),
                ("b_to_s", b_to_s_delta),
                ("compress_transition", compress_delta),
                ("s_to_b", s_to_b_delta),
                ("compressed_propagation", compressed_prop_delta),
            ):
                state_norm, val_norm = self._delta_norm(delta)
                stats[f"{prefix}_state_norm"] = state_norm
                stats[f"{prefix}_val_norm"] = val_norm
            self.last_runtime_stats = stats
        return s_layer, next_compressed


class ProgressiveBExampleLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        dim: int,
        seq_nodes: int,
        warmup_layers: int = 2,
        stage_specs: Sequence[ProgressiveBStageSpec] | None = None,
        final_refine_layers: int = 2,
        s_window: int = 4,
        route_topk: int = 4,
        expanded_topk: int = 4,
        compressed_topk: int = 4,
        sequence_sparse_type: str = "window",
        expanded_sparse_type: str = "topk",
        compressed_sparse_type: str = "topk",
        route_mode: str = "topk",
        value_norm_kind: str = "layernorm",
        norm_position: str = "post",
        expanded_window: int | None = None,
        compressed_window: int | None = None,
        implementation: str = "streaming",
        propagation_residual: bool = True,
        value_residual_scale: float = 1.0,
        state_residual_scale: float = 1.0,
        alpha_scale: float = 1.0,
        beta_s_to_b_scale: float = 1.0,
        beta_b_to_s_scale: float = 1.0,
        s_delta_scale: float = 0.25,
        b_delta_scale: float = 0.20,
        cross_delta_scale: float = 0.15,
        route_temperature: float = 1.0,
        route_kind: str = "diagonal_bilinear",
        route_hidden_dim: int | None = None,
        edge_dropout_p: float = 0.0,
        usage_dropout_base: float = 0.0,
        usage_dropout_scale: float = 0.0,
        usage_dropout_max: float = 0.0,
        usage_ema_decay: float = 0.99,
        state_init_mode: str = "zero",
        pairwise_kind: str = "diagonal_bilinear",
        pairwise_hidden_dim: int | None = None,
        prediction_slot_index: int = -1,
        query_topk: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_nodes = seq_nodes
        self.prediction_slot_index = prediction_slot_index
        self.propagation_residual = propagation_residual
        self.route_mode = route_mode
        self.sequence_sparse_type = sequence_sparse_type
        self.value_norm_kind = value_norm_kind
        self.norm_position = norm_position
        self.value_residual_scale = value_residual_scale
        self.state_residual_scale = state_residual_scale
        self.alpha_scale = alpha_scale
        self.beta_s_to_b_scale = beta_s_to_b_scale
        self.beta_b_to_s_scale = beta_b_to_s_scale
        self.s_delta_scale = s_delta_scale
        self.b_delta_scale = b_delta_scale
        self.cross_delta_scale = cross_delta_scale
        self.route_temperature = route_temperature
        self.route_kind = route_kind
        self.route_hidden_dim = route_hidden_dim
        self.edge_dropout_p = edge_dropout_p
        self.usage_dropout_base = usage_dropout_base
        self.usage_dropout_scale = usage_dropout_scale
        self.usage_dropout_max = usage_dropout_max
        self.usage_ema_decay = usage_ema_decay
        self.state_init_mode = state_init_mode
        self.pairwise_kind = pairwise_kind
        self.pairwise_hidden_dim = pairwise_hidden_dim
        self.query_topk = query_topk or route_topk
        if self.query_topk <= 0:
            raise ValueError("query_topk must be positive.")
        self.stage_specs = tuple(stage_specs or build_progressive_b_stage_specs(seq_nodes))
        self.track_stats = False
        self.last_runtime_stats: dict[str, float] | None = None

        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_encoding = LearnedPositionEncoding(dim)
        self.state_init = nn.Linear(dim, 1)
        if state_init_mode == "zero":
            nn.init.zeros_(self.state_init.weight)
            nn.init.zeros_(self.state_init.bias)
        elif state_init_mode == "neg_half":
            nn.init.zeros_(self.state_init.weight)
            nn.init.constant_(self.state_init.bias, -0.5)
        elif state_init_mode == "normal":
            nn.init.normal_(self.state_init.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.state_init.bias)
        else:
            raise ValueError(f"Unsupported state_init_mode: {state_init_mode!r}.")
        self.sequence_val_norm = _make_value_norm(dim, value_norm_kind)
        if pairwise_kind == "diagonal_bilinear":
            pairwise_factory = lambda: DiagonalBilinearPairwise(dim=dim)
        elif pairwise_kind == "low_rank_bilinear":
            pairwise_factory = lambda: LowRankBilinearPairwise(
                dim=dim,
                rank=pairwise_hidden_dim or max(1, dim // 2),
            )
        elif pairwise_kind == "hadamard_mlp":
            pairwise_factory = lambda: HadamardMLPPairwise(dim=dim, hidden_dim=pairwise_hidden_dim)
        else:
            raise ValueError(f"Unsupported pairwise_kind: {pairwise_kind!r}.")
        shared_s_pairwise = pairwise_factory()
        shared_expanded_pairwise = pairwise_factory()
        shared_compressed_pairwise = pairwise_factory()
        query_pairwise = pairwise_factory()
        self.query_val_norm = _make_value_norm(dim, value_norm_kind)
        self.query_transition = _make_transition(
            dim=dim,
            route_topk=self.query_topk,
            route_mode=route_mode,
            route_kind=route_kind,
            route_hidden_dim=route_hidden_dim,
            route_temperature=route_temperature,
            implementation=implementation,
            edge_dropout_p=edge_dropout_p,
            usage_dropout_base=usage_dropout_base,
            usage_dropout_scale=usage_dropout_scale,
            usage_dropout_max=usage_dropout_max,
            usage_ema_decay=usage_ema_decay,
        )
        self.query_propagation = QueryTopKPropagation(
            query_pairwise,
            topk=self.query_topk,
            implementation=implementation,
        )
        self.s_warmup = nn.ModuleList(
            [
                _make_sparse_or_dense_propagation(
                    dim=dim,
                    sparse_type=sequence_sparse_type,
                    window=s_window,
                    implementation=implementation,
                    pairwise_fn=shared_s_pairwise,
                    residual=propagation_residual,
                )
                for _ in range(warmup_layers)
            ]
        )
        self.joint_blocks = nn.ModuleList(
            [
                ProgressiveBJointBlock(
                    dim=dim,
                    seq_nodes=seq_nodes,
                    expanded_nodes=stage.expanded_nodes,
                    compressed_nodes=stage.compressed_nodes,
                    alpha_b=stage.alpha_b,
                    beta_s_to_b=stage.beta_s_to_b,
                    beta_b_to_s=stage.beta_b_to_s,
                    s_window=s_window,
                    route_topk=route_topk,
                    expanded_topk=expanded_topk,
                    compressed_topk=compressed_topk,
                    sequence_sparse_type=sequence_sparse_type,
                    expanded_sparse_type=expanded_sparse_type,
                    compressed_sparse_type=compressed_sparse_type,
                    route_mode=route_mode,
                    value_norm_kind=value_norm_kind,
                    norm_position=norm_position,
                    expanded_window=expanded_window,
                    compressed_window=compressed_window,
                    implementation=implementation,
                    propagation_residual=propagation_residual,
                    value_residual_scale=value_residual_scale,
                    state_residual_scale=state_residual_scale,
                    alpha_scale=alpha_scale,
                    beta_s_to_b_scale=beta_s_to_b_scale,
                    beta_b_to_s_scale=beta_b_to_s_scale,
                    s_delta_scale=s_delta_scale,
                    b_delta_scale=b_delta_scale,
                    cross_delta_scale=cross_delta_scale,
                    route_temperature=route_temperature,
                    route_kind=route_kind,
                    route_hidden_dim=route_hidden_dim,
                    edge_dropout_p=edge_dropout_p,
                    usage_dropout_base=usage_dropout_base,
                    usage_dropout_scale=usage_dropout_scale,
                    usage_dropout_max=usage_dropout_max,
                    usage_ema_decay=usage_ema_decay,
                    s_pairwise_fn=shared_s_pairwise,
                    expanded_pairwise_fn=shared_expanded_pairwise,
                    compressed_pairwise_fn=shared_compressed_pairwise,
                )
                for stage in self.stage_specs
                for _ in range(stage.num_layers)
            ]
        )
        self.s_refine = nn.ModuleList(
            [
                _make_sparse_or_dense_propagation(
                    dim=dim,
                    sparse_type=sequence_sparse_type,
                    window=s_window,
                    implementation=implementation,
                    pairwise_fn=shared_s_pairwise,
                    residual=propagation_residual,
                )
                for _ in range(final_refine_layers)
            ]
        )
        self.readout_norm = _make_value_norm(dim, value_norm_kind)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.response_decoder = nn.GRU(
            input_size=dim,
            hidden_size=dim,
            batch_first=True,
        )
        self.response_head_norm = _make_value_norm(dim, value_norm_kind)
        self.response_head = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, vocab_size),
        )
        self.query_head_norm = _make_value_norm(dim, value_norm_kind)
        self.query_head = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, vocab_size),
        )

    def set_b_schedule_scale(self, scale: float) -> None:
        for block in self.joint_blocks:
            block.set_b_schedule_scale(scale)

    def set_track_stats(self, enabled: bool) -> None:
        self.track_stats = enabled
        self.last_runtime_stats = None
        for block in self.joint_blocks:
            block.set_track_stats(enabled)

    def initialize_sequence_layer(
        self,
        token_ids: Tensor,
        *,
        slot_mask: Tensor | None = None,
    ) -> Layer:
        num_nodes = token_ids.shape[-1]
        token_val = self.token_embedding(token_ids)
        position_val = self.position_encoding(
            num_nodes,
            device=token_val.device,
            dtype=token_val.dtype,
        ).unsqueeze(0)
        combined = token_val + position_val
        state = self.state_init(combined).squeeze(-1)
        layer = Layer(
            dim=self.dim,
            num_nodes=num_nodes,
            state=state,
            val=combined,
        )
        layer = _stabilize_layer(
            layer,
            self.sequence_val_norm,
            apply_value_norm=self.norm_position == "post",
        )
        if slot_mask is not None:
            layer = _apply_layer_slot_mask(layer, slot_mask)
        return layer

    def _prepare_sequence_input(self, s_layer: Layer) -> Layer:
        if self.norm_position != "pre":
            return s_layer
        return _normalize_layer_values(s_layer, self.sequence_val_norm)

    def _apply_sequence_delta(self, s_layer: Layer, delta: LayerDelta, *, scale: float) -> Layer:
        return _apply_scaled_delta(
            s_layer,
            delta,
            state_scale=scale * self.state_residual_scale,
            val_scale=scale * self.value_residual_scale,
        )

    def _finalize_sequence_layer(self, s_layer: Layer) -> Layer:
        return _stabilize_layer(
            s_layer,
            self.sequence_val_norm,
            apply_value_norm=self.norm_position == "post",
        )

    def _collect_layer_stats(self, prefix: str, layer: Layer | None) -> dict[str, float]:
        if layer is None:
            return {}
        state = layer.state.detach()
        val = layer.val.detach()
        softplus_state = F.softplus(state)
        stats = {
            f"{prefix}_state_mean": float(state.mean().item()),
            f"{prefix}_state_std": float(state.std(unbiased=False).item()),
            f"{prefix}_softplus_state_mean": float(softplus_state.mean().item()),
            f"{prefix}_softplus_state_std": float(softplus_state.std(unbiased=False).item()),
            f"{prefix}_saturation_ratio": float((state.abs() > 4.0).to(dtype=torch.float32).mean().item()),
            f"{prefix}_val_variance": float(val.var(unbiased=False).item()),
            f"{prefix}_active_slot_ratio": float((softplus_state > 1e-3).to(dtype=torch.float32).mean().item()),
        }
        if layer.num_nodes > 1:
            flat_val = val.reshape(-1, layer.num_nodes, self.dim)
            left = flat_val[:, :-1, :]
            right = flat_val[:, 1:, :]
            cosine = F.cosine_similarity(left, right, dim=-1).mean()
            stats[f"{prefix}_adjacent_cosine"] = float(cosine.item())
        return stats

    def collect_runtime_stats(self, token_ids: Tensor) -> dict[str, float]:
        was_training = self.training
        self.eval()
        self.set_track_stats(True)
        try:
            with torch.no_grad():
                device = next(self.parameters()).device
                token_ids = token_ids.to(device)
                _, s_layer, compressed_b = self(token_ids, return_layers=True)
            stats = {}
            stats.update(self._collect_layer_stats("s", s_layer))
            stats.update(self._collect_layer_stats("b", compressed_b))
            block_stats: list[dict[str, float]] = [
                block.last_runtime_stats
                for block in self.joint_blocks
                if block.last_runtime_stats
            ]
            if block_stats:
                keys = sorted({key for item in block_stats for key in item})
                for key in keys:
                    values = [item[key] for item in block_stats if key in item]
                    if values:
                        stats[f"blocks/{key}_mean"] = float(sum(values) / len(values))
            self.last_runtime_stats = stats
            return stats
        finally:
            self.set_track_stats(False)
            if was_training:
                self.train()

    def read_prediction_slot(self, s_layer: Layer) -> Tensor:
        index = self.prediction_slot_index
        if index < 0:
            index = s_layer.num_nodes + index
        return s_layer.val[..., index, :]

    def read_prediction_slots(self, s_layer: Layer, indices: Tensor) -> Tensor:
        if tuple(indices.shape) != tuple(s_layer.batch_shape):
            raise ValueError(
                "indices must match s_layer batch shape, "
                f"expected {tuple(s_layer.batch_shape)}, got {tuple(indices.shape)}."
            )
        flat_val = s_layer.val.reshape(-1, s_layer.num_nodes, self.dim)
        flat_indices = indices.reshape(-1)
        batch_index = torch.arange(flat_val.shape[0], device=flat_val.device)
        gathered = flat_val[batch_index, flat_indices]
        return gathered.reshape(*indices.shape, self.dim)

    def read_response_slots(self, s_layer: Layer, response_len: int) -> Tensor:
        if response_len <= 0:
            raise ValueError("response_len must be positive.")
        values = self.readout_norm(s_layer.val)
        if response_len <= s_layer.num_nodes:
            return values[..., -response_len:, :]
        extra = values[..., -1:, :].expand(*values.shape[:-2], response_len - s_layer.num_nodes, self.dim)
        return torch.cat((values, extra), dim=-2)

    def _apply_sequence_mask(self, s_layer: Layer, slot_mask: Tensor) -> Layer:
        return _apply_layer_slot_mask(s_layer, slot_mask)

    def encode_prefix(self, token_ids: Tensor) -> tuple[Layer, Layer | None]:
        s_layer = self.initialize_sequence_layer(token_ids)
        compressed_b: Layer | None = None

        for op in self.s_warmup:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)
        for block in self.joint_blocks:
            s_layer, compressed_b = block(s_layer, compressed_b)
        for op in self.s_refine:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)

        if self.track_stats:
            stats = {}
            stats.update(self._collect_layer_stats("s", s_layer))
            stats.update(self._collect_layer_stats("b", compressed_b))
            block_stats = [block.last_runtime_stats for block in self.joint_blocks if block.last_runtime_stats]
            if block_stats:
                keys = sorted({key for item in block_stats for key in item})
                for key in keys:
                    values = [item[key] for item in block_stats if key in item]
                    if values:
                        stats[f"blocks/{key}_mean"] = float(sum(values) / len(values))
            self.last_runtime_stats = stats
        return s_layer, compressed_b

    @staticmethod
    def _concat_layers(first: Layer, second: Layer) -> Layer:
        return Layer(
            dim=first.dim,
            num_nodes=first.num_nodes + second.num_nodes,
            state=torch.cat((first.state, second.state), dim=-1),
            val=torch.cat((first.val, second.val), dim=-2),
        )

    def initialize_query_layer(self, reference: Layer, *, query_nodes: int = 1) -> Layer:
        if query_nodes <= 0:
            raise ValueError("query_nodes must be positive.")
        batch_shape = tuple(reference.batch_shape)
        query_position = self.position_encoding(
            reference.num_nodes + query_nodes,
            device=reference.val.device,
            dtype=reference.val.dtype,
        )[-query_nodes:, :]
        query_val = _expand_position_encoding(query_position, batch_shape)
        query_state = self.state_init(query_val).squeeze(-1)
        query_layer = Layer(
            dim=self.dim,
            num_nodes=query_nodes,
            state=query_state,
            val=query_val,
        )
        return _stabilize_layer(
            query_layer,
            self.query_val_norm,
            apply_value_norm=self.norm_position == "post",
        )

    def _prepare_query_input(self, query_layer: Layer) -> Layer:
        if self.norm_position != "pre":
            return query_layer
        return _normalize_layer_values(query_layer, self.query_val_norm)

    def _finalize_query_layer(self, query_layer: Layer) -> Layer:
        return _stabilize_layer(
            query_layer,
            self.query_val_norm,
            apply_value_norm=self.norm_position == "post",
        )

    def apply_query_transition(
        self,
        source_layer: Layer,
        query_layer: Layer | None = None,
    ) -> Layer:
        if query_layer is None:
            query_layer = self.initialize_query_layer(source_layer)
        transition_delta = self.query_transition.compute_delta(
            self._prepare_sequence_input(source_layer),
            self._prepare_query_input(query_layer),
        )
        query_layer = _apply_scaled_delta(
            query_layer,
            transition_delta,
            state_scale=self.cross_delta_scale * self.state_residual_scale,
            val_scale=self.cross_delta_scale * self.value_residual_scale,
        )
        return self._finalize_query_layer(query_layer)

    def apply_query_propagation(
        self,
        query_layer: Layer,
        propagation_source_layer: Layer | None = None,
    ) -> Layer:
        source_layer = query_layer if propagation_source_layer is None else propagation_source_layer
        propagation_delta = self.query_propagation.compute_delta(
            self._prepare_query_input(query_layer),
            self._prepare_query_input(source_layer),
        )
        query_layer = _apply_scaled_delta(
            query_layer,
            propagation_delta,
            state_scale=self.s_delta_scale * self.state_residual_scale,
            val_scale=self.s_delta_scale * self.value_residual_scale,
        )
        return self._finalize_query_layer(query_layer)

    def update_query_slot(
        self,
        source_layer: Layer,
        query_layer: Layer | None = None,
        *,
        propagation_source_layer: Layer | None = None,
    ) -> Layer:
        query_layer = self.apply_query_transition(source_layer, query_layer)
        if propagation_source_layer is None:
            propagation_source_layer = query_layer
        return self.apply_query_propagation(query_layer, propagation_source_layer)

    def forward_query_block(
        self,
        token_ids: Tensor,
        *,
        target_len: int,
        return_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if target_len <= 0:
            raise ValueError("target_len must be positive.")
        s_layer, compressed_b = self.encode_prefix(token_ids)
        query_template = self.initialize_query_layer(s_layer, query_nodes=target_len)
        transitioned_queries = self.apply_query_transition(s_layer, query_template)
        updated_prefix: Layer | None = None
        for index in range(target_len):
            query_slot = Layer(
                dim=self.dim,
                num_nodes=1,
                state=transitioned_queries.state[..., index : index + 1],
                val=transitioned_queries.val[..., index : index + 1, :],
            )
            propagation_source = query_slot
            if updated_prefix is not None:
                propagation_source = self._concat_layers(updated_prefix, query_slot)
            query_slot = self.apply_query_propagation(query_slot, propagation_source)
            updated_prefix = (
                query_slot
                if updated_prefix is None
                else self._concat_layers(updated_prefix, query_slot)
            )
        assert updated_prefix is not None
        query_slots = self.query_head_norm(updated_prefix.val)
        logits = self.query_head(query_slots)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits

    def forward_query_next_token(
        self,
        token_ids: Tensor,
        *,
        return_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        s_layer, compressed_b = self.encode_prefix(token_ids)
        query_layer = self.apply_query_transition(s_layer)
        query_layer = self.apply_query_propagation(query_layer)
        query_slot = self.query_head_norm(query_layer.val[..., 0, :])
        logits = self.query_head(query_slot)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits

    def forward_response(
        self,
        prefix_token_ids: Tensor,
        decoder_input_ids: Tensor,
        *,
        return_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if decoder_input_ids.ndim != prefix_token_ids.ndim:
            raise ValueError("decoder_input_ids must have the same rank as prefix_token_ids.")
        if tuple(decoder_input_ids.shape[:-1]) != tuple(prefix_token_ids.shape[:-1]):
            raise ValueError("decoder_input_ids batch shape must match prefix_token_ids.")
        response_len = decoder_input_ids.shape[-1]
        s_layer, compressed_b = self.encode_prefix(prefix_token_ids)
        read_slots = self.read_response_slots(s_layer, response_len)
        decoder_inputs = self.token_embedding(decoder_input_ids) + read_slots
        initial_hidden = self.readout_norm(s_layer.val[..., -1, :]).unsqueeze(0)
        decoded, _ = self.response_decoder(decoder_inputs, initial_hidden)
        logits = self.response_head(self.response_head_norm(decoded + read_slots))
        if return_layers:
            return logits, s_layer, compressed_b
        return logits

    def forward_full_sequence_causal(
        self, token_ids: Tensor, *, return_layers: bool = False
    ) -> Tensor | tuple[Tensor, Layer, None]:
        # This path keeps strict causality by using the causal S operators only.
        s_layer = self.initialize_sequence_layer(token_ids)
        for op in self.s_warmup:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)
        for block in self.joint_blocks:
            s_layer = block.forward_sequence_only(s_layer)
        for op in self.s_refine:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)

        logits = self.lm_head(self.readout_norm(s_layer.val))
        if return_layers:
            return logits, s_layer, None
        return logits

    def _forward_teacher_forcing_chunk(
        self,
        base_layer: Layer,
        prediction_indices: Tensor,
    ) -> tuple[Tensor, Layer, Layer | None]:
        batch_shape = tuple(base_layer.batch_shape)
        flat_batch = math.prod(batch_shape) if batch_shape else 1
        device = base_layer.state.device
        chunk_size = prediction_indices.numel()
        num_nodes = base_layer.num_nodes

        base_mask = prediction_indices.view(chunk_size, 1) >= torch.arange(
            num_nodes, device=device
        ).view(1, num_nodes)
        slot_mask = base_mask.unsqueeze(0).expand(flat_batch, -1, -1)
        prediction_indices = prediction_indices.view(1, chunk_size).expand(flat_batch, -1)

        expanded_state = (
            base_layer.state.reshape(flat_batch, num_nodes)
            .unsqueeze(1)
            .expand(flat_batch, chunk_size, num_nodes)
            .reshape(flat_batch * chunk_size, num_nodes)
        )
        expanded_val = (
            base_layer.val.reshape(flat_batch, num_nodes, self.dim)
            .unsqueeze(1)
            .expand(flat_batch, chunk_size, num_nodes, self.dim)
            .reshape(flat_batch * chunk_size, num_nodes, self.dim)
        )
        slot_mask_flat = slot_mask.reshape(flat_batch * chunk_size, num_nodes)
        prediction_indices_flat = prediction_indices.reshape(flat_batch * chunk_size)

        s_layer = Layer(
            dim=self.dim,
            num_nodes=num_nodes,
            state=expanded_state,
            val=expanded_val,
        )
        s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)
        compressed_b: Layer | None = None

        for op in self.s_warmup:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)
            s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)
        for block in self.joint_blocks:
            s_layer, compressed_b = block(s_layer, compressed_b)
            s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)
        for op in self.s_refine:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)
            s_layer = self._apply_sequence_mask(s_layer, slot_mask_flat)

        prediction_slot = self.readout_norm(
            self.read_prediction_slots(s_layer, prediction_indices_flat)
        )
        logits = self.lm_head(prediction_slot).reshape(
            *batch_shape,
            chunk_size,
            self.vocab_size,
        )
        return logits, s_layer, compressed_b

    def forward_teacher_forcing(
        self,
        token_ids: Tensor,
        *,
        return_layers: bool = False,
        teacher_forcing_chunk_size: int | None = None,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if teacher_forcing_chunk_size is not None and teacher_forcing_chunk_size <= 0:
            raise ValueError("teacher_forcing_chunk_size must be positive when provided.")

        base_layer = self.initialize_sequence_layer(token_ids)
        num_nodes = base_layer.num_nodes
        chunk_size = teacher_forcing_chunk_size or num_nodes
        if chunk_size >= num_nodes:
            logits, s_layer, compressed_b = self._forward_teacher_forcing_chunk(
                base_layer,
                torch.arange(num_nodes, device=token_ids.device),
            )
            if return_layers:
                return logits, s_layer, compressed_b
            return logits

        if return_layers:
            raise ValueError(
                "return_layers is not supported when teacher_forcing_chunk_size is smaller "
                "than seq_nodes."
            )

        logits_chunks: list[Tensor] = []
        for chunk_start in range(0, num_nodes, chunk_size):
            chunk_end = min(num_nodes, chunk_start + chunk_size)
            prediction_indices = torch.arange(chunk_start, chunk_end, device=token_ids.device)
            logits_chunk, _, _ = self._forward_teacher_forcing_chunk(
                base_layer,
                prediction_indices,
            )
            logits_chunks.append(logits_chunk)
        return torch.cat(logits_chunks, dim=-2)

    def forward(
        self,
        token_ids: Tensor,
        *,
        return_layers: bool = False,
        teacher_forcing: bool = False,
        full_sequence_causal: bool = False,
        teacher_forcing_chunk_size: int | None = None,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if teacher_forcing and full_sequence_causal:
            raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
        if full_sequence_causal:
            return self.forward_full_sequence_causal(token_ids, return_layers=return_layers)
        if teacher_forcing:
            return self.forward_teacher_forcing(
                token_ids,
                return_layers=return_layers,
                teacher_forcing_chunk_size=teacher_forcing_chunk_size,
            )
        s_layer, compressed_b = self.encode_prefix(token_ids)

        prediction_slot = self.readout_norm(self.read_prediction_slot(s_layer))
        logits = self.lm_head(prediction_slot)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits


@dataclass(frozen=True, slots=True)
class CharVocab:
    stoi: dict[str, int]
    itos: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)


@dataclass(frozen=True, slots=True)
class NextTokenBatch:
    context: Tensor
    target: Tensor


@dataclass(frozen=True, slots=True)
class PrefixResponseBatch:
    context: Tensor
    decoder_input: Tensor
    target: Tensor
    loss_mask: Tensor


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    eval_steps: tuple[int, ...]
    train_step_losses: tuple[float, ...]
    grad_norms: tuple[float, ...]
    train_losses: tuple[float, ...]
    val_losses: tuple[float, ...]


def build_char_vocab(text: str) -> CharVocab:
    symbols = tuple(sorted(set(text)))
    return CharVocab(stoi={ch: idx for idx, ch in enumerate(symbols)}, itos=symbols)


def split_train_val(tokens: Tensor, *, train_fraction: float = 0.9) -> tuple[Tensor, Tensor]:
    split_index = int(tokens.numel() * train_fraction)
    split_index = max(2, min(split_index, tokens.numel() - 2))
    return tokens[:split_index], tokens[split_index:]


def sample_next_token_batch(
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
    target_len: int = 1,
) -> NextTokenBatch:
    if teacher_forcing and full_sequence_causal:
        raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
    if target_len <= 0:
        raise ValueError("target_len must be positive.")
    forecast_len = 1 if teacher_forcing or full_sequence_causal else target_len
    max_start = tokens.numel() - seq_len - forecast_len
    if max_start < 0:
        raise ValueError("The tokenized corpus must be longer than seq_len + target_len.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    context = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
    if teacher_forcing or full_sequence_causal:
        target = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts], dim=0)
    elif target_len > 1:
        target = torch.stack(
            [tokens[start + seq_len : start + seq_len + target_len] for start in starts],
            dim=0,
        )
    else:
        target = torch.stack([tokens[start + seq_len] for start in starts], dim=0)
    return NextTokenBatch(context=context.to(device), target=target.to(device))


class NextTokenBatchDataset(IterableDataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        tokens: Tensor,
        *,
        seq_len: int,
        batch_size: int,
        teacher_forcing: bool = False,
        full_sequence_causal: bool = False,
        target_len: int = 1,
    ) -> None:
        super().__init__()
        self.tokens = tokens.detach().cpu().contiguous()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.teacher_forcing = teacher_forcing
        self.full_sequence_causal = full_sequence_causal
        self.target_len = target_len

    def __iter__(self):
        while True:
            batch = sample_next_token_batch(
                self.tokens,
                seq_len=self.seq_len,
                batch_size=self.batch_size,
                device="cpu",
                teacher_forcing=self.teacher_forcing,
                full_sequence_causal=self.full_sequence_causal,
                target_len=self.target_len,
            )
            yield batch.context, batch.target


def _pad_or_trim_prefix(token_ids: Tensor, *, seq_len: int, pad_token_id: int) -> Tensor:
    token_ids = token_ids.detach().cpu().to(dtype=torch.long)
    if token_ids.numel() >= seq_len:
        return token_ids[-seq_len:].contiguous()
    padding = torch.full((seq_len - token_ids.numel(),), pad_token_id, dtype=torch.long)
    return torch.cat((padding, token_ids), dim=0)


def _make_response_targets(
    response_ids: Tensor,
    *,
    response_len: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor]:
    response_ids = response_ids.detach().cpu().to(dtype=torch.long)
    target = torch.full((response_len,), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros(response_len, dtype=torch.float32)
    content_budget = max(0, response_len - 1)
    content = response_ids[:content_budget]
    used = int(content.numel())
    if used > 0:
        target[:used] = content
        loss_mask[:used] = 1.0
    if used < response_len:
        target[used] = eos_token_id
        loss_mask[used] = 1.0
    decoder_input = torch.full((response_len,), pad_token_id, dtype=torch.long)
    decoder_input[0] = bos_token_id
    if response_len > 1:
        decoder_input[1:] = target[:-1]
    return decoder_input, target, loss_mask


def sample_prefix_response_batch(
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    response_len: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
) -> PrefixResponseBatch:
    if not pairs:
        raise ValueError("pairs must not be empty.")
    if response_len <= 0:
        raise ValueError("response_len must be positive.")
    indices = torch.randint(0, len(pairs), (batch_size,))
    contexts: list[Tensor] = []
    decoder_inputs: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    for index in indices.tolist():
        prefix_ids, response_ids = pairs[index]
        contexts.append(
            _pad_or_trim_prefix(prefix_ids, seq_len=seq_len, pad_token_id=pad_token_id)
        )
        decoder_input, target, loss_mask = _make_response_targets(
            response_ids,
            response_len=response_len,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        decoder_inputs.append(decoder_input)
        targets.append(target)
        masks.append(loss_mask)
    target_device = torch.device(device)
    return PrefixResponseBatch(
        context=torch.stack(contexts, dim=0).to(target_device),
        decoder_input=torch.stack(decoder_inputs, dim=0).to(target_device),
        target=torch.stack(targets, dim=0).to(target_device),
        loss_mask=torch.stack(masks, dim=0).to(target_device),
    )


class PrefixResponseBatchDataset(
    IterableDataset[tuple[Tensor, Tensor, Tensor, Tensor]]
):
    def __init__(
        self,
        pairs: Sequence[tuple[Tensor, Tensor]],
        *,
        seq_len: int,
        response_len: int,
        batch_size: int,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.pairs = tuple((prefix.detach().cpu(), response.detach().cpu()) for prefix, response in pairs)
        self.seq_len = seq_len
        self.response_len = response_len
        self.batch_size = batch_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __iter__(self):
        while True:
            batch = sample_prefix_response_batch(
                self.pairs,
                seq_len=self.seq_len,
                response_len=self.response_len,
                batch_size=self.batch_size,
                bos_token_id=self.bos_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                device="cpu",
            )
            yield batch.context, batch.decoder_input, batch.target, batch.loss_mask


def _make_train_batch_iterator(
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    teacher_forcing: bool,
    full_sequence_causal: bool,
    data_workers: int,
    prefetch_factor: int,
    target_len: int = 1,
):
    target_device = torch.device(device)
    if data_workers <= 0:
        while True:
            yield sample_next_token_batch(
                tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                device=target_device,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                target_len=target_len,
            )
    else:
        dataset = NextTokenBatchDataset(
            tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
            target_len=target_len,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=data_workers,
            pin_memory=target_device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=max(1, prefetch_factor),
        )
        for context, target in loader:
            yield NextTokenBatch(
                context=context.to(target_device, non_blocking=True),
                target=target.to(target_device, non_blocking=True),
            )


def _make_prefix_response_batch_iterator(
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    response_len: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
    data_workers: int,
    prefetch_factor: int,
):
    target_device = torch.device(device)
    if data_workers <= 0:
        while True:
            yield sample_prefix_response_batch(
                pairs,
                seq_len=seq_len,
                response_len=response_len,
                batch_size=batch_size,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                device=target_device,
            )
    else:
        dataset = PrefixResponseBatchDataset(
            pairs,
            seq_len=seq_len,
            response_len=response_len,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=data_workers,
            pin_memory=target_device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=max(1, prefetch_factor),
        )
        for context, decoder_input, target, loss_mask in loader:
            yield PrefixResponseBatch(
                context=context.to(target_device, non_blocking=True),
                decoder_input=decoder_input.to(target_device, non_blocking=True),
                target=target.to(target_device, non_blocking=True),
                loss_mask=loss_mask.to(target_device, non_blocking=True),
            )


def compute_next_token_loss(
    model: nn.Module,
    batch: NextTokenBatch,
    *,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
    query_next_token: bool = False,
    query_block: bool = False,
    teacher_forcing_chunk_size: int | None = None,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if teacher_forcing and full_sequence_causal:
        raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
    if query_next_token and query_block:
        raise ValueError("query_next_token and query_block cannot both be enabled.")
    if (query_next_token or query_block) and (teacher_forcing or full_sequence_causal):
        raise ValueError("query objectives cannot be combined with teacher forcing modes.")
    use_autocast = autocast_device_type is not None and autocast_dtype is not None
    with torch.autocast(
        device_type=autocast_device_type or "cpu",
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        if query_block:
            target_len = batch.target.shape[-1] if batch.target.ndim > 1 else 1
            logits = model.forward_query_block(batch.context, target_len=target_len)
        elif query_next_token:
            logits = model.forward_query_next_token(batch.context)
        else:
            logits = model(
                batch.context,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                teacher_forcing_chunk_size=teacher_forcing_chunk_size,
            )
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.target.reshape(-1),
        )
    return loss, logits


def compute_prefix_response_loss(
    model: nn.Module,
    batch: PrefixResponseBatch,
    *,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    use_autocast = autocast_device_type is not None and autocast_dtype is not None
    with torch.autocast(
        device_type=autocast_device_type or "cpu",
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        logits = model.forward_response(batch.context, batch.decoder_input)
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch.target.reshape(-1),
            reduction="none",
        ).reshape_as(batch.target)
        mask = batch.loss_mask.to(dtype=token_losses.dtype)
        loss = (token_losses * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, logits


@torch.no_grad()
def estimate_next_token_loss(
    model: nn.Module,
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    eval_steps: int,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
    query_next_token: bool = False,
    query_block: bool = False,
    target_len: int = 1,
    teacher_forcing_chunk_size: int | None = None,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        batch = sample_next_token_batch(
            tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
            target_len=target_len,
        )
        loss, _ = compute_next_token_loss(
            model,
            batch,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
            query_next_token=query_next_token,
            query_block=query_block,
            teacher_forcing_chunk_size=teacher_forcing_chunk_size,
            autocast_device_type=autocast_device_type,
            autocast_dtype=autocast_dtype,
        )
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def estimate_prefix_response_loss(
    model: nn.Module,
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    response_len: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
    eval_steps: int,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        batch = sample_prefix_response_batch(
            pairs,
            seq_len=seq_len,
            response_len=response_len,
            batch_size=batch_size,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            device=device,
        )
        loss, _ = compute_prefix_response_loss(
            model,
            batch,
            autocast_device_type=autocast_device_type,
            autocast_dtype=autocast_dtype,
        )
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def train_next_token_model(
    model: nn.Module,
    train_tokens: Tensor,
    val_tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    steps: int,
    eval_interval: int,
    eval_steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
    query_next_token: bool = False,
    query_block: bool = False,
    target_len: int = 1,
    teacher_forcing_chunk_size: int | None = None,
    data_workers: int = 0,
    prefetch_factor: int = 2,
    step_setup_callback: Callable[[int, int], None] | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    step_callback: Callable[[int, float, float], None] | None = None,
    eval_callback: Callable[[int, float, float], None] | None = None,
    checkpoint_callback: Callable[[int, float, float, nn.Module, torch.optim.Optimizer], None] | None = None,
    eval_on_first_step: bool = True,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> TrainingHistory:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive.")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    eval_steps_seen: list[int] = []
    train_step_losses: list[float] = []
    grad_norms: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    batch_iter = _make_train_batch_iterator(
        train_tokens,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
        teacher_forcing=teacher_forcing,
        full_sequence_causal=full_sequence_causal,
        data_workers=data_workers,
        prefetch_factor=prefetch_factor,
        target_len=target_len,
    )

    for step in range(1, steps + 1):
        if step_setup_callback is not None:
            step_setup_callback(step, steps)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(grad_accum_steps):
            batch = next(batch_iter)
            loss, _ = compute_next_token_loss(
                model,
                batch,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                query_next_token=query_next_token,
                query_block=query_block,
                teacher_forcing_chunk_size=teacher_forcing_chunk_size,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            accumulated_loss += float(loss.item())
            (loss / grad_accum_steps).backward()
        grad_norm = 0.0
        if grad_clip is not None:
            grad_norm = float(
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip).item()
            )
        else:
            grads = [parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]
            if grads:
                grad_norm = float(torch.stack(grads).norm().item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite grad norm at step {step}: {grad_norm}")
        optimizer.step()
        step_loss = accumulated_loss / grad_accum_steps
        train_step_losses.append(step_loss)
        grad_norms.append(grad_norm)
        if progress_callback is not None:
            progress_callback(step, steps, step_loss)
        if step_callback is not None:
            step_callback(step, step_loss, grad_norm)

        if step % eval_interval == 0 or (eval_on_first_step and step == 1) or step == steps:
            eval_steps_seen.append(step)
            train_eval = estimate_next_token_loss(
                model,
                train_tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                eval_steps=eval_steps,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                query_next_token=query_next_token,
                query_block=query_block,
                target_len=target_len,
                teacher_forcing_chunk_size=teacher_forcing_chunk_size,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
            val_eval = estimate_next_token_loss(
                model,
                val_tokens,
                seq_len=seq_len,
                batch_size=batch_size,
                device=device,
                eval_steps=eval_steps,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                query_next_token=query_next_token,
                query_block=query_block,
                target_len=target_len,
                teacher_forcing_chunk_size=teacher_forcing_chunk_size,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
            train_losses.append(train_eval)
            val_losses.append(val_eval)
            if checkpoint_callback is not None:
                checkpoint_callback(step, train_eval, val_eval, model, optimizer)
            if eval_callback is not None:
                eval_callback(step, train_eval, val_eval)

    return TrainingHistory(
        eval_steps=tuple(eval_steps_seen),
        train_step_losses=tuple(train_step_losses),
        grad_norms=tuple(grad_norms),
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
    )


def train_prefix_response_model(
    model: nn.Module,
    train_pairs: Sequence[tuple[Tensor, Tensor]],
    val_pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    response_len: int,
    batch_size: int,
    bos_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
    steps: int,
    eval_interval: int,
    eval_steps: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    grad_clip: float | None = 1.0,
    grad_accum_steps: int = 1,
    data_workers: int = 0,
    prefetch_factor: int = 2,
    step_setup_callback: Callable[[int, int], None] | None = None,
    progress_callback: Callable[[int, int, float], None] | None = None,
    step_callback: Callable[[int, float, float], None] | None = None,
    eval_callback: Callable[[int, float, float], None] | None = None,
    checkpoint_callback: Callable[[int, float, float, nn.Module, torch.optim.Optimizer], None] | None = None,
    eval_on_first_step: bool = True,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> TrainingHistory:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive.")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    eval_steps_seen: list[int] = []
    train_step_losses: list[float] = []
    grad_norms: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    batch_iter = _make_prefix_response_batch_iterator(
        train_pairs,
        seq_len=seq_len,
        response_len=response_len,
        batch_size=batch_size,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        device=device,
        data_workers=data_workers,
        prefetch_factor=prefetch_factor,
    )

    for step in range(1, steps + 1):
        if step_setup_callback is not None:
            step_setup_callback(step, steps)
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(grad_accum_steps):
            batch = next(batch_iter)
            loss, _ = compute_prefix_response_loss(
                model,
                batch,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            accumulated_loss += float(loss.item())
            (loss / grad_accum_steps).backward()
        grad_norm = 0.0
        if grad_clip is not None:
            grad_norm = float(
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip).item()
            )
        else:
            grads = [parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]
            if grads:
                grad_norm = float(torch.stack(grads).norm().item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite grad norm at step {step}: {grad_norm}")
        optimizer.step()
        step_loss = accumulated_loss / grad_accum_steps
        train_step_losses.append(step_loss)
        grad_norms.append(grad_norm)
        if progress_callback is not None:
            progress_callback(step, steps, step_loss)
        if step_callback is not None:
            step_callback(step, step_loss, grad_norm)

        if step % eval_interval == 0 or (eval_on_first_step and step == 1) or step == steps:
            eval_steps_seen.append(step)
            train_eval = estimate_prefix_response_loss(
                model,
                train_pairs,
                seq_len=seq_len,
                response_len=response_len,
                batch_size=batch_size,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                device=device,
                eval_steps=eval_steps,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
            val_eval = estimate_prefix_response_loss(
                model,
                val_pairs,
                seq_len=seq_len,
                response_len=response_len,
                batch_size=batch_size,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                device=device,
                eval_steps=eval_steps,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
            train_losses.append(train_eval)
            val_losses.append(val_eval)
            if checkpoint_callback is not None:
                checkpoint_callback(step, train_eval, val_eval, model, optimizer)
            if eval_callback is not None:
                eval_callback(step, train_eval, val_eval)

    return TrainingHistory(
        eval_steps=tuple(eval_steps_seen),
        train_step_losses=tuple(train_step_losses),
        grad_norms=tuple(grad_norms),
        train_losses=tuple(train_losses),
        val_losses=tuple(val_losses),
    )


@torch.no_grad()
def generate_next_tokens(
    model: nn.Module,
    prompt: Tensor,
    *,
    max_new_tokens: int,
    seq_len: int,
    device: torch.device | str,
) -> Tensor:
    was_training = model.training
    model.eval()
    generated = prompt.to(device).clone()
    for _ in range(max_new_tokens):
        context = generated[-seq_len:].unsqueeze(0)
        logits = model(context)
        next_token = torch.argmax(logits, dim=-1)
        generated = torch.cat((generated, next_token), dim=0)
    if was_training:
        model.train()
    return generated


def perplexity_from_loss(loss: float) -> float:
    if loss > 80.0:
        return float("inf")
    return float(math.exp(loss))
