from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from jakal_net import (
    AdditiveLowRankPairwise,
    AdditiveLowRankRoute,
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
    propagation_query_dense_native,
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
    merge_mode: str = "replace",
) -> Layer:
    if state_scale == 0.0 and val_scale == 0.0:
        return layer
    return layer.apply_delta(
        _scale_delta(delta, state_scale=state_scale, val_scale=val_scale),
        merge_mode=merge_mode,
    )


def _zero_loss(reference: Tensor) -> Tensor:
    return reference.float().sum() * 0.0


def _route_destination_concentration_loss(
    logits: Tensor,
    *,
    load_cap: float,
) -> tuple[Tensor, Tensor]:
    if logits.shape[-1] <= 1:
        return _zero_loss(logits), _zero_loss(logits)
    routes = torch.softmax(logits.float(), dim=-1)
    destination_load = routes.sum(dim=-2)
    destination_load = destination_load / destination_load.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    max_load = destination_load.max(dim=-1).values
    loss = (max_load - load_cap).clamp_min(0.0).square().mean()
    return loss, max_load.mean()


def _edge_probability_concentration_loss(
    scores: Tensor,
    *,
    prob_cap: float,
) -> tuple[Tensor, Tensor]:
    if scores.shape[-1] <= 1:
        return _zero_loss(scores), _zero_loss(scores)
    edge_probs = torch.softmax(scores.float(), dim=-1)
    max_prob = edge_probs.max(dim=-1).values
    loss = (max_prob - prob_cap).clamp_min(0.0).square().mean()
    return loss, max_prob.mean()


def _transition_route_concentration_loss(
    transition: Transition | SparseTransition,
    src_layer: Layer,
    dst_layer: Layer,
    *,
    load_cap: float,
) -> tuple[Tensor, Tensor]:
    logits = transition.compute_route_logits(src_layer, dst_layer)
    return _route_destination_concentration_loss(logits, load_cap=load_cap)


def _layer_cosine_duplicate_loss(
    layer: Layer | None,
    *,
    margin: float,
    reference: Tensor,
) -> tuple[Tensor, Tensor]:
    if layer is None or layer.num_nodes <= 1:
        zero = _zero_loss(reference)
        return zero, zero
    values = F.normalize(layer.val.float(), dim=-1)
    cosine = torch.matmul(values, values.transpose(-1, -2))
    off_diagonal_mask = ~torch.eye(layer.num_nodes, device=cosine.device, dtype=torch.bool)
    off_diagonal = cosine[..., off_diagonal_mask]
    over_margin = (off_diagonal - margin).clamp_min(0.0)
    loss = over_margin.mean()
    mean_high_cosine = over_margin.mean()
    return loss, mean_high_cosine


def _set_optimizer_hyperparams(
    optimizer: torch.optim.Optimizer,
    *,
    learning_rate: float,
    weight_decay: float | None = None,
) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
        if weight_decay is not None:
            param_group["weight_decay"] = weight_decay


def _resolve_scheduled_learning_rate(
    *,
    base_learning_rate: float,
    schedule: str,
    warmup_steps: int,
    warmup_start_learning_rate: float | None,
    min_learning_rate_ratio: float,
    step: int,
    start_step: int,
    total_steps: int,
) -> float:
    if schedule == "none":
        return base_learning_rate
    if schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {schedule!r}.")
    relative_step = max(1, step - start_step)
    if warmup_steps > 0 and relative_step <= warmup_steps:
        if warmup_start_learning_rate is not None:
            if warmup_steps == 1:
                return base_learning_rate
            warmup_progress = (relative_step - 1) / (warmup_steps - 1)
            return warmup_start_learning_rate + (
                base_learning_rate - warmup_start_learning_rate
            ) * warmup_progress
        return base_learning_rate * (relative_step / warmup_steps)
    decay_steps = max(1, total_steps - start_step - warmup_steps)
    decay_progress = min(1.0, max(0.0, (relative_step - warmup_steps) / decay_steps))
    cosine_scale = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return base_learning_rate * (
        min_learning_rate_ratio + (1.0 - min_learning_rate_ratio) * cosine_scale
    )


def _grad_group_keys(parameter_name: str, *, key_prefix: str = "grad") -> tuple[str, ...]:
    groups: list[str] = []
    if parameter_name.startswith("joint_blocks."):
        parts = parameter_name.split(".")
        if len(parts) >= 3:
            block_index = parts[1]
            component = parts[2]
            groups.append(f"{key_prefix}/joint_blocks/{block_index}/total")
            groups.append(f"{key_prefix}/joint_blocks/{block_index}/{component}")
        return tuple(groups)
    direct_prefixes = (
        "token_embedding",
        "position_encoding",
        "state_init",
        "query_transition",
        "query_propagation",
        "query_head",
        "query_head_norm",
        "query_feedback_proj",
        "query_feedback_norm",
        "sequence_val_norm",
        "query_val_norm",
    )
    for prefix in direct_prefixes:
        if parameter_name.startswith(prefix):
            return (f"{key_prefix}/{prefix}",)
    if parameter_name.startswith("s_warmup."):
        parts = parameter_name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            layer_index = parts[1]
            return (
                f"{key_prefix}/s_warmup",
                f"{key_prefix}/s_warmup/{layer_index}",
            )
        return (f"{key_prefix}/s_warmup",)
    if parameter_name.startswith("s_refine."):
        return (f"{key_prefix}/s_refine",)
    return (f"{key_prefix}/other",)


def _collect_grad_group_norms(
    model: nn.Module,
    *,
    key_prefix: str = "grad",
) -> dict[str, float]:
    grad_square_sums: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        square_sum = float(grad.square().sum().item())
        for group in _grad_group_keys(name, key_prefix=key_prefix):
            grad_square_sums[group] = grad_square_sums.get(group, 0.0) + square_sum
    return {
        key: math.sqrt(value)
        for key, value in sorted(grad_square_sums.items())
        if value > 0.0
    }


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


def _sample_slot_keep_mask(layer: Layer, *, dropout_p: float) -> Tensor | None:
    if dropout_p <= 0.0 or dropout_p >= 1.0 or layer.num_nodes <= 1:
        return None
    mask = (
        torch.rand(
            *layer.batch_shape,
            layer.num_nodes,
            device=layer.val.device,
        )
        >= dropout_p
    )
    flat_mask = mask.reshape(-1, layer.num_nodes)
    empty_rows = ~flat_mask.any(dim=-1)
    if empty_rows.any():
        random_indices = torch.randint(
            0,
            layer.num_nodes,
            (int(empty_rows.sum().item()),),
            device=layer.val.device,
        )
        flat_mask[empty_rows, random_indices] = True
    return flat_mask.reshape(*layer.batch_shape, layer.num_nodes)


def _apply_scaled_slot_mask(layer: Layer, slot_mask: Tensor, *, keep_prob: float) -> Layer:
    scale = slot_mask.to(dtype=layer.val.dtype) / keep_prob
    return layer.with_tensors(
        state=layer.state * scale.to(dtype=layer.state.dtype),
        val=layer.val * scale.unsqueeze(-1),
    )


def _apply_scaled_delta_slot_mask(
    delta: LayerDelta,
    slot_mask: Tensor,
    *,
    keep_prob: float,
) -> LayerDelta:
    scale = slot_mask.to(dtype=delta.delta_val.dtype) / keep_prob
    return LayerDelta(
        delta_state=delta.delta_state * scale.to(dtype=delta.delta_state.dtype),
        delta_val=delta.delta_val * scale.unsqueeze(-1),
    )


def _expand_position_encoding(
    position_encoding: Tensor,
    batch_shape: tuple[int, ...],
) -> Tensor:
    view_shape = (1,) * len(batch_shape) + tuple(position_encoding.shape)
    return position_encoding.view(view_shape).expand(*batch_shape, *position_encoding.shape)


def _resize_slot_parameter(
    parameter: Tensor,
    *,
    num_nodes: int,
) -> Tensor:
    if parameter.shape[0] == num_nodes:
        return parameter
    if parameter.ndim == 1:
        resized = F.interpolate(
            parameter.view(1, 1, -1),
            size=num_nodes,
            mode="linear",
            align_corners=True,
        )
        return resized.view(num_nodes)
    if parameter.ndim == 2:
        resized = F.interpolate(
            parameter.transpose(0, 1).unsqueeze(0),
            size=num_nodes,
            mode="linear",
            align_corners=True,
        )
        return resized.squeeze(0).transpose(0, 1)
    raise ValueError(f"Unsupported slot parameter rank: {parameter.ndim}.")


def _expand_slot_template(slot_parameter: Tensor, batch_shape: tuple[int, ...]) -> Tensor:
    view_shape = (1,) * len(batch_shape) + tuple(slot_parameter.shape)
    return slot_parameter.view(view_shape).expand(*batch_shape, *slot_parameter.shape)


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


def _make_route_fn(
    *,
    dim: int,
    route_kind: str,
    route_hidden_dim: int | None,
    route_temperature: float,
) -> nn.Module:
    if route_kind == "diagonal_bilinear":
        route_fn: nn.Module = DiagonalBilinearRoute(src_dim=dim, dst_dim=dim)
    elif route_kind == "low_rank_bilinear":
        route_fn = LowRankBilinearRoute(
            src_dim=dim,
            dst_dim=dim,
            rank=route_hidden_dim or max(1, dim // 2),
        )
    elif route_kind == "additive_low_rank":
        route_fn = AdditiveLowRankRoute(
            src_dim=dim,
            dst_dim=dim,
            hidden_dim=route_hidden_dim or max(1, dim // 16),
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
    return route_fn


def _dense_causal_block_mask(
    *,
    target_start: int,
    target_end: int,
    source_start: int,
    source_end: int,
    device: torch.device,
    batch_shape: tuple[int, ...],
) -> Tensor:
    target_positions = torch.arange(target_start, target_end, device=device)
    source_positions = torch.arange(source_start, source_end, device=device)
    mask_2d = source_positions.unsqueeze(0) <= target_positions.unsqueeze(1)
    return mask_2d.view((1,) * len(batch_shape) + mask_2d.shape)


def _compute_dense_causal_propagation_delta(
    propagation: Propagation,
    layer: Layer,
) -> LayerDelta:
    projected_state, projected_val = propagation._project_inputs(layer)
    delta_state, delta_val = propagation._allocate_delta_buffers(
        layer, projected_state, projected_val
    )
    target_block_size = propagation.target_block_size or layer.num_nodes
    source_block_size = propagation.source_block_size or layer.num_nodes
    if isinstance(propagation.pairwise_fn, HadamardMLPPairwise):
        target_block_size = min(target_block_size, 8)
        source_block_size = min(source_block_size, 8)
    state_acc_dtype = delta_state.dtype
    val_acc_dtype = delta_val.dtype

    for target_start in range(0, layer.num_nodes, target_block_size):
        target_end = min(layer.num_nodes, target_start + target_block_size)
        target_val = layer.val[..., target_start:target_end, :]

        for source_start in range(0, target_end, source_block_size):
            source_end = min(target_end, source_start + source_block_size)
            source_val = layer.val[..., source_start:source_end, :]
            scores = propagation.pairwise_fn(target_val, source_val)
            edges = propagation.edge_compress_fn(scores) * _dense_causal_block_mask(
                target_start=target_start,
                target_end=target_end,
                source_start=source_start,
                source_end=source_end,
                device=scores.device,
                batch_shape=layer.batch_shape,
            ).to(dtype=scores.dtype)
            delta_state[..., target_start:target_end] += torch.einsum(
                "...ij,...j->...i",
                edges.to(dtype=state_acc_dtype),
                projected_state[..., source_start:source_end].to(dtype=state_acc_dtype),
            )
            delta_val[..., target_start:target_end, :] += torch.einsum(
                "...ij,...jd->...id",
                edges.to(dtype=val_acc_dtype),
                projected_val[..., source_start:source_end, :].to(dtype=val_acc_dtype),
            )

    return LayerDelta(
        delta_state=delta_state.to(dtype=projected_state.dtype),
        delta_val=delta_val.to(dtype=projected_val.dtype),
    )


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
            query_state_acc = torch.zeros(
                *query_layer.batch_shape,
                query_end - query_start,
                device=query_layer.state.device,
                dtype=projected_state.dtype,
            )
            query_val_acc = torch.zeros(
                *query_layer.batch_shape,
                query_end - query_start,
                query_layer.dim,
                device=query_layer.val.device,
                dtype=projected_val.dtype,
            )
            for source_start in range(0, source_nodes, self.source_block_size or source_nodes):
                source_end = min(source_nodes, source_start + (self.source_block_size or source_nodes))
                source_val = source_layer.val[..., source_start:source_end, :]
                scores = self.pairwise_fn(query_val, source_val)
                edges = self.edge_compress_fn(scores)
                query_state_acc += torch.einsum(
                    "...ij,...j->...i",
                    edges.to(projected_state.dtype),
                    projected_state[..., source_start:source_end].to(projected_state.dtype),
                )
                query_val_acc += torch.einsum(
                    "...ij,...jd->...id",
                    edges.to(projected_val.dtype),
                    projected_val[..., source_start:source_end, :].to(projected_val.dtype),
                )

            delta_state[..., query_start:query_end] = query_state_acc
            delta_val[..., query_start:query_end, :] = query_val_acc

        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def compute_delta(self, query_layer: Layer, source_layer: Layer) -> LayerDelta:
        edge_compress_name = self._native_edge_compress_name()
        if (
            self.implementation == "native"
            and edge_compress_name is not None
            and supports_pairwise_kernel(self.pairwise_fn)
            and native_supports("propagation_query_dense")
            and native_supports_device(query_layer.val.device.type)
        ):
            return propagation_query_dense_native(
                pairwise_fn=self.pairwise_fn,
                edge_compress_name=edge_compress_name,
                query_val=query_layer.val,
                source_val=source_layer.val,
                projected_state=source_layer.state,
                projected_val=source_layer.val,
                query_block_size=self.query_block_size or query_layer.num_nodes,
                source_block_size=self.source_block_size or source_layer.num_nodes,
            )
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
    route_fn: nn.Module | None = None,
    merge_mode: str = "add",
    edge_dropout_p: float = 0.0,
    usage_dropout_base: float = 0.0,
    usage_dropout_scale: float = 0.0,
    usage_dropout_max: float = 0.0,
    usage_ema_decay: float = 0.99,
) -> Transition | SparseTransition:
    if route_fn is None:
        route_fn = _make_route_fn(
            dim=dim,
            route_kind=route_kind,
            route_hidden_dim=route_hidden_dim,
            route_temperature=route_temperature,
        )
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
        block_residual: bool = False,
        value_residual_scale: float = 1.0,
        state_residual_scale: float = 1.0,
        alpha_scale: float = 1.0,
        beta_s_to_b_scale: float = 1.0,
        beta_b_to_s_scale: float = 1.0,
        warmup_delta_scale: float = 0.0,
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
        b_slot_dropout_p: float = 0.0,
        s_to_b_slot_dropout_p: float = 0.0,
        b_to_s_slot_dropout_p: float = 0.0,
        s_pairwise_fn: nn.Module | None = None,
        expanded_pairwise_fn: nn.Module | None = None,
        compressed_pairwise_fn: nn.Module | None = None,
        expand_route_fn: nn.Module | None = None,
        b_to_s_route_fn: nn.Module | None = None,
        compress_route_fn: nn.Module | None = None,
        s_to_b_route_fn: nn.Module | None = None,
        compressed_adapter_route_fn: nn.Module | None = None,
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
        self.block_residual = block_residual
        self.warmup_delta_scale = warmup_delta_scale
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
        self.b_slot_dropout_p = b_slot_dropout_p
        self.s_to_b_slot_dropout_p = s_to_b_slot_dropout_p
        self.b_to_s_slot_dropout_p = b_to_s_slot_dropout_p
        self.track_stats = False
        self.last_runtime_stats: dict[str, float] | None = None
        self.collect_aux_losses = False
        self.route_load_cap = 0.25
        self.last_aux_losses: dict[str, Tensor] | None = None
        self.last_aux_stats: dict[str, float] | None = None
        self.s_val_norm = _make_value_norm(dim, value_norm_kind)
        self.expanded_val_norm = _make_value_norm(dim, value_norm_kind)
        self.compressed_val_norm = _make_value_norm(dim, value_norm_kind)
        self.expanded_slot_state = nn.Parameter(torch.empty(expanded_nodes))
        self.expanded_slot_val = nn.Parameter(torch.empty(expanded_nodes, dim))
        self.compressed_slot_state = nn.Parameter(torch.empty(compressed_nodes))
        self.compressed_slot_val = nn.Parameter(torch.empty(compressed_nodes, dim))
        nn.init.normal_(self.expanded_slot_state, mean=0.0, std=0.02)
        nn.init.normal_(self.expanded_slot_val, mean=0.0, std=0.02)
        nn.init.normal_(self.compressed_slot_state, mean=0.0, std=0.02)
        nn.init.normal_(self.compressed_slot_val, mean=0.0, std=0.02)

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
            route_fn=expand_route_fn,
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
            route_fn=b_to_s_route_fn,
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
            route_fn=compress_route_fn,
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
            route_fn=s_to_b_route_fn,
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
            route_fn=(
                BilinearPairwiseRoute(src_dim=dim, dst_dim=dim)
                if compressed_adapter_route_fn is None
                else compressed_adapter_route_fn
            ),
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

    def set_aux_loss_collection(self, enabled: bool, *, route_load_cap: float) -> None:
        self.collect_aux_losses = enabled
        self.route_load_cap = route_load_cap
        self.last_aux_losses = None
        self.last_aux_stats = None

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
            merge_mode="add" if self.block_residual else "replace",
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
        *,
        slot_state_template: Tensor,
        slot_val_template: Tensor,
    ) -> Layer:
        batch_shape = tuple(reference.batch_shape)
        slot_state = _resize_slot_parameter(
            slot_state_template,
            num_nodes=num_nodes,
        )
        slot_val = _resize_slot_parameter(
            slot_val_template,
            num_nodes=num_nodes,
        )
        state = _expand_slot_template(slot_state, batch_shape).to(
            device=reference.state.device,
            dtype=reference.state.dtype,
        )
        val = _expand_slot_template(slot_val, batch_shape).to(
            device=reference.val.device,
            dtype=reference.val.dtype,
        )
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
                slot_state_template=self.compressed_slot_state,
                slot_val_template=self.compressed_slot_val,
            )
        if compressed_b.num_nodes == compressed_nodes:
            return compressed_b
        adapted = self._make_b_layer(
            s_layer,
            compressed_nodes,
            slot_state_template=self.compressed_slot_state,
            slot_val_template=self.compressed_slot_val,
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
        aux_losses: dict[str, Tensor] = {}
        aux_stats: dict[str, float] = {}

        def add_route_concentration(
            name: str,
            transition: Transition | SparseTransition,
            src: Layer,
            dst: Layer,
        ) -> None:
            if not self.collect_aux_losses:
                return
            loss, max_load = _transition_route_concentration_loss(
                transition,
                src,
                dst,
                load_cap=self.route_load_cap,
            )
            aux_losses[f"route_concentration/{name}"] = loss
            aux_stats[f"route_concentration/{name}_max_load"] = float(max_load.detach().item())

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
        if self.training and self.b_slot_dropout_p > 0.0:
            compressed_slot_mask = _sample_slot_keep_mask(
                compressed_b,
                dropout_p=self.b_slot_dropout_p,
            )
            assert compressed_slot_mask is not None
            compressed_b = _apply_scaled_slot_mask(
                compressed_b,
                compressed_slot_mask,
                keep_prob=1.0 - self.b_slot_dropout_p,
            )

        expanded_b = self._make_b_layer(
            s_layer,
            expanded_nodes,
            slot_state_template=self.expanded_slot_state,
            slot_val_template=self.expanded_slot_val,
        )
        expand_src = self._prepare_operator_input(compressed_b, self.compressed_val_norm)
        expand_dst = self._prepare_operator_input(expanded_b, self.expanded_val_norm)
        add_route_concentration("expand", self.expand_transition, expand_src, expand_dst)
        expand_transition_delta = self.expand_transition.compute_delta(
            expand_src,
            expand_dst,
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
        if self.training and self.b_slot_dropout_p > 0.0:
            expanded_slot_mask = _sample_slot_keep_mask(
                expanded_b,
                dropout_p=self.b_slot_dropout_p,
            )
            assert expanded_slot_mask is not None
            expanded_b = _apply_scaled_slot_mask(
                expanded_b,
                expanded_slot_mask,
                keep_prob=1.0 - self.b_slot_dropout_p,
            )

        b_to_s_src = self._prepare_operator_input(expanded_b, self.expanded_val_norm)
        b_to_s_dst = self._prepare_operator_input(s_layer, self.s_val_norm)
        b_to_s_slot_mask = None
        if self.training and self.b_to_s_slot_dropout_p > 0.0:
            b_to_s_slot_mask = _sample_slot_keep_mask(
                b_to_s_dst,
                dropout_p=self.b_to_s_slot_dropout_p,
            )
            assert b_to_s_slot_mask is not None
            b_to_s_dst = _apply_scaled_slot_mask(
                b_to_s_dst,
                b_to_s_slot_mask,
                keep_prob=1.0 - self.b_to_s_slot_dropout_p,
            )
        add_route_concentration("b_to_s", self.b_to_s, b_to_s_src, b_to_s_dst)
        b_to_s_delta = self.b_to_s.compute_delta(
            b_to_s_src,
            b_to_s_dst,
        )
        if b_to_s_slot_mask is not None:
            b_to_s_delta = _apply_scaled_delta_slot_mask(
                b_to_s_delta,
                b_to_s_slot_mask,
                keep_prob=1.0 - self.b_to_s_slot_dropout_p,
            )
        s_layer = self._scaled_apply(
            s_layer,
            b_to_s_delta,
            path_scale=alpha_b * beta_b_to_s * self.cross_delta_scale,
        )
        s_layer = self._finalize_layer(s_layer, self.s_val_norm)

        next_compressed = compressed_b.clone()
        compress_src = self._prepare_operator_input(expanded_b, self.expanded_val_norm)
        compress_dst = self._prepare_operator_input(next_compressed, self.compressed_val_norm)
        add_route_concentration("compress", self.compress_transition, compress_src, compress_dst)
        compress_delta = self.compress_transition.compute_delta(
            compress_src,
            compress_dst,
        )
        next_compressed = self._scaled_apply(
            next_compressed,
            compress_delta,
            path_scale=alpha_b * self.b_delta_scale,
        )
        s_to_b_src = self._prepare_operator_input(s_layer, self.s_val_norm)
        s_to_b_dst = self._prepare_operator_input(next_compressed, self.compressed_val_norm)
        s_to_b_slot_mask = None
        if self.training and self.s_to_b_slot_dropout_p > 0.0:
            s_to_b_slot_mask = _sample_slot_keep_mask(
                s_to_b_dst,
                dropout_p=self.s_to_b_slot_dropout_p,
            )
            assert s_to_b_slot_mask is not None
            s_to_b_dst = _apply_scaled_slot_mask(
                s_to_b_dst,
                s_to_b_slot_mask,
                keep_prob=1.0 - self.s_to_b_slot_dropout_p,
            )
        add_route_concentration("s_to_b", self.s_to_b, s_to_b_src, s_to_b_dst)
        s_to_b_delta = self.s_to_b.compute_delta(
            s_to_b_src,
            s_to_b_dst,
        )
        if s_to_b_slot_mask is not None:
            s_to_b_delta = _apply_scaled_delta_slot_mask(
                s_to_b_delta,
                s_to_b_slot_mask,
                keep_prob=1.0 - self.s_to_b_slot_dropout_p,
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
        if self.training and self.b_slot_dropout_p > 0.0:
            next_compressed_slot_mask = _sample_slot_keep_mask(
                next_compressed,
                dropout_p=self.b_slot_dropout_p,
            )
            assert next_compressed_slot_mask is not None
            next_compressed = _apply_scaled_slot_mask(
                next_compressed,
                next_compressed_slot_mask,
                keep_prob=1.0 - self.b_slot_dropout_p,
            )
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
        self.last_aux_losses = aux_losses if aux_losses else None
        self.last_aux_stats = aux_stats if aux_stats else None
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
        query_refine_layers: int | None = None,
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
        block_residual: bool = False,
        query_residual: bool = False,
        value_residual_scale: float = 1.0,
        state_residual_scale: float = 1.0,
        alpha_scale: float = 1.0,
        beta_s_to_b_scale: float = 1.0,
        beta_b_to_s_scale: float = 1.0,
        warmup_delta_scale: float = 0.0,
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
        b_slot_dropout_p: float = 0.0,
        s_to_b_slot_dropout_p: float = 0.0,
        b_to_s_slot_dropout_p: float = 0.0,
        state_init_mode: str = "zero",
        pairwise_kind: str = "diagonal_bilinear",
        pairwise_hidden_dim: int | None = None,
        prediction_slot_index: int = -1,
        query_topk: int | None = None,
        include_query_head: bool = True,
        include_query_rnn_head: bool = False,
        query_rnn_hidden_dim: int | None = None,
        query_rnn_head_width: int | None = None,
        share_route_families: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_nodes = seq_nodes
        self.prediction_slot_index = prediction_slot_index
        self.propagation_residual = propagation_residual
        self.block_residual = block_residual
        self.query_residual = query_residual
        self.route_mode = route_mode
        self.sequence_sparse_type = sequence_sparse_type
        self.value_norm_kind = value_norm_kind
        self.norm_position = norm_position
        self.value_residual_scale = value_residual_scale
        self.state_residual_scale = state_residual_scale
        self.alpha_scale = alpha_scale
        self.beta_s_to_b_scale = beta_s_to_b_scale
        self.beta_b_to_s_scale = beta_b_to_s_scale
        self.warmup_delta_scale = warmup_delta_scale
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
        self.b_slot_dropout_p = b_slot_dropout_p
        self.s_to_b_slot_dropout_p = s_to_b_slot_dropout_p
        self.b_to_s_slot_dropout_p = b_to_s_slot_dropout_p
        self.state_init_mode = state_init_mode
        self.pairwise_kind = pairwise_kind
        self.pairwise_hidden_dim = pairwise_hidden_dim
        self.query_topk = query_topk or route_topk
        self.include_query_head = include_query_head
        # Keep the legacy arguments accepted for checkpoint/backward compatibility,
        # but the architecture now uses query-slot self-conditioning instead of an
        # auxiliary RNN readout path.
        self.include_query_rnn_head = False
        self.query_rnn_hidden_dim = query_rnn_hidden_dim or max(1, dim // 4)
        self.query_rnn_head_width = query_rnn_head_width
        self.share_route_families = share_route_families
        self.last_query_rnn_states: Tensor | None = None
        self.query_refine_layers = max(
            1,
            final_refine_layers if query_refine_layers is None else int(query_refine_layers),
        )
        if self.query_topk <= 0:
            raise ValueError("query_topk must be positive.")
        if self.include_query_rnn_head and not self.include_query_head:
            raise ValueError("include_query_rnn_head requires include_query_head.")
        self.stage_specs = tuple(stage_specs or build_progressive_b_stage_specs(seq_nodes))
        self.track_stats = False
        self.last_runtime_stats: dict[str, float] | None = None
        self.collect_aux_losses = False
        self.route_load_cap = 0.25
        self.edge_prob_cap = 0.55
        self.last_aux_losses: dict[str, Tensor] | None = None
        self.last_aux_stats: dict[str, float] | None = None
        self.collect_b_diversity_losses = False
        self.b_diversity_early_margin = 0.35
        self.b_diversity_final_margin = 0.20
        self.last_b_diversity_losses: dict[str, Tensor] | None = None
        self.last_b_diversity_stats: dict[str, float] | None = None
        self.track_activation_grads = False
        self._activation_grad_samples: dict[str, list[float]] = {}

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
        elif pairwise_kind == "additive_low_rank":
            pairwise_factory = lambda: AdditiveLowRankPairwise(
                dim=dim,
                rank=pairwise_hidden_dim or max(1, dim // 16),
            )
        elif pairwise_kind == "hadamard_mlp":
            pairwise_factory = lambda: HadamardMLPPairwise(dim=dim, hidden_dim=pairwise_hidden_dim)
        else:
            raise ValueError(f"Unsupported pairwise_kind: {pairwise_kind!r}.")
        shared_s_pairwise = pairwise_factory()
        shared_expanded_pairwise = pairwise_factory()
        shared_compressed_pairwise = pairwise_factory()
        query_pairwise = pairwise_factory()
        shared_query_transition_route_fn: nn.Module | None = None
        shared_expand_route_fn: nn.Module | None = None
        shared_b_to_s_route_fn: nn.Module | None = None
        shared_compress_route_fn: nn.Module | None = None
        shared_s_to_b_route_fn: nn.Module | None = None
        shared_compressed_adapter_route_fn: nn.Module | None = None
        if share_route_families:
            shared_expand_route_fn = _make_route_fn(
                dim=dim,
                route_kind=route_kind,
                route_hidden_dim=route_hidden_dim,
                route_temperature=route_temperature,
            )
            shared_b_to_s_route_fn = _make_route_fn(
                dim=dim,
                route_kind=route_kind,
                route_hidden_dim=route_hidden_dim,
                route_temperature=route_temperature,
            )
            shared_compress_route_fn = _make_route_fn(
                dim=dim,
                route_kind=route_kind,
                route_hidden_dim=route_hidden_dim,
                route_temperature=route_temperature,
            )
            shared_s_to_b_route_fn = _make_route_fn(
                dim=dim,
                route_kind=route_kind,
                route_hidden_dim=route_hidden_dim,
                route_temperature=route_temperature,
            )
            shared_compressed_adapter_route_fn = BilinearPairwiseRoute(src_dim=dim, dst_dim=dim)
        if include_query_head:
            self.query_val_norm = _make_value_norm(dim, value_norm_kind)
            if share_route_families:
                shared_query_transition_route_fn = _make_route_fn(
                    dim=dim,
                    route_kind=route_kind,
                    route_hidden_dim=route_hidden_dim,
                    route_temperature=route_temperature,
                )
            self.query_transition = _make_transition(
                dim=dim,
                route_topk=self.query_topk,
                route_mode=route_mode,
                route_kind=route_kind,
                route_hidden_dim=route_hidden_dim,
                route_temperature=route_temperature,
                implementation=implementation,
                route_fn=shared_query_transition_route_fn,
                edge_dropout_p=edge_dropout_p,
                usage_dropout_base=usage_dropout_base,
                usage_dropout_scale=usage_dropout_scale,
                usage_dropout_max=usage_dropout_max,
                usage_ema_decay=usage_ema_decay,
            )
            self.query_propagation = _make_sparse_or_dense_propagation(
                dim=dim,
                sparse_type="dense",
                implementation=implementation,
                pairwise_fn=query_pairwise,
                residual=propagation_residual,
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
                    block_residual=block_residual,
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
                    b_slot_dropout_p=b_slot_dropout_p,
                    s_to_b_slot_dropout_p=s_to_b_slot_dropout_p,
                    b_to_s_slot_dropout_p=b_to_s_slot_dropout_p,
                    s_pairwise_fn=shared_s_pairwise,
                    expanded_pairwise_fn=shared_expanded_pairwise,
                    compressed_pairwise_fn=shared_compressed_pairwise,
                    expand_route_fn=shared_expand_route_fn,
                    b_to_s_route_fn=shared_b_to_s_route_fn,
                    compress_route_fn=shared_compress_route_fn,
                    s_to_b_route_fn=shared_s_to_b_route_fn,
                    compressed_adapter_route_fn=shared_compressed_adapter_route_fn,
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
        if include_query_head:
            self.query_head_norm = _make_value_norm(dim, value_norm_kind)
            self.query_feedback_norm = _make_value_norm(dim, value_norm_kind)
            self.query_feedback_proj = nn.Linear(dim, dim)
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

    def set_aux_loss_collection(
        self,
        enabled: bool,
        *,
        route_load_cap: float = 0.25,
        edge_prob_cap: float = 0.55,
    ) -> None:
        self.collect_aux_losses = enabled
        self.route_load_cap = route_load_cap
        self.edge_prob_cap = edge_prob_cap
        self.last_aux_losses = None
        self.last_aux_stats = None
        for block in self.joint_blocks:
            block.set_aux_loss_collection(enabled, route_load_cap=route_load_cap)

    def set_b_diversity_collection(
        self,
        enabled: bool,
        *,
        early_margin: float,
        final_margin: float,
    ) -> None:
        self.collect_b_diversity_losses = enabled
        self.b_diversity_early_margin = early_margin
        self.b_diversity_final_margin = final_margin
        self.last_b_diversity_losses = None
        self.last_b_diversity_stats = None

    def _resolve_b_diversity_margin(self, block_index: int) -> float:
        total_blocks = len(self.joint_blocks)
        if total_blocks <= 1:
            return self.b_diversity_final_margin
        progress = block_index / (total_blocks - 1)
        return self.b_diversity_early_margin + (
            self.b_diversity_final_margin - self.b_diversity_early_margin
        ) * progress

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

    def reset_activation_grad_stats(self) -> None:
        self._activation_grad_samples = {}

    def set_activation_grad_tracking(self, enabled: bool) -> None:
        self.track_activation_grads = enabled
        if not enabled:
            self._activation_grad_samples = {}

    def consume_activation_grad_stats(self) -> dict[str, float]:
        stats = {
            key: sum(values) / len(values)
            for key, values in sorted(self._activation_grad_samples.items())
            if values
        }
        self._activation_grad_samples = {}
        return stats

    def _record_activation_grad(self, key: str, grad: Tensor) -> None:
        grad_norm = float(grad.detach().float().norm().item())
        self._activation_grad_samples.setdefault(key, []).append(grad_norm)

    def _register_layer_activation_grad(self, prefix: str, layer: Layer) -> None:
        if not self.track_activation_grads or not torch.is_grad_enabled():
            return
        if layer.val.requires_grad:
            layer.val.register_hook(
                lambda grad, key=f"activation_grad/{prefix}/val": self._record_activation_grad(
                    key, grad
                )
            )
        if layer.state.requires_grad:
            layer.state.register_hook(
                lambda grad, key=f"activation_grad/{prefix}/state": self._record_activation_grad(
                    key, grad
                )
            )

    def collect_runtime_stats(self, token_ids: Tensor) -> dict[str, float]:
        if token_ids.numel() == 0 or token_ids.shape[-1] == 0:
            return {}
        was_training = self.training
        self.eval()
        self.set_track_stats(True)
        try:
            with torch.no_grad():
                device = next(self.parameters()).device
                token_ids = token_ids.to(device)
                s_layer, compressed_b = self.encode_prefix(token_ids)
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
        raise RuntimeError("Response decoding was removed; this model is query-head only.")

    def _apply_sequence_mask(self, s_layer: Layer, slot_mask: Tensor) -> Layer:
        return _apply_layer_slot_mask(s_layer, slot_mask)

    def encode_prefix(self, token_ids: Tensor) -> tuple[Layer, Layer | None]:
        s_layer = self.initialize_sequence_layer(token_ids)
        compressed_b: Layer | None = None
        b_diversity_losses: dict[str, Tensor] = {}
        b_diversity_stats: dict[str, float] = {}

        for warmup_index, op in enumerate(self.s_warmup):
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(
                s_layer,
                delta,
                scale=self.warmup_delta_scale,
            )
            s_layer = self._finalize_sequence_layer(s_layer)
            self._register_layer_activation_grad(f"s_warmup/{warmup_index}", s_layer)
        for block_index, block in enumerate(self.joint_blocks):
            s_layer, compressed_b = block(s_layer, compressed_b)
            if self.collect_b_diversity_losses and compressed_b is not None:
                margin = self._resolve_b_diversity_margin(block_index)
                block_loss, block_high_cosine = _layer_cosine_duplicate_loss(
                    compressed_b,
                    margin=margin,
                    reference=s_layer.val,
                )
                key = f"b_diversity/block_{block_index:02d}"
                b_diversity_losses[key] = block_loss
                b_diversity_stats[f"{key}_margin"] = margin
                b_diversity_stats[f"{key}_over_margin_mean"] = float(
                    block_high_cosine.detach().item()
                )
        for op in self.s_refine:
            delta = op.compute_delta(self._prepare_sequence_input(s_layer))
            s_layer = self._apply_sequence_delta(s_layer, delta, scale=0.25)
            s_layer = self._finalize_sequence_layer(s_layer)

        self.last_b_diversity_losses = b_diversity_losses if b_diversity_losses else None
        self.last_b_diversity_stats = b_diversity_stats if b_diversity_stats else None

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

    def _require_query_head(self) -> None:
        if not hasattr(self, "query_transition") or not hasattr(self, "query_head"):
            raise RuntimeError("This model was built without the query head.")

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
            merge_mode="add" if self.query_residual else "replace",
        )
        return self._finalize_query_layer(query_layer)

    def apply_query_propagation(
        self,
        query_layer: Layer,
    ) -> Layer:
        self._require_query_head()
        prepared_query = self._prepare_query_input(query_layer)
        propagation_delta = _compute_dense_causal_propagation_delta(
            self.query_propagation,
            prepared_query,
        )
        query_layer = _apply_scaled_delta(
            query_layer,
            propagation_delta,
            state_scale=self.s_delta_scale * self.state_residual_scale,
            val_scale=self.s_delta_scale * self.value_residual_scale,
            merge_mode="add" if self.query_residual else "replace",
        )
        return self._finalize_query_layer(query_layer)

    def refine_query_layer(
        self,
        source_layer: Layer,
        query_layer: Layer | None = None,
    ) -> Layer:
        if query_layer is None:
            query_layer = self.initialize_query_layer(source_layer)
        for _ in range(self.query_refine_layers):
            query_layer = self.apply_query_transition(source_layer, query_layer)
            query_layer = self.apply_query_propagation(query_layer)
        return query_layer

    def _soft_token_feedback(self, logits: Tensor) -> Tensor:
        token_probs = torch.softmax(logits.float(), dim=-1)
        token_table = self.token_embedding.weight.float()
        feedback = torch.matmul(token_probs, token_table)
        feedback = feedback.to(dtype=self.token_embedding.weight.dtype)
        feedback = self.query_feedback_norm(feedback)
        return self.query_feedback_proj(feedback)

    def _token_id_feedback(self, token_ids: Tensor) -> Tensor:
        feedback = self.token_embedding(token_ids)
        feedback = self.query_feedback_norm(feedback)
        return self.query_feedback_proj(feedback)

    def _inject_query_feedback(
        self,
        query_layer: Layer,
        feedback: Tensor,
    ) -> Layer:
        updated_val = query_layer.val + feedback.to(query_layer.val.dtype)
        return query_layer.with_tensors(val=updated_val)

    def update_query_slot(
        self,
        source_layer: Layer,
        query_layer: Layer | None = None,
        *,
        propagation_source_layer: Layer | None = None,
    ) -> Layer:
        del propagation_source_layer
        return self.refine_query_layer(source_layer, query_layer)

    def forward_query_block(
        self,
        token_ids: Tensor,
        *,
        target_len: int,
        query_seed_token_ids: Tensor | None = None,
        query_feedback_token_ids: Tensor | None = None,
        return_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        self._require_query_head()
        if target_len <= 0:
            raise ValueError("target_len must be positive.")
        if query_seed_token_ids is not None:
            if query_seed_token_ids.ndim != 2:
                raise ValueError("query_seed_token_ids must have shape [batch, seed_len].")
            if query_seed_token_ids.shape[0] != token_ids.shape[0]:
                raise ValueError("query_seed_token_ids batch size must match token_ids.")
            if query_seed_token_ids.shape[1] > target_len:
                raise ValueError("query_seed_token_ids cannot be longer than target_len.")
        if query_feedback_token_ids is not None:
            if query_feedback_token_ids.ndim != 2:
                raise ValueError("query_feedback_token_ids must have shape [batch, target_len].")
            if query_feedback_token_ids.shape != (token_ids.shape[0], target_len):
                raise ValueError("query_feedback_token_ids shape must match [batch, target_len].")
        s_layer, compressed_b = self.encode_prefix(token_ids)
        aux_losses: dict[str, Tensor] = {}
        aux_stats: dict[str, float] = {}
        if self.collect_aux_losses:
            block_route_losses: list[Tensor] = []
            block_stat_values: dict[str, list[float]] = {}
            for block in self.joint_blocks:
                if block.last_aux_losses:
                    block_route_losses.extend(block.last_aux_losses.values())
                if block.last_aux_stats:
                    for key, value in block.last_aux_stats.items():
                        block_stat_values.setdefault(key, []).append(value)
            if block_route_losses:
                aux_losses["route_concentration/blocks"] = torch.stack(block_route_losses).mean()
            for key, values in block_stat_values.items():
                if values:
                    aux_stats[f"blocks/{key}_mean"] = float(sum(values) / len(values))
        query_layer = self.initialize_query_layer(s_layer, query_nodes=target_len)
        query_transition_losses: list[Tensor] = []
        query_transition_max_loads: list[Tensor] = []

        def refine_with_feedback(current_query_layer: Layer) -> Layer:
            nonlocal query_transition_losses, query_transition_max_loads
            for _ in range(self.query_refine_layers):
                if self.collect_aux_losses:
                    query_transition_loss, query_transition_max_load = _transition_route_concentration_loss(
                        self.query_transition,
                        self._prepare_sequence_input(s_layer),
                        self._prepare_query_input(current_query_layer),
                        load_cap=self.route_load_cap,
                    )
                    query_transition_losses.append(query_transition_loss)
                    query_transition_max_loads.append(query_transition_max_load)
                current_query_layer = self.apply_query_transition(s_layer, current_query_layer)
                current_query_layer = self.apply_query_propagation(current_query_layer)
            return current_query_layer

        query_layer = refine_with_feedback(query_layer)
        draft_query = self.query_head_norm(query_layer.val)
        draft_logits = self.query_head(draft_query)
        seed_len = 0 if query_seed_token_ids is None else query_seed_token_ids.shape[1]
        if query_feedback_token_ids is not None:
            feedback = self._token_id_feedback(query_feedback_token_ids)
        else:
            feedback = self._soft_token_feedback(draft_logits)
        if seed_len > 0:
            seed_feedback = self._token_id_feedback(query_seed_token_ids)
            feedback = feedback.clone()
            feedback[:, :seed_len, :] = seed_feedback.to(feedback.dtype)
        query_layer = self._inject_query_feedback(query_layer, feedback)
        query_layer = refine_with_feedback(query_layer)

        if self.collect_aux_losses and query_transition_losses:
            aux_losses["route_concentration/query_transition"] = torch.stack(query_transition_losses).mean()
            aux_stats["route_concentration/query_transition_max_load"] = float(
                torch.stack(query_transition_max_loads).mean().detach().item()
            )
        self.last_aux_losses = aux_losses if aux_losses else None
        self.last_aux_stats = aux_stats if aux_stats else None
        final_query = self.query_head_norm(query_layer.val)
        logits = self.query_head(final_query)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits

    def forward_query_next_token(
        self,
        token_ids: Tensor,
        *,
        return_layers: bool = False,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        self._require_query_head()
        s_layer, compressed_b = self.encode_prefix(token_ids)
        query_layer = self.refine_query_layer(s_layer)
        query_slot = self.query_head_norm(query_layer.val[..., 0, :])
        logits = self.query_head(query_slot)
        if return_layers:
            return logits, s_layer, compressed_b
        return logits

    def forward(
        self,
        token_ids: Tensor,
        *,
        return_layers: bool = False,
        teacher_forcing: bool = False,
        full_sequence_causal: bool = False,
        teacher_forcing_chunk_size: int | None = None,
    ) -> Tensor | tuple[Tensor, Layer, Layer | None]:
        if teacher_forcing or full_sequence_causal or teacher_forcing_chunk_size is not None:
            raise RuntimeError(
                "ProgressiveBExampleLM is query-head only. Use forward_query_next_token "
                "or forward_query_block for non-legacy training objectives."
            )
        return self.forward_query_next_token(token_ids, return_layers=return_layers)


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
    loss_mask: Tensor | None = None


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
    train_step_stats: tuple[dict[str, float], ...] = ()


def build_char_vocab(text: str) -> CharVocab:
    symbols = tuple(sorted(set(text)))
    return CharVocab(stoi={ch: idx for idx, ch in enumerate(symbols)}, itos=symbols)


def split_train_val(tokens: Tensor, *, train_fraction: float = 0.9) -> tuple[Tensor, Tensor]:
    split_index = int(tokens.numel() * train_fraction)
    split_index = max(2, min(split_index, tokens.numel() - 2))
    return tokens[:split_index], tokens[split_index:]


def _sample_next_token_item(
    tokens: Tensor,
    *,
    seq_len: int,
    forecast_len: int,
    teacher_forcing: bool,
    full_sequence_causal: bool,
    target_len: int,
    query_block_start_token_id: int | None = None,
) -> tuple[Tensor, Tensor]:
    max_start = tokens.numel() - seq_len - forecast_len
    if max_start < 0:
        raise ValueError("The tokenized corpus must be longer than seq_len + target_len.")
    start = int(torch.randint(0, max_start + 1, (1,)).item())
    context = tokens[start : start + seq_len]
    if teacher_forcing or full_sequence_causal:
        target = tokens[start + 1 : start + seq_len + 1]
    else:
        future_tokens = tokens[start + seq_len : start + seq_len + target_len]
        if query_block_start_token_id is not None:
            start_token = torch.tensor([query_block_start_token_id], dtype=torch.long)
            target = torch.cat((start_token, future_tokens), dim=0)
        elif target_len > 1:
            target = future_tokens
        else:
            target = future_tokens[0]
    return context, target


def sample_next_token_batch(
    tokens: Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    teacher_forcing: bool = False,
    full_sequence_causal: bool = False,
    target_len: int = 1,
    balanced_token_groups: Sequence[Tensor] | None = None,
    query_block_start_token_id: int | None = None,
) -> NextTokenBatch:
    if teacher_forcing and full_sequence_causal:
        raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
    if target_len <= 0:
        raise ValueError("target_len must be positive.")
    forecast_len = 1 if teacher_forcing or full_sequence_causal else target_len
    groups = [group.detach().cpu().contiguous() for group in balanced_token_groups or () if group.numel() > seq_len + forecast_len]
    if groups:
        contexts: list[Tensor] = []
        targets: list[Tensor] = []
        offset = int(torch.randint(0, len(groups), (1,)).item())
        for item_index in range(batch_size):
            group = groups[(offset + item_index) % len(groups)]
            context, target = _sample_next_token_item(
                group,
                seq_len=seq_len,
                forecast_len=forecast_len,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                target_len=target_len,
                query_block_start_token_id=query_block_start_token_id,
            )
            contexts.append(context)
            targets.append(target)
        context_batch = torch.stack(contexts, dim=0)
        target_batch = torch.stack(targets, dim=0)
        return NextTokenBatch(
            context=context_batch.to(device),
            target=target_batch.to(device),
            loss_mask=None,
        )
    max_start = tokens.numel() - seq_len - forecast_len
    if max_start < 0:
        raise ValueError("The tokenized corpus must be longer than seq_len + target_len.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    context = torch.stack([tokens[start : start + seq_len] for start in starts], dim=0)
    if teacher_forcing or full_sequence_causal:
        target = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts], dim=0)
    else:
        future_tokens = torch.stack(
            [tokens[start + seq_len : start + seq_len + target_len] for start in starts],
            dim=0,
        )
        if query_block_start_token_id is not None:
            start_tokens = torch.full((batch_size, 1), query_block_start_token_id, dtype=torch.long)
            target = torch.cat((start_tokens, future_tokens), dim=1)
        elif target_len > 1:
            target = future_tokens
        else:
            target = future_tokens[:, 0]
    return NextTokenBatch(
        context=context.to(device),
        target=target.to(device),
        loss_mask=None,
    )


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
        balanced_token_groups: Sequence[Tensor] | None = None,
        query_block_start_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.tokens = tokens.detach().cpu().contiguous()
        self.balanced_token_groups = tuple(
            group.detach().cpu().contiguous() for group in balanced_token_groups or ()
        )
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.teacher_forcing = teacher_forcing
        self.full_sequence_causal = full_sequence_causal
        self.target_len = target_len
        self.query_block_start_token_id = query_block_start_token_id

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
                balanced_token_groups=self.balanced_token_groups,
                query_block_start_token_id=self.query_block_start_token_id,
            )
            yield batch.context, batch.target, batch.loss_mask


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


def _make_query_block_targets(
    response_ids: Tensor,
    *,
    target_len: int,
    query_block_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[Tensor, Tensor]:
    if target_len <= 0:
        raise ValueError('target_len must be positive.')
    response_ids = response_ids.detach().cpu().to(dtype=torch.long)
    total_len = target_len + 1
    target = torch.full((total_len,), pad_token_id, dtype=torch.long)
    loss_mask = torch.zeros(total_len, dtype=torch.float32)
    # query_block_start is a structural buffer token, not a supervised content target.
    target[0] = query_block_start_token_id
    content = response_ids[:target_len]
    used = int(content.numel())
    if used > 0:
        target[1 : 1 + used] = content
        loss_mask[1 : 1 + used] = 1.0
    if used < target_len:
        target[1 + used] = eos_token_id
        loss_mask[1 + used] = 1.0
    return target, loss_mask


def sample_query_block_pair_batch(
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    target_len: int,
    batch_size: int,
    query_block_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
) -> NextTokenBatch:
    if not pairs:
        raise ValueError('pairs must not be empty.')
    if target_len <= 0:
        raise ValueError('target_len must be positive.')
    if balanced_index_groups:
        groups = [tuple(int(index) for index in group) for group in balanced_index_groups if group]
        if not groups:
            raise ValueError('balanced_index_groups must contain at least one non-empty group.')
        sampled: list[int] = []
        offset = int(torch.randint(0, len(groups), (1,)).item())
        for item_index in range(batch_size):
            group = groups[(offset + item_index) % len(groups)]
            sampled.append(group[int(torch.randint(0, len(group), (1,)).item())])
        indices = torch.tensor(sampled, dtype=torch.long)
    else:
        indices = torch.randint(0, len(pairs), (batch_size,))
    contexts: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    for index in indices.tolist():
        prefix_ids, response_ids = pairs[index]
        contexts.append(_pad_or_trim_prefix(prefix_ids, seq_len=seq_len, pad_token_id=pad_token_id))
        target, loss_mask = _make_query_block_targets(
            response_ids,
            target_len=target_len,
            query_block_start_token_id=query_block_start_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        targets.append(target)
        masks.append(loss_mask)
    target_device = torch.device(device)
    return NextTokenBatch(
        context=torch.stack(contexts, dim=0).to(target_device),
        target=torch.stack(targets, dim=0).to(target_device),
        loss_mask=torch.stack(masks, dim=0).to(target_device),
    )


class QueryBlockPairBatchDataset(IterableDataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        pairs: Sequence[tuple[Tensor, Tensor]],
        *,
        seq_len: int,
        target_len: int,
        batch_size: int,
        query_block_start_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        balanced_index_groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.pairs = tuple((prefix.detach().cpu(), response.detach().cpu()) for prefix, response in pairs)
        self.balanced_index_groups = (
            tuple(tuple(int(index) for index in group) for group in balanced_index_groups if group)
            if balanced_index_groups
            else None
        )
        self.seq_len = seq_len
        self.target_len = target_len
        self.batch_size = batch_size
        self.query_block_start_token_id = query_block_start_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def __iter__(self):
        while True:
            batch = sample_query_block_pair_batch(
                self.pairs,
                seq_len=self.seq_len,
                target_len=self.target_len,
                batch_size=self.batch_size,
                query_block_start_token_id=self.query_block_start_token_id,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.pad_token_id,
                device='cpu',
                balanced_index_groups=self.balanced_index_groups,
            )
            assert batch.loss_mask is not None
            yield batch.context, batch.target, batch.loss_mask


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
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
) -> PrefixResponseBatch:
    if not pairs:
        raise ValueError("pairs must not be empty.")
    if response_len <= 0:
        raise ValueError("response_len must be positive.")
    if balanced_index_groups:
        groups = [tuple(int(index) for index in group) for group in balanced_index_groups if group]
        if not groups:
            raise ValueError("balanced_index_groups must contain at least one non-empty group.")
        sampled: list[int] = []
        offset = int(torch.randint(0, len(groups), (1,)).item())
        for item_index in range(batch_size):
            group = groups[(offset + item_index) % len(groups)]
            sampled.append(group[int(torch.randint(0, len(group), (1,)).item())])
        indices = torch.tensor(sampled, dtype=torch.long)
    else:
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
        balanced_index_groups: Sequence[Sequence[int]] | None = None,
    ) -> None:
        super().__init__()
        self.pairs = tuple((prefix.detach().cpu(), response.detach().cpu()) for prefix, response in pairs)
        self.balanced_index_groups = (
            tuple(tuple(int(index) for index in group) for group in balanced_index_groups if group)
            if balanced_index_groups
            else None
        )
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
                balanced_index_groups=self.balanced_index_groups,
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
    balanced_token_groups: Sequence[Tensor] | None = None,
    query_block_start_token_id: int | None = None,
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
                balanced_token_groups=balanced_token_groups,
                query_block_start_token_id=query_block_start_token_id,
            )
    else:
        dataset = NextTokenBatchDataset(
            tokens,
            seq_len=seq_len,
            batch_size=batch_size,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
            target_len=target_len,
            balanced_token_groups=balanced_token_groups,
            query_block_start_token_id=query_block_start_token_id,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=data_workers,
            pin_memory=target_device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=max(1, prefetch_factor),
        )
        for context, target, loss_mask in loader:
            yield NextTokenBatch(
                context=context.to(target_device, non_blocking=True),
                target=target.to(target_device, non_blocking=True),
                loss_mask=None if loss_mask is None else loss_mask.to(target_device, non_blocking=True),
            )


def _make_query_block_pair_batch_iterator(
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    target_len: int,
    batch_size: int,
    query_block_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
    data_workers: int,
    prefetch_factor: int,
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
):
    target_device = torch.device(device)
    if data_workers <= 0:
        while True:
            yield sample_query_block_pair_batch(
                pairs,
                seq_len=seq_len,
                target_len=target_len,
                batch_size=batch_size,
                query_block_start_token_id=query_block_start_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                device=target_device,
                balanced_index_groups=balanced_index_groups,
            )
    else:
        dataset = QueryBlockPairBatchDataset(
            pairs,
            seq_len=seq_len,
            target_len=target_len,
            batch_size=batch_size,
            query_block_start_token_id=query_block_start_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            balanced_index_groups=balanced_index_groups,
        )
        loader = DataLoader(
            dataset,
            batch_size=None,
            num_workers=data_workers,
            pin_memory=target_device.type == 'cuda',
            persistent_workers=True,
            prefetch_factor=max(1, prefetch_factor),
        )
        for context, target, loss_mask in loader:
            yield NextTokenBatch(
                context=context.to(target_device, non_blocking=True),
                target=target.to(target_device, non_blocking=True),
                loss_mask=loss_mask.to(target_device, non_blocking=True),
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
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
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
                balanced_index_groups=balanced_index_groups,
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
            balanced_index_groups=balanced_index_groups,
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


def _cross_entropy_with_optional_mask(
    logits: Tensor,
    target: Tensor,
    loss_mask: Tensor | None,
    *,
    position_weights: Tensor | None = None,
) -> Tensor:
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction='none',
    )
    flat_weights = None
    if loss_mask is not None:
        flat_weights = loss_mask.reshape(-1).to(device=losses.device, dtype=losses.dtype)
    if position_weights is not None:
        scaled_weights = position_weights.reshape(-1).to(device=losses.device, dtype=losses.dtype)
        flat_weights = scaled_weights if flat_weights is None else (flat_weights * scaled_weights)
    if flat_weights is None:
        return losses.mean()
    mask_total = flat_weights.sum()
    if mask_total.item() <= 0:
        raise ValueError('loss_mask must select at least one target position.')
    return (losses * flat_weights).sum() / mask_total


def _chunked_cross_entropy_with_optional_mask_from_states(
    states: Tensor,
    target: Tensor,
    loss_mask: Tensor | None,
    *,
    head_norm: nn.Module,
    head: nn.Module,
    position_weights: Tensor | None = None,
    chunk_size: int = 16,
) -> Tensor:
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive.')
    weighted_total = states.new_zeros((), dtype=torch.float32)
    weight_total = states.new_zeros((), dtype=torch.float32)
    for start in range(0, target.shape[-1], chunk_size):
        end = min(target.shape[-1], start + chunk_size)
        chunk_states = head_norm(states[:, start:end, :])
        chunk_logits = head(chunk_states)
        chunk_losses = F.cross_entropy(
            chunk_logits.reshape(-1, chunk_logits.shape[-1]),
            target[:, start:end].reshape(-1),
            reduction='none',
        )
        chunk_weights = None
        if loss_mask is not None:
            chunk_weights = loss_mask[:, start:end].reshape(-1).to(
                device=chunk_losses.device,
                dtype=chunk_losses.dtype,
            )
        if position_weights is not None:
            chunk_position_weights = position_weights[:, start:end].reshape(-1).to(
                device=chunk_losses.device,
                dtype=chunk_losses.dtype,
            )
            chunk_weights = (
                chunk_position_weights if chunk_weights is None else (chunk_weights * chunk_position_weights)
            )
        if chunk_weights is None:
            chunk_weights = torch.ones_like(chunk_losses)
        weighted_total = weighted_total + (chunk_losses * chunk_weights).sum()
        weight_total = weight_total + chunk_weights.sum()
    if weight_total.item() <= 0:
        raise ValueError('loss_mask must select at least one target position.')
    return weighted_total / weight_total


def _query_block_position_weights(
    target: Tensor,
    *,
    front_weight: float,
) -> Tensor | None:
    if front_weight == 1.0:
        return None
    target_width = target.shape[-1]
    if target_width <= 1:
        return None
    weights = torch.ones(target_width, device=target.device, dtype=torch.float32)
    weights[1:] = torch.exp(
        torch.linspace(
            math.log(front_weight),
            0.0,
            steps=target_width - 1,
            device=target.device,
            dtype=torch.float32,
        )
    )
    weights[0] = 0.0
    return weights.unsqueeze(0).expand(target.shape[0], -1)


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
    b_diversity_loss_weight: float = 0.0,
    b_diversity_unweighted_per_layer: bool = False,
    b_diversity_per_layer_weight: float = 1.0,
    b_cosine_margin: float = 0.20,
    b_cosine_early_margin: float = 0.35,
    route_concentration_loss_weight: float = 0.0,
    route_load_cap: float = 0.25,
    edge_prob_cap: float = 0.55,
    query_block_front_weight: float = 1.0,
) -> tuple[Tensor, Tensor]:
    if teacher_forcing and full_sequence_causal:
        raise ValueError("teacher_forcing and full_sequence_causal cannot both be enabled.")
    if query_next_token and query_block:
        raise ValueError("query_next_token and query_block cannot both be enabled.")
    if (query_next_token or query_block) and (teacher_forcing or full_sequence_causal):
        raise ValueError("query objectives cannot be combined with teacher forcing modes.")
    use_autocast = autocast_device_type is not None and autocast_dtype is not None
    loss_stats: dict[str, float] = {}
    collect_aux_losses = route_concentration_loss_weight > 0.0 and hasattr(
        model,
        "set_aux_loss_collection",
    )
    collect_b_diversity_losses = (
        query_block
        and b_diversity_unweighted_per_layer
        and hasattr(model, "set_b_diversity_collection")
    )
    if hasattr(model, "last_loss_stats"):
        model.last_loss_stats = {}
    if collect_aux_losses:
        model.set_aux_loss_collection(
            True,
            route_load_cap=route_load_cap,
            edge_prob_cap=edge_prob_cap,
        )
    if collect_b_diversity_losses:
        model.set_b_diversity_collection(
            True,
            early_margin=b_cosine_early_margin,
            final_margin=b_cosine_margin,
        )
    with torch.autocast(
        device_type=autocast_device_type or "cpu",
        dtype=autocast_dtype,
        enabled=use_autocast,
    ):
        if query_block:
            target_len = batch.target.shape[-1] if batch.target.ndim > 1 else 1
            query_seed_token_ids = None
            query_feedback_token_ids = None
            if batch.target.ndim > 1 and batch.target.shape[-1] > 0:
                query_seed_token_ids = batch.target[:, :1]
                query_feedback_token_ids = torch.cat((query_seed_token_ids, batch.target[:, :-1]), dim=1)
            logits, _, compressed_b = model.forward_query_block(
                batch.context,
                target_len=target_len,
                query_seed_token_ids=query_seed_token_ids,
                query_feedback_token_ids=query_feedback_token_ids,
                return_layers=True,
            )
        elif query_next_token:
            logits = model.forward_query_next_token(batch.context)
            compressed_b = None
        else:
            logits = model(
                batch.context,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                teacher_forcing_chunk_size=teacher_forcing_chunk_size,
            )
            compressed_b = None
        position_weights = None
        if query_block and batch.target.ndim > 1:
            position_weights = _query_block_position_weights(
                batch.target,
                front_weight=query_block_front_weight,
            )
        average_main_loss = _cross_entropy_with_optional_mask(
            logits,
            batch.target,
            batch.loss_mask,
        )
        main_loss = _cross_entropy_with_optional_mask(
            logits,
            batch.target,
            batch.loss_mask,
            position_weights=position_weights,
        )
        loss = main_loss
        loss_stats["loss/query_head"] = float(main_loss.detach().item())
        loss_stats["loss/avg_query_head"] = float(average_main_loss.detach().item())
        if query_block and batch.target.ndim > 1:
            target_width = batch.target.shape[-1]
            for start, end in ((0, 16), (16, 32), (32, 64), (64, 128)):
                if start >= target_width:
                    continue
                clipped_end = min(end, target_width)
                bucket_mask = (
                    None
                    if batch.loss_mask is None
                    else batch.loss_mask[:, start:clipped_end]
                )
                if bucket_mask is not None and bucket_mask.sum().item() <= 0:
                    continue
                bucket_loss = _cross_entropy_with_optional_mask(
                    logits[:, start:clipped_end, :],
                    batch.target[:, start:clipped_end],
                    bucket_mask,
                )
                loss_stats[f"query_block_pos_loss/{start:03d}_{clipped_end - 1:03d}"] = float(
                    bucket_loss.detach().item()
                )
        if b_diversity_loss_weight > 0.0:
            b_loss, b_high_cosine = _layer_cosine_duplicate_loss(
                compressed_b,
                margin=b_cosine_margin,
                reference=loss,
            )
            loss = loss + b_diversity_loss_weight * b_loss
            loss_stats["aux/b_cosine_duplicate_loss"] = float(b_loss.detach().item())
            loss_stats["aux/b_cosine_over_margin_mean"] = float(b_high_cosine.detach().item())
        if collect_b_diversity_losses:
            b_losses = getattr(model, "last_b_diversity_losses", None)
            b_stats = getattr(model, "last_b_diversity_stats", None)
            if b_losses:
                per_layer_b_loss = torch.stack(list(b_losses.values())).mean()
                weighted_per_layer_b_loss = b_diversity_per_layer_weight * per_layer_b_loss
                loss = loss + weighted_per_layer_b_loss
                loss_stats["aux/b_cosine_per_layer_loss"] = float(per_layer_b_loss.detach().item())
                loss_stats["aux/b_cosine_per_layer_weighted_loss"] = float(
                    weighted_per_layer_b_loss.detach().item()
                )
                for key, value in sorted(b_losses.items()):
                    loss_stats[f"aux/{key}_loss"] = float(value.detach().item())
            if b_stats:
                for key, value in sorted(b_stats.items()):
                    loss_stats[f"aux/{key}"] = value
        if route_concentration_loss_weight > 0.0:
            aux_losses = getattr(model, "last_aux_losses", None)
            aux_stats = getattr(model, "last_aux_stats", None)
            if aux_losses:
                route_loss = torch.stack(list(aux_losses.values())).mean()
                loss = loss + route_concentration_loss_weight * route_loss
                loss_stats["aux/route_edge_concentration_loss"] = float(route_loss.detach().item())
                for key, value in sorted(aux_losses.items()):
                    loss_stats[f"aux/{key}_loss"] = float(value.detach().item())
            if aux_stats:
                for key, value in sorted(aux_stats.items()):
                    loss_stats[f"aux/{key}"] = value
        loss_stats["loss/total"] = float(loss.detach().item())
    if collect_aux_losses:
        model.set_aux_loss_collection(False)
    if collect_b_diversity_losses:
        model.set_b_diversity_collection(
            False,
            early_margin=b_cosine_early_margin,
            final_margin=b_cosine_margin,
        )
    if hasattr(model, "__dict__"):
        model.last_loss_stats = loss_stats
    return loss, logits


def compute_prefix_response_loss(
    model: nn.Module,
    batch: PrefixResponseBatch,
    *,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    raise RuntimeError("Prefix-response training was removed; use query objectives.")


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
    balanced_token_groups: Sequence[Tensor] | None = None,
    query_block_start_token_id: int | None = None,
    query_block_front_weight: float = 1.0,
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
            balanced_token_groups=balanced_token_groups,
            query_block_start_token_id=query_block_start_token_id,
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
            query_block_front_weight=query_block_front_weight,
        )
        losses.append(float(loss.item()))
    if was_training:
        model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def estimate_query_block_pair_loss(
    model: nn.Module,
    pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    target_len: int,
    batch_size: int,
    query_block_start_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
    device: torch.device | str,
    eval_steps: int,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
    query_block_front_weight: float = 1.0,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        batch = sample_query_block_pair_batch(
            pairs,
            seq_len=seq_len,
            target_len=target_len,
            batch_size=batch_size,
            query_block_start_token_id=query_block_start_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            device=device,
            balanced_index_groups=balanced_index_groups,
        )
        loss, _ = compute_next_token_loss(
            model,
            batch,
            query_block=True,
            autocast_device_type=autocast_device_type,
            autocast_dtype=autocast_dtype,
            query_block_front_weight=query_block_front_weight,
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
    loss_stats_callback: Callable[[int, dict[str, float]], None] | None = None,
    grad_stats_callback: Callable[[int, dict[str, float]], None] | None = None,
    eval_callback: Callable[[int, float, float], None] | None = None,
    checkpoint_callback: Callable[[int, float, float, nn.Module, torch.optim.Optimizer], None] | None = None,
    eval_on_first_step: bool = True,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
    balanced_token_groups: Sequence[Tensor] | None = None,
    val_balanced_token_groups: Sequence[Tensor] | None = None,
    query_block_start_token_id: int | None = None,
    start_step: int = 0,
    total_steps: int | None = None,
    optimizer_state_dict: dict[str, object] | None = None,
    checkpoint_interval: int | None = None,
    b_diversity_loss_weight: float = 0.0,
    b_diversity_unweighted_per_layer: bool = False,
    b_diversity_per_layer_weight: float = 1.0,
    b_cosine_margin: float = 0.20,
    b_cosine_early_margin: float = 0.35,
    route_concentration_loss_weight: float = 0.0,
    route_load_cap: float = 0.25,
    edge_prob_cap: float = 0.55,
    learning_rate_schedule: str = "none",
    learning_rate_warmup_steps: int = 0,
    learning_rate_warmup_start: float | None = None,
    learning_rate_min_ratio: float = 0.0,
    query_block_front_weight: float = 1.0,
    grad_breakdown_start_step: int | None = None,
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
    loaded_optimizer_state = False
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            loaded_optimizer_state = True
        except ValueError as exc:
            print(f"optimizer_state_load_skipped | reason={exc}")
    if loaded_optimizer_state:
        _set_optimizer_hyperparams(
            optimizer,
            learning_rate=optimizer.param_groups[0]["lr"],
            weight_decay=optimizer.param_groups[0].get("weight_decay"),
        )
    else:
        _set_optimizer_hyperparams(
            optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    eval_steps_seen: list[int] = []
    train_step_losses: list[float] = []
    grad_norms: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_step_stats: list[dict[str, float]] = []
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
        balanced_token_groups=balanced_token_groups,
        query_block_start_token_id=query_block_start_token_id,
    )

    schedule_total_steps = total_steps if total_steps is not None else start_step + steps
    for relative_step in range(1, steps + 1):
        step = start_step + relative_step
        if step_setup_callback is not None:
            step_setup_callback(step, schedule_total_steps)
        if not loaded_optimizer_state:
            current_learning_rate = _resolve_scheduled_learning_rate(
                base_learning_rate=learning_rate,
                schedule=learning_rate_schedule,
                warmup_steps=learning_rate_warmup_steps,
                warmup_start_learning_rate=learning_rate_warmup_start,
                min_learning_rate_ratio=learning_rate_min_ratio,
                step=step,
                start_step=start_step,
                total_steps=schedule_total_steps,
            )
            _set_optimizer_hyperparams(
                optimizer,
                learning_rate=current_learning_rate,
            )
        optimizer.zero_grad(set_to_none=True)
        track_activation_grads = (
            grad_stats_callback is not None
            and (grad_breakdown_start_step is None or step >= grad_breakdown_start_step)
        )
        set_activation_tracking = getattr(model, "set_activation_grad_tracking", None)
        if callable(set_activation_tracking):
            set_activation_tracking(track_activation_grads)
        if track_activation_grads:
            reset_activation_stats = getattr(model, "reset_activation_grad_stats", None)
            if callable(reset_activation_stats):
                reset_activation_stats()
        accumulated_loss = 0.0
        accumulated_loss_stats: dict[str, list[float]] = {}
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
                b_diversity_loss_weight=b_diversity_loss_weight,
                b_diversity_unweighted_per_layer=b_diversity_unweighted_per_layer,
                b_diversity_per_layer_weight=b_diversity_per_layer_weight,
                b_cosine_margin=b_cosine_margin,
                b_cosine_early_margin=b_cosine_early_margin,
                route_concentration_loss_weight=route_concentration_loss_weight,
                route_load_cap=route_load_cap,
                edge_prob_cap=edge_prob_cap,
                query_block_front_weight=query_block_front_weight,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            accumulated_loss += float(loss.item())
            loss_stats = getattr(model, "last_loss_stats", None)
            if loss_stats:
                for key, value in loss_stats.items():
                    accumulated_loss_stats.setdefault(key, []).append(value)
            (loss / grad_accum_steps).backward()
        grad_stats: dict[str, float] | None = None
        if (
            grad_stats_callback is not None
            and (grad_breakdown_start_step is None or step >= grad_breakdown_start_step)
        ):
            grad_stats = _collect_grad_group_norms(model, key_prefix="grad_preclip")
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
        if grad_stats is not None:
            grad_stats.update(_collect_grad_group_norms(model, key_prefix="grad_postclip"))
            consume_activation_stats = getattr(model, "consume_activation_grad_stats", None)
            if callable(consume_activation_stats):
                grad_stats.update(consume_activation_stats())
            grad_stats_callback(step, grad_stats)
        optimizer.step()
        step_loss = accumulated_loss / grad_accum_steps
        train_step_losses.append(step_loss)
        grad_norms.append(grad_norm)
        averaged_loss_stats = {
            key: sum(values) / len(values)
            for key, values in sorted(accumulated_loss_stats.items())
            if values
        }
        train_step_stats.append(averaged_loss_stats)
        if progress_callback is not None:
            progress_callback(step, schedule_total_steps, step_loss)
        if step_callback is not None:
            step_callback(step, step_loss, grad_norm)
        if loss_stats_callback is not None and averaged_loss_stats:
            loss_stats_callback(step, averaged_loss_stats)
        if (
            checkpoint_callback is not None
            and checkpoint_interval is not None
            and checkpoint_interval > 0
            and step % checkpoint_interval == 0
        ):
            checkpoint_callback(step, step_loss, None, model, optimizer)

        if step % eval_interval == 0 or (eval_on_first_step and step == start_step + 1) or step == schedule_total_steps:
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
                balanced_token_groups=balanced_token_groups,
                query_block_start_token_id=query_block_start_token_id,
                query_block_front_weight=query_block_front_weight,
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
                balanced_token_groups=val_balanced_token_groups,
                query_block_start_token_id=query_block_start_token_id,
                query_block_front_weight=query_block_front_weight,
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
        train_step_stats=tuple(train_step_stats),
    )


def train_query_block_pair_model(
    model: nn.Module,
    train_pairs: Sequence[tuple[Tensor, Tensor]],
    val_pairs: Sequence[tuple[Tensor, Tensor]],
    *,
    seq_len: int,
    target_len: int,
    batch_size: int,
    query_block_start_token_id: int,
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
    loss_stats_callback: Callable[[int, dict[str, float]], None] | None = None,
    grad_stats_callback: Callable[[int, dict[str, float]], None] | None = None,
    eval_callback: Callable[[int, float, float], None] | None = None,
    checkpoint_callback: Callable[[int, float, float, nn.Module, torch.optim.Optimizer], None] | None = None,
    eval_on_first_step: bool = True,
    autocast_device_type: str | None = None,
    autocast_dtype: torch.dtype | None = None,
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
    val_balanced_index_groups: Sequence[Sequence[int]] | None = None,
    start_step: int = 0,
    total_steps: int | None = None,
    optimizer_state_dict: dict[str, object] | None = None,
    checkpoint_interval: int | None = None,
    b_diversity_loss_weight: float = 0.0,
    b_diversity_unweighted_per_layer: bool = False,
    b_diversity_per_layer_weight: float = 1.0,
    b_cosine_margin: float = 0.20,
    b_cosine_early_margin: float = 0.35,
    route_concentration_loss_weight: float = 0.0,
    route_load_cap: float = 0.25,
    edge_prob_cap: float = 0.55,
    learning_rate_schedule: str = 'none',
    learning_rate_warmup_steps: int = 0,
    learning_rate_warmup_start: float | None = None,
    learning_rate_min_ratio: float = 0.0,
    query_block_front_weight: float = 1.0,
    grad_breakdown_start_step: int | None = None,
) -> TrainingHistory:
    if grad_accum_steps <= 0:
        raise ValueError('grad_accum_steps must be positive.')
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loaded_optimizer_state = False
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            loaded_optimizer_state = True
        except ValueError as exc:
            print(f'optimizer_state_load_skipped | reason={exc}')
    if loaded_optimizer_state:
        _set_optimizer_hyperparams(
            optimizer,
            learning_rate=optimizer.param_groups[0]['lr'],
            weight_decay=optimizer.param_groups[0].get('weight_decay'),
        )
    else:
        _set_optimizer_hyperparams(
            optimizer,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    eval_steps_seen: list[int] = []
    train_step_losses: list[float] = []
    grad_norms: list[float] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_step_stats: list[dict[str, float]] = []
    batch_iter = _make_query_block_pair_batch_iterator(
        train_pairs,
        seq_len=seq_len,
        target_len=target_len,
        batch_size=batch_size,
        query_block_start_token_id=query_block_start_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        device=device,
        data_workers=data_workers,
        prefetch_factor=prefetch_factor,
        balanced_index_groups=balanced_index_groups,
    )

    schedule_total_steps = total_steps if total_steps is not None else start_step + steps
    for relative_step in range(1, steps + 1):
        step = start_step + relative_step
        if step_setup_callback is not None:
            step_setup_callback(step, schedule_total_steps)
        if not loaded_optimizer_state:
            current_learning_rate = _resolve_scheduled_learning_rate(
                base_learning_rate=learning_rate,
                schedule=learning_rate_schedule,
                warmup_steps=learning_rate_warmup_steps,
                warmup_start_learning_rate=learning_rate_warmup_start,
                min_learning_rate_ratio=learning_rate_min_ratio,
                step=step,
                start_step=start_step,
                total_steps=schedule_total_steps,
            )
            _set_optimizer_hyperparams(
                optimizer,
                learning_rate=current_learning_rate,
            )
        optimizer.zero_grad(set_to_none=True)
        track_activation_grads = (
            grad_stats_callback is not None
            and (grad_breakdown_start_step is None or step >= grad_breakdown_start_step)
        )
        set_activation_tracking = getattr(model, "set_activation_grad_tracking", None)
        if callable(set_activation_tracking):
            set_activation_tracking(track_activation_grads)
        if track_activation_grads:
            reset_activation_stats = getattr(model, "reset_activation_grad_stats", None)
            if callable(reset_activation_stats):
                reset_activation_stats()
        accumulated_loss = 0.0
        accumulated_loss_stats: dict[str, list[float]] = {}
        for _ in range(grad_accum_steps):
            batch = next(batch_iter)
            loss, _ = compute_next_token_loss(
                model,
                batch,
                query_block=True,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
                b_diversity_loss_weight=b_diversity_loss_weight,
                b_diversity_unweighted_per_layer=b_diversity_unweighted_per_layer,
                b_diversity_per_layer_weight=b_diversity_per_layer_weight,
                b_cosine_margin=b_cosine_margin,
                b_cosine_early_margin=b_cosine_early_margin,
                route_concentration_loss_weight=route_concentration_loss_weight,
                route_load_cap=route_load_cap,
                edge_prob_cap=edge_prob_cap,
                query_block_front_weight=query_block_front_weight,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Non-finite loss at step {step}: {loss.item()}')
            accumulated_loss += float(loss.item())
            loss_stats = getattr(model, 'last_loss_stats', None)
            if loss_stats:
                for key, value in loss_stats.items():
                    accumulated_loss_stats.setdefault(key, []).append(value)
            (loss / grad_accum_steps).backward()
        grad_stats: dict[str, float] | None = None
        if (
            grad_stats_callback is not None
            and (grad_breakdown_start_step is None or step >= grad_breakdown_start_step)
        ):
            grad_stats = _collect_grad_group_norms(model, key_prefix="grad_preclip")
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
            raise FloatingPointError(f'Non-finite grad norm at step {step}: {grad_norm}')
        if grad_stats is not None:
            grad_stats.update(_collect_grad_group_norms(model, key_prefix="grad_postclip"))
            consume_activation_stats = getattr(model, "consume_activation_grad_stats", None)
            if callable(consume_activation_stats):
                grad_stats.update(consume_activation_stats())
            grad_stats_callback(step, grad_stats)
        optimizer.step()
        step_loss = accumulated_loss / grad_accum_steps
        train_step_losses.append(step_loss)
        grad_norms.append(grad_norm)
        averaged_loss_stats = {
            key: sum(values) / len(values)
            for key, values in sorted(accumulated_loss_stats.items())
            if values
        }
        train_step_stats.append(averaged_loss_stats)
        if progress_callback is not None:
            progress_callback(step, schedule_total_steps, step_loss)
        if step_callback is not None:
            step_callback(step, step_loss, grad_norm)
        if loss_stats_callback is not None and averaged_loss_stats:
            loss_stats_callback(step, averaged_loss_stats)
        if (
            checkpoint_callback is not None
            and checkpoint_interval is not None
            and checkpoint_interval > 0
            and step % checkpoint_interval == 0
        ):
            checkpoint_callback(step, step_loss, None, model, optimizer)

        if step % eval_interval == 0 or (eval_on_first_step and step == start_step + 1) or step == schedule_total_steps:
            eval_steps_seen.append(step)
            train_eval = estimate_query_block_pair_loss(
                model,
                train_pairs,
                seq_len=seq_len,
                target_len=target_len,
                batch_size=batch_size,
                query_block_start_token_id=query_block_start_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                device=device,
                eval_steps=eval_steps,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
                balanced_index_groups=balanced_index_groups,
                query_block_front_weight=query_block_front_weight,
            )
            val_eval = estimate_query_block_pair_loss(
                model,
                val_pairs,
                seq_len=seq_len,
                target_len=target_len,
                batch_size=batch_size,
                query_block_start_token_id=query_block_start_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                device=device,
                eval_steps=eval_steps,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
                balanced_index_groups=val_balanced_index_groups,
                query_block_front_weight=query_block_front_weight,
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
        train_step_stats=tuple(train_step_stats),
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
    balanced_index_groups: Sequence[Sequence[int]] | None = None,
    start_step: int = 0,
    total_steps: int | None = None,
    optimizer_state_dict: dict[str, object] | None = None,
    checkpoint_interval: int | None = None,
    learning_rate_schedule: str = "none",
    learning_rate_warmup_steps: int = 0,
    learning_rate_warmup_start: float | None = None,
    learning_rate_min_ratio: float = 0.0,
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
    loaded_optimizer_state = False
    if optimizer_state_dict is not None:
        try:
            optimizer.load_state_dict(optimizer_state_dict)
            loaded_optimizer_state = True
        except ValueError as exc:
            print(f"optimizer_state_load_skipped | reason={exc}")
    if loaded_optimizer_state:
        _set_optimizer_hyperparams(
            optimizer,
            learning_rate=optimizer.param_groups[0]["lr"],
            weight_decay=optimizer.param_groups[0].get("weight_decay"),
        )
    else:
        _set_optimizer_hyperparams(
            optimizer,
            learning_rate=learning_rate,
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
        balanced_index_groups=balanced_index_groups,
    )

    schedule_total_steps = total_steps if total_steps is not None else start_step + steps
    for relative_step in range(1, steps + 1):
        step = start_step + relative_step
        if step_setup_callback is not None:
            step_setup_callback(step, schedule_total_steps)
        if not loaded_optimizer_state:
            current_learning_rate = _resolve_scheduled_learning_rate(
                base_learning_rate=learning_rate,
                schedule=learning_rate_schedule,
                warmup_steps=learning_rate_warmup_steps,
                warmup_start_learning_rate=learning_rate_warmup_start,
                min_learning_rate_ratio=learning_rate_min_ratio,
                step=step,
                start_step=start_step,
                total_steps=schedule_total_steps,
            )
            _set_optimizer_hyperparams(
                optimizer,
                learning_rate=current_learning_rate,
            )
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
            progress_callback(step, schedule_total_steps, step_loss)
        if step_callback is not None:
            step_callback(step, step_loss, grad_norm)
        if (
            checkpoint_callback is not None
            and checkpoint_interval is not None
            and checkpoint_interval > 0
            and step % checkpoint_interval == 0
        ):
            checkpoint_callback(step, step_loss, None, model, optimizer)

        if step % eval_interval == 0 or (eval_on_first_step and step == start_step + 1) or step == schedule_total_steps:
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
