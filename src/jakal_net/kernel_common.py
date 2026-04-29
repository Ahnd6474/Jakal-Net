from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
from torch import Tensor
from torch.nn import functional as F

from jakal_net.modules import (
    BilinearPairwise,
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    HadamardMLPPairwise,
    LinearRoute,
    LowRankBilinearPairwise,
    LowRankBilinearRoute,
    MLPRoute,
    BilinearPairwiseRoute,
    QueryNormalizedDotRoute,
    ScaledCosinePairwise,
    SourceTargetHadamardMLPRoute,
    MultiHeadPairwise,
    MultiHeadRoute,
)


@dataclass(slots=True)
class OnlineSoftmaxState:
    max_logits: Tensor
    exp_sums: Tensor
    weighted_values: Tensor


@dataclass(slots=True)
class OnlineSoftmaxStats:
    max_logits: Tensor
    exp_sums: Tensor


@dataclass(frozen=True, slots=True)
class PairwiseKernelSpec:
    kind: str
    weight: Tensor
    bias: Tensor | None
    in_weight: Tensor | None = None
    in_bias: Tensor | None = None
    out_weight: Tensor | None = None
    out_bias: Tensor | None = None


@dataclass(frozen=True, slots=True)
class RouteKernelSpec:
    kind: str
    in_weight: Tensor
    in_bias: Tensor | None
    out_weight: Tensor | None = None
    out_bias: Tensor | None = None


@dataclass(frozen=True, slots=True)
class PairwiseRouteKernelSpec:
    kind: str
    source_weight: Tensor | None
    source_bias: Tensor | None
    target_weight: Tensor | None
    target_bias: Tensor | None
    core_weight: Tensor
    bias: Tensor | None
    hidden_weight: Tensor | None = None
    hidden_bias: Tensor | None = None
    out_weight: Tensor | None = None
    out_bias: Tensor | None = None
    temperature: float = 1.0


def flatten_state(state: Tensor) -> Tensor:
    return state.reshape(prod(state.shape[:-1]) or 1, state.shape[-1])


def flatten_val(val: Tensor) -> Tensor:
    return val.reshape(prod(val.shape[:-2]) or 1, val.shape[-2], val.shape[-1])


def reshape_state(
    flat_state: Tensor, batch_shape: torch.Size | tuple[int, ...], nodes: int
) -> Tensor:
    return flat_state.reshape(*batch_shape, nodes)


def reshape_val(
    flat_val: Tensor,
    batch_shape: torch.Size | tuple[int, ...],
    nodes: int,
    dim: int,
) -> Tensor:
    return flat_val.reshape(*batch_shape, nodes, dim)


def supports_pairwise_kernel(pairwise_fn: object) -> bool:
    if isinstance(pairwise_fn, MultiHeadPairwise):
        return (
            pairwise_fn.aggregate in {"max", "smoothmax", "signed_smoothmax"}
            and len(pairwise_fn.heads) > 0
            and all(supports_pairwise_kernel(head) for head in pairwise_fn.heads)
            and len({pairwise_kernel_spec(head).kind for head in pairwise_fn.heads}) == 1
        )
    return isinstance(
        pairwise_fn,
        (
            DiagonalBilinearPairwise,
            BilinearPairwise,
            LowRankBilinearPairwise,
            HadamardMLPPairwise,
            ScaledCosinePairwise,
        ),
    )


def supports_route_kernel(route_fn: object) -> bool:
    return isinstance(route_fn, (LinearRoute, MLPRoute))


def _unwrap_temperature_scaled_route(route_fn: object) -> tuple[object, float]:
    temperature = float(getattr(route_fn, "temperature", 1.0))
    inner = getattr(route_fn, "route_fn", route_fn)
    return inner, temperature


def supports_pairwise_route_kernel(route_fn: object) -> bool:
    inner, _ = _unwrap_temperature_scaled_route(route_fn)
    if isinstance(inner, MultiHeadRoute):
        return (
            inner.aggregate == "max"
            and len(inner.heads) > 0
            and all(supports_pairwise_route_kernel(head) for head in inner.heads)
            and len({pairwise_route_kernel_spec(head).kind for head in inner.heads}) == 1
        )
    return isinstance(
        inner,
        (
            DiagonalBilinearRoute,
            LowRankBilinearRoute,
            BilinearPairwiseRoute,
            QueryNormalizedDotRoute,
            SourceTargetHadamardMLPRoute,
        ),
    )


def pairwise_kernel_spec(pairwise_fn: object) -> PairwiseKernelSpec:
    if isinstance(pairwise_fn, MultiHeadPairwise):
        if pairwise_fn.aggregate not in {"max", "smoothmax", "signed_smoothmax"}:
            raise TypeError("Only max-, smoothmax-, or signed_smoothmax-aggregated MultiHeadPairwise is supported for native/kernel spec.")
        head_specs = [pairwise_kernel_spec(head) for head in pairwise_fn.heads]
        if not head_specs:
            raise TypeError("MultiHeadPairwise requires at least one head.")
        base_kind = head_specs[0].kind
        if any(spec.kind != base_kind for spec in head_specs[1:]):
            raise TypeError("All MultiHeadPairwise heads must share the same kernel kind.")

        def _stack_optional(name: str) -> Tensor | None:
            values = [getattr(spec, name) for spec in head_specs]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise TypeError(f"MultiHeadPairwise head specs disagree on optional field {name!r}.")
            return torch.stack(values)

        return PairwiseKernelSpec(
            kind=f"multihead_{pairwise_fn.aggregate}_{base_kind}",
            weight=torch.stack([spec.weight for spec in head_specs]),
            bias=_stack_optional("bias"),
            in_weight=_stack_optional("in_weight"),
            in_bias=_stack_optional("in_bias"),
            out_weight=_stack_optional("out_weight"),
            out_bias=_stack_optional("out_bias"),
        )
    if isinstance(pairwise_fn, DiagonalBilinearPairwise):
        return PairwiseKernelSpec(
            kind="diagonal_bilinear",
            weight=pairwise_fn.normalized_weight(),
            bias=pairwise_fn.bias,
        )
    if isinstance(pairwise_fn, LowRankBilinearPairwise):
        return PairwiseKernelSpec(
            kind="low_rank_bilinear",
            weight=pairwise_fn.normalized_weight(),
            bias=pairwise_fn.bias,
            in_weight=pairwise_fn.source_proj.weight,
            out_weight=pairwise_fn.target_proj.weight,
        )
    if isinstance(pairwise_fn, BilinearPairwise):
        return PairwiseKernelSpec(
            kind="bilinear",
            weight=pairwise_fn.normalized_weight(),
            bias=pairwise_fn.bias,
        )
    if isinstance(pairwise_fn, ScaledCosinePairwise):
        return PairwiseKernelSpec(
            kind="scaled_cosine",
            weight=pairwise_fn.scale_buffer,
            bias=pairwise_fn.eps_buffer,
        )
    if isinstance(pairwise_fn, HadamardMLPPairwise):
        return PairwiseKernelSpec(
            kind="hadamard_mlp",
            weight=pairwise_fn.proj_in.weight,
            bias=pairwise_fn.proj_out.bias,
            in_weight=pairwise_fn.proj_in.weight,
            in_bias=pairwise_fn.proj_in.bias,
            out_weight=pairwise_fn.proj_out.weight,
            out_bias=pairwise_fn.proj_out.bias,
        )
    raise TypeError("Unsupported pairwise_fn for native/kernel spec.")


def route_kernel_spec(route_fn: object) -> RouteKernelSpec:
    if isinstance(route_fn, LinearRoute):
        return RouteKernelSpec(
            kind="linear",
            in_weight=route_fn.linear.weight,
            in_bias=route_fn.linear.bias,
        )
    if isinstance(route_fn, MLPRoute):
        first = route_fn.net[0]
        second = route_fn.net[2]
        return RouteKernelSpec(
            kind="mlp",
            in_weight=first.weight,
            in_bias=first.bias,
            out_weight=second.weight,
            out_bias=second.bias,
        )
    raise TypeError("Unsupported route_fn for native/kernel spec.")


def pairwise_route_kernel_spec(route_fn: object) -> PairwiseRouteKernelSpec:
    inner, temperature = _unwrap_temperature_scaled_route(route_fn)
    if isinstance(inner, MultiHeadRoute):
        if inner.aggregate != "max":
            raise TypeError("Only max-aggregated MultiHeadRoute is supported for native/kernel spec.")
        head_specs = [pairwise_route_kernel_spec(head) for head in inner.heads]
        if not head_specs:
            raise TypeError("MultiHeadRoute requires at least one head.")
        base_kind = head_specs[0].kind
        if any(spec.kind != base_kind for spec in head_specs[1:]):
            raise TypeError("All MultiHeadRoute heads must share the same kernel kind.")
        if any(spec.temperature != temperature for spec in head_specs):
            raise TypeError("All MultiHeadRoute heads must share the same temperature.")

        def _stack_optional(name: str) -> Tensor | None:
            values = [getattr(spec, name) for spec in head_specs]
            if all(value is None for value in values):
                return None
            if any(value is None for value in values):
                raise TypeError(f"MultiHeadRoute head specs disagree on optional field {name!r}.")
            return torch.stack(values)

        return PairwiseRouteKernelSpec(
            kind=f"multihead_max_{base_kind}",
            source_weight=_stack_optional("source_weight"),
            source_bias=_stack_optional("source_bias"),
            target_weight=_stack_optional("target_weight"),
            target_bias=_stack_optional("target_bias"),
            core_weight=torch.stack([spec.core_weight for spec in head_specs]),
            bias=_stack_optional("bias"),
            hidden_weight=_stack_optional("hidden_weight"),
            hidden_bias=_stack_optional("hidden_bias"),
            out_weight=_stack_optional("out_weight"),
            out_bias=_stack_optional("out_bias"),
            temperature=temperature,
        )
    if isinstance(inner, DiagonalBilinearRoute):
        return PairwiseRouteKernelSpec(
            kind="diagonal_bilinear_route",
            source_weight=None,
            source_bias=None,
            target_weight=None,
            target_bias=None,
            core_weight=inner.normalized_weight(),
            bias=inner.bias,
            temperature=temperature,
        )
    if isinstance(inner, LowRankBilinearRoute):
        return PairwiseRouteKernelSpec(
            kind="low_rank_bilinear_route",
            source_weight=inner.source_proj.weight,
            source_bias=None,
            target_weight=inner.target_proj.weight,
            target_bias=None,
            core_weight=inner.weight,
            bias=inner.bias,
            temperature=temperature,
        )
    if isinstance(inner, BilinearPairwiseRoute):
        return PairwiseRouteKernelSpec(
            kind="full_bilinear_route",
            source_weight=inner.source_proj.weight,
            source_bias=None,
            target_weight=inner.target_proj.weight,
            target_bias=None,
            core_weight=inner.weight,
            bias=inner.bias,
            temperature=temperature,
        )
    if isinstance(inner, QueryNormalizedDotRoute):
        return PairwiseRouteKernelSpec(
            kind="query_normalized_dot_route",
            source_weight=None,
            source_bias=None,
            target_weight=None,
            target_bias=None,
            core_weight=inner.scale_buffer,
            bias=inner.eps_buffer,
            temperature=temperature,
        )
    if isinstance(inner, SourceTargetHadamardMLPRoute):
        return PairwiseRouteKernelSpec(
            kind="source_target_hadamard_mlp_route",
            source_weight=inner.source_proj.weight,
            source_bias=inner.source_proj.bias,
            target_weight=inner.target_proj.weight,
            target_bias=inner.target_proj.bias,
            core_weight=inner.proj_in.weight,
            bias=None,
            hidden_weight=inner.proj_in.weight,
            hidden_bias=inner.proj_in.bias,
            out_weight=inner.proj_out.weight,
            out_bias=inner.proj_out.bias,
            temperature=temperature,
        )
    raise TypeError("Unsupported pairwise route_fn for native/kernel spec.")


def pairwise_scores_dense(
    pairwise_fn: object, target_val: Tensor, source_val: Tensor
) -> Tensor:
    if isinstance(pairwise_fn, DiagonalBilinearPairwise):
        target_proj = target_val * pairwise_fn.normalized_weight().view(1, 1, -1)
        scores = torch.bmm(target_proj, source_val.transpose(1, 2))
        if pairwise_fn.bias is not None:
            scores = scores + pairwise_fn.bias
        return scores

    if isinstance(pairwise_fn, LowRankBilinearPairwise):
        projected_target = pairwise_fn.target_proj(target_val)
        projected_source = pairwise_fn.source_proj(source_val)
        projected_source = projected_source * pairwise_fn.normalized_weight()
        scores = torch.bmm(projected_target, projected_source.transpose(1, 2))
        if pairwise_fn.bias is not None:
            scores = scores + pairwise_fn.bias
        return scores

    if isinstance(pairwise_fn, BilinearPairwise):
        target_proj = torch.matmul(target_val, pairwise_fn.weight)
        scores = torch.bmm(target_proj, source_val.transpose(1, 2))
        if pairwise_fn.bias is not None:
            scores = scores + pairwise_fn.bias
        return scores

    if isinstance(pairwise_fn, HadamardMLPPairwise):
        interaction = target_val.unsqueeze(-2) * source_val.unsqueeze(-3)
        hidden = pairwise_fn.activation(pairwise_fn.proj_in(interaction))
        return pairwise_fn.proj_out(hidden).squeeze(-1)

    raise TypeError("Unsupported pairwise_fn for kernel path.")


def route_logits(route_fn: object, src_val: Tensor) -> Tensor:
    if isinstance(route_fn, LinearRoute):
        return F.linear(src_val, route_fn.linear.weight, route_fn.linear.bias)
    if isinstance(route_fn, MLPRoute):
        hidden = prepare_route_context(route_fn, src_val)
        return route_block_logits(
            route_fn,
            hidden,
            start=0,
            end=route_fn.net[2].out_features,
        )
    return route_fn(src_val)


def prepare_route_context(route_fn: object, src_val: Tensor) -> Tensor:
    if isinstance(route_fn, LinearRoute):
        return src_val
    if isinstance(route_fn, MLPRoute):
        first = route_fn.net[0]
        hidden = F.linear(src_val, first.weight, first.bias)
        return F.silu(hidden)
    raise TypeError("Unsupported route_fn for block kernel path.")


def route_block_logits(
    route_fn: object,
    route_context: Tensor,
    *,
    start: int,
    end: int,
) -> Tensor:
    if start < 0:
        raise ValueError("start must be non-negative.")
    if end < start:
        raise ValueError("end must be greater than or equal to start.")

    if isinstance(route_fn, LinearRoute):
        weight = route_fn.linear.weight[start:end]
        bias = None if route_fn.linear.bias is None else route_fn.linear.bias[start:end]
        return F.linear(route_context, weight, bias)

    if isinstance(route_fn, MLPRoute):
        second = route_fn.net[2]
        weight = second.weight[start:end]
        bias = None if second.bias is None else second.bias[start:end]
        return F.linear(route_context, weight, bias)

    raise TypeError("Unsupported route_fn for block kernel path.")


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked_logits = torch.where(mask, logits, masked_logits)
    return torch.softmax(masked_logits, dim=dim)


def select_topk(scores: Tensor, topk: int, *, dim: int = -1) -> torch.return_types.topk:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    k = min(topk, scores.shape[dim])
    return scores.topk(k=k, dim=dim)


def build_topk_mask(scores: Tensor, topk: int, *, dim: int = -1) -> Tensor:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if topk >= scores.shape[dim]:
        return torch.ones_like(scores, dtype=torch.bool)
    indices = select_topk(scores, topk, dim=dim).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(dim, indices, True)


def causal_window_mask(
    target_start: int,
    target_end: int,
    source_start: int,
    source_end: int,
    window: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    if window < 0:
        raise ValueError("window must be non-negative.")
    target_idx = torch.arange(target_start, target_end, device=device).unsqueeze(-1)
    source_idx = torch.arange(source_start, source_end, device=device).unsqueeze(0)
    return (source_idx <= target_idx) & (source_idx >= target_idx - window)


def normalize_slot_mask(
    slot_mask: Tensor,
    *,
    batch_shape: torch.Size | tuple[int, ...],
    num_nodes: int,
    device: torch.device | str | None = None,
) -> Tensor:
    expected_shape = (*batch_shape, num_nodes)
    mask = slot_mask.to(device=device, dtype=torch.bool)
    if tuple(mask.shape) == expected_shape:
        return mask
    if tuple(mask.shape) == (num_nodes,):
        view = (1,) * len(batch_shape) + (num_nodes,)
        return mask.view(view).expand(expected_shape)
    raise ValueError(
        f"slot_mask must have shape {expected_shape} or {(num_nodes,)}, got {tuple(mask.shape)}."
    )


def apply_slot_mask_to_state(state: Tensor, slot_mask: Tensor) -> Tensor:
    normalized = normalize_slot_mask(
        slot_mask,
        batch_shape=state.shape[:-1],
        num_nodes=state.shape[-1],
        device=state.device,
    )
    return state * normalized.to(dtype=state.dtype)


def apply_slot_mask_to_val(val: Tensor, slot_mask: Tensor) -> Tensor:
    normalized = normalize_slot_mask(
        slot_mask,
        batch_shape=val.shape[:-2],
        num_nodes=val.shape[-2],
        device=val.device,
    )
    return val * normalized.to(dtype=val.dtype).unsqueeze(-1)


def pairwise_slot_mask(target_slot_mask: Tensor, source_slot_mask: Tensor) -> Tensor:
    if target_slot_mask.shape[:-1] != source_slot_mask.shape[:-1]:
        raise ValueError("target_slot_mask and source_slot_mask must share batch shape.")
    return target_slot_mask.unsqueeze(-1) & source_slot_mask.unsqueeze(-2)


def route_slot_mask(source_slot_mask: Tensor, dst_slot_mask: Tensor) -> Tensor:
    if source_slot_mask.shape[:-1] != dst_slot_mask.shape[:-1]:
        raise ValueError("source_slot_mask and dst_slot_mask must share batch shape.")
    return source_slot_mask.unsqueeze(-1) & dst_slot_mask.unsqueeze(-2)


def gather_state_by_indices(projected_state: Tensor, indices: Tensor) -> Tensor:
    target_nodes = indices.shape[-2]
    source_nodes = projected_state.shape[-1]
    expanded_state = projected_state.unsqueeze(-2).expand(
        *projected_state.shape[:-1], target_nodes, source_nodes
    )
    return torch.take_along_dim(expanded_state, indices, dim=-1)


def gather_val_by_indices(projected_val: Tensor, indices: Tensor) -> Tensor:
    target_nodes = indices.shape[-2]
    source_nodes = projected_val.shape[-2]
    dim = projected_val.shape[-1]
    expanded_val = projected_val.unsqueeze(-3).expand(
        *projected_val.shape[:-2], target_nodes, source_nodes, dim
    )
    gather_indices = indices.unsqueeze(-1).expand(*indices.shape, dim)
    return torch.take_along_dim(expanded_val, gather_indices, dim=-2)


def online_softmax_reduce_step(
    state: OnlineSoftmaxState | None, logits: Tensor, values: Tensor
) -> OnlineSoftmaxState:
    if logits.shape[:-1] != values.shape[:-2]:
        raise ValueError("values must match logits on every dimension except the last.")
    if logits.shape[-1] != values.shape[-2]:
        raise ValueError("values must align with the logits reduction dimension.")

    block_max = logits.max(dim=-1).values
    block_exp = torch.exp(logits - block_max.unsqueeze(-1))
    block_sum = block_exp.sum(dim=-1)
    block_weighted = (block_exp.unsqueeze(-1) * values).sum(dim=-2)

    if state is None:
        return OnlineSoftmaxState(
            max_logits=block_max,
            exp_sums=block_sum,
            weighted_values=block_weighted,
        )

    next_max = torch.maximum(state.max_logits, block_max)
    state_scale = torch.exp(state.max_logits - next_max)
    block_scale = torch.exp(block_max - next_max)
    extra_dims = (1,) * (state.weighted_values.ndim - state.max_logits.ndim)
    return OnlineSoftmaxState(
        max_logits=next_max,
        exp_sums=state.exp_sums * state_scale + block_sum * block_scale,
        weighted_values=state.weighted_values * state_scale.view(*state_scale.shape, *extra_dims)
        + block_weighted * block_scale.view(*block_scale.shape, *extra_dims),
    )


def finalize_online_softmax(state: OnlineSoftmaxState, *, eps: float = 1e-12) -> Tensor:
    denom = state.exp_sums.clamp_min(eps)
    extra_dims = (1,) * (state.weighted_values.ndim - denom.ndim)
    return state.weighted_values / denom.view(*denom.shape, *extra_dims)


def online_softmax_stats_step(
    state: OnlineSoftmaxStats | None, logits: Tensor
) -> OnlineSoftmaxStats:
    block_max = logits.max(dim=-1).values
    block_exp = torch.exp(logits - block_max.unsqueeze(-1))
    block_sum = block_exp.sum(dim=-1)

    if state is None:
        return OnlineSoftmaxStats(max_logits=block_max, exp_sums=block_sum)

    next_max = torch.maximum(state.max_logits, block_max)
    state_scale = torch.exp(state.max_logits - next_max)
    block_scale = torch.exp(block_max - next_max)
    return OnlineSoftmaxStats(
        max_logits=next_max,
        exp_sums=state.exp_sums * state_scale + block_sum * block_scale,
    )


def normalize_with_online_softmax(
    logits: Tensor,
    stats: OnlineSoftmaxStats,
    *,
    eps: float = 1e-12,
) -> Tensor:
    denom = stats.exp_sums.clamp_min(eps).unsqueeze(-1)
    return torch.exp(logits - stats.max_logits.unsqueeze(-1)) / denom
