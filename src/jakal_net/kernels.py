from __future__ import annotations

from math import prod
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F

from jakal_net.core import LayerDelta
from jakal_net.modules import BilinearPairwise, DiagonalBilinearPairwise, LinearRoute


def _flatten_state(state: Tensor) -> Tensor:
    return state.reshape(prod(state.shape[:-1]) or 1, state.shape[-1])


def _flatten_val(val: Tensor) -> Tensor:
    return val.reshape(prod(val.shape[:-2]) or 1, val.shape[-2], val.shape[-1])


def _reshape_state(flat_state: Tensor, batch_shape: torch.Size | tuple[int, ...], nodes: int) -> Tensor:
    return flat_state.reshape(*batch_shape, nodes)


def _reshape_val(
    flat_val: Tensor,
    batch_shape: torch.Size | tuple[int, ...],
    nodes: int,
    dim: int,
) -> Tensor:
    return flat_val.reshape(*batch_shape, nodes, dim)


def supports_pairwise_kernel(pairwise_fn: object) -> bool:
    return isinstance(pairwise_fn, (DiagonalBilinearPairwise, BilinearPairwise))


def _pairwise_scores_dense(pairwise_fn: object, target_val: Tensor, source_val: Tensor) -> Tensor:
    if isinstance(pairwise_fn, DiagonalBilinearPairwise):
        target_proj = target_val * pairwise_fn.weight.view(1, 1, -1)
        scores = torch.bmm(target_proj, source_val.transpose(1, 2))
        if pairwise_fn.bias is not None:
            scores = scores + pairwise_fn.bias
        return scores

    if isinstance(pairwise_fn, BilinearPairwise):
        target_proj = torch.matmul(target_val, pairwise_fn.weight)
        scores = torch.bmm(target_proj, source_val.transpose(1, 2))
        if pairwise_fn.bias is not None:
            scores = scores + pairwise_fn.bias
        return scores

    raise TypeError("Unsupported pairwise_fn for kernel path.")


def propagation_dense_kernel(
    *,
    pairwise_fn: object,
    edge_compress_fn: Callable[[Tensor], Tensor],
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
) -> LayerDelta:
    batch_shape = layer_val.shape[:-2]
    num_nodes = layer_val.shape[-2]
    out_dim = projected_val.shape[-1]

    flat_val = _flatten_val(layer_val)
    flat_projected_state = _flatten_state(projected_state)
    flat_projected_val = _flatten_val(projected_val)

    scores = _pairwise_scores_dense(pairwise_fn, flat_val, flat_val)
    edges = edge_compress_fn(scores)
    delta_state = torch.bmm(edges, flat_projected_state.unsqueeze(-1)).squeeze(-1)
    delta_val = torch.bmm(edges, flat_projected_val)

    return LayerDelta(
        delta_state=_reshape_state(delta_state, batch_shape, num_nodes),
        delta_val=_reshape_val(delta_val, batch_shape, num_nodes, out_dim),
    )


def propagation_window_kernel(
    *,
    pairwise_fn: object,
    edge_compress_fn: Callable[[Tensor], Tensor],
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    window: int,
) -> LayerDelta:
    if window < 0:
        raise ValueError("window must be non-negative.")

    batch_shape = layer_val.shape[:-2]
    num_nodes = layer_val.shape[-2]
    val_dim = layer_val.shape[-1]
    out_dim = projected_val.shape[-1]
    flat_val = _flatten_val(layer_val)
    flat_projected_state = _flatten_state(projected_state)
    flat_projected_val = _flatten_val(projected_val)

    if isinstance(pairwise_fn, DiagonalBilinearPairwise):
        target_proj = flat_val * pairwise_fn.weight.view(1, 1, val_dim)
        source_basis = flat_val
        bias = pairwise_fn.bias
    elif isinstance(pairwise_fn, BilinearPairwise):
        target_proj = torch.matmul(flat_val, pairwise_fn.weight)
        source_basis = flat_val
        bias = pairwise_fn.bias
    else:
        raise TypeError("Unsupported pairwise_fn for kernel path.")

    padded_source = F.pad(source_basis, (0, 0, window, 0))
    source_windows = padded_source.unfold(1, window + 1, 1).permute(0, 1, 3, 2)
    scores = (target_proj.unsqueeze(-2) * source_windows).sum(dim=-1)
    if bias is not None:
        scores = scores + bias

    edges = edge_compress_fn(scores)

    padded_state = F.pad(flat_projected_state, (window, 0))
    state_windows = padded_state.unfold(1, window + 1, 1)
    delta_state = (edges * state_windows).sum(dim=-1)

    padded_val = F.pad(flat_projected_val, (0, 0, window, 0))
    val_windows = padded_val.unfold(1, window + 1, 1).permute(0, 1, 3, 2)
    delta_val = (edges.unsqueeze(-1) * val_windows).sum(dim=-2)

    return LayerDelta(
        delta_state=_reshape_state(delta_state, batch_shape, num_nodes),
        delta_val=_reshape_val(delta_val, batch_shape, num_nodes, out_dim),
    )


def propagation_topk_kernel(
    *,
    pairwise_fn: object,
    edge_compress_fn: Callable[[Tensor], Tensor],
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    topk: int,
) -> LayerDelta:
    if topk <= 0:
        raise ValueError("topk must be positive.")

    batch_shape = layer_val.shape[:-2]
    num_nodes = layer_val.shape[-2]
    out_dim = projected_val.shape[-1]
    flat_val = _flatten_val(layer_val)
    flat_projected_state = _flatten_state(projected_state)
    flat_projected_val = _flatten_val(projected_val)

    scores = _pairwise_scores_dense(pairwise_fn, flat_val, flat_val)
    k = min(topk, num_nodes)
    topk_values, topk_indices = scores.topk(k=k, dim=-1)
    edges = edge_compress_fn(topk_values)

    state_index = topk_indices
    state_source = flat_projected_state.unsqueeze(-2).expand(-1, num_nodes, -1)
    selected_state = torch.take_along_dim(state_source, state_index, dim=-1)
    delta_state = (edges * selected_state).sum(dim=-1)

    val_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, out_dim)
    val_source = flat_projected_val.unsqueeze(-3).expand(-1, num_nodes, -1, -1)
    selected_val = torch.take_along_dim(val_source, val_index, dim=-2)
    delta_val = (edges.unsqueeze(-1) * selected_val).sum(dim=-2)

    return LayerDelta(
        delta_state=_reshape_state(delta_state, batch_shape, num_nodes),
        delta_val=_reshape_val(delta_val, batch_shape, num_nodes, out_dim),
    )


def _route_logits(route_fn: object, src_val: Tensor) -> Tensor:
    if isinstance(route_fn, LinearRoute):
        return F.linear(src_val, route_fn.linear.weight, route_fn.linear.bias)
    return route_fn(src_val)


def transition_dense_kernel(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    dst_nodes: int,
) -> LayerDelta:
    batch_shape = src_val.shape[:-2]
    src_nodes = src_val.shape[-2]
    out_dim = projected_val.shape[-1]

    flat_src_val = _flatten_val(src_val)
    flat_sender_strength = _flatten_state(sender_strength)
    flat_projected_state = _flatten_state(projected_state)
    flat_projected_val = _flatten_val(projected_val)

    logits = _route_logits(route_fn, flat_src_val)
    if logits.shape != (flat_src_val.shape[0], src_nodes, dst_nodes):
        raise ValueError("route_fn returned an unexpected shape in kernel path.")
    routes = torch.softmax(logits, dim=-1)
    weighted_routes = routes * flat_sender_strength.unsqueeze(-1)
    transport = weighted_routes.transpose(1, 2).contiguous()

    delta_state = torch.bmm(transport, flat_projected_state.unsqueeze(-1)).squeeze(-1)
    delta_val = torch.bmm(transport, flat_projected_val)

    return LayerDelta(
        delta_state=_reshape_state(delta_state, batch_shape, dst_nodes),
        delta_val=_reshape_val(delta_val, batch_shape, dst_nodes, out_dim),
    )


def transition_topk_kernel(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    dst_nodes: int,
    topk: int,
) -> LayerDelta:
    if topk <= 0:
        raise ValueError("topk must be positive.")

    batch_shape = src_val.shape[:-2]
    src_nodes = src_val.shape[-2]
    out_dim = projected_val.shape[-1]
    batch_flat = prod(batch_shape) or 1

    flat_src_val = _flatten_val(src_val)
    flat_sender_strength = _flatten_state(sender_strength)
    flat_projected_state = _flatten_state(projected_state)
    flat_projected_val = _flatten_val(projected_val)

    logits = _route_logits(route_fn, flat_src_val)
    if logits.shape != (batch_flat, src_nodes, dst_nodes):
        raise ValueError("route_fn returned an unexpected shape in kernel path.")

    k = min(topk, dst_nodes)
    if k == dst_nodes:
        return transition_dense_kernel(
            route_fn=route_fn,
            sender_strength=sender_strength,
            src_val=src_val,
            projected_state=projected_state,
            projected_val=projected_val,
            dst_nodes=dst_nodes,
        )

    topk_values, topk_indices = logits.topk(k=k, dim=-1)
    routes = torch.softmax(topk_values, dim=-1)
    weighted_routes = routes * flat_sender_strength.unsqueeze(-1)

    delta_state = torch.zeros(
        batch_flat, dst_nodes, device=src_val.device, dtype=projected_state.dtype
    )
    state_contrib = (
        weighted_routes * flat_projected_state.unsqueeze(-1)
    ).reshape(batch_flat, -1)
    delta_state.scatter_add_(dim=-1, index=topk_indices.reshape(batch_flat, -1), src=state_contrib)

    delta_val = torch.zeros(
        batch_flat, dst_nodes, out_dim, device=src_val.device, dtype=projected_val.dtype
    )
    val_contrib = (
        weighted_routes.unsqueeze(-1) * flat_projected_val.unsqueeze(-2)
    ).reshape(batch_flat, -1, out_dim)
    scatter_index = topk_indices.unsqueeze(-1).expand(-1, -1, -1, out_dim).reshape(
        batch_flat, -1, out_dim
    )
    delta_val.scatter_add_(dim=1, index=scatter_index, src=val_contrib)

    return LayerDelta(
        delta_state=_reshape_state(delta_state, batch_shape, dst_nodes),
        delta_val=_reshape_val(delta_val, batch_shape, dst_nodes, out_dim),
    )
