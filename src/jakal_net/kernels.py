from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from jakal_net.core import LayerDelta, allocate_accumulator, iter_blocks
from jakal_net.kernel_common import (
    causal_window_mask,
    flatten_state,
    flatten_val,
    gather_state_by_indices,
    gather_val_by_indices,
    normalize_with_online_softmax,
    online_softmax_stats_step,
    pairwise_scores_dense,
    prepare_route_context,
    reshape_state,
    reshape_val,
    route_block_logits,
    select_topk,
    supports_pairwise_kernel,
    supports_route_kernel,
)


def _signed_abs_softmax(logits: Tensor, *, dim: int = -1) -> Tensor:
    clean_logits = torch.nan_to_num(logits)
    return torch.sign(clean_logits) * torch.softmax(clean_logits.abs(), dim=dim)


def _edge_compress_name(edge_compress_fn: Callable[[Tensor], Tensor]) -> str | None:
    return getattr(edge_compress_fn, "__name__", "")


def _compress_edges(
    scores: Tensor,
    edge_compress_fn: Callable[[Tensor], Tensor],
    *,
    mask: Tensor | None = None,
) -> Tensor:
    if _edge_compress_name(edge_compress_fn) in {"signed_abs_softmax", "_signed_abs_softmax_edges"}:
        clean_scores = torch.nan_to_num(scores)
        signs = torch.sign(clean_scores)
        magnitudes = clean_scores.abs()
        if mask is not None:
            bool_mask = mask.to(dtype=torch.bool)
            signs = signs * bool_mask.to(dtype=signs.dtype)
            magnitudes = magnitudes.masked_fill(~bool_mask, -torch.inf)
        return torch.nan_to_num(signs * torch.softmax(magnitudes, dim=-1))
    edges = edge_compress_fn(scores)
    if mask is not None:
        edges = edges * mask.to(dtype=edges.dtype)
    return edges


def _compress_routes(logits: Tensor, *, route_compress_name: str) -> Tensor:
    if route_compress_name == "softmax":
        return torch.softmax(logits, dim=-1)
    if route_compress_name == "signed_abs_softmax":
        return _signed_abs_softmax(logits, dim=-1)
    raise ValueError(f"Unsupported route_compress_name: {route_compress_name!r}.")


def _weight_edges(edges: Tensor, flat_source_state: Tensor | None, source_start: int, source_end: int) -> Tensor:
    if flat_source_state is None:
        return edges
    return edges * flat_source_state[:, source_start:source_end].unsqueeze(1)


def propagation_dense_kernel(
    *,
    pairwise_fn: object,
    edge_compress_fn: Callable[[Tensor], Tensor],
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_state: Tensor | None = None,
    target_block_size: int | None = 128,
    source_block_size: int | None = 128,
    accumulator_dtype: torch.dtype | None = None,
) -> LayerDelta:
    batch_shape = layer_val.shape[:-2]
    num_nodes = layer_val.shape[-2]
    out_dim = projected_val.shape[-1]

    flat_val = flatten_val(layer_val)
    flat_projected_state = flatten_state(projected_state)
    flat_projected_val = flatten_val(projected_val)
    flat_source_state = None if source_state is None else flatten_state(source_state)
    state_acc_dtype = allocate_accumulator(
        (1,),
        device=layer_val.device,
        tensor_dtype=projected_state.dtype,
        accumulator_dtype=accumulator_dtype,
    ).dtype
    val_acc_dtype = allocate_accumulator(
        (1,),
        device=layer_val.device,
        tensor_dtype=projected_val.dtype,
        accumulator_dtype=accumulator_dtype,
    ).dtype
    state_blocks: list[Tensor] = []
    val_blocks: list[Tensor] = []

    for target_start, target_end in iter_blocks(
        num_nodes, target_block_size, name="target_block_size"
    ):
        target_val = flat_val[:, target_start:target_end, :]
        target_state_acc = allocate_accumulator(
            (flat_val.shape[0], target_end - target_start),
            device=layer_val.device,
            tensor_dtype=projected_state.dtype,
            accumulator_dtype=accumulator_dtype,
        )
        target_val_acc = allocate_accumulator(
            (flat_val.shape[0], target_end - target_start, out_dim),
            device=layer_val.device,
            tensor_dtype=projected_val.dtype,
            accumulator_dtype=accumulator_dtype,
        )
        for source_start, source_end in iter_blocks(
            num_nodes, source_block_size, name="source_block_size"
        ):
            source_val = flat_val[:, source_start:source_end, :]
            scores = pairwise_scores_dense(pairwise_fn, target_val, source_val)
            edges = _weight_edges(
                _compress_edges(scores, edge_compress_fn),
                flat_source_state,
                source_start,
                source_end,
            )
            state_edges = edges.to(dtype=state_acc_dtype)
            val_edges = edges.to(dtype=val_acc_dtype)

            target_state_acc += torch.bmm(
                state_edges,
                flat_projected_state[:, source_start:source_end]
                .to(dtype=state_acc_dtype)
                .unsqueeze(-1),
            ).squeeze(-1)
            target_val_acc += torch.bmm(
                val_edges,
                flat_projected_val[:, source_start:source_end, :].to(dtype=val_acc_dtype),
            )
        state_blocks.append(target_state_acc.to(dtype=projected_state.dtype))
        val_blocks.append(target_val_acc.to(dtype=projected_val.dtype))

    return LayerDelta(
        delta_state=reshape_state(torch.cat(state_blocks, dim=1), batch_shape, num_nodes),
        delta_val=reshape_val(torch.cat(val_blocks, dim=1), batch_shape, num_nodes, out_dim),
    )


def propagation_window_kernel(
    *,
    pairwise_fn: object,
    edge_compress_fn: Callable[[Tensor], Tensor],
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_state: Tensor | None = None,
    window: int,
    target_block_size: int | None = 128,
    source_block_size: int | None = 128,
    accumulator_dtype: torch.dtype | None = None,
) -> LayerDelta:
    if window < 0:
        raise ValueError("window must be non-negative.")

    batch_shape = layer_val.shape[:-2]
    num_nodes = layer_val.shape[-2]
    out_dim = projected_val.shape[-1]

    flat_val = flatten_val(layer_val)
    flat_projected_state = flatten_state(projected_state)
    flat_projected_val = flatten_val(projected_val)
    flat_source_state = None if source_state is None else flatten_state(source_state)
    state_acc_dtype = allocate_accumulator(
        (1,),
        device=layer_val.device,
        tensor_dtype=projected_state.dtype,
        accumulator_dtype=accumulator_dtype,
    ).dtype
    val_acc_dtype = allocate_accumulator(
        (1,),
        device=layer_val.device,
        tensor_dtype=projected_val.dtype,
        accumulator_dtype=accumulator_dtype,
    ).dtype
    state_blocks: list[Tensor] = []
    val_blocks: list[Tensor] = []

    for target_start, target_end in iter_blocks(
        num_nodes, target_block_size, name="target_block_size"
    ):
        target_val = flat_val[:, target_start:target_end, :]
        source_floor = max(0, target_start - window)
        source_ceiling = target_end
        target_state_acc = allocate_accumulator(
            (flat_val.shape[0], target_end - target_start),
            device=layer_val.device,
            tensor_dtype=projected_state.dtype,
            accumulator_dtype=accumulator_dtype,
        )
        target_val_acc = allocate_accumulator(
            (flat_val.shape[0], target_end - target_start, out_dim),
            device=layer_val.device,
            tensor_dtype=projected_val.dtype,
            accumulator_dtype=accumulator_dtype,
        )

        source_width = source_ceiling - source_floor
        if _edge_compress_name(edge_compress_fn) in {"signed_abs_softmax", "_signed_abs_softmax_edges"}:
            source_blocks = ((0, source_width),)
        else:
            source_blocks = iter_blocks(
                source_width,
                source_block_size,
                name="source_block_size",
            )

        for source_offset_start, source_offset_end in source_blocks:
            source_start = source_floor + source_offset_start
            source_end = source_floor + source_offset_end
            source_val = flat_val[:, source_start:source_end, :]
            scores = pairwise_scores_dense(pairwise_fn, target_val, source_val)
            mask = causal_window_mask(
                target_start,
                target_end,
                source_start,
                source_end,
                window,
                device=layer_val.device,
            ).view(1, target_end - target_start, source_end - source_start)
            edges = _compress_edges(scores, edge_compress_fn, mask=mask)
            edges = _weight_edges(edges, flat_source_state, source_start, source_end)
            state_edges = edges.to(dtype=state_acc_dtype)
            val_edges = edges.to(dtype=val_acc_dtype)

            target_state_acc += torch.bmm(
                state_edges,
                flat_projected_state[:, source_start:source_end]
                .to(dtype=state_acc_dtype)
                .unsqueeze(-1),
            ).squeeze(-1)
            target_val_acc += torch.bmm(
                val_edges,
                flat_projected_val[:, source_start:source_end, :].to(dtype=val_acc_dtype),
            )
        state_blocks.append(target_state_acc.to(dtype=projected_state.dtype))
        val_blocks.append(target_val_acc.to(dtype=projected_val.dtype))

    return LayerDelta(
        delta_state=reshape_state(torch.cat(state_blocks, dim=1), batch_shape, num_nodes),
        delta_val=reshape_val(torch.cat(val_blocks, dim=1), batch_shape, num_nodes, out_dim),
    )


def propagation_topk_kernel(
    *,
    pairwise_fn: object,
    edge_compress_fn: Callable[[Tensor], Tensor],
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_state: Tensor | None = None,
    topk: int,
    target_block_size: int | None = 128,
    source_block_size: int | None = 128,
    accumulator_dtype: torch.dtype | None = None,
) -> LayerDelta:
    if topk <= 0:
        raise ValueError("topk must be positive.")

    batch_shape = layer_val.shape[:-2]
    num_nodes = layer_val.shape[-2]
    out_dim = projected_val.shape[-1]
    k = min(topk, num_nodes)

    flat_val = flatten_val(layer_val)
    flat_projected_state = flatten_state(projected_state)
    flat_projected_val = flatten_val(projected_val)
    state_acc_dtype = allocate_accumulator(
        (1,),
        device=layer_val.device,
        tensor_dtype=projected_state.dtype,
        accumulator_dtype=accumulator_dtype,
    ).dtype
    val_acc_dtype = allocate_accumulator(
        (1,),
        device=layer_val.device,
        tensor_dtype=projected_val.dtype,
        accumulator_dtype=accumulator_dtype,
    ).dtype
    state_blocks: list[Tensor] = []
    val_blocks: list[Tensor] = []

    if k == num_nodes:
        return propagation_dense_kernel(
            pairwise_fn=pairwise_fn,
            edge_compress_fn=edge_compress_fn,
            layer_val=layer_val,
            projected_state=projected_state,
            projected_val=projected_val,
            source_state=source_state,
            target_block_size=target_block_size,
            source_block_size=source_block_size,
            accumulator_dtype=accumulator_dtype,
        )

    for target_start, target_end in iter_blocks(
        num_nodes, target_block_size, name="target_block_size"
    ):
        target_val = flat_val[:, target_start:target_end, :]
        target_nodes = target_end - target_start
        best_scores: Tensor | None = None
        best_indices: Tensor | None = None

        for source_start, source_end in iter_blocks(
            num_nodes, source_block_size, name="source_block_size"
        ):
            source_val = flat_val[:, source_start:source_end, :]
            scores = pairwise_scores_dense(pairwise_fn, target_val, source_val)

            if best_scores is None or best_indices is None:
                best_scores = torch.full(
                    (flat_val.shape[0], target_nodes, k),
                    fill_value=torch.finfo(scores.dtype).min,
                    device=layer_val.device,
                    dtype=scores.dtype,
                )
                best_indices = torch.zeros(
                    (flat_val.shape[0], target_nodes, k),
                    device=layer_val.device,
                    dtype=torch.long,
                )

            source_indices = torch.arange(
                source_start, source_end, device=layer_val.device, dtype=torch.long
            ).view(1, 1, source_end - source_start)
            source_indices = source_indices.expand(flat_val.shape[0], target_nodes, -1)

            candidate_scores = torch.cat((best_scores, scores), dim=-1)
            candidate_indices = torch.cat((best_indices, source_indices), dim=-1)
            selected = select_topk(candidate_scores, k, dim=-1)
            best_scores = selected.values
            best_indices = torch.take_along_dim(
                candidate_indices, selected.indices, dim=-1
            )

        if best_scores is None or best_indices is None:
            continue

        edges = _compress_edges(best_scores, edge_compress_fn)
        if source_state is not None:
            selected_source_state = gather_state_by_indices(
                flatten_state(source_state),
                best_indices,
            )
            edges = edges * selected_source_state
        selected_state = gather_state_by_indices(flat_projected_state, best_indices)
        selected_val = gather_val_by_indices(flat_projected_val, best_indices)

        state_blocks.append(
            (
                edges.to(dtype=state_acc_dtype)
                * selected_state.to(dtype=state_acc_dtype)
            ).sum(dim=-1).to(dtype=projected_state.dtype)
        )
        val_blocks.append(
            (
                edges.to(dtype=val_acc_dtype).unsqueeze(-1)
                * selected_val.to(dtype=val_acc_dtype)
            ).sum(dim=-2).to(dtype=projected_val.dtype)
        )

    return LayerDelta(
        delta_state=reshape_state(torch.cat(state_blocks, dim=1), batch_shape, num_nodes),
        delta_val=reshape_val(torch.cat(val_blocks, dim=1), batch_shape, num_nodes, out_dim),
    )


def transition_dense_kernel(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    dst_nodes: int,
    route_compress_name: str = "softmax",
    src_block_size: int | None = 128,
    dst_block_size: int | None = 128,
    accumulator_dtype: torch.dtype | None = None,
) -> LayerDelta:
    if not supports_route_kernel(route_fn):
        raise TypeError("Unsupported route_fn for dense transition kernel path.")

    batch_shape = src_val.shape[:-2]
    src_nodes = src_val.shape[-2]
    out_dim = projected_val.shape[-1]

    flat_src_val = flatten_val(src_val)
    flat_sender_strength = flatten_state(sender_strength)
    flat_projected_state = flatten_state(projected_state)
    flat_projected_val = flatten_val(projected_val)
    batch_flat = flat_src_val.shape[0]

    delta_state = allocate_accumulator(
        (batch_flat, dst_nodes),
        device=src_val.device,
        tensor_dtype=projected_state.dtype,
        accumulator_dtype=accumulator_dtype,
    )
    delta_val = allocate_accumulator(
        (batch_flat, dst_nodes, out_dim),
        device=src_val.device,
        tensor_dtype=projected_val.dtype,
        accumulator_dtype=accumulator_dtype,
    )
    state_acc_dtype = delta_state.dtype
    val_acc_dtype = delta_val.dtype

    for src_start, src_end in iter_blocks(src_nodes, src_block_size, name="src_block_size"):
        src_block = flat_src_val[:, src_start:src_end, :]
        block_nodes = src_end - src_start
        route_context = prepare_route_context(route_fn, src_block)
        softmax_stats = None

        for dst_start, dst_end in iter_blocks(
            dst_nodes, dst_block_size, name="dst_block_size"
        ):
            logits_block = route_block_logits(
                route_fn,
                route_context,
                start=dst_start,
                end=dst_end,
            )
            expected_shape = (batch_flat, block_nodes, dst_end - dst_start)
            if tuple(logits_block.shape) != expected_shape:
                raise ValueError(
                    "route_fn returned an unexpected block shape in kernel path, "
                    f"expected {expected_shape}, got {tuple(logits_block.shape)}."
                )
            stats_logits = logits_block.abs() if route_compress_name == "signed_abs_softmax" else logits_block
            softmax_stats = online_softmax_stats_step(softmax_stats, stats_logits)

        if softmax_stats is None:
            continue

        state_sender = (
            flat_sender_strength[:, src_start:src_end].to(dtype=state_acc_dtype)
            * flat_projected_state[:, src_start:src_end].to(dtype=state_acc_dtype)
        )
        val_sender = (
            flat_sender_strength[:, src_start:src_end]
            .to(dtype=val_acc_dtype)
            .unsqueeze(-1)
            * flat_projected_val[:, src_start:src_end, :].to(dtype=val_acc_dtype)
        )

        for dst_start, dst_end in iter_blocks(
            dst_nodes, dst_block_size, name="dst_block_size"
        ):
            logits_block = route_block_logits(
                route_fn,
                route_context,
                start=dst_start,
                end=dst_end,
            )
            if route_compress_name == "softmax":
                routes_block = normalize_with_online_softmax(
                    logits_block, softmax_stats
                )
            elif route_compress_name == "signed_abs_softmax":
                routes_block = torch.sign(torch.nan_to_num(logits_block)) * normalize_with_online_softmax(
                    logits_block.abs(), softmax_stats
                )
            else:
                raise ValueError(f"Unsupported route_compress_name: {route_compress_name!r}.")
            delta_state[:, dst_start:dst_end] += torch.bmm(
                routes_block.to(dtype=state_acc_dtype).transpose(1, 2).contiguous(),
                state_sender.unsqueeze(-1),
            ).squeeze(-1)
            delta_val[:, dst_start:dst_end, :] += torch.bmm(
                routes_block.to(dtype=val_acc_dtype).transpose(1, 2).contiguous(),
                val_sender,
            )

    return LayerDelta(
        delta_state=reshape_state(
            delta_state.to(dtype=projected_state.dtype), batch_shape, dst_nodes
        ),
        delta_val=reshape_val(
            delta_val.to(dtype=projected_val.dtype), batch_shape, dst_nodes, out_dim
        ),
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
    route_compress_name: str = "softmax",
    src_block_size: int | None = 128,
    dst_block_size: int | None = 128,
    accumulator_dtype: torch.dtype | None = None,
) -> LayerDelta:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if not supports_route_kernel(route_fn):
        raise TypeError("Unsupported route_fn for sparse transition kernel path.")

    batch_shape = src_val.shape[:-2]
    src_nodes = src_val.shape[-2]
    out_dim = projected_val.shape[-1]
    k = min(topk, dst_nodes)

    if k == dst_nodes:
        return transition_dense_kernel(
            route_fn=route_fn,
            sender_strength=sender_strength,
            src_val=src_val,
            projected_state=projected_state,
            projected_val=projected_val,
            dst_nodes=dst_nodes,
            route_compress_name=route_compress_name,
            src_block_size=src_block_size,
            dst_block_size=dst_block_size,
            accumulator_dtype=accumulator_dtype,
        )

    flat_src_val = flatten_val(src_val)
    flat_sender_strength = flatten_state(sender_strength)
    flat_projected_state = flatten_state(projected_state)
    flat_projected_val = flatten_val(projected_val)
    batch_flat = flat_src_val.shape[0]

    delta_state = allocate_accumulator(
        (batch_flat, dst_nodes),
        device=src_val.device,
        tensor_dtype=projected_state.dtype,
        accumulator_dtype=accumulator_dtype,
    )
    delta_val = allocate_accumulator(
        (batch_flat, dst_nodes, out_dim),
        device=src_val.device,
        tensor_dtype=projected_val.dtype,
        accumulator_dtype=accumulator_dtype,
    )
    state_acc_dtype = delta_state.dtype
    val_acc_dtype = delta_val.dtype

    for src_start, src_end in iter_blocks(src_nodes, src_block_size, name="src_block_size"):
        src_block = flat_src_val[:, src_start:src_end, :]
        block_nodes = src_end - src_start
        route_context = prepare_route_context(route_fn, src_block)
        best_values = torch.full(
            (batch_flat, block_nodes, k),
            fill_value=torch.finfo(src_block.dtype).min,
            device=src_val.device,
            dtype=src_block.dtype,
        )
        best_indices = torch.zeros(
            (batch_flat, block_nodes, k),
            device=src_val.device,
            dtype=torch.long,
        )

        for dst_start, dst_end in iter_blocks(
            dst_nodes, dst_block_size, name="dst_block_size"
        ):
            logits_block = route_block_logits(
                route_fn,
                route_context,
                start=dst_start,
                end=dst_end,
            )
            expected_shape = (batch_flat, block_nodes, dst_end - dst_start)
            if tuple(logits_block.shape) != expected_shape:
                raise ValueError(
                    "route_fn returned an unexpected block shape in kernel path, "
                    f"expected {expected_shape}, got {tuple(logits_block.shape)}."
                )

            block_indices = torch.arange(
                dst_start, dst_end, device=src_val.device, dtype=torch.long
            ).view(1, 1, dst_end - dst_start)
            block_indices = block_indices.expand(batch_flat, block_nodes, -1)
            candidate_values = torch.cat((best_values, logits_block), dim=-1)
            candidate_indices = torch.cat((best_indices, block_indices), dim=-1)
            selected = select_topk(candidate_values, k, dim=-1)
            best_values = selected.values
            best_indices = torch.take_along_dim(
                candidate_indices, selected.indices, dim=-1
            )

        routes = _compress_routes(best_values, route_compress_name=route_compress_name)
        state_contrib = (
            routes.to(dtype=state_acc_dtype)
            * (
                flat_sender_strength[:, src_start:src_end].to(dtype=state_acc_dtype)
                * flat_projected_state[:, src_start:src_end].to(dtype=state_acc_dtype)
            ).unsqueeze(-1)
        ).reshape(batch_flat, -1)
        delta_state.scatter_add_(
            dim=-1,
            index=best_indices.reshape(batch_flat, -1),
            src=state_contrib,
        )

        val_contrib = (
            routes.to(dtype=val_acc_dtype).unsqueeze(-1)
            * (
                flat_sender_strength[:, src_start:src_end]
                .to(dtype=val_acc_dtype)
                .unsqueeze(-1)
                * flat_projected_val[:, src_start:src_end, :].to(dtype=val_acc_dtype)
            ).unsqueeze(-2)
        ).reshape(batch_flat, -1, out_dim)
        scatter_index = best_indices.unsqueeze(-1).expand(-1, -1, -1, out_dim).reshape(
            batch_flat, -1, out_dim
        )
        delta_val.scatter_add_(dim=1, index=scatter_index, src=val_contrib)

    return LayerDelta(
        delta_state=reshape_state(
            delta_state.to(dtype=projected_state.dtype), batch_shape, dst_nodes
        ),
        delta_val=reshape_val(
            delta_val.to(dtype=projected_val.dtype), batch_shape, dst_nodes, out_dim
        ),
    )
