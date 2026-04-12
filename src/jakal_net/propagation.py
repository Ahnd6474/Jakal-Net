from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net.core import (
    ImplementationMode,
    Layer,
    LayerDelta,
    SparsePropagationType,
    apply_optional_layer_fn,
    iter_blocks,
    resolve_accumulator_dtype,
    validate_implementation,
    validate_pairwise_block_scores,
    validate_pairwise_scores,
    validate_projected_state,
    validate_projected_val,
)
from jakal_net.kernels import (
    propagation_dense_kernel,
    propagation_topk_kernel,
    propagation_window_kernel,
    supports_pairwise_kernel,
)


def _causal_window_mask(
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


def _topk_mask(scores: Tensor, topk: int) -> Tensor:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    k = min(topk, scores.shape[-1])
    if k == scores.shape[-1]:
        return torch.ones_like(scores, dtype=torch.bool)
    indices = scores.topk(k=k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, indices, True)
    return mask


def _gather_state_by_indices(projected_state: Tensor, indices: Tensor) -> Tensor:
    target_nodes = indices.shape[-2]
    source_nodes = projected_state.shape[-1]
    expanded_state = projected_state.unsqueeze(-2).expand(
        *projected_state.shape[:-1], target_nodes, source_nodes
    )
    return torch.take_along_dim(expanded_state, indices, dim=-1)


def _gather_val_by_indices(projected_val: Tensor, indices: Tensor) -> Tensor:
    target_nodes = indices.shape[-2]
    source_nodes = projected_val.shape[-2]
    dim = projected_val.shape[-1]
    expanded_val = projected_val.unsqueeze(-3).expand(
        *projected_val.shape[:-2], target_nodes, source_nodes, dim
    )
    gather_indices = indices.unsqueeze(-1).expand(*indices.shape, dim)
    return torch.take_along_dim(expanded_val, gather_indices, dim=-2)


class Propagation(nn.Module):
    def __init__(
        self,
        pairwise_fn: Callable[[Tensor, Tensor], Tensor] | nn.Module,
        *,
        edge_compress_fn: Callable[[Tensor], Tensor] = F.softsign,
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        norm_fn: Callable[[Layer], Layer] | None = None,
        residual: bool = True,
        return_delta: bool = True,
        implementation: ImplementationMode = "streaming",
        target_block_size: int | None = 128,
        source_block_size: int | None = 128,
        accumulator_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        validate_implementation(implementation)
        self.pairwise_fn = pairwise_fn
        self.edge_compress_fn = edge_compress_fn
        self.val_proj_fn = nn.Identity() if val_proj_fn is None else val_proj_fn
        self.state_proj_fn = nn.Identity() if state_proj_fn is None else state_proj_fn
        self.norm_fn = norm_fn
        self.residual = residual
        self.return_delta = return_delta
        self.implementation = implementation
        self.target_block_size = target_block_size
        self.source_block_size = source_block_size
        self.accumulator_dtype = accumulator_dtype

    def compute_scores(self, layer: Layer) -> Tensor:
        scores = self.pairwise_fn(layer.val, layer.val)
        validate_pairwise_scores(scores, layer)
        return scores

    def compute_edges(self, layer: Layer) -> Tensor:
        return self.edge_compress_fn(self.compute_scores(layer))

    def _project_inputs(self, layer: Layer) -> tuple[Tensor, Tensor]:
        projected_state = self.state_proj_fn(layer.state)
        projected_val = self.val_proj_fn(layer.val)
        validate_projected_state(projected_state, layer)
        validate_projected_val(projected_val, layer)
        return projected_state, projected_val

    def _allocate_delta_buffers(
        self, layer: Layer, projected_state: Tensor, projected_val: Tensor
    ) -> tuple[Tensor, Tensor]:
        state_acc_dtype = resolve_accumulator_dtype(
            projected_state.dtype, self.accumulator_dtype
        )
        val_acc_dtype = resolve_accumulator_dtype(
            projected_val.dtype, self.accumulator_dtype
        )
        delta_state = torch.zeros(
            layer.state.shape, device=layer.state.device, dtype=state_acc_dtype
        )
        delta_val = torch.zeros(
            layer.val.shape, device=layer.val.device, dtype=val_acc_dtype
        )
        return delta_state, delta_val

    def _compute_delta_reference(self, layer: Layer) -> LayerDelta:
        edges = self.compute_edges(layer)
        projected_state, projected_val = self._project_inputs(layer)

        delta_state = torch.einsum("...ij,...j->...i", edges, projected_state)
        delta_val = torch.einsum("...ij,...jd->...id", edges, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_delta_streaming(self, layer: Layer) -> LayerDelta:
        projected_state, projected_val = self._project_inputs(layer)
        delta_state, delta_val = self._allocate_delta_buffers(
            layer, projected_state, projected_val
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype

        for target_start, target_end in iter_blocks(
            layer.num_nodes, self.target_block_size, name="target_block_size"
        ):
            target_val = layer.val[..., target_start:target_end, :]
            for source_start, source_end in iter_blocks(
                layer.num_nodes, self.source_block_size, name="source_block_size"
            ):
                source_val = layer.val[..., source_start:source_end, :]
                scores = self.pairwise_fn(target_val, source_val)
                validate_pairwise_block_scores(
                    scores=scores,
                    batch_shape=layer.batch_shape,
                    target_nodes=target_end - target_start,
                    source_nodes=source_end - source_start,
                )
                edges = self.edge_compress_fn(scores)
                state_edges = edges.to(dtype=state_acc_dtype)
                val_edges = edges.to(dtype=val_acc_dtype)

                delta_state[..., target_start:target_end] += torch.einsum(
                    "...ij,...j->...i",
                    state_edges,
                    projected_state[..., source_start:source_end].to(state_acc_dtype),
                )
                delta_val[..., target_start:target_end, :] += torch.einsum(
                    "...ij,...jd->...id",
                    val_edges,
                    projected_val[..., source_start:source_end, :].to(val_acc_dtype),
                )

        return LayerDelta(
            delta_state=delta_state.to(projected_state.dtype),
            delta_val=delta_val.to(projected_val.dtype),
        )

    def compute_delta(self, layer: Layer) -> LayerDelta:
        if (
            layer.val.device.type == "privateuseone"
            and self.implementation != "reference"
            and supports_pairwise_kernel(self.pairwise_fn)
        ):
            projected_state, projected_val = self._project_inputs(layer)
            return propagation_dense_kernel(
                pairwise_fn=self.pairwise_fn,
                edge_compress_fn=self.edge_compress_fn,
                layer_val=layer.val,
                projected_state=projected_state,
                projected_val=projected_val,
            )
        if self.implementation == "reference":
            return self._compute_delta_reference(layer)
        if self.implementation == "kernel":
            if supports_pairwise_kernel(self.pairwise_fn):
                projected_state, projected_val = self._project_inputs(layer)
                return propagation_dense_kernel(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_fn=self.edge_compress_fn,
                    layer_val=layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                )
            return self._compute_delta_streaming(layer)
        return self._compute_delta_streaming(layer)

    def forward(self, layer: Layer) -> LayerDelta | Layer:
        delta = self.compute_delta(layer)
        if self.return_delta:
            return delta

        merge_mode = "add" if self.residual else "replace"
        updated = layer.apply_delta(delta, merge_mode=merge_mode)
        return apply_optional_layer_fn(updated, self.norm_fn)


class SparsePropagation(Propagation):
    def __init__(
        self,
        pairwise_fn: Callable[[Tensor, Tensor], Tensor] | nn.Module,
        *,
        sparse_type: SparsePropagationType,
        edge_compress_fn: Callable[[Tensor], Tensor] = F.softsign,
        topk: int | None = None,
        window: int | None = None,
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        norm_fn: Callable[[Layer], Layer] | None = None,
        residual: bool = True,
        return_delta: bool = True,
        implementation: ImplementationMode = "streaming",
        target_block_size: int | None = 128,
        source_block_size: int | None = 128,
        accumulator_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            pairwise_fn=pairwise_fn,
            edge_compress_fn=edge_compress_fn,
            val_proj_fn=val_proj_fn,
            state_proj_fn=state_proj_fn,
            norm_fn=norm_fn,
            residual=residual,
            return_delta=return_delta,
            implementation=implementation,
            target_block_size=target_block_size,
            source_block_size=source_block_size,
            accumulator_dtype=accumulator_dtype,
        )
        if sparse_type not in {"window", "topk"}:
            raise ValueError(f"Unsupported sparse_type: {sparse_type!r}.")
        if sparse_type == "window" and window is None:
            raise ValueError("window sparse propagation requires window.")
        if sparse_type == "topk" and topk is None:
            raise ValueError("topk sparse propagation requires topk.")
        self.sparse_type = sparse_type
        self.topk = topk
        self.window = window

    def _compute_delta_reference(self, layer: Layer) -> LayerDelta:
        projected_state, projected_val = self._project_inputs(layer)
        scores = self.compute_scores(layer)
        edges = self.edge_compress_fn(scores)

        if self.sparse_type == "window":
            mask_2d = _causal_window_mask(
                0,
                layer.num_nodes,
                0,
                layer.num_nodes,
                self.window or 0,
                device=scores.device,
            )
            view_shape = (1,) * (scores.ndim - 2) + mask_2d.shape
            mask = mask_2d.view(view_shape)
        else:
            mask = _topk_mask(scores, self.topk or layer.num_nodes)

        masked_edges = edges * mask.to(dtype=edges.dtype)
        delta_state = torch.einsum("...ij,...j->...i", masked_edges, projected_state)
        delta_val = torch.einsum("...ij,...jd->...id", masked_edges, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_window_delta_streaming(self, layer: Layer) -> LayerDelta:
        projected_state, projected_val = self._project_inputs(layer)
        delta_state, delta_val = self._allocate_delta_buffers(
            layer, projected_state, projected_val
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype
        window = self.window or 0

        for target_start, target_end in iter_blocks(
            layer.num_nodes, self.target_block_size, name="target_block_size"
        ):
            target_val = layer.val[..., target_start:target_end, :]
            source_floor = max(0, target_start - window)
            source_ceiling = target_end

            for source_start, source_end in iter_blocks(
                source_ceiling - source_floor,
                self.source_block_size,
                name="source_block_size",
            ):
                global_source_start = source_floor + source_start
                global_source_end = source_floor + source_end
                source_val = layer.val[..., global_source_start:global_source_end, :]
                scores = self.pairwise_fn(target_val, source_val)
                validate_pairwise_block_scores(
                    scores=scores,
                    batch_shape=layer.batch_shape,
                    target_nodes=target_end - target_start,
                    source_nodes=global_source_end - global_source_start,
                )
                mask_2d = _causal_window_mask(
                    target_start,
                    target_end,
                    global_source_start,
                    global_source_end,
                    window,
                    device=scores.device,
                )
                mask = mask_2d.view((1,) * len(layer.batch_shape) + mask_2d.shape)
                edges = self.edge_compress_fn(scores) * mask.to(dtype=scores.dtype)

                delta_state[..., target_start:target_end] += torch.einsum(
                    "...ij,...j->...i",
                    edges.to(dtype=state_acc_dtype),
                    projected_state[..., global_source_start:global_source_end].to(
                        state_acc_dtype
                    ),
                )
                delta_val[..., target_start:target_end, :] += torch.einsum(
                    "...ij,...jd->...id",
                    edges.to(dtype=val_acc_dtype),
                    projected_val[..., global_source_start:global_source_end, :].to(
                        val_acc_dtype
                    ),
                )

        return LayerDelta(
            delta_state=delta_state.to(projected_state.dtype),
            delta_val=delta_val.to(projected_val.dtype),
        )

    def _compute_topk_delta_streaming(self, layer: Layer) -> LayerDelta:
        projected_state, projected_val = self._project_inputs(layer)
        delta_state, delta_val = self._allocate_delta_buffers(
            layer, projected_state, projected_val
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype
        k = min(self.topk or layer.num_nodes, layer.num_nodes)

        if k == layer.num_nodes:
            return super()._compute_delta_streaming(layer)

        for target_start, target_end in iter_blocks(
            layer.num_nodes, self.target_block_size, name="target_block_size"
        ):
            target_val = layer.val[..., target_start:target_end, :]
            target_nodes = target_end - target_start
            best_scores: Tensor | None = None
            best_indices: Tensor | None = None

            for source_start, source_end in iter_blocks(
                layer.num_nodes, self.source_block_size, name="source_block_size"
            ):
                source_val = layer.val[..., source_start:source_end, :]
                scores = self.pairwise_fn(target_val, source_val)
                validate_pairwise_block_scores(
                    scores=scores,
                    batch_shape=layer.batch_shape,
                    target_nodes=target_nodes,
                    source_nodes=source_end - source_start,
                )

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
                    source_start, source_end, device=scores.device, dtype=torch.long
                )
                source_indices = source_indices.view(
                    (1,) * (scores.ndim - 1) + (source_end - source_start,)
                ).expand_as(scores)

                candidate_scores = torch.cat((best_scores, scores), dim=-1)
                candidate_indices = torch.cat((best_indices, source_indices), dim=-1)
                selected = candidate_scores.topk(k=k, dim=-1)
                best_scores = selected.values
                best_indices = torch.take_along_dim(
                    candidate_indices, selected.indices, dim=-1
                )

            if best_scores is None or best_indices is None:
                continue

            compressed_edges = self.edge_compress_fn(best_scores)
            selected_state = _gather_state_by_indices(projected_state, best_indices)
            selected_val = _gather_val_by_indices(projected_val, best_indices)

            delta_state[..., target_start:target_end] = (
                compressed_edges.to(dtype=state_acc_dtype)
                * selected_state.to(dtype=state_acc_dtype)
            ).sum(dim=-1)
            delta_val[..., target_start:target_end, :] = (
                compressed_edges.to(dtype=val_acc_dtype).unsqueeze(-1)
                * selected_val.to(dtype=val_acc_dtype)
            ).sum(dim=-2)

        return LayerDelta(
            delta_state=delta_state.to(projected_state.dtype),
            delta_val=delta_val.to(projected_val.dtype),
        )

    def _compute_delta_streaming(self, layer: Layer) -> LayerDelta:
        if self.sparse_type == "window":
            return self._compute_window_delta_streaming(layer)
        return self._compute_topk_delta_streaming(layer)

    def compute_delta(self, layer: Layer) -> LayerDelta:
        if layer.val.device.type == "privateuseone" and supports_pairwise_kernel(
            self.pairwise_fn
        ):
            projected_state, projected_val = self._project_inputs(layer)
            if self.sparse_type == "window":
                if self.implementation != "reference":
                    return propagation_window_kernel(
                        pairwise_fn=self.pairwise_fn,
                        edge_compress_fn=self.edge_compress_fn,
                        layer_val=layer.val,
                        projected_state=projected_state,
                        projected_val=projected_val,
                        window=self.window or 0,
                    )
            elif self.sparse_type == "topk":
                if self.implementation != "reference":
                    return propagation_topk_kernel(
                        pairwise_fn=self.pairwise_fn,
                        edge_compress_fn=self.edge_compress_fn,
                        layer_val=layer.val,
                        projected_state=projected_state,
                        projected_val=projected_val,
                        topk=self.topk or layer.num_nodes,
                    )
                return propagation_topk_kernel(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_fn=self.edge_compress_fn,
                    layer_val=layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    topk=self.topk or layer.num_nodes,
                )
        if self.implementation == "reference":
            return self._compute_delta_reference(layer)
        if self.implementation == "kernel":
            if supports_pairwise_kernel(self.pairwise_fn):
                projected_state, projected_val = self._project_inputs(layer)
                if self.sparse_type == "window":
                    return propagation_window_kernel(
                        pairwise_fn=self.pairwise_fn,
                        edge_compress_fn=self.edge_compress_fn,
                        layer_val=layer.val,
                        projected_state=projected_state,
                        projected_val=projected_val,
                        window=self.window or 0,
                    )
                return propagation_topk_kernel(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_fn=self.edge_compress_fn,
                    layer_val=layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    topk=self.topk or layer.num_nodes,
                )
            return self._compute_delta_streaming(layer)
        return self._compute_delta_streaming(layer)
