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
    allocate_accumulator,
    apply_optional_layer_fn,
    iter_blocks,
    validate_implementation,
    validate_pairwise_block_scores,
    validate_pairwise_scores,
    validate_projected_state,
    validate_projected_val,
)
from jakal_net.kernel_common import (
    build_topk_mask,
    causal_window_mask,
    gather_state_by_indices,
    gather_val_by_indices,
    select_topk,
)
from jakal_net.kernels import (
    propagation_dense_kernel,
    propagation_topk_kernel,
    propagation_window_kernel,
    signed_entmax15,
    supports_pairwise_kernel,
)
from jakal_net.native_backend import (
    native_supports,
    native_supports_device,
    propagation_dense_native,
    propagation_topk_native,
    propagation_window_native,
)
from jakal_net.modules import LowRankBilinearPairwise, MultiHeadPairwise


def _cuda_graph_capture_active(device_type: str) -> bool:
    if device_type != "cuda":
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _edge_compress_name(edge_compress_fn: Callable[[Tensor], Tensor]) -> str | None:
    name = getattr(edge_compress_fn, "__name__", "")
    if edge_compress_fn is F.softsign or name == "softsign":
        return "softsign"
    if name in {"signed_abs_softmax", "signed_abs_softmax_edges", "_signed_abs_softmax_edges"}:
        return "signed_abs_softmax"
    if name in {"signed_entmax15", "_signed_entmax15_edges"}:
        return "signed_entmax15"
    return None


def _compress_edges(
    scores: Tensor,
    edge_compress_fn: Callable[[Tensor], Tensor],
    *,
    mask: Tensor | None = None,
) -> Tensor:
    edge_name = _edge_compress_name(edge_compress_fn)
    if edge_name == "signed_abs_softmax":
        clean_scores = torch.nan_to_num(scores)
        signs = torch.sign(clean_scores)
        magnitudes = clean_scores.abs()
        if mask is not None:
            bool_mask = mask.to(dtype=torch.bool)
            signs = signs * bool_mask.to(dtype=signs.dtype)
            magnitudes = magnitudes.masked_fill(~bool_mask, -torch.inf)
        return torch.nan_to_num(signs * torch.softmax(magnitudes, dim=-1))
    if edge_name == "signed_entmax15":
        return signed_entmax15(scores, dim=-1, mask=mask)
    edges = edge_compress_fn(scores)
    if mask is not None:
        edges = edges * mask.to(dtype=edges.dtype)
    return edges


def _native_edge_compress_name(edge_compress_fn: Callable[[Tensor], Tensor]) -> str | None:
    name = _edge_compress_name(edge_compress_fn)
    if name in {"softsign", "signed_abs_softmax", "signed_entmax15"}:
        return name
    return None


def _disable_native_multihead_signed_smoothmax_path(pairwise_fn: object) -> bool:
    if not isinstance(pairwise_fn, MultiHeadPairwise):
        return False
    if pairwise_fn.aggregate != "signed_smoothmax":
        return False
    if len(pairwise_fn.heads) == 0:
        return False
    return all(isinstance(head, LowRankBilinearPairwise) for head in pairwise_fn.heads)


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
        state_weight_edges: bool = False,
        implementation: ImplementationMode = "streaming",
        target_block_size: int | None = 128,
        source_block_size: int | None = 128,
        accumulator_dtype: torch.dtype | None = None,
        use_direction_only: bool = False,
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
        self.state_weight_edges = state_weight_edges
        self.implementation = implementation
        self.target_block_size = target_block_size
        self.source_block_size = source_block_size
        self.accumulator_dtype = accumulator_dtype
        self.use_direction_only = use_direction_only

    def _directional_val(self, state: Tensor, val: Tensor) -> Tensor:
        return val

    def compute_scores(self, layer: Layer) -> Tensor:
        directional_val = self._directional_val(layer.state, layer.val)
        scores = self.pairwise_fn(directional_val, directional_val)
        validate_pairwise_scores(scores, layer)
        return scores

    def _supports_multihead_vectorized_fast_path(self) -> bool:
        return isinstance(self.pairwise_fn, MultiHeadPairwise)

    def compute_edges(self, layer: Layer) -> Tensor:
        return self.edge_compress_fn(self.compute_scores(layer))

    def _edge_source_strength(self, source_state: Tensor) -> Tensor:
        return F.softplus(source_state)

    def _weight_edges(self, edges: Tensor, source_state: Tensor) -> Tensor:
        if not self.state_weight_edges:
            return edges
        return edges * self._edge_source_strength(source_state).unsqueeze(-2)

    def _fold_state_weight_into_projected_inputs(
        self,
        projected_state: Tensor,
        projected_val: Tensor,
        source_state: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not self.state_weight_edges:
            return projected_state, projected_val
        source_strength = self._edge_source_strength(source_state)
        return source_strength, projected_val * source_strength.unsqueeze(-1)

    def _project_inputs(
        self,
        layer: Layer,
        *,
        directional_val: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        projected_state = self.state_proj_fn(layer.state)
        projected_val = self.val_proj_fn(
            self._directional_val(layer.state, layer.val) if directional_val is None else directional_val
        )
        validate_projected_state(projected_state, layer)
        validate_projected_val(projected_val, layer)
        return projected_state, projected_val

    def _allocate_delta_buffers(
        self, layer: Layer, projected_state: Tensor, projected_val: Tensor
    ) -> tuple[Tensor, Tensor]:
        delta_state = allocate_accumulator(
            layer.state.shape,
            device=layer.state.device,
            tensor_dtype=projected_state.dtype,
            accumulator_dtype=self.accumulator_dtype,
        )
        delta_val = allocate_accumulator(
            layer.val.shape,
            device=layer.val.device,
            tensor_dtype=projected_val.dtype,
            accumulator_dtype=self.accumulator_dtype,
        )
        return delta_state, delta_val

    def _compute_delta_reference(self, layer: Layer) -> LayerDelta:
        directional_val = self._directional_val(layer.state, layer.val)
        scores = self.pairwise_fn(directional_val, directional_val)
        validate_pairwise_scores(scores, layer)
        edges = self.edge_compress_fn(scores)
        edges = self._weight_edges(edges, layer.state)
        projected_state, projected_val = self._project_inputs(layer, directional_val=directional_val)

        if self.state_weight_edges:
            delta_state = edges.sum(dim=-1)
        else:
            delta_state = torch.einsum("...ij,...j->...i", edges, projected_state)
        delta_val = torch.einsum("...ij,...jd->...id", edges, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_delta_streaming(self, layer: Layer) -> LayerDelta:
        directional_val = self._directional_val(layer.state, layer.val)
        projected_state, projected_val = self._project_inputs(layer, directional_val=directional_val)
        delta_state, delta_val = self._allocate_delta_buffers(
            layer, projected_state, projected_val
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype

        for target_start, target_end in iter_blocks(
            layer.num_nodes, self.target_block_size, name="target_block_size"
        ):
            target_val = directional_val[..., target_start:target_end, :]
            for source_start, source_end in iter_blocks(
                layer.num_nodes, self.source_block_size, name="source_block_size"
            ):
                source_val = directional_val[..., source_start:source_end, :]
                scores = self.pairwise_fn(target_val, source_val)
                validate_pairwise_block_scores(
                    scores=scores,
                    batch_shape=layer.batch_shape,
                    target_nodes=target_end - target_start,
                    source_nodes=source_end - source_start,
                )
                edges = self._weight_edges(
                    self.edge_compress_fn(scores),
                    layer.state[..., source_start:source_end],
                )
                state_edges = edges.to(dtype=state_acc_dtype)
                val_edges = edges.to(dtype=val_acc_dtype)

                if self.state_weight_edges:
                    delta_state[..., target_start:target_end] += state_edges.sum(dim=-1)
                else:
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

    def _compute_delta_kernel_preferred(self, layer: Layer) -> LayerDelta:
        signed_smoothmax_lowrank_native = (
            _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn)
            and self.implementation == "native"
            and self.state_weight_edges
        )
        if _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn) and not signed_smoothmax_lowrank_native:
            return self._compute_delta_reference(layer)
        edge_compress_name = _native_edge_compress_name(self.edge_compress_fn)
        directional_layer_val = self._directional_val(layer.state, layer.val)
        if (
            edge_compress_name is not None
            and supports_pairwise_kernel(self.pairwise_fn)
            and native_supports("propagation_dense")
            and native_supports_device(layer.val.device.type)
        ):
            projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
            if signed_smoothmax_lowrank_native:
                projected_state = self._edge_source_strength(layer.state)
            else:
                projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                    projected_state,
                    projected_val,
                    layer.state,
                )
            return propagation_dense_native(
                pairwise_fn=self.pairwise_fn,
                edge_compress_name=edge_compress_name,
                layer_val=directional_layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_state=None,
                target_block_size=self.target_block_size or layer.num_nodes,
                source_block_size=self.source_block_size or layer.num_nodes,
            )
        if (
            layer.val.device.type == "privateuseone"
            and supports_pairwise_kernel(self.pairwise_fn)
        ):
            projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
            projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                projected_state,
                projected_val,
                layer.state,
            )
            return propagation_dense_kernel(
                pairwise_fn=self.pairwise_fn,
                edge_compress_fn=self.edge_compress_fn,
                layer_val=directional_layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_state=None,
                target_block_size=self.target_block_size,
                source_block_size=self.source_block_size,
                accumulator_dtype=self.accumulator_dtype,
            )
        if supports_pairwise_kernel(self.pairwise_fn):
            projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
            projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                projected_state,
                projected_val,
                layer.state,
            )
            return propagation_dense_kernel(
                pairwise_fn=self.pairwise_fn,
                edge_compress_fn=self.edge_compress_fn,
                layer_val=layer.val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_state=None,
                target_block_size=self.target_block_size,
                source_block_size=self.source_block_size,
                accumulator_dtype=self.accumulator_dtype,
            )
        return Propagation._compute_delta_streaming(self, layer)

    def compute_delta(self, layer: Layer) -> LayerDelta:
        signed_smoothmax_lowrank_native = (
            self.implementation == "native"
            and _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn)
            and self.state_weight_edges
        )
        if self.implementation == "native" and _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn) and not signed_smoothmax_lowrank_native:
            return self._compute_delta_reference(layer)
        if self.implementation == "native" and _cuda_graph_capture_active(layer.val.device.type):
            return self._compute_delta_reference(layer)
        if self._supports_multihead_vectorized_fast_path() and self.implementation != "native":
            return self._compute_delta_reference(layer)
        if self.implementation == "native":
            edge_compress_name = _native_edge_compress_name(self.edge_compress_fn)
            directional_layer_val = self._directional_val(layer.state, layer.val)
            if (
                edge_compress_name is not None
                and supports_pairwise_kernel(self.pairwise_fn)
                and native_supports("propagation_dense")
                and native_supports_device(layer.val.device.type)
            ):
                projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
                if signed_smoothmax_lowrank_native:
                    projected_state = self._edge_source_strength(layer.state)
                else:
                    projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                        projected_state,
                        projected_val,
                        layer.state,
                    )
                return propagation_dense_native(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_name=edge_compress_name,
                    layer_val=directional_layer_val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    source_state=None,
                    target_block_size=self.target_block_size or layer.num_nodes,
                    source_block_size=self.source_block_size or layer.num_nodes,
                )
            return self._compute_delta_kernel_preferred(layer)
        if self.implementation == "reference":
            return self._compute_delta_reference(layer)
        if self.implementation == "kernel":
            return self._compute_delta_kernel_preferred(layer)
        if layer.val.device.type == "privateuseone":
            return self._compute_delta_reference(layer)
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
        state_weight_edges: bool = False,
        implementation: ImplementationMode = "streaming",
        target_block_size: int | None = 128,
        source_block_size: int | None = 128,
        accumulator_dtype: torch.dtype | None = None,
        use_direction_only: bool = False,
    ) -> None:
        super().__init__(
            pairwise_fn=pairwise_fn,
            edge_compress_fn=edge_compress_fn,
            val_proj_fn=val_proj_fn,
            state_proj_fn=state_proj_fn,
            norm_fn=norm_fn,
            residual=residual,
            return_delta=return_delta,
            state_weight_edges=state_weight_edges,
            implementation=implementation,
            target_block_size=target_block_size,
            source_block_size=source_block_size,
            accumulator_dtype=accumulator_dtype,
            use_direction_only=use_direction_only,
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
        directional_val = self._directional_val(layer.state, layer.val)
        projected_state, projected_val = self._project_inputs(layer, directional_val=directional_val)
        scores = self.pairwise_fn(directional_val, directional_val)
        validate_pairwise_scores(scores, layer)

        if self.sparse_type == "window":
            mask_2d = causal_window_mask(
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
            mask = build_topk_mask(scores, self.topk or layer.num_nodes)

        masked_edges = _compress_edges(scores, self.edge_compress_fn, mask=mask)
        masked_edges = self._weight_edges(masked_edges, layer.state)
        if self.state_weight_edges:
            delta_state = masked_edges.sum(dim=-1)
        else:
            delta_state = torch.einsum("...ij,...j->...i", masked_edges, projected_state)
        delta_val = torch.einsum("...ij,...jd->...id", masked_edges, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_window_delta_streaming(self, layer: Layer) -> LayerDelta:
        if _edge_compress_name(self.edge_compress_fn) == "signed_abs_softmax":
            return self._compute_delta_reference(layer)
        directional_val = self._directional_val(layer.state, layer.val)
        projected_state, projected_val = self._project_inputs(layer, directional_val=directional_val)
        delta_state, delta_val = self._allocate_delta_buffers(
            layer, projected_state, projected_val
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype
        window = self.window or 0

        for target_start, target_end in iter_blocks(
            layer.num_nodes, self.target_block_size, name="target_block_size"
        ):
            target_val = directional_val[..., target_start:target_end, :]
            source_floor = max(0, target_start - window)
            source_ceiling = target_end

            for source_start, source_end in iter_blocks(
                source_ceiling - source_floor,
                self.source_block_size,
                name="source_block_size",
            ):
                global_source_start = source_floor + source_start
                global_source_end = source_floor + source_end
                source_val = directional_val[..., global_source_start:global_source_end, :]
                scores = self.pairwise_fn(target_val, source_val)
                validate_pairwise_block_scores(
                    scores=scores,
                    batch_shape=layer.batch_shape,
                    target_nodes=target_end - target_start,
                    source_nodes=global_source_end - global_source_start,
                )
                mask_2d = causal_window_mask(
                    target_start,
                    target_end,
                    global_source_start,
                    global_source_end,
                    window,
                    device=scores.device,
                )
                mask = mask_2d.view((1,) * len(layer.batch_shape) + mask_2d.shape)
                edges = _compress_edges(scores, self.edge_compress_fn, mask=mask)
                edges = self._weight_edges(
                    edges,
                    layer.state[..., global_source_start:global_source_end],
                )

                state_edges = edges.to(dtype=state_acc_dtype)
                if self.state_weight_edges:
                    delta_state[..., target_start:target_end] += state_edges.sum(dim=-1)
                else:
                    delta_state[..., target_start:target_end] += torch.einsum(
                        "...ij,...j->...i",
                        state_edges,
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
        directional_val = self._directional_val(layer.state, layer.val)
        projected_state, projected_val = self._project_inputs(layer, directional_val=directional_val)
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
            target_val = directional_val[..., target_start:target_end, :]
            target_nodes = target_end - target_start
            best_scores: Tensor | None = None
            best_indices: Tensor | None = None

            for source_start, source_end in iter_blocks(
                layer.num_nodes, self.source_block_size, name="source_block_size"
            ):
                source_val = directional_val[..., source_start:source_end, :]
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
                selected = select_topk(candidate_scores, k, dim=-1)
                best_scores = selected.values
                best_indices = torch.take_along_dim(
                    candidate_indices, selected.indices, dim=-1
                )

            if best_scores is None or best_indices is None:
                continue

            compressed_edges = _compress_edges(best_scores, self.edge_compress_fn)
            if self.state_weight_edges:
                selected_source_state = gather_state_by_indices(layer.state, best_indices)
                compressed_edges = compressed_edges * self._edge_source_strength(selected_source_state)
            selected_state = gather_state_by_indices(projected_state, best_indices)
            selected_val = gather_val_by_indices(projected_val, best_indices)

            state_edges = compressed_edges.to(dtype=state_acc_dtype)
            if self.state_weight_edges:
                delta_state[..., target_start:target_end] = state_edges.sum(dim=-1)
            else:
                delta_state[..., target_start:target_end] = (
                    state_edges * selected_state.to(dtype=state_acc_dtype)
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

    def _compute_delta_kernel_preferred(self, layer: Layer) -> LayerDelta:
        signed_smoothmax_lowrank_dense_native = (
            _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn)
            and self.implementation == "native"
            and self.sparse_type == "window"
            and int(self.window or 0) + 1 >= int(layer.num_nodes)
            and self.state_weight_edges
        )
        if _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn) and not signed_smoothmax_lowrank_dense_native:
            return self._compute_delta_reference(layer)
        edge_compress_name = _native_edge_compress_name(self.edge_compress_fn)
        directional_layer_val = self._directional_val(layer.state, layer.val)
        if (
            edge_compress_name is not None
            and supports_pairwise_kernel(self.pairwise_fn)
            and (not _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn) or signed_smoothmax_lowrank_dense_native)
            and native_supports_device(layer.val.device.type)
        ):
            projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
            if signed_smoothmax_lowrank_dense_native:
                projected_state = self._edge_source_strength(layer.state)
            else:
                projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                    projected_state,
                    projected_val,
                    layer.state,
                )
            if self.sparse_type == "window" and native_supports("propagation_window"):
                return propagation_window_native(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_name=edge_compress_name,
                    layer_val=directional_layer_val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    window=self.window or 0,
                    target_block_size=self.target_block_size or layer.num_nodes,
                    source_block_size=self.source_block_size or layer.num_nodes,
                )
            if self.sparse_type == "topk" and native_supports("propagation_topk"):
                return propagation_topk_native(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_name=edge_compress_name,
                    layer_val=directional_layer_val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    topk=self.topk or layer.num_nodes,
                    target_block_size=self.target_block_size or layer.num_nodes,
                    source_block_size=self.source_block_size or layer.num_nodes,
                )
        if layer.val.device.type == "privateuseone" and supports_pairwise_kernel(
            self.pairwise_fn
        ):
            projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
            projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                projected_state,
                projected_val,
                layer.state,
            )
            if self.sparse_type == "window":
                return propagation_window_kernel(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_fn=self.edge_compress_fn,
                    layer_val=directional_layer_val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    source_state=None,
                    window=self.window or 0,
                    target_block_size=self.target_block_size,
                    source_block_size=self.source_block_size,
                    accumulator_dtype=self.accumulator_dtype,
                )
            return propagation_topk_kernel(
                pairwise_fn=self.pairwise_fn,
                edge_compress_fn=self.edge_compress_fn,
                layer_val=directional_layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_state=None,
                topk=self.topk or layer.num_nodes,
                target_block_size=self.target_block_size,
                source_block_size=self.source_block_size,
                accumulator_dtype=self.accumulator_dtype,
            )
        if supports_pairwise_kernel(self.pairwise_fn):
            projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
            projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                projected_state,
                projected_val,
                layer.state,
            )
            if self.sparse_type == "window":
                return propagation_window_kernel(
                    pairwise_fn=self.pairwise_fn,
                    edge_compress_fn=self.edge_compress_fn,
                    layer_val=directional_layer_val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    source_state=None,
                    window=self.window or 0,
                    target_block_size=self.target_block_size,
                    source_block_size=self.source_block_size,
                    accumulator_dtype=self.accumulator_dtype,
                )
            return propagation_topk_kernel(
                pairwise_fn=self.pairwise_fn,
                edge_compress_fn=self.edge_compress_fn,
                layer_val=directional_layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_state=None,
                topk=self.topk or layer.num_nodes,
                target_block_size=self.target_block_size,
                source_block_size=self.source_block_size,
                accumulator_dtype=self.accumulator_dtype,
            )
        return self._compute_delta_streaming(layer)

    def compute_delta(self, layer: Layer) -> LayerDelta:
        signed_smoothmax_lowrank_dense_native = (
            self.implementation == "native"
            and _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn)
            and self.sparse_type == "window"
            and int(self.window or 0) + 1 >= int(layer.num_nodes)
            and self.state_weight_edges
        )
        if self.implementation == "native" and _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn) and not signed_smoothmax_lowrank_dense_native:
            return self._compute_delta_reference(layer)
        if self.implementation == "native" and _cuda_graph_capture_active(layer.val.device.type):
            return self._compute_delta_reference(layer)
        if self._supports_multihead_vectorized_fast_path() and self.implementation != "native":
            return self._compute_delta_reference(layer)
        if self.implementation == "native":
            edge_compress_name = _native_edge_compress_name(self.edge_compress_fn)
            directional_layer_val = self._directional_val(layer.state, layer.val)
            if (
                edge_compress_name is not None
                and (supports_pairwise_kernel(self.pairwise_fn) or isinstance(self.pairwise_fn, MultiHeadPairwise))
                and (not _disable_native_multihead_signed_smoothmax_path(self.pairwise_fn) or signed_smoothmax_lowrank_dense_native)
                and native_supports_device(layer.val.device.type)
            ):
                projected_state, projected_val = self._project_inputs(layer, directional_val=directional_layer_val)
                if signed_smoothmax_lowrank_dense_native:
                    projected_state = self._edge_source_strength(layer.state)
                else:
                    projected_state, projected_val = self._fold_state_weight_into_projected_inputs(
                        projected_state,
                        projected_val,
                        layer.state,
                    )
                if self.sparse_type == "window" and native_supports("propagation_window"):
                    return propagation_window_native(
                        pairwise_fn=self.pairwise_fn,
                        edge_compress_name=edge_compress_name,
                        layer_val=directional_layer_val,
                        projected_state=projected_state,
                        projected_val=projected_val,
                        window=self.window or 0,
                        target_block_size=self.target_block_size or layer.num_nodes,
                        source_block_size=self.source_block_size or layer.num_nodes,
                    )
                if self.sparse_type == "topk" and native_supports("propagation_topk"):
                    return propagation_topk_native(
                        pairwise_fn=self.pairwise_fn,
                        edge_compress_name=edge_compress_name,
                        layer_val=directional_layer_val,
                        projected_state=projected_state,
                        projected_val=projected_val,
                        topk=self.topk or layer.num_nodes,
                        target_block_size=self.target_block_size or layer.num_nodes,
                        source_block_size=self.source_block_size or layer.num_nodes,
                    )
            return self._compute_delta_kernel_preferred(layer)
        if layer.val.device.type == "privateuseone":
            if self.sparse_type == "topk":
                return self._compute_delta_kernel_preferred(layer)
            if self.implementation == "streaming":
                return self._compute_delta_reference(layer)
        if self.implementation == "reference":
            return self._compute_delta_reference(layer)
        if self.implementation == "kernel":
            return self._compute_delta_kernel_preferred(layer)
        return self._compute_delta_streaming(layer)
