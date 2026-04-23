from __future__ import annotations

import inspect
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from jakal_net.core import (
    ImplementationMode,
    Layer,
    LayerDelta,
    MergeMode,
    allocate_accumulator,
    apply_optional_layer_fn,
    iter_blocks,
    validate_implementation,
    validate_merge_mode,
    validate_projected_state,
    validate_route_block_logits,
    validate_route_logits,
)
from jakal_net.kernel_common import (
    build_topk_mask,
    masked_softmax,
    select_topk,
    supports_pairwise_route_kernel,
    supports_route_kernel,
)
from jakal_net.kernels import signed_entmax15, transition_dense_kernel, transition_topk_kernel
from jakal_net.native_backend import (
    native_supports,
    native_supports_device,
    transition_dense_native,
    transition_pairwise_dense_native,
    transition_pairwise_topk_native,
    transition_topk_native,
)
from jakal_net.modules import MultiHeadRoute


def signed_abs_softmax(
    logits: Tensor,
    *,
    dim: int = -1,
    mask: Tensor | None = None,
) -> Tensor:
    clean_logits = torch.nan_to_num(logits)
    signs = torch.sign(clean_logits)
    magnitudes = clean_logits.abs()
    if mask is not None:
        bool_mask = mask.to(dtype=torch.bool)
        signs = signs * bool_mask.to(dtype=signs.dtype)
        magnitudes = magnitudes.masked_fill(~bool_mask, -torch.inf)
    routes = signs * torch.softmax(magnitudes, dim=dim)
    return torch.nan_to_num(routes)


def signed_entmax15_routes(
    logits: Tensor,
    *,
    dim: int = -1,
    mask: Tensor | None = None,
) -> Tensor:
    return signed_entmax15(logits, dim=dim, mask=mask)


def _route_uses_pairwise_inputs(route_fn: object) -> bool:
    if getattr(route_fn, "expects_pairwise_inputs", False):
        return True

    fn = route_fn.forward if isinstance(route_fn, nn.Module) else route_fn
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False

    positional_count = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            positional_count += 1
    return positional_count >= 2


def _summarize_routes(
    routes: Tensor,
    *,
    topk_indices: Tensor | None = None,
) -> dict[str, float]:
    probs = routes.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1).mean()
    destination_load = routes.sum(dim=-2)
    load_variance = destination_load.var(unbiased=False)
    active_slots = (destination_load > 1e-6).to(dtype=torch.float32)
    dead_slot_ratio = 1.0 - active_slots.mean()
    overlap = torch.tensor(0.0, device=routes.device)
    if topk_indices is not None and topk_indices.shape[-2] > 1:
        current = topk_indices[..., 1:, :]
        previous = topk_indices[..., :-1, :]
        pairwise_overlap = (current.unsqueeze(-1) == previous.unsqueeze(-2)).any(dim=-1)
        overlap = pairwise_overlap.to(dtype=torch.float32).mean()
    return {
        "entropy": float(entropy.item()),
        "destination_load_variance": float(load_variance.item()),
        "dead_slot_ratio": float(dead_slot_ratio.item()),
        "topk_overlap": float(overlap.item()),
    }


class Transition(nn.Module):
    def __init__(
        self,
        route_fn: Callable[[Tensor], Tensor] | nn.Module,
        *,
        norm_fn: Callable[[Layer], Layer] | None = None,
        state_activation_fn: Callable[[Tensor], Tensor] = F.softplus,
        route_compress_name: str = "softmax",
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        merge_mode: MergeMode = "add",
        implementation: ImplementationMode = "streaming",
        src_block_size: int | None = 128,
        dst_block_size: int | None = 128,
        accumulator_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        validate_merge_mode(merge_mode)
        validate_implementation(implementation)
        self.route_fn = route_fn
        self.norm_fn = norm_fn
        self.state_activation_fn = state_activation_fn
        if route_compress_name not in {"softmax", "signed_abs_softmax", "signed_entmax15"}:
            raise ValueError(f"Unsupported route_compress_name: {route_compress_name!r}.")
        self.route_compress_name = route_compress_name
        self.val_proj_fn = nn.Identity() if val_proj_fn is None else val_proj_fn
        self.state_proj_fn = nn.Identity() if state_proj_fn is None else state_proj_fn
        self.merge_mode = merge_mode
        self.implementation = implementation
        self.src_block_size = src_block_size
        self.dst_block_size = dst_block_size
        self.accumulator_dtype = accumulator_dtype
        self.track_stats = False
        self.last_stats: dict[str, float] | None = None

    def _route_logits(self, src_val: Tensor, dst_val: Tensor) -> Tensor:
        if _route_uses_pairwise_inputs(self.route_fn):
            return self.route_fn(src_val, dst_val)
        return self.route_fn(src_val)

    def compute_route_logits(self, src_layer: Layer, dst_layer: Layer) -> Tensor:
        logits = self._route_logits(src_layer.val, dst_layer.val)
        validate_route_logits(logits, src_layer, dst_layer)
        return logits

    def _supports_multihead_vectorized_fast_path(self) -> bool:
        return isinstance(self.route_fn, MultiHeadRoute)

    def compute_routes(self, src_layer: Layer, dst_layer: Layer) -> Tensor:
        logits = self.compute_route_logits(src_layer, dst_layer)
        return self._compress_routes(logits)

    def _compress_routes(self, logits: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.route_compress_name == "signed_abs_softmax":
            return signed_abs_softmax(logits, dim=-1, mask=mask)
        if self.route_compress_name == "signed_entmax15":
            return signed_entmax15_routes(logits, dim=-1, mask=mask)
        if mask is not None:
            return masked_softmax(logits, mask, dim=-1)
        return torch.softmax(logits, dim=-1)

    def _project_inputs(
        self, src_layer: Layer, dst_layer: Layer
    ) -> tuple[Tensor, Tensor, Tensor]:
        projected_val = self.val_proj_fn(src_layer.val)
        projected_state = self.state_proj_fn(src_layer.state)
        sender_strength = self.state_activation_fn(src_layer.state)

        if tuple(projected_val.shape) != (*src_layer.state.shape, dst_layer.dim):
            raise ValueError(
                "val_proj_fn must return [..., src_nodes, dst_dim], "
                f"expected {(*src_layer.state.shape, dst_layer.dim)}, "
                f"got {tuple(projected_val.shape)}."
            )
        validate_projected_state(projected_state, src_layer)
        if sender_strength.shape != src_layer.state.shape:
            raise ValueError(
                "state_activation_fn must preserve the src_layer.state shape."
            )
        return projected_val, projected_state, sender_strength

    def _allocate_delta_buffers(
        self, dst_layer: Layer, projected_val: Tensor, projected_state: Tensor
    ) -> tuple[Tensor, Tensor]:
        delta_state = allocate_accumulator(
            dst_layer.state.shape,
            device=dst_layer.state.device,
            tensor_dtype=projected_state.dtype,
            accumulator_dtype=self.accumulator_dtype,
        )
        delta_val = allocate_accumulator(
            dst_layer.val.shape,
            device=dst_layer.val.device,
            tensor_dtype=projected_val.dtype,
            accumulator_dtype=self.accumulator_dtype,
        )
        return delta_state, delta_val

    def _compute_delta_reference(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        routes = self.compute_routes(src_layer, dst_layer)
        projected_val, projected_state, sender_strength = self._project_inputs(
            src_layer, dst_layer
        )
        weighted_routes = routes * sender_strength.unsqueeze(-1)

        delta_state = torch.einsum("...jk,...j->...k", weighted_routes, projected_state)
        delta_val = torch.einsum("...jk,...jd->...kd", weighted_routes, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_delta_streaming(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        projected_val, projected_state, sender_strength = self._project_inputs(
            src_layer, dst_layer
        )
        delta_state, delta_val = self._allocate_delta_buffers(
            dst_layer, projected_val, projected_state
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype

        for src_start, src_end in iter_blocks(
            src_layer.num_nodes, self.src_block_size, name="src_block_size"
        ):
            src_val = src_layer.val[..., src_start:src_end, :]
            logits = self._route_logits(src_val, dst_layer.val)
            validate_route_block_logits(
                logits=logits,
                batch_shape=src_layer.batch_shape,
                source_nodes=src_end - src_start,
                dst_nodes=dst_layer.num_nodes,
            )
            routes = self._compress_routes(logits)
            weighted_routes = routes * sender_strength[..., src_start:src_end].unsqueeze(-1)

            delta_state += torch.einsum(
                "...jk,...j->...k",
                weighted_routes.to(dtype=state_acc_dtype),
                projected_state[..., src_start:src_end].to(dtype=state_acc_dtype),
            )
            delta_val += torch.einsum(
                "...jk,...jd->...kd",
                weighted_routes.to(dtype=val_acc_dtype),
                projected_val[..., src_start:src_end, :].to(dtype=val_acc_dtype),
            )

        return LayerDelta(
            delta_state=delta_state.to(projected_state.dtype),
            delta_val=delta_val.to(projected_val.dtype),
        )

    def _compute_delta_kernel_preferred(
        self, src_layer: Layer, dst_layer: Layer
    ) -> LayerDelta:
        if src_layer.val.device.type == "privateuseone":
            return self._compute_delta_reference(src_layer, dst_layer)
        if not _route_uses_pairwise_inputs(self.route_fn) and supports_route_kernel(
            self.route_fn
        ):
            projected_val, projected_state, sender_strength = self._project_inputs(
                src_layer, dst_layer
            )
            return transition_dense_kernel(
                route_fn=self.route_fn,
                sender_strength=sender_strength,
                src_val=src_layer.val,
                projected_state=projected_state,
                projected_val=projected_val,
                dst_nodes=dst_layer.num_nodes,
                route_compress_name=self.route_compress_name,
                src_block_size=self.src_block_size,
                dst_block_size=self.dst_block_size,
                accumulator_dtype=self.accumulator_dtype,
            )
        return self._compute_delta_streaming(src_layer, dst_layer)

    def compute_delta(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        if self.track_stats:
            logits = self.compute_route_logits(src_layer, dst_layer)
            routes = self._compress_routes(logits)
            self.last_stats = _summarize_routes(routes)
        if self._supports_multihead_vectorized_fast_path() and self.implementation != "native":
            return self._compute_delta_reference(src_layer, dst_layer)
        if self.implementation == "native":
            if (
                self.route_compress_name in {"softmax", "signed_entmax15"}
                and
                supports_pairwise_route_kernel(self.route_fn)
                and native_supports("transition_pairwise_dense")
                and native_supports_device(src_layer.val.device.type)
            ):
                projected_val, projected_state, sender_strength = self._project_inputs(
                    src_layer, dst_layer
                )
                return transition_pairwise_dense_native(
                    route_fn=self.route_fn,
                    sender_strength=sender_strength,
                    src_val=src_layer.val,
                    dst_val=dst_layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    src_block_size=self.src_block_size or src_layer.num_nodes,
                    dst_block_size=self.dst_block_size or dst_layer.num_nodes,
                    route_compress_name=self.route_compress_name,
                )
            if (
                self.route_compress_name in {"softmax", "signed_entmax15"}
                and
                not _route_uses_pairwise_inputs(self.route_fn)
                and supports_route_kernel(self.route_fn)
                and native_supports("transition_dense")
                and native_supports_device(src_layer.val.device.type)
            ):
                projected_val, projected_state, sender_strength = self._project_inputs(
                    src_layer, dst_layer
                )
                return transition_dense_native(
                    route_fn=self.route_fn,
                    sender_strength=sender_strength,
                    src_val=src_layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    dst_nodes=dst_layer.num_nodes,
                    src_block_size=self.src_block_size or src_layer.num_nodes,
                    dst_block_size=self.dst_block_size or dst_layer.num_nodes,
                    route_compress_name=self.route_compress_name,
                )
            return self._compute_delta_kernel_preferred(src_layer, dst_layer)
        if self.implementation == "reference":
            return self._compute_delta_reference(src_layer, dst_layer)
        if self.implementation == "kernel":
            return self._compute_delta_kernel_preferred(src_layer, dst_layer)
        return self._compute_delta_streaming(src_layer, dst_layer)

    def forward(self, src_layer: Layer, dst_layer: Layer) -> Layer:
        delta = self.compute_delta(src_layer, dst_layer)
        updated = dst_layer.apply_delta(delta, merge_mode=self.merge_mode)
        return apply_optional_layer_fn(updated, self.norm_fn)


class SparseTransition(Transition):
    def __init__(
        self,
        route_fn: Callable[[Tensor], Tensor] | nn.Module,
        *,
        topk: int,
        norm_fn: Callable[[Layer], Layer] | None = None,
        state_activation_fn: Callable[[Tensor], Tensor] = F.softplus,
        route_compress_name: str = "softmax",
        val_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        state_proj_fn: Callable[[Tensor], Tensor] | nn.Module | None = None,
        merge_mode: MergeMode = "add",
        implementation: ImplementationMode = "streaming",
        src_block_size: int | None = 128,
        dst_block_size: int | None = 128,
        accumulator_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            route_fn=route_fn,
            norm_fn=norm_fn,
            state_activation_fn=state_activation_fn,
            route_compress_name=route_compress_name,
            val_proj_fn=val_proj_fn,
            state_proj_fn=state_proj_fn,
            merge_mode=merge_mode,
            implementation=implementation,
            src_block_size=src_block_size,
            dst_block_size=dst_block_size,
            accumulator_dtype=accumulator_dtype,
        )
        if topk <= 0:
            raise ValueError("topk must be positive.")
        self.topk = topk

    def _compute_delta_reference(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        logits = self.compute_route_logits(src_layer, dst_layer)
        k = min(self.topk, dst_layer.num_nodes)
        if k == dst_layer.num_nodes:
            routes = self.compute_routes(src_layer, dst_layer)
        else:
            mask = build_topk_mask(logits, k, dim=-1)
            routes = self._compress_routes(logits, mask=mask)

        projected_val, projected_state, sender_strength = self._project_inputs(
            src_layer, dst_layer
        )
        weighted_routes = routes * sender_strength.unsqueeze(-1)
        delta_state = torch.einsum("...jk,...j->...k", weighted_routes, projected_state)
        delta_val = torch.einsum("...jk,...jd->...kd", weighted_routes, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_delta_directml_fallback(
        self, src_layer: Layer, dst_layer: Layer
    ) -> LayerDelta:
        logits = self.compute_route_logits(src_layer, dst_layer)
        k = min(self.topk, dst_layer.num_nodes)
        if k == dst_layer.num_nodes:
            routes = self._compress_routes(logits)
        else:
            topk_indices = select_topk(logits, k, dim=-1).indices
            dst_index = torch.arange(
                dst_layer.num_nodes, device=logits.device, dtype=topk_indices.dtype
            )
            dst_index = dst_index.view((1,) * topk_indices.ndim + (dst_layer.num_nodes,))
            mask = (topk_indices.unsqueeze(-1) == dst_index).any(dim=-2)
            routes = self._compress_routes(logits, mask=mask)

        projected_val, projected_state, sender_strength = self._project_inputs(
            src_layer, dst_layer
        )
        weighted_routes = routes * sender_strength.unsqueeze(-1)
        delta_state = torch.einsum("...jk,...j->...k", weighted_routes, projected_state)
        delta_val = torch.einsum("...jk,...jd->...kd", weighted_routes, projected_val)
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)

    def _compute_delta_streaming(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        projected_val, projected_state, sender_strength = self._project_inputs(
            src_layer, dst_layer
        )
        delta_state, delta_val = self._allocate_delta_buffers(
            dst_layer, projected_val, projected_state
        )
        state_acc_dtype = delta_state.dtype
        val_acc_dtype = delta_val.dtype
        k = min(self.topk, dst_layer.num_nodes)

        if k == dst_layer.num_nodes:
            return super()._compute_delta_streaming(src_layer, dst_layer)

        pairwise_route = _route_uses_pairwise_inputs(self.route_fn)
        for src_start, src_end in iter_blocks(
            src_layer.num_nodes, self.src_block_size, name="src_block_size"
        ):
            src_val = src_layer.val[..., src_start:src_end, :]
            if pairwise_route:
                topk_values: Tensor | None = None
                topk_indices: Tensor | None = None
                for dst_start, dst_end in iter_blocks(
                    dst_layer.num_nodes, self.dst_block_size, name="dst_block_size"
                ):
                    dst_val = dst_layer.val[..., dst_start:dst_end, :]
                    logits = self._route_logits(src_val, dst_val)
                    validate_route_block_logits(
                        logits=logits,
                        batch_shape=src_layer.batch_shape,
                        source_nodes=src_end - src_start,
                        dst_nodes=dst_end - dst_start,
                    )
                    dst_indices = torch.arange(
                        dst_start, dst_end, device=logits.device, dtype=torch.long
                    )
                    dst_indices = dst_indices.view(
                        (1,) * (logits.ndim - 1) + (dst_end - dst_start,)
                    ).expand_as(logits)
                    if topk_values is None or topk_indices is None:
                        topk_values = torch.full(
                            (*logits.shape[:-1], k),
                            fill_value=torch.finfo(logits.dtype).min,
                            device=logits.device,
                            dtype=logits.dtype,
                        )
                        topk_indices = torch.zeros(
                            (*logits.shape[:-1], k),
                            device=logits.device,
                            dtype=torch.long,
                        )
                    candidate_values = torch.cat((topk_values, logits), dim=-1)
                    candidate_indices = torch.cat((topk_indices, dst_indices), dim=-1)
                    selected = select_topk(candidate_values, k, dim=-1)
                    topk_values = selected.values
                    topk_indices = torch.take_along_dim(
                        candidate_indices, selected.indices, dim=-1
                    )
                if topk_values is None or topk_indices is None:
                    continue
            else:
                logits = self._route_logits(src_val, dst_layer.val)
                validate_route_block_logits(
                    logits=logits,
                    batch_shape=src_layer.batch_shape,
                    source_nodes=src_end - src_start,
                    dst_nodes=dst_layer.num_nodes,
                )
                topk_selected = select_topk(logits, k, dim=-1)
                topk_values = topk_selected.values
                topk_indices = topk_selected.indices

            routes = self._compress_routes(topk_values)
            weighted_routes = routes * sender_strength[..., src_start:src_end].unsqueeze(-1)

            state_contrib = (
                weighted_routes.to(dtype=state_acc_dtype)
                * projected_state[..., src_start:src_end]
                .to(dtype=state_acc_dtype)
                .unsqueeze(-1)
            )
            flat_indices = topk_indices.reshape(*topk_indices.shape[:-2], -1)
            flat_state_contrib = state_contrib.reshape(*state_contrib.shape[:-2], -1)
            delta_state.scatter_add_(
                dim=-1, index=flat_indices, src=flat_state_contrib
            )

            val_contrib = (
                weighted_routes.to(dtype=val_acc_dtype).unsqueeze(-1)
                * projected_val[..., src_start:src_end, :]
                .to(dtype=val_acc_dtype)
                .unsqueeze(-2)
            )
            flat_val_contrib = val_contrib.reshape(
                *val_contrib.shape[:-3], -1, dst_layer.dim
            )
            scatter_index = flat_indices.unsqueeze(-1).expand(
                *flat_indices.shape, dst_layer.dim
            )
            delta_val.scatter_add_(dim=-2, index=scatter_index, src=flat_val_contrib)

        return LayerDelta(
            delta_state=delta_state.to(projected_state.dtype),
            delta_val=delta_val.to(projected_val.dtype),
        )

    def _compute_delta_kernel_preferred(
        self, src_layer: Layer, dst_layer: Layer
    ) -> LayerDelta:
        if src_layer.val.device.type == "privateuseone":
            return self._compute_delta_directml_fallback(src_layer, dst_layer)
        if (
            supports_pairwise_route_kernel(self.route_fn)
            and native_supports("transition_pairwise_topk")
            and native_supports_device(src_layer.val.device.type)
        ):
            projected_val, projected_state, sender_strength = self._project_inputs(
                src_layer, dst_layer
            )
            try:
                return transition_pairwise_topk_native(
                    route_fn=self.route_fn,
                    sender_strength=sender_strength,
                    src_val=src_layer.val,
                    dst_val=dst_layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    topk=self.topk,
                    src_block_size=self.src_block_size or src_layer.num_nodes,
                    dst_block_size=self.dst_block_size or dst_layer.num_nodes,
                    route_compress_name=self.route_compress_name,
                )
            except TypeError:
                if self.route_compress_name != "softmax":
                    return self._compute_delta_streaming(src_layer, dst_layer)
                raise
        if not _route_uses_pairwise_inputs(self.route_fn) and supports_route_kernel(
            self.route_fn
        ):
            projected_val, projected_state, sender_strength = self._project_inputs(
                src_layer, dst_layer
            )
            return transition_topk_kernel(
                route_fn=self.route_fn,
                sender_strength=sender_strength,
                src_val=src_layer.val,
                projected_state=projected_state,
                projected_val=projected_val,
                dst_nodes=dst_layer.num_nodes,
                route_compress_name=self.route_compress_name,
                topk=self.topk,
                src_block_size=self.src_block_size,
                dst_block_size=self.dst_block_size,
                accumulator_dtype=self.accumulator_dtype,
            )
        return self._compute_delta_streaming(src_layer, dst_layer)

    def compute_delta(self, src_layer: Layer, dst_layer: Layer) -> LayerDelta:
        if self.track_stats:
            logits = self.compute_route_logits(src_layer, dst_layer)
            k = min(self.topk, dst_layer.num_nodes)
            if k == dst_layer.num_nodes:
                routes = self._compress_routes(logits)
                topk_indices = None
            else:
                topk_selected = select_topk(logits, k, dim=-1)
                topk_indices = topk_selected.indices
                routes = self._compress_routes(topk_selected.values)
                dense_routes = torch.zeros_like(logits)
                dense_routes.scatter_(-1, topk_indices, routes)
                routes = dense_routes
            self.last_stats = _summarize_routes(routes, topk_indices=topk_indices)
        if self._supports_multihead_vectorized_fast_path():
            return self._compute_delta_reference(src_layer, dst_layer)
        if self.implementation == "native":
            if (
                supports_pairwise_route_kernel(self.route_fn)
                and native_supports("transition_pairwise_topk")
                and native_supports_device(src_layer.val.device.type)
            ):
                projected_val, projected_state, sender_strength = self._project_inputs(
                    src_layer, dst_layer
                )
                try:
                    return transition_pairwise_topk_native(
                        route_fn=self.route_fn,
                        sender_strength=sender_strength,
                        src_val=src_layer.val,
                        dst_val=dst_layer.val,
                        projected_state=projected_state,
                        projected_val=projected_val,
                        topk=self.topk,
                        src_block_size=self.src_block_size or src_layer.num_nodes,
                        dst_block_size=self.dst_block_size or dst_layer.num_nodes,
                        route_compress_name=self.route_compress_name,
                    )
                except TypeError:
                    if self.route_compress_name != "softmax":
                        return self._compute_delta_streaming(src_layer, dst_layer)
                    raise
            if (
                self.route_compress_name == "softmax"
                and
                not _route_uses_pairwise_inputs(self.route_fn)
                and supports_route_kernel(self.route_fn)
                and native_supports("transition_topk")
                and native_supports_device(src_layer.val.device.type)
            ):
                projected_val, projected_state, sender_strength = self._project_inputs(
                    src_layer, dst_layer
                )
                return transition_topk_native(
                    route_fn=self.route_fn,
                    sender_strength=sender_strength,
                    src_val=src_layer.val,
                    projected_state=projected_state,
                    projected_val=projected_val,
                    dst_nodes=dst_layer.num_nodes,
                    topk=self.topk,
                    src_block_size=self.src_block_size or src_layer.num_nodes,
                    dst_block_size=self.dst_block_size or dst_layer.num_nodes,
                )
            return self._compute_delta_kernel_preferred(src_layer, dst_layer)
        if src_layer.val.device.type == "privateuseone":
            return self._compute_delta_directml_fallback(src_layer, dst_layer)
        if self.implementation == "reference":
            return self._compute_delta_reference(src_layer, dst_layer)
        if self.implementation == "kernel":
            return self._compute_delta_kernel_preferred(src_layer, dst_layer)
        return self._compute_delta_streaming(src_layer, dst_layer)
