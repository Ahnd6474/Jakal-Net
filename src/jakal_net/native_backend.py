from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.autograd import Function

from jakal_net.core import LayerDelta
from jakal_net.kernel_common import (
    pairwise_kernel_spec,
    pairwise_route_kernel_spec,
    route_kernel_spec,
    supports_pairwise_kernel,
    supports_pairwise_route_kernel,
    supports_route_kernel,
)
from jakal_net.modules import (
    DiagonalBilinearPairwise,
    DiagonalBilinearRoute,
    LowRankBilinearPairwise,
    LowRankBilinearRoute,
)

DEFAULT_NATIVE_MODULE = "jakal_net_native"
NATIVE_MODULE_ENV = "JAKAL_NET_NATIVE_MODULE"
DISABLE_NATIVE_ENV = "JAKAL_NET_DISABLE_NATIVE"


@dataclass(frozen=True, slots=True)
class NativeStatus:
    available: bool
    module_name: str
    backend_name: str | None
    supported_ops: tuple[str, ...]
    supported_devices: tuple[str, ...]
    error: str | None


_NATIVE_MODULE: Any | None = None
_NATIVE_STATUS: NativeStatus | None = None


def _module_name() -> str:
    return os.environ.get(NATIVE_MODULE_ENV, DEFAULT_NATIVE_MODULE)


def _candidate_module_paths() -> tuple[Path, ...]:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root / "build_native",
        repo_root / "native" / "build",
        repo_root,
    )


def _import_native_module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as primary_exc:  # noqa: BLE001
        attempted: list[str] = [f"default import: {type(primary_exc).__name__}: {primary_exc}"]

        for candidate in _candidate_module_paths():
            if not candidate.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                return importlib.import_module(module_name)
            except Exception as candidate_exc:  # noqa: BLE001
                attempted.append(
                    f"{candidate_str}: {type(candidate_exc).__name__}: {candidate_exc}"
                )

        raise ImportError("; ".join(attempted)) from primary_exc


def _load_native_module(force_reload: bool = False) -> NativeStatus:
    global _NATIVE_MODULE, _NATIVE_STATUS

    if os.environ.get(DISABLE_NATIVE_ENV, "").strip() in {"1", "true", "TRUE", "yes"}:
        _NATIVE_MODULE = None
        _NATIVE_STATUS = NativeStatus(
            available=False,
            module_name=_module_name(),
            backend_name=None,
            supported_ops=(),
            supported_devices=(),
            error=f"{DISABLE_NATIVE_ENV} is set.",
        )
        return _NATIVE_STATUS

    if _NATIVE_STATUS is not None and not force_reload:
        return _NATIVE_STATUS

    module_name = _module_name()
    try:
        module = _import_native_module(module_name)
    except Exception as exc:  # noqa: BLE001
        _NATIVE_MODULE = None
        _NATIVE_STATUS = NativeStatus(
            available=False,
            module_name=module_name,
            backend_name=None,
            supported_ops=(),
            supported_devices=(),
            error=f"{type(exc).__name__}: {exc}",
        )
        return _NATIVE_STATUS

    supported = getattr(module, "supported_ops", None)
    if callable(supported):
        supported_ops = tuple(str(name) for name in supported())
    else:
        supported_ops = tuple(str(name) for name in getattr(module, "SUPPORTED_OPS", ()))

    supported_devices_fn = getattr(module, "supported_devices", None)
    if callable(supported_devices_fn):
        supported_devices = tuple(str(name) for name in supported_devices_fn())
    else:
        supported_devices = tuple(
            str(name) for name in getattr(module, "SUPPORTED_DEVICES", ("cpu",))
        )

    backend_name = getattr(module, "backend_name", None)
    if callable(backend_name):
        backend_name = str(backend_name())
    elif backend_name is not None:
        backend_name = str(backend_name)

    _NATIVE_MODULE = module
    _NATIVE_STATUS = NativeStatus(
        available=True,
        module_name=module_name,
        backend_name=backend_name,
        supported_ops=supported_ops,
        supported_devices=supported_devices,
        error=None,
    )
    return _NATIVE_STATUS


def native_status(*, force_reload: bool = False) -> NativeStatus:
    return _load_native_module(force_reload=force_reload)


def native_available() -> bool:
    return native_status().available


def native_supports(op_name: str) -> bool:
    status = native_status()
    return status.available and op_name in status.supported_ops


def native_supports_device(device_type: str) -> bool:
    status = native_status()
    return status.available and device_type in status.supported_devices


def _native_module() -> Any:
    status = native_status()
    if not status.available or _NATIVE_MODULE is None:
        raise RuntimeError(status.error or "Native backend is unavailable.")
    return _NATIVE_MODULE


def _to_layer_delta(result: Any) -> LayerDelta:
    if isinstance(result, LayerDelta):
        return result
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], Tensor)
        and isinstance(result[1], Tensor)
    ):
        return LayerDelta(delta_state=result[0], delta_val=result[1])
    raise TypeError("Native backend must return LayerDelta or (delta_state, delta_val).")


def _cuda_float_tensor(tensor: Tensor | None) -> bool:
    return (
        tensor is not None
        and tensor.device.type == "cuda"
        and tensor.dtype in {torch.float16, torch.bfloat16, torch.float32}
    )


def _query_backward_ops_available() -> bool:
    return all(
        native_supports(name)
        for name in (
            "query_topk_reduce_backward_cuda",
            "softsign_backward_cuda",
            "softmax_backward_cuda",
        )
    )


def _flatten_query_tensors(
    query_val: Tensor,
    source_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, tuple[int, ...], int, int, int]:
    batch_shape = tuple(query_val.shape[:-2])
    query_nodes = query_val.shape[-2]
    source_nodes = source_val.shape[-2]
    out_dim = projected_val.shape[-1]
    return (
        query_val.reshape(-1, query_nodes, query_val.shape[-1]).contiguous(),
        source_val.reshape(-1, source_nodes, source_val.shape[-1]).contiguous(),
        projected_state.reshape(-1, source_nodes).contiguous(),
        projected_val.reshape(-1, source_nodes, out_dim).contiguous(),
        batch_shape,
        query_nodes,
        source_nodes,
        out_dim,
    )


def _coerce_query_reduce_backward_inputs(
    edges: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    dtypes = {
        edges.dtype,
        projected_state.dtype,
        projected_val.dtype,
        grad_delta_state.dtype,
        grad_delta_val.dtype,
    }
    if len(dtypes) == 1:
        return edges, projected_state, projected_val, grad_delta_state, grad_delta_val

    target_dtype = projected_state.dtype
    return (
        edges.to(dtype=target_dtype),
        projected_state.to(dtype=target_dtype),
        projected_val.to(dtype=target_dtype),
        grad_delta_state.to(dtype=target_dtype),
        grad_delta_val.to(dtype=target_dtype),
    )


class _DiagonalPropagationQueryTopK(Function):
    @staticmethod
    def forward(
        ctx: Any,
        query_val: Tensor,
        source_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        topk: int,
        query_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().propagation_query_topk_select(
            "diagonal_bilinear",
            weight,
            bias,
            "softsign",
            query_val,
            source_val,
            projected_state,
            projected_val,
            topk,
            query_block_size,
            source_block_size,
            True,
        )
        delta_state, delta_val, scores, indices = result
        ctx.has_bias = bias is not None
        ctx.save_for_backward(query_val, source_val, projected_state, projected_val, weight, scores, indices)
        return delta_state, delta_val

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        query_val, source_val, projected_state, projected_val, weight, scores, indices = ctx.saved_tensors
        (
            flat_query,
            flat_source,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            query_nodes,
            source_nodes,
            out_dim,
        ) = _flatten_query_tensors(query_val, source_val, projected_state, projected_val)
        flat_scores = scores.reshape(-1, query_nodes, scores.shape[-1]).contiguous()
        flat_indices = indices.reshape(-1, query_nodes, indices.shape[-1]).contiguous()
        flat_edges = torch.nn.functional.softsign(flat_scores).contiguous()
        flat_grad_state = grad_delta_state.reshape(-1, query_nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, query_nodes, out_dim).contiguous()
        (
            flat_edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        ) = _coerce_query_reduce_backward_inputs(
            flat_edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        module = _native_module()
        grad_edges, grad_projected_state, grad_projected_val = module.query_topk_reduce_backward_cuda(
            flat_edges,
            flat_indices,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        grad_scores = module.softsign_backward_cuda(flat_scores, grad_edges.contiguous())
        grad_query, grad_source, grad_weight, grad_bias = module.diagonal_pairwise_topk_backward_cuda(
            flat_query,
            flat_source,
            weight.contiguous(),
            flat_indices,
            grad_scores.contiguous(),
            1.0,
        )
        return (
            grad_query.reshape_as(query_val),
            grad_source.reshape_as(source_val),
            grad_projected_state.reshape_as(projected_state),
            grad_projected_val.reshape_as(projected_val),
            grad_weight,
            grad_bias if ctx.has_bias else None,
            None,
            None,
            None,
        )


class _LowRankPropagationQueryTopK(Function):
    @staticmethod
    def forward(
        ctx: Any,
        query_val: Tensor,
        source_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        full_weight: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        topk: int,
        query_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().propagation_query_topk_select(
            "bilinear",
            full_weight,
            bias,
            "softsign",
            query_val,
            source_val,
            projected_state,
            projected_val,
            topk,
            query_block_size,
            source_block_size,
            True,
        )
        delta_state, delta_val, scores, indices = result
        ctx.has_bias = bias is not None
        ctx.save_for_backward(
            query_val,
            source_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            scores,
            indices,
        )
        return delta_state, delta_val

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            query_val,
            source_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            scores,
            indices,
        ) = ctx.saved_tensors
        (
            flat_query,
            flat_source,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            query_nodes,
            _source_nodes,
            out_dim,
        ) = _flatten_query_tensors(query_val, source_val, projected_state, projected_val)
        flat_scores = scores.reshape(-1, query_nodes, scores.shape[-1]).contiguous()
        flat_indices = indices.reshape(-1, query_nodes, indices.shape[-1]).contiguous()
        flat_edges = torch.nn.functional.softsign(flat_scores).contiguous()
        flat_grad_state = grad_delta_state.reshape(-1, query_nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, query_nodes, out_dim).contiguous()
        (
            flat_edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        ) = _coerce_query_reduce_backward_inputs(
            flat_edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        module = _native_module()
        grad_edges, grad_projected_state, grad_projected_val = module.query_topk_reduce_backward_cuda(
            flat_edges,
            flat_indices,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        grad_scores = module.softsign_backward_cuda(flat_scores, grad_edges.contiguous())
        projected_query = torch.matmul(flat_query, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_source, source_weight.t()).contiguous()
        (
            grad_query,
            grad_source,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias,
        ) = module.low_rank_pairwise_topk_backward_cuda(
            flat_query,
            flat_source,
            source_weight.contiguous(),
            target_weight.contiguous(),
            core_weight.contiguous(),
            projected_query,
            projected_source,
            flat_indices,
            grad_scores.contiguous(),
            1.0,
        )
        return (
            grad_query.reshape_as(query_val),
            grad_source.reshape_as(source_val),
            grad_projected_state.reshape_as(projected_state),
            grad_projected_val.reshape_as(projected_val),
            None,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias if ctx.has_bias else None,
            None,
            None,
            None,
        )


class _DiagonalTransitionQueryTopK(Function):
    @staticmethod
    def forward(
        ctx: Any,
        sender_strength: Tensor,
        src_val: Tensor,
        query_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        temperature: float,
        topk: int,
        query_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().transition_query_topk_select(
            "diagonal_bilinear_route",
            None,
            None,
            weight,
            bias,
            float(temperature),
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            topk,
            query_block_size,
            source_block_size,
            True,
        )
        delta_state, delta_val, scores, indices = result
        routes = torch.softmax(scores, dim=-1)
        ctx.has_bias = bias is not None
        ctx.temperature = float(temperature)
        ctx.save_for_backward(
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            weight,
            routes,
            indices,
        )
        return delta_state, delta_val

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        sender_strength, src_val, query_val, projected_state, projected_val, weight, routes, indices = ctx.saved_tensors
        (
            flat_query,
            flat_source,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            query_nodes,
            source_nodes,
            out_dim,
        ) = _flatten_query_tensors(query_val, src_val, projected_state, projected_val)
        flat_sender = sender_strength.reshape(-1, source_nodes).contiguous()
        weighted_state = (flat_sender * flat_projected_state).contiguous()
        weighted_val = (flat_sender.unsqueeze(-1) * flat_projected_val).contiguous()
        flat_routes = routes.reshape(-1, query_nodes, routes.shape[-1]).contiguous()
        flat_indices = indices.reshape(-1, query_nodes, indices.shape[-1]).contiguous()
        flat_grad_state = grad_delta_state.reshape(-1, query_nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, query_nodes, out_dim).contiguous()
        (
            flat_routes,
            weighted_state,
            weighted_val,
            flat_grad_state,
            flat_grad_val,
        ) = _coerce_query_reduce_backward_inputs(
            flat_routes,
            weighted_state,
            weighted_val,
            flat_grad_state,
            flat_grad_val,
        )
        module = _native_module()
        grad_routes, grad_weighted_state, grad_weighted_val = module.query_topk_reduce_backward_cuda(
            flat_routes,
            flat_indices,
            weighted_state,
            weighted_val,
            flat_grad_state,
            flat_grad_val,
        )
        grad_scores = module.softmax_backward_cuda(flat_routes, grad_routes.contiguous())
        grad_query, grad_source, grad_weight, grad_bias = module.diagonal_pairwise_topk_backward_cuda(
            flat_query,
            flat_source,
            weight.contiguous(),
            flat_indices,
            grad_scores.contiguous(),
            ctx.temperature,
        )
        grad_sender = (
            grad_weighted_state * flat_projected_state
            + (grad_weighted_val * flat_projected_val).sum(dim=-1)
        )
        grad_projected_state = grad_weighted_state * flat_sender
        grad_projected_val = grad_weighted_val * flat_sender.unsqueeze(-1)
        return (
            grad_sender.reshape_as(sender_strength),
            grad_source.reshape_as(src_val),
            grad_query.reshape_as(query_val),
            grad_projected_state.reshape_as(projected_state),
            grad_projected_val.reshape_as(projected_val),
            grad_weight,
            grad_bias if ctx.has_bias else None,
            None,
            None,
            None,
            None,
        )


class _LowRankTransitionQueryTopK(Function):
    @staticmethod
    def forward(
        ctx: Any,
        sender_strength: Tensor,
        src_val: Tensor,
        query_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        temperature: float,
        topk: int,
        query_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().transition_query_topk_select(
            "low_rank_bilinear_route",
            source_weight,
            target_weight,
            core_weight,
            bias,
            float(temperature),
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            topk,
            query_block_size,
            source_block_size,
            True,
        )
        delta_state, delta_val, scores, indices = result
        routes = torch.softmax(scores, dim=-1)
        ctx.has_bias = bias is not None
        ctx.temperature = float(temperature)
        ctx.save_for_backward(
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            routes,
            indices,
        )
        return delta_state, delta_val

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            routes,
            indices,
        ) = ctx.saved_tensors
        (
            flat_query,
            flat_source,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            query_nodes,
            source_nodes,
            out_dim,
        ) = _flatten_query_tensors(query_val, src_val, projected_state, projected_val)
        flat_sender = sender_strength.reshape(-1, source_nodes).contiguous()
        weighted_state = (flat_sender * flat_projected_state).contiguous()
        weighted_val = (flat_sender.unsqueeze(-1) * flat_projected_val).contiguous()
        flat_routes = routes.reshape(-1, query_nodes, routes.shape[-1]).contiguous()
        flat_indices = indices.reshape(-1, query_nodes, indices.shape[-1]).contiguous()
        flat_grad_state = grad_delta_state.reshape(-1, query_nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, query_nodes, out_dim).contiguous()
        (
            flat_routes,
            weighted_state,
            weighted_val,
            flat_grad_state,
            flat_grad_val,
        ) = _coerce_query_reduce_backward_inputs(
            flat_routes,
            weighted_state,
            weighted_val,
            flat_grad_state,
            flat_grad_val,
        )
        module = _native_module()
        grad_routes, grad_weighted_state, grad_weighted_val = module.query_topk_reduce_backward_cuda(
            flat_routes,
            flat_indices,
            weighted_state,
            weighted_val,
            flat_grad_state,
            flat_grad_val,
        )
        grad_scores = module.softmax_backward_cuda(flat_routes, grad_routes.contiguous())
        projected_query = torch.matmul(flat_query, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_source, source_weight.t()).contiguous()
        (
            grad_query,
            grad_source,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias,
        ) = module.low_rank_pairwise_topk_backward_cuda(
            flat_query,
            flat_source,
            source_weight.contiguous(),
            target_weight.contiguous(),
            core_weight.contiguous(),
            projected_query,
            projected_source,
            flat_indices,
            grad_scores.contiguous(),
            ctx.temperature,
        )
        grad_sender = (
            grad_weighted_state * flat_projected_state
            + (grad_weighted_val * flat_projected_val).sum(dim=-1)
        )
        grad_projected_state = grad_weighted_state * flat_sender
        grad_projected_val = grad_weighted_val * flat_sender.unsqueeze(-1)
        return (
            grad_sender.reshape_as(sender_strength),
            grad_source.reshape_as(src_val),
            grad_query.reshape_as(query_val),
            grad_projected_state.reshape_as(projected_state),
            grad_projected_val.reshape_as(projected_val),
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias if ctx.has_bias else None,
            None,
            None,
            None,
            None,
        )


def propagation_dense_native(
    *,
    pairwise_fn: object,
    edge_compress_name: str,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    target_block_size: int,
    source_block_size: int,
) -> Any:
    if not supports_pairwise_kernel(pairwise_fn):
        raise TypeError("Unsupported pairwise_fn for native propagation.")
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_dense(
        spec.kind,
        spec.weight,
        spec.bias,
        edge_compress_name,
        layer_val,
        projected_state,
        projected_val,
        target_block_size,
        source_block_size,
    ))


def propagation_window_native(
    *,
    pairwise_fn: object,
    edge_compress_name: str,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    window: int,
    target_block_size: int,
    source_block_size: int,
) -> Any:
    if not supports_pairwise_kernel(pairwise_fn):
        raise TypeError("Unsupported pairwise_fn for native propagation.")
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_window(
        spec.kind,
        spec.weight,
        spec.bias,
        edge_compress_name,
        layer_val,
        projected_state,
        projected_val,
        window,
        target_block_size,
        source_block_size,
    ))


def propagation_topk_native(
    *,
    pairwise_fn: object,
    edge_compress_name: str,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    topk: int,
    target_block_size: int,
    source_block_size: int,
) -> Any:
    if not supports_pairwise_kernel(pairwise_fn):
        raise TypeError("Unsupported pairwise_fn for native propagation.")
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_topk(
        spec.kind,
        spec.weight,
        spec.bias,
        edge_compress_name,
        layer_val,
        projected_state,
        projected_val,
        topk,
        target_block_size,
        source_block_size,
    ))


def propagation_query_topk_native(
    *,
    pairwise_fn: object,
    edge_compress_name: str,
    query_val: Tensor,
    source_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    topk: int,
    query_block_size: int,
    source_block_size: int,
    use_cuda_reduce: bool = True,
) -> Any:
    if not supports_pairwise_kernel(pairwise_fn):
        raise TypeError("Unsupported pairwise_fn for native query propagation.")
    use_cuda_autograd = (
        _query_backward_ops_available()
        and native_supports("propagation_query_topk_select")
        and native_supports("diagonal_pairwise_topk_backward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and _cuda_float_tensor(query_val)
        and _cuda_float_tensor(source_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    if use_cuda_autograd and isinstance(pairwise_fn, DiagonalBilinearPairwise):
        delta_state, delta_val = _DiagonalPropagationQueryTopK.apply(
            query_val,
            source_val,
            projected_state,
            projected_val,
            pairwise_fn.weight,
            pairwise_fn.bias,
            topk,
            query_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    if use_cuda_autograd and isinstance(pairwise_fn, LowRankBilinearPairwise):
        delta_state, delta_val = _LowRankPropagationQueryTopK.apply(
            query_val,
            source_val,
            projected_state,
            projected_val,
            pairwise_fn.effective_weight(),
            pairwise_fn.source_proj.weight,
            pairwise_fn.target_proj.weight,
            pairwise_fn.weight,
            pairwise_fn.bias,
            topk,
            query_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_query_topk(
        spec.kind,
        spec.weight,
        spec.bias,
        edge_compress_name,
        query_val,
        source_val,
        projected_state,
        projected_val,
        topk,
        query_block_size,
        source_block_size,
        use_cuda_reduce,
    ))


def transition_dense_native(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    dst_nodes: int,
    src_block_size: int,
    dst_block_size: int,
) -> Any:
    if not supports_route_kernel(route_fn):
        raise TypeError("Unsupported route_fn for native transition.")
    spec = route_kernel_spec(route_fn)
    return _to_layer_delta(_native_module().transition_dense(
        spec.kind,
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
        sender_strength,
        src_val,
        projected_state,
        projected_val,
        dst_nodes,
        src_block_size,
        dst_block_size,
    ))


def transition_topk_native(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    dst_nodes: int,
    topk: int,
    src_block_size: int,
    dst_block_size: int,
) -> Any:
    if not supports_route_kernel(route_fn):
        raise TypeError("Unsupported route_fn for native transition.")
    spec = route_kernel_spec(route_fn)
    return _to_layer_delta(_native_module().transition_topk(
        spec.kind,
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
        sender_strength,
        src_val,
        projected_state,
        projected_val,
        dst_nodes,
        topk,
        src_block_size,
        dst_block_size,
    ))


def transition_query_topk_native(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    query_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    topk: int,
    query_block_size: int,
    source_block_size: int,
    use_cuda_reduce: bool = True,
) -> Any:
    if not supports_pairwise_route_kernel(route_fn):
        raise TypeError("Unsupported pairwise route_fn for native query transition.")
    inner = getattr(route_fn, "route_fn", route_fn)
    temperature = float(getattr(route_fn, "temperature", 1.0))
    use_cuda_autograd = (
        _query_backward_ops_available()
        and native_supports("transition_query_topk_select")
        and native_supports("diagonal_pairwise_topk_backward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and _cuda_float_tensor(sender_strength)
        and _cuda_float_tensor(src_val)
        and _cuda_float_tensor(query_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    if use_cuda_autograd and isinstance(inner, DiagonalBilinearRoute):
        delta_state, delta_val = _DiagonalTransitionQueryTopK.apply(
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            inner.weight,
            inner.bias,
            temperature,
            topk,
            query_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    if use_cuda_autograd and isinstance(inner, LowRankBilinearRoute):
        delta_state, delta_val = _LowRankTransitionQueryTopK.apply(
            sender_strength,
            src_val,
            query_val,
            projected_state,
            projected_val,
            inner.source_proj.weight,
            inner.target_proj.weight,
            inner.weight,
            inner.bias,
            temperature,
            topk,
            query_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_route_kernel_spec(route_fn)
    return _to_layer_delta(_native_module().transition_query_topk(
        spec.kind,
        spec.source_weight,
        spec.target_weight,
        spec.core_weight,
        spec.bias,
        float(spec.temperature),
        sender_strength,
        src_val,
        query_val,
        projected_state,
        projected_val,
        topk,
        query_block_size,
        source_block_size,
        use_cuda_reduce,
    ))


def transition_pairwise_topk_native(
    *,
    route_fn: object,
    sender_strength: Tensor,
    src_val: Tensor,
    dst_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    topk: int,
    src_block_size: int,
    dst_block_size: int,
) -> Any:
    if not supports_pairwise_route_kernel(route_fn):
        raise TypeError("Unsupported pairwise route_fn for native sparse transition.")
    spec = pairwise_route_kernel_spec(route_fn)
    return _to_layer_delta(_native_module().transition_pairwise_topk(
        spec.kind,
        spec.source_weight,
        spec.target_weight,
        spec.core_weight,
        spec.bias,
        float(spec.temperature),
        sender_strength,
        src_val,
        dst_val,
        projected_state,
        projected_val,
        topk,
        src_block_size,
        dst_block_size,
    ))
