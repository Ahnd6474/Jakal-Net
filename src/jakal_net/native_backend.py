from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from jakal_net.core import LayerDelta
from jakal_net.kernel_common import (
    pairwise_kernel_spec,
    route_kernel_spec,
    supports_pairwise_kernel,
    supports_route_kernel,
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
