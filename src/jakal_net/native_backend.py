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
from torch.nn import functional as F

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
    HadamardMLPPairwise,
    LowRankBilinearPairwise,
    LowRankBilinearRoute,
    SourceTargetHadamardMLPRoute,
)

DEFAULT_NATIVE_MODULE = "jakal_net_native"
NATIVE_MODULE_ENV = "JAKAL_NET_NATIVE_MODULE"
DISABLE_NATIVE_ENV = "JAKAL_NET_DISABLE_NATIVE"
EXPERIMENTAL_FUSED_TRAINING_ENV = "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING"
EXPERIMENTAL_FUSED_TRAINING_CHECKPOINT_STRIDE_ENV = "JAKAL_NET_FUSED_TRAINING_CHECKPOINT_STRIDE"
EXPERIMENTAL_SCAN_BACKWARD_CUDA_ENV = "JAKAL_NET_ENABLE_EXPERIMENTAL_SCAN_BACKWARD_CUDA"


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


def _native_scan_uses_legacy_low_rank_extension(
    route_kind_name: str,
    propagation_pairwise_kind: str,
) -> bool:
    return (
        route_kind_name in {"low_rank_bilinear", "low_rank_bilinear_route"}
        and propagation_pairwise_kind == "low_rank_bilinear"
    ) or (
        route_kind_name == "multihead_max_low_rank_bilinear_route"
        and propagation_pairwise_kind == "multihead_max_low_rank_bilinear"
    ) or (
        route_kind_name == "diagonal_bilinear_route"
        and propagation_pairwise_kind == "diagonal_bilinear"
    ) or (
        route_kind_name == "multihead_max_diagonal_bilinear_route"
        and propagation_pairwise_kind == "multihead_max_diagonal_bilinear"
    )


def causal_memory_scan_fused_trace_native(
    *,
    aligned_s: Tensor,
    flat_memory: tuple[Tensor, ...],
    value_to_state_weight: Tensor,
    value_to_state_bias: Tensor | None,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
    write_source_weights: tuple[Tensor, ...],
    write_target_weights: tuple[Tensor, ...],
    write_core_weights: tuple[Tensor, ...],
    write_biases: tuple[Tensor, ...],
    write_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_source_weights: tuple[Tensor, ...],
    propagation_target_weights: tuple[Tensor, ...],
    propagation_core_weights: tuple[Tensor, ...],
    propagation_biases: tuple[Tensor, ...],
    propagation_topks: tuple[int, ...],
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
    val_norm_weights: tuple[Tensor, ...],
    val_norm_biases: tuple[Tensor, ...],
    level_transition_source_weights: tuple[Tensor, ...],
    level_transition_target_weights: tuple[Tensor, ...],
    level_transition_core_weights: tuple[Tensor, ...],
    level_transition_biases: tuple[Tensor, ...],
    level_transition_topks: tuple[int, ...],
    level_norm_weights: tuple[Tensor, ...],
    level_norm_biases: tuple[Tensor, ...],
    skip_source_weights: tuple[Tensor, ...],
    skip_target_weights: tuple[Tensor, ...],
    skip_core_weights: tuple[Tensor, ...],
    skip_biases: tuple[Tensor, ...],
    skip_gates: tuple[Tensor, ...],
    skip_topks: tuple[int, ...],
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    if _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind):
        result = _native_module().causal_memory_scan_fused_trace(
            aligned_s,
            list(flat_memory),
            value_to_state_weight,
            value_to_state_bias,
            s_prediction_weight,
            prediction_input_norm_weight,
            prediction_input_norm_bias,
            read_template_val,
            list(read_projection_weights),
            list(read_gates),
            list(write_source_weights),
            list(write_target_weights),
            list(write_core_weights),
            list(write_biases),
            list(write_topks),
            transition_compress_name,
            list(propagation_source_weights),
            list(propagation_target_weights),
            list(propagation_core_weights),
            list(propagation_biases),
            list(propagation_topks),
            propagation_compress_name,
            list(val_norm_weights),
            list(val_norm_biases),
            list(level_transition_source_weights),
            list(level_transition_target_weights),
            list(level_transition_core_weights),
            list(level_transition_biases),
            list(level_transition_topks),
            list(level_norm_weights),
            list(level_norm_biases),
            list(skip_source_weights),
            list(skip_target_weights),
            list(skip_core_weights),
            list(skip_biases),
            list(skip_gates),
            list(skip_topks),
        )
    else:
        result = _native_module().causal_memory_scan_fused_trace(
            aligned_s,
            list(flat_memory),
            value_to_state_weight,
            value_to_state_bias,
            s_prediction_weight,
            prediction_input_norm_weight,
            prediction_input_norm_bias,
            read_template_val,
            list(read_projection_weights),
            list(read_gates),
            list(write_source_weights),
            list(write_target_weights),
            list(write_core_weights),
            list(write_biases),
            list(write_topks),
            route_kind_name,
            transition_compress_name,
            list(propagation_source_weights),
            list(propagation_target_weights),
            list(propagation_core_weights),
            list(propagation_biases),
            list(propagation_topks),
            propagation_pairwise_kind,
            propagation_compress_name,
            list(val_norm_weights),
            list(val_norm_biases),
            list(level_transition_source_weights),
            list(level_transition_target_weights),
            list(level_transition_core_weights),
            list(level_transition_biases),
            list(level_transition_topks),
            list(level_norm_weights),
            list(level_norm_biases),
            list(skip_source_weights),
            list(skip_target_weights),
            list(skip_core_weights),
            list(skip_biases),
            list(skip_gates),
            list(skip_topks),
        )
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError("causal_memory_scan_fused_trace must return (query_val, flat_memory_tensors, trace_tensors).")
    query_val, next_memory, trace_tensors = result
    return query_val, tuple(next_memory), tuple(trace_tensors)


def causal_memory_scan_fused_checkpoints_native(
    *,
    checkpoint_stride: int,
    aligned_s: Tensor,
    flat_memory: tuple[Tensor, ...],
    value_to_state_weight: Tensor,
    value_to_state_bias: Tensor | None,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
    write_source_weights: tuple[Tensor, ...],
    write_target_weights: tuple[Tensor, ...],
    write_core_weights: tuple[Tensor, ...],
    write_biases: tuple[Tensor, ...],
    write_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_source_weights: tuple[Tensor, ...],
    propagation_target_weights: tuple[Tensor, ...],
    propagation_core_weights: tuple[Tensor, ...],
    propagation_biases: tuple[Tensor, ...],
    propagation_topks: tuple[int, ...],
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
    val_norm_weights: tuple[Tensor, ...],
    val_norm_biases: tuple[Tensor, ...],
    level_transition_source_weights: tuple[Tensor, ...],
    level_transition_target_weights: tuple[Tensor, ...],
    level_transition_core_weights: tuple[Tensor, ...],
    level_transition_biases: tuple[Tensor, ...],
    level_transition_topks: tuple[int, ...],
    level_norm_weights: tuple[Tensor, ...],
    level_norm_biases: tuple[Tensor, ...],
    skip_source_weights: tuple[Tensor, ...],
    skip_target_weights: tuple[Tensor, ...],
    skip_core_weights: tuple[Tensor, ...],
    skip_biases: tuple[Tensor, ...],
    skip_gates: tuple[Tensor, ...],
    skip_topks: tuple[int, ...],
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    if _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind):
        result = _native_module().causal_memory_scan_fused_checkpoints(
            aligned_s,
            list(flat_memory),
            value_to_state_weight,
            value_to_state_bias,
            s_prediction_weight,
            prediction_input_norm_weight,
            prediction_input_norm_bias,
            read_template_val,
            list(read_projection_weights),
            list(read_gates),
            list(write_source_weights),
            list(write_target_weights),
            list(write_core_weights),
            list(write_biases),
            list(write_topks),
            transition_compress_name,
            list(propagation_source_weights),
            list(propagation_target_weights),
            list(propagation_core_weights),
            list(propagation_biases),
            list(propagation_topks),
            propagation_compress_name,
            list(val_norm_weights),
            list(val_norm_biases),
            list(level_transition_source_weights),
            list(level_transition_target_weights),
            list(level_transition_core_weights),
            list(level_transition_biases),
            list(level_transition_topks),
            list(level_norm_weights),
            list(level_norm_biases),
            list(skip_source_weights),
            list(skip_target_weights),
            list(skip_core_weights),
            list(skip_biases),
            list(skip_gates),
            list(skip_topks),
            int(checkpoint_stride),
        )
    else:
        result = _native_module().causal_memory_scan_fused_checkpoints(
            aligned_s,
            list(flat_memory),
            value_to_state_weight,
            value_to_state_bias,
            s_prediction_weight,
            prediction_input_norm_weight,
            prediction_input_norm_bias,
            read_template_val,
            list(read_projection_weights),
            list(read_gates),
            list(write_source_weights),
            list(write_target_weights),
            list(write_core_weights),
            list(write_biases),
            list(write_topks),
            route_kind_name,
            transition_compress_name,
            list(propagation_source_weights),
            list(propagation_target_weights),
            list(propagation_core_weights),
            list(propagation_biases),
            list(propagation_topks),
            propagation_pairwise_kind,
            propagation_compress_name,
            list(val_norm_weights),
            list(val_norm_biases),
            list(level_transition_source_weights),
            list(level_transition_target_weights),
            list(level_transition_core_weights),
            list(level_transition_biases),
            list(level_transition_topks),
            list(level_norm_weights),
            list(level_norm_biases),
            list(skip_source_weights),
            list(skip_target_weights),
            list(skip_core_weights),
            list(skip_biases),
            list(skip_gates),
            list(skip_topks),
            int(checkpoint_stride),
        )
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError("causal_memory_scan_fused_checkpoints must return (query_val, flat_memory_tensors, checkpoint_tensors).")
    query_val, next_memory, checkpoint_tensors = result
    return query_val, tuple(next_memory), tuple(checkpoint_tensors)


def causal_memory_scan_fused_native(
    *,
    aligned_s: Tensor,
    flat_memory: tuple[Tensor, ...],
    value_to_state_weight: Tensor,
    value_to_state_bias: Tensor | None,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
    write_source_weights: tuple[Tensor, ...],
    write_target_weights: tuple[Tensor, ...],
    write_core_weights: tuple[Tensor, ...],
    write_biases: tuple[Tensor, ...],
    write_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_source_weights: tuple[Tensor, ...],
    propagation_target_weights: tuple[Tensor, ...],
    propagation_core_weights: tuple[Tensor, ...],
    propagation_biases: tuple[Tensor, ...],
    propagation_topks: tuple[int, ...],
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
    val_norm_weights: tuple[Tensor, ...],
    val_norm_biases: tuple[Tensor, ...],
    level_transition_source_weights: tuple[Tensor, ...],
    level_transition_target_weights: tuple[Tensor, ...],
    level_transition_core_weights: tuple[Tensor, ...],
    level_transition_biases: tuple[Tensor, ...],
    level_transition_topks: tuple[int, ...],
    level_norm_weights: tuple[Tensor, ...],
    level_norm_biases: tuple[Tensor, ...],
    skip_source_weights: tuple[Tensor, ...],
    skip_target_weights: tuple[Tensor, ...],
    skip_core_weights: tuple[Tensor, ...],
    skip_biases: tuple[Tensor, ...],
    skip_gates: tuple[Tensor, ...],
    skip_topks: tuple[int, ...],
) -> tuple[Tensor, tuple[Tensor, ...]]:
    tensor_args, meta_args = _flatten_causal_memory_scan_args(
        aligned_s=aligned_s,
        flat_memory=flat_memory,
        value_to_state_weight=value_to_state_weight,
        value_to_state_bias=value_to_state_bias,
        s_prediction_weight=s_prediction_weight,
        prediction_input_norm_weight=prediction_input_norm_weight,
        prediction_input_norm_bias=prediction_input_norm_bias,
        read_template_val=read_template_val,
        read_projection_weights=read_projection_weights,
        read_gates=read_gates,
        write_source_weights=write_source_weights,
        write_target_weights=write_target_weights,
        write_core_weights=write_core_weights,
        write_biases=write_biases,
        write_topks=write_topks,
        route_kind_name=route_kind_name,
        transition_compress_name=transition_compress_name,
        propagation_source_weights=propagation_source_weights,
        propagation_target_weights=propagation_target_weights,
        propagation_core_weights=propagation_core_weights,
        propagation_biases=propagation_biases,
        propagation_topks=propagation_topks,
        propagation_pairwise_kind=propagation_pairwise_kind,
        propagation_compress_name=propagation_compress_name,
        val_norm_weights=val_norm_weights,
        val_norm_biases=val_norm_biases,
        level_transition_source_weights=level_transition_source_weights,
        level_transition_target_weights=level_transition_target_weights,
        level_transition_core_weights=level_transition_core_weights,
        level_transition_biases=level_transition_biases,
        level_transition_topks=level_transition_topks,
        level_norm_weights=level_norm_weights,
        level_norm_biases=level_norm_biases,
        skip_source_weights=skip_source_weights,
        skip_target_weights=skip_target_weights,
        skip_core_weights=skip_core_weights,
        skip_biases=skip_biases,
        skip_gates=skip_gates,
        skip_topks=skip_topks,
    )
    if torch.is_grad_enabled() and _experimental_fused_training_enabled():
        outputs = _CausalMemoryScanFusedFunction.apply(*tensor_args, *meta_args)
        return outputs[0], tuple(outputs[1:])

    query_val, next_memory = _causal_memory_scan_fused_native_forward(
        *tensor_args,
        num_levels=meta_args[0],
        write_topks=meta_args[1],
        propagation_topks=meta_args[2],
        level_transition_topks=meta_args[3],
        skip_topks=meta_args[4],
        route_kind_name=meta_args[5],
        transition_compress_name=meta_args[6],
        propagation_pairwise_kind=meta_args[7],
        propagation_compress_name=meta_args[8],
    )
    return query_val, tuple(next_memory)


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


def _signed_entmax15_ops_available() -> bool:
    try:
        getattr(torch.ops.jakal_net, "signed_entmax15")
        getattr(torch.ops.jakal_net, "signed_entmax15_backward")
    except (AttributeError, RuntimeError):
        return False
    return True


def _experimental_fused_training_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_FUSED_TRAINING_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _experimental_fused_training_checkpoint_stride(seq_len: int) -> int | None:
    raw_value = os.environ.get(EXPERIMENTAL_FUSED_TRAINING_CHECKPOINT_STRIDE_ENV, "").strip()
    if not raw_value:
        return None
    stride = int(raw_value)
    if stride <= 0:
        return None
    return min(seq_len, stride)


def _experimental_scan_backward_cuda_enabled() -> bool:
    return os.environ.get(EXPERIMENTAL_SCAN_BACKWARD_CUDA_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _flatten_causal_memory_scan_args(
    *,
    aligned_s: Tensor,
    flat_memory: tuple[Tensor, ...],
    value_to_state_weight: Tensor,
    value_to_state_bias: Tensor | None,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
    write_source_weights: tuple[Tensor, ...],
    write_target_weights: tuple[Tensor, ...],
    write_core_weights: tuple[Tensor, ...],
    write_biases: tuple[Tensor, ...],
    write_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_source_weights: tuple[Tensor, ...],
    propagation_target_weights: tuple[Tensor, ...],
    propagation_core_weights: tuple[Tensor, ...],
    propagation_biases: tuple[Tensor, ...],
    propagation_topks: tuple[int, ...],
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
    val_norm_weights: tuple[Tensor, ...],
    val_norm_biases: tuple[Tensor, ...],
    level_transition_source_weights: tuple[Tensor, ...],
    level_transition_target_weights: tuple[Tensor, ...],
    level_transition_core_weights: tuple[Tensor, ...],
    level_transition_biases: tuple[Tensor, ...],
    level_transition_topks: tuple[int, ...],
    level_norm_weights: tuple[Tensor, ...],
    level_norm_biases: tuple[Tensor, ...],
    skip_source_weights: tuple[Tensor, ...],
    skip_target_weights: tuple[Tensor, ...],
    skip_core_weights: tuple[Tensor, ...],
    skip_biases: tuple[Tensor, ...],
    skip_gates: tuple[Tensor, ...],
    skip_topks: tuple[int, ...],
) -> tuple[
    tuple[Tensor, ...],
    tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], str, str, str, str],
]:
    num_levels = len(read_projection_weights)
    tensor_args: tuple[Tensor, ...] = (
        aligned_s,
        *flat_memory,
        value_to_state_weight,
        _save_optional_tensor(value_to_state_bias, aligned_s),
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, aligned_s),
        read_template_val,
        *read_projection_weights,
        *read_gates,
        *write_source_weights,
        *write_target_weights,
        *write_core_weights,
        *write_biases,
        *propagation_source_weights,
        *propagation_target_weights,
        *propagation_core_weights,
        *propagation_biases,
        *val_norm_weights,
        *val_norm_biases,
        *level_transition_source_weights,
        *level_transition_target_weights,
        *level_transition_core_weights,
        *level_transition_biases,
        *level_norm_weights,
        *level_norm_biases,
        *skip_source_weights,
        *skip_target_weights,
        *skip_core_weights,
        *skip_biases,
        *skip_gates,
    )
    meta_args = (
        num_levels,
        tuple(int(v) for v in write_topks),
        tuple(int(v) for v in propagation_topks),
        tuple(int(v) for v in level_transition_topks),
        tuple(int(v) for v in skip_topks),
        str(route_kind_name),
        str(transition_compress_name),
        str(propagation_pairwise_kind),
        str(propagation_compress_name),
    )
    return tensor_args, meta_args


def _unpack_causal_memory_scan_tensor_args(
    tensor_args: tuple[Tensor, ...],
    num_levels: int,
) -> dict[str, Any]:
    idx = 0
    out: dict[str, Any] = {}
    out["aligned_s"] = tensor_args[idx]
    idx += 1
    out["flat_memory"] = tensor_args[idx: idx + (2 * num_levels)]
    idx += 2 * num_levels
    out["value_to_state_weight"] = tensor_args[idx]
    idx += 1
    out["value_to_state_bias"] = tensor_args[idx]
    idx += 1
    out["s_prediction_weight"] = tensor_args[idx]
    idx += 1
    out["prediction_input_norm_weight"] = tensor_args[idx]
    idx += 1
    out["prediction_input_norm_bias"] = tensor_args[idx]
    idx += 1
    out["read_template_val"] = tensor_args[idx]
    idx += 1

    for name, count in (
        ("read_projection_weights", num_levels),
        ("read_gates", num_levels),
        ("write_source_weights", num_levels),
        ("write_target_weights", num_levels),
        ("write_core_weights", num_levels),
        ("write_biases", num_levels),
        ("propagation_source_weights", num_levels),
        ("propagation_target_weights", num_levels),
        ("propagation_core_weights", num_levels),
        ("propagation_biases", num_levels),
        ("val_norm_weights", num_levels),
        ("val_norm_biases", num_levels),
        ("level_transition_source_weights", max(num_levels - 1, 0)),
        ("level_transition_target_weights", max(num_levels - 1, 0)),
        ("level_transition_core_weights", max(num_levels - 1, 0)),
        ("level_transition_biases", max(num_levels - 1, 0)),
        ("level_norm_weights", num_levels),
        ("level_norm_biases", num_levels),
        ("skip_source_weights", max(num_levels - 1, 0)),
        ("skip_target_weights", max(num_levels - 1, 0)),
        ("skip_core_weights", max(num_levels - 1, 0)),
        ("skip_biases", max(num_levels - 1, 0)),
        ("skip_gates", max(num_levels - 1, 0)),
    ):
        out[name] = tensor_args[idx: idx + count]
        idx += count

    if idx != len(tensor_args):
        raise RuntimeError("unexpected causal-memory scan tensor arg count")
    return out


def _native_scan_layer_norm(input: Tensor, weight: Tensor, packed_bias: Tensor) -> Tensor:
    if weight.numel() == 0:
        return input
    return F.layer_norm(
        input,
        (input.shape[-1],),
        weight.to(dtype=input.dtype),
        None if packed_bias.numel() == 0 else packed_bias.to(dtype=input.dtype),
    )


def _native_scan_signed_softmax_state(state: Tensor) -> Tensor:
    clean_state = torch.nan_to_num(state)
    magnitude = torch.softmax(clean_state.abs(), dim=-1)
    return torch.sign(clean_state) * magnitude * float(state.shape[-1])


def _native_scan_signed_abs_softmax(scores: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    return torch.nan_to_num(torch.sign(clean_scores) * torch.softmax(clean_scores.abs(), dim=-1))


def _native_scan_value_to_state(
    val: Tensor,
    weight: Tensor,
    packed_bias: Tensor,
) -> Tensor:
    if weight.numel() == 0:
        return torch.linalg.vector_norm(val, ord=2, dim=-1)
    return F.linear(
        val,
        weight.to(dtype=val.dtype),
        None if packed_bias.numel() == 0 else packed_bias.to(dtype=val.dtype),
    ).squeeze(-1)


def _native_scan_pairwise_scores(
    pairwise_kind: str,
    src_val: Tensor,
    dst_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    packed_bias: Tensor,
) -> Tensor:
    if pairwise_kind.startswith("multihead_max_"):
        base_kind = pairwise_kind[len("multihead_max_") :]
        if base_kind == "low_rank_bilinear":
            projected_source = torch.einsum(
                "...id,hrd->...hir",
                src_val,
                source_weight.to(dtype=src_val.dtype),
            )
            projected_target = torch.einsum(
                "...kd,hrd->...hkr",
                dst_val,
                target_weight.to(dtype=dst_val.dtype),
            )
            core_view_shape = [1] * projected_source.dim()
            core_view_shape[-3] = core_weight.shape[0]
            core_view_shape[-1] = core_weight.shape[1]
            weighted_source = projected_source * core_weight.to(dtype=src_val.dtype).view(*core_view_shape)
            scores = torch.einsum("...hir,...hkr->...hik", weighted_source, projected_target)
            if packed_bias.numel() != 0:
                scores = scores + packed_bias.to(dtype=scores.dtype).view(
                    *([1] * (scores.dim() - 3)), packed_bias.shape[0], 1, 1
                )
            return scores.max(dim=-3).values
        num_heads = int(core_weight.shape[0])
        best_scores: Tensor | None = None
        for head_index in range(num_heads):
            head_scores = _native_scan_pairwise_scores(
                base_kind,
                src_val,
                dst_val,
                source_weight[head_index] if source_weight.numel() != 0 else source_weight,
                target_weight[head_index] if target_weight.numel() != 0 else target_weight,
                core_weight[head_index],
                packed_bias[head_index] if packed_bias.numel() != 0 else packed_bias,
            )
            best_scores = head_scores if best_scores is None else torch.maximum(best_scores, head_scores)
        assert best_scores is not None
        return best_scores
    if pairwise_kind == "low_rank_bilinear":
        projected_source = F.linear(src_val, source_weight.to(dtype=src_val.dtype), None)
        projected_source = projected_source * core_weight.to(dtype=src_val.dtype)
        projected_target = F.linear(dst_val, target_weight.to(dtype=dst_val.dtype), None)
        scores = torch.einsum("...ir,...kr->...ik", projected_source, projected_target)
        if packed_bias.numel() != 0:
            scores = scores + packed_bias.to(dtype=scores.dtype)
        return scores
    if pairwise_kind == "diagonal_bilinear":
        weighted_target = dst_val * core_weight.to(dtype=dst_val.dtype).view(1, 1, -1)
        scores = torch.einsum("...id,...kd->...ik", weighted_target, src_val)
        if packed_bias.numel() != 0:
            scores = scores + packed_bias.to(dtype=scores.dtype)
        return scores
    if pairwise_kind == "bilinear":
        projected_target = torch.matmul(dst_val, core_weight.to(dtype=dst_val.dtype))
        scores = torch.einsum("...id,...kd->...ik", projected_target, src_val)
        if packed_bias.numel() != 0:
            scores = scores + packed_bias.to(dtype=scores.dtype)
        return scores
    if pairwise_kind == "scaled_cosine":
        scale = core_weight.reshape(-1)[0].to(dtype=dst_val.dtype)
        eps = packed_bias.reshape(-1)[0].item() if packed_bias.numel() != 0 else 1e-6
        normalized_src = src_val / src_val.norm(dim=-1, keepdim=True).clamp_min(eps)
        normalized_dst = dst_val / dst_val.norm(dim=-1, keepdim=True).clamp_min(eps)
        return torch.einsum("...id,...kd->...ik", normalized_dst, normalized_src) * scale
    raise RuntimeError(f"Unsupported native scan pairwise kind: {pairwise_kind!r}")


def _native_scan_route_scores(
    route_kind_name: str,
    src_val: Tensor,
    dst_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    packed_bias: Tensor,
) -> Tensor:
    if route_kind_name.startswith("multihead_max_"):
        base_kind = route_kind_name[len("multihead_max_") :]
        if base_kind == "low_rank_bilinear_route":
            projected_source = torch.einsum(
                "...id,hrd->...hir",
                src_val,
                source_weight.to(dtype=src_val.dtype),
            )
            projected_target = torch.einsum(
                "...kd,hrd->...hkr",
                dst_val,
                target_weight.to(dtype=dst_val.dtype),
            )
            core_view_shape = [1] * projected_source.dim()
            core_view_shape[-3] = core_weight.shape[0]
            core_view_shape[-1] = core_weight.shape[1]
            weighted_source = projected_source * core_weight.to(dtype=src_val.dtype).view(*core_view_shape)
            scores = torch.einsum("...hir,...hkr->...hik", weighted_source, projected_target)
            if packed_bias.numel() != 0:
                scores = scores + packed_bias.to(dtype=scores.dtype).view(
                    *([1] * (scores.dim() - 3)), packed_bias.shape[0], 1, 1
                )
            return scores.max(dim=-3).values
        num_heads = int(core_weight.shape[0])
        best_scores: Tensor | None = None
        for head_index in range(num_heads):
            head_scores = _native_scan_route_scores(
                base_kind,
                src_val,
                dst_val,
                source_weight[head_index] if source_weight.numel() != 0 else source_weight,
                target_weight[head_index] if target_weight.numel() != 0 else target_weight,
                core_weight[head_index],
                packed_bias[head_index] if packed_bias.numel() != 0 else packed_bias,
            )
            best_scores = head_scores if best_scores is None else torch.maximum(best_scores, head_scores)
        assert best_scores is not None
        return best_scores
    if route_kind_name == "low_rank_bilinear_route":
        projected_source = F.linear(src_val, source_weight.to(dtype=src_val.dtype), None)
        projected_source = projected_source * core_weight.to(dtype=src_val.dtype)
        projected_target = F.linear(dst_val, target_weight.to(dtype=dst_val.dtype), None)
        scores = torch.einsum("...ir,...kr->...ik", projected_source, projected_target)
    elif route_kind_name == "diagonal_bilinear_route":
        weighted_source = src_val * core_weight.to(dtype=src_val.dtype).view(1, 1, -1)
        scores = torch.einsum("...id,...kd->...ik", weighted_source, dst_val)
    elif route_kind_name == "full_bilinear_route":
        projected_source = F.linear(src_val, source_weight.to(dtype=src_val.dtype), None)
        projected_target = F.linear(dst_val, target_weight.to(dtype=dst_val.dtype), None)
        weighted_source = torch.matmul(projected_source, core_weight.to(dtype=src_val.dtype))
        scores = torch.einsum("...id,...kd->...ik", weighted_source, projected_target)
    elif route_kind_name == "query_normalized_dot_route":
        scale = core_weight.reshape(-1)[0].to(dtype=src_val.dtype)
        eps = packed_bias.reshape(-1)[0].item() if packed_bias.numel() != 0 else 1e-6
        numerators = torch.einsum("...id,...kd->...ik", src_val, dst_val)
        denominators = src_val.square().sum(dim=-1, keepdim=True).clamp_min(eps)
        scores = numerators / denominators * scale
    else:
        raise RuntimeError(f"Unsupported native scan route kind: {route_kind_name!r}")
    if packed_bias.numel() != 0 and route_kind_name != "query_normalized_dot_route":
        scores = scores + packed_bias.to(dtype=scores.dtype)
    return scores


def _native_scan_full_topk_indices(scores: Tensor) -> Tensor:
    return torch.arange(scores.shape[-1], device=scores.device, dtype=torch.long).view(1, 1, -1).expand(scores.shape[0], scores.shape[1], scores.shape[2])


def _native_scan_gather_state(source: Tensor, indices: Tensor) -> Tensor:
    expanded = source.unsqueeze(1).expand(source.shape[0], indices.shape[1], source.shape[1])
    return expanded.gather(2, indices)


def _native_scan_gather_val(source: Tensor, indices: Tensor) -> Tensor:
    expanded = source.unsqueeze(1).expand(source.shape[0], indices.shape[1], source.shape[1], source.shape[2])
    gather_index = indices.unsqueeze(-1).expand(indices.shape[0], indices.shape[1], indices.shape[2], source.shape[2])
    return expanded.gather(2, gather_index)


def _native_scan_transition_pairwise_topk_signed_abs(
    route_kind_name: str,
    sender_strength: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    src_val: Tensor,
    dst_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    packed_bias: Tensor,
    topk: int,
) -> tuple[Tensor, Tensor]:
    scores = _native_scan_route_scores(route_kind_name, src_val, dst_val, source_weight, target_weight, core_weight, packed_bias)
    k = min(max(1, int(topk)), dst_val.shape[1])
    if k == dst_val.shape[1]:
        selected_scores = scores
        selected_indices = _native_scan_full_topk_indices(scores)
    else:
        selected_scores, selected_indices = torch.topk(scores, k, dim=-1)
    routes = _native_scan_signed_abs_softmax(selected_scores)
    weighted_routes = routes * sender_strength.unsqueeze(-1)
    delta_state = torch.zeros(projected_state.shape[0], dst_val.shape[1], device=projected_state.device, dtype=projected_state.dtype)
    flat_indices = selected_indices.reshape(selected_indices.shape[0], -1)
    state_contrib = (weighted_routes * projected_state.unsqueeze(-1)).reshape(projected_state.shape[0], -1)
    delta_state.scatter_add_(1, flat_indices, state_contrib)
    delta_val = torch.zeros(projected_val.shape[0], dst_val.shape[1], projected_val.shape[2], device=projected_val.device, dtype=projected_val.dtype)
    val_contrib = (weighted_routes.unsqueeze(-1) * projected_val.unsqueeze(-2)).reshape(projected_val.shape[0], -1, projected_val.shape[2])
    scatter_index = flat_indices.unsqueeze(-1).expand(flat_indices.shape[0], flat_indices.shape[1], projected_val.shape[2])
    delta_val.scatter_add_(1, scatter_index, val_contrib)
    return delta_state, delta_val


def _native_scan_propagation_topk_signed_abs(
    pairwise_kind: str,
    layer_state: Tensor,
    layer_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    packed_bias: Tensor,
    topk: int,
) -> tuple[Tensor, Tensor]:
    scores = _native_scan_pairwise_scores(pairwise_kind, layer_val, layer_val, source_weight, target_weight, core_weight, packed_bias)
    k = min(max(1, int(topk)), layer_val.shape[1])
    if k == layer_val.shape[1]:
        selected_scores = scores
        selected_indices = _native_scan_full_topk_indices(scores)
    else:
        selected_scores, selected_indices = torch.topk(scores, k, dim=-1)
    edges = _native_scan_signed_abs_softmax(selected_scores)
    selected_state = _native_scan_gather_state(layer_state, selected_indices)
    weighted_edges = edges * selected_state
    selected_val = _native_scan_gather_val(layer_val, selected_indices)
    delta_state = (weighted_edges * selected_state).sum(dim=-1)
    delta_val = (weighted_edges.unsqueeze(-1) * selected_val).sum(dim=-2)
    return delta_state, delta_val


def _native_scan_apply_delta(
    layer_state: Tensor,
    layer_val: Tensor,
    delta_state: Tensor,
    delta_val: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor,
) -> tuple[Tensor, Tensor]:
    next_state = _native_scan_signed_softmax_state(layer_state + delta_state)
    next_val = _native_scan_layer_norm(layer_val + delta_val, val_norm_weight, val_norm_bias)
    return next_state, next_val


def _native_scan_read_memory(
    memory_state: list[tuple[Tensor, Tensor]],
    val_norm_weights: tuple[Tensor, ...],
    val_norm_biases: tuple[Tensor, ...],
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
) -> Tensor:
    read_terms: list[Tensor] = []
    for index, (state, val) in enumerate(memory_state):
        read_val = _native_scan_layer_norm(val, val_norm_weights[index], val_norm_biases[index])
        sender_strength = F.softplus(state).unsqueeze(-1)
        read_summary = (sender_strength * read_val).sum(dim=-2)
        read_summary = read_summary + read_template_val.to(dtype=read_summary.dtype).unsqueeze(0)
        projected = F.linear(read_summary, read_projection_weights[index].to(dtype=read_summary.dtype), None)
        read_terms.append(torch.sigmoid(read_gates[index].to(dtype=read_summary.dtype)) * projected)
    return torch.stack(read_terms, dim=0).sum(dim=0)


def _causal_memory_scan_fused_reference(
    tensor_args: tuple[Tensor, ...],
    num_levels: int,
    write_topks: tuple[int, ...],
    propagation_topks: tuple[int, ...],
    level_transition_topks: tuple[int, ...],
    skip_topks: tuple[int, ...],
    route_kind_name: str,
    propagation_pairwise_kind: str,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    args = _unpack_causal_memory_scan_tensor_args(tensor_args, num_levels)
    aligned_s = args["aligned_s"]
    projected_s = F.linear(aligned_s, args["s_prediction_weight"].to(dtype=aligned_s.dtype), None)
    current_memory = [
        (args["flat_memory"][index * 2], args["flat_memory"][index * 2 + 1])
        for index in range(num_levels)
    ]
    query_steps: list[Tensor] = []

    value_to_state_bias = _load_optional_tensor(args["value_to_state_bias"])
    prediction_input_norm_bias = args["prediction_input_norm_bias"]

    for time_index in range(aligned_s.shape[1]):
        token_val = aligned_s[:, time_index : time_index + 1, :]
        token_state = _native_scan_value_to_state(
            token_val,
            args["value_to_state_weight"],
            args["value_to_state_bias"],
        )
        next_memory: list[tuple[Tensor, Tensor]] = []

        first_normed_val = _native_scan_layer_norm(
            current_memory[0][1],
            args["val_norm_weights"][0],
            args["val_norm_biases"][0],
        )
        first_write_delta = _native_scan_transition_pairwise_topk_signed_abs(
            route_kind_name,
            F.softplus(token_state),
            token_state,
            token_val,
            token_val,
            first_normed_val,
            args["write_source_weights"][0],
            args["write_target_weights"][0],
            args["write_core_weights"][0],
            args["write_biases"][0],
            write_topks[0],
        )
        level_state, level_val = _native_scan_apply_delta(
            current_memory[0][0],
            current_memory[0][1],
            first_write_delta[0],
            first_write_delta[1],
            args["val_norm_weights"][0],
            args["val_norm_biases"][0],
        )
        level_for_prop_val = _native_scan_layer_norm(level_val, args["val_norm_weights"][0], args["val_norm_biases"][0])
        first_prop_delta = _native_scan_propagation_topk_signed_abs(
            propagation_pairwise_kind,
            level_state,
            level_for_prop_val,
            args["propagation_source_weights"][0],
            args["propagation_target_weights"][0],
            args["propagation_core_weights"][0],
            args["propagation_biases"][0],
            propagation_topks[0],
        )
        level_state, level_val = _native_scan_apply_delta(
            level_state,
            level_val,
            first_prop_delta[0],
            first_prop_delta[1],
            args["val_norm_weights"][0],
            args["val_norm_biases"][0],
        )
        next_memory.append((level_state, level_val))

        for level_index in range(1, num_levels):
            current_state, current_val = current_memory[level_index]
            normalized_level_val = _native_scan_layer_norm(
                current_val,
                args["val_norm_weights"][level_index],
                args["val_norm_biases"][level_index],
            )
            normalized_parent_val = _native_scan_layer_norm(
                next_memory[level_index - 1][1],
                args["level_norm_weights"][level_index - 1],
                args["level_norm_biases"][level_index - 1],
            )
            parent_delta = _native_scan_transition_pairwise_topk_signed_abs(
                route_kind_name,
                F.softplus(next_memory[level_index - 1][0]),
                next_memory[level_index - 1][0],
                normalized_parent_val,
                normalized_parent_val,
                normalized_level_val,
                args["level_transition_source_weights"][level_index - 1],
                args["level_transition_target_weights"][level_index - 1],
                args["level_transition_core_weights"][level_index - 1],
                args["level_transition_biases"][level_index - 1],
                level_transition_topks[level_index - 1],
            )
            updated_state, updated_val = _native_scan_apply_delta(
                current_state,
                current_val,
                parent_delta[0],
                parent_delta[1],
                args["val_norm_weights"][level_index],
                args["val_norm_biases"][level_index],
            )

            if level_index == 1 and num_levels > 1:
                skip_gate = torch.sigmoid(args["skip_gates"][0].to(dtype=token_val.dtype))
                skip_delta = _native_scan_transition_pairwise_topk_signed_abs(
                    route_kind_name,
                    F.softplus(token_state),
                    token_state,
                    token_val,
                    token_val,
                    normalized_level_val,
                    args["skip_source_weights"][0],
                    args["skip_target_weights"][0],
                    args["skip_core_weights"][0],
                    args["skip_biases"][0],
                    skip_topks[0],
                )
                updated_state, updated_val = _native_scan_apply_delta(
                    updated_state,
                    updated_val,
                    skip_delta[0] * skip_gate,
                    skip_delta[1] * skip_gate,
                    args["val_norm_weights"][level_index],
                    args["val_norm_biases"][level_index],
                )

            if level_index >= 2:
                skip_index = level_index - 1
                normalized_skip_source_val = _native_scan_layer_norm(
                    next_memory[level_index - 2][1],
                    args["level_norm_weights"][level_index - 2],
                    args["level_norm_biases"][level_index - 2],
                )
                skip_gate = torch.sigmoid(args["skip_gates"][skip_index].to(dtype=normalized_skip_source_val.dtype))
                skip_delta = _native_scan_transition_pairwise_topk_signed_abs(
                    route_kind_name,
                    F.softplus(next_memory[level_index - 2][0]),
                    next_memory[level_index - 2][0],
                    normalized_skip_source_val,
                    normalized_skip_source_val,
                    normalized_level_val,
                    args["skip_source_weights"][skip_index],
                    args["skip_target_weights"][skip_index],
                    args["skip_core_weights"][skip_index],
                    args["skip_biases"][skip_index],
                    skip_topks[skip_index],
                )
                updated_state, updated_val = _native_scan_apply_delta(
                    updated_state,
                    updated_val,
                    skip_delta[0] * skip_gate,
                    skip_delta[1] * skip_gate,
                    args["val_norm_weights"][level_index],
                    args["val_norm_biases"][level_index],
                )

            updated_level_for_prop_val = _native_scan_layer_norm(
                updated_val,
                args["val_norm_weights"][level_index],
                args["val_norm_biases"][level_index],
            )
            prop_delta = _native_scan_propagation_topk_signed_abs(
                propagation_pairwise_kind,
                updated_state,
                updated_level_for_prop_val,
                args["propagation_source_weights"][level_index],
                args["propagation_target_weights"][level_index],
                args["propagation_core_weights"][level_index],
                args["propagation_biases"][level_index],
                propagation_topks[level_index],
            )
            updated_state, updated_val = _native_scan_apply_delta(
                updated_state,
                updated_val,
                prop_delta[0],
                prop_delta[1],
                args["val_norm_weights"][level_index],
                args["val_norm_biases"][level_index],
            )
            next_memory.append((updated_state, updated_val))

        current_memory = next_memory
        read_vector = _native_scan_read_memory(
            current_memory,
            args["val_norm_weights"],
            args["val_norm_biases"],
            args["read_template_val"],
            args["read_projection_weights"],
            args["read_gates"],
        )
        query_input = projected_s[:, time_index, :] + read_vector
        query_steps.append(
            _native_scan_layer_norm(
                query_input,
                args["prediction_input_norm_weight"],
                prediction_input_norm_bias,
            ).unsqueeze(1)
        )

    query_val = torch.cat(query_steps, dim=1) if query_steps else aligned_s.new_empty((aligned_s.shape[0], 0, aligned_s.shape[2]))
    flat_next_memory: list[Tensor] = []
    for state, val in current_memory:
        flat_next_memory.extend((state, val))
    return query_val, tuple(flat_next_memory)


def _causal_memory_scan_fused_native_forward(
    *tensor_args: Tensor,
    num_levels: int,
    write_topks: tuple[int, ...],
    propagation_topks: tuple[int, ...],
    level_transition_topks: tuple[int, ...],
    skip_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    args = _unpack_causal_memory_scan_tensor_args(tensor_args, num_levels)
    if (
        (route_kind_name.startswith("multihead_max_") or propagation_pairwise_kind.startswith("multihead_max_"))
        and not _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind)
    ):
        return _causal_memory_scan_fused_reference(
            tensor_args,
            num_levels=num_levels,
            write_topks=write_topks,
            propagation_topks=propagation_topks,
            level_transition_topks=level_transition_topks,
            skip_topks=skip_topks,
            route_kind_name=route_kind_name,
            propagation_pairwise_kind=propagation_pairwise_kind,
        )
    if _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind):
        result = _native_module().causal_memory_scan_fused(
            args["aligned_s"],
            list(args["flat_memory"]),
            args["value_to_state_weight"],
            _load_optional_tensor(args["value_to_state_bias"]),
            args["s_prediction_weight"],
            args["prediction_input_norm_weight"],
            _load_optional_tensor(args["prediction_input_norm_bias"]),
            args["read_template_val"],
            list(args["read_projection_weights"]),
            list(args["read_gates"]),
            list(args["write_source_weights"]),
            list(args["write_target_weights"]),
            list(args["write_core_weights"]),
            list(args["write_biases"]),
            list(write_topks),
            transition_compress_name,
            list(args["propagation_source_weights"]),
            list(args["propagation_target_weights"]),
            list(args["propagation_core_weights"]),
            list(args["propagation_biases"]),
            list(propagation_topks),
            propagation_compress_name,
            list(args["val_norm_weights"]),
            list(args["val_norm_biases"]),
            list(args["level_transition_source_weights"]),
            list(args["level_transition_target_weights"]),
            list(args["level_transition_core_weights"]),
            list(args["level_transition_biases"]),
            list(level_transition_topks),
            list(args["level_norm_weights"]),
            list(args["level_norm_biases"]),
            list(args["skip_source_weights"]),
            list(args["skip_target_weights"]),
            list(args["skip_core_weights"]),
            list(args["skip_biases"]),
            list(args["skip_gates"]),
            list(skip_topks),
        )
    else:
        result = _native_module().causal_memory_scan_fused(
            args["aligned_s"],
            list(args["flat_memory"]),
            args["value_to_state_weight"],
            _load_optional_tensor(args["value_to_state_bias"]),
            args["s_prediction_weight"],
            args["prediction_input_norm_weight"],
            _load_optional_tensor(args["prediction_input_norm_bias"]),
            args["read_template_val"],
            list(args["read_projection_weights"]),
            list(args["read_gates"]),
            list(args["write_source_weights"]),
            list(args["write_target_weights"]),
            list(args["write_core_weights"]),
            list(args["write_biases"]),
            list(write_topks),
            route_kind_name,
            transition_compress_name,
            list(args["propagation_source_weights"]),
            list(args["propagation_target_weights"]),
            list(args["propagation_core_weights"]),
            list(args["propagation_biases"]),
            list(propagation_topks),
            propagation_pairwise_kind,
            propagation_compress_name,
            list(args["val_norm_weights"]),
            list(args["val_norm_biases"]),
            list(args["level_transition_source_weights"]),
            list(args["level_transition_target_weights"]),
            list(args["level_transition_core_weights"]),
            list(args["level_transition_biases"]),
            list(level_transition_topks),
            list(args["level_norm_weights"]),
            list(args["level_norm_biases"]),
            list(args["skip_source_weights"]),
            list(args["skip_target_weights"]),
            list(args["skip_core_weights"]),
            list(args["skip_biases"]),
            list(args["skip_gates"]),
            list(skip_topks),
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("causal_memory_scan_fused must return (query_val, flat_memory_tensors).")
    query_val, next_memory = result
    if not isinstance(query_val, Tensor):
        raise TypeError("causal_memory_scan_fused query_val must be a Tensor.")
    if not isinstance(next_memory, (list, tuple)):
        raise TypeError("causal_memory_scan_fused flat memory must be a sequence of Tensors.")
    return query_val, tuple(next_memory)


def _causal_memory_scan_fused_native_forward_with_checkpoints(
    *tensor_args: Tensor,
    num_levels: int,
    write_topks: tuple[int, ...],
    propagation_topks: tuple[int, ...],
    level_transition_topks: tuple[int, ...],
    skip_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
    checkpoint_stride: int,
) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
    args = _unpack_causal_memory_scan_tensor_args(tensor_args, num_levels)
    query_val, next_memory, checkpoint_tensors = causal_memory_scan_fused_checkpoints_native(
        checkpoint_stride=checkpoint_stride,
        aligned_s=args["aligned_s"],
        flat_memory=tuple(args["flat_memory"]),
        value_to_state_weight=args["value_to_state_weight"],
        value_to_state_bias=_load_optional_tensor(args["value_to_state_bias"]),
        s_prediction_weight=args["s_prediction_weight"],
        prediction_input_norm_weight=args["prediction_input_norm_weight"],
        prediction_input_norm_bias=_load_optional_tensor(args["prediction_input_norm_bias"]),
        read_template_val=args["read_template_val"],
        read_projection_weights=tuple(args["read_projection_weights"]),
        read_gates=tuple(args["read_gates"]),
        write_source_weights=tuple(args["write_source_weights"]),
        write_target_weights=tuple(args["write_target_weights"]),
        write_core_weights=tuple(args["write_core_weights"]),
        write_biases=tuple(args["write_biases"]),
        write_topks=write_topks,
        route_kind_name=route_kind_name,
        transition_compress_name=transition_compress_name,
        propagation_source_weights=tuple(args["propagation_source_weights"]),
        propagation_target_weights=tuple(args["propagation_target_weights"]),
        propagation_core_weights=tuple(args["propagation_core_weights"]),
        propagation_biases=tuple(args["propagation_biases"]),
        propagation_topks=propagation_topks,
        propagation_pairwise_kind=propagation_pairwise_kind,
        propagation_compress_name=propagation_compress_name,
        val_norm_weights=tuple(args["val_norm_weights"]),
        val_norm_biases=tuple(args["val_norm_biases"]),
        level_transition_source_weights=tuple(args["level_transition_source_weights"]),
        level_transition_target_weights=tuple(args["level_transition_target_weights"]),
        level_transition_core_weights=tuple(args["level_transition_core_weights"]),
        level_transition_biases=tuple(args["level_transition_biases"]),
        level_transition_topks=level_transition_topks,
        level_norm_weights=tuple(args["level_norm_weights"]),
        level_norm_biases=tuple(args["level_norm_biases"]),
        skip_source_weights=tuple(args["skip_source_weights"]),
        skip_target_weights=tuple(args["skip_target_weights"]),
        skip_core_weights=tuple(args["skip_core_weights"]),
        skip_biases=tuple(args["skip_biases"]),
        skip_gates=tuple(args["skip_gates"]),
        skip_topks=skip_topks,
    )
    return query_val, next_memory, checkpoint_tensors


def _repack_causal_memory_scan_chunk_args(
    args: dict[str, Any],
    *,
    aligned_s: Tensor,
    flat_memory: tuple[Tensor, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
) -> tuple[Tensor, ...]:
    tensor_args, _ = _flatten_causal_memory_scan_args(
        aligned_s=aligned_s,
        flat_memory=flat_memory,
        value_to_state_weight=args["value_to_state_weight"],
        value_to_state_bias=_load_optional_tensor(args["value_to_state_bias"]),
        s_prediction_weight=args["s_prediction_weight"],
        prediction_input_norm_weight=args["prediction_input_norm_weight"],
        prediction_input_norm_bias=_load_optional_tensor(args["prediction_input_norm_bias"]),
        read_template_val=args["read_template_val"],
        read_projection_weights=tuple(args["read_projection_weights"]),
        read_gates=tuple(args["read_gates"]),
        write_source_weights=tuple(args["write_source_weights"]),
        write_target_weights=tuple(args["write_target_weights"]),
        write_core_weights=tuple(args["write_core_weights"]),
        write_biases=tuple(args["write_biases"]),
        write_topks=(),
        route_kind_name=route_kind_name,
        transition_compress_name=transition_compress_name,
        propagation_source_weights=tuple(args["propagation_source_weights"]),
        propagation_target_weights=tuple(args["propagation_target_weights"]),
        propagation_core_weights=tuple(args["propagation_core_weights"]),
        propagation_biases=tuple(args["propagation_biases"]),
        propagation_topks=(),
        propagation_pairwise_kind=propagation_pairwise_kind,
        propagation_compress_name=propagation_compress_name,
        val_norm_weights=tuple(args["val_norm_weights"]),
        val_norm_biases=tuple(args["val_norm_biases"]),
        level_transition_source_weights=tuple(args["level_transition_source_weights"]),
        level_transition_target_weights=tuple(args["level_transition_target_weights"]),
        level_transition_core_weights=tuple(args["level_transition_core_weights"]),
        level_transition_biases=tuple(args["level_transition_biases"]),
        level_transition_topks=(),
        level_norm_weights=tuple(args["level_norm_weights"]),
        level_norm_biases=tuple(args["level_norm_biases"]),
        skip_source_weights=tuple(args["skip_source_weights"]),
        skip_target_weights=tuple(args["skip_target_weights"]),
        skip_core_weights=tuple(args["skip_core_weights"]),
        skip_biases=tuple(args["skip_biases"]),
        skip_gates=tuple(args["skip_gates"]),
        skip_topks=(),
    )
    return tensor_args


def _causal_memory_scan_fused_backward_cuda(
    tensor_args: tuple[Tensor, ...],
    *,
    num_levels: int,
    write_topks: tuple[int, ...],
    propagation_topks: tuple[int, ...],
    level_transition_topks: tuple[int, ...],
    skip_topks: tuple[int, ...],
    route_kind_name: str,
    transition_compress_name: str,
    propagation_pairwise_kind: str,
    propagation_compress_name: str,
    trace_tensors: tuple[Tensor, ...] = (),
    grad_query_val: Tensor,
    grad_next_memory: tuple[Tensor, ...],
) -> tuple[Tensor | None, ...]:
    args = _unpack_causal_memory_scan_tensor_args(tensor_args, num_levels)
    with torch.enable_grad():
        result = _native_module().causal_memory_scan_fused_backward_cuda(
        args["aligned_s"],
        list(args["flat_memory"]),
        args["value_to_state_weight"],
        args["value_to_state_bias"],
        args["s_prediction_weight"],
        args["prediction_input_norm_weight"],
        args["prediction_input_norm_bias"],
        args["read_template_val"],
        list(args["read_projection_weights"]),
        list(args["read_gates"]),
        list(args["write_source_weights"]),
        list(args["write_target_weights"]),
        list(args["write_core_weights"]),
        list(args["write_biases"]),
        list(write_topks),
        route_kind_name,
        transition_compress_name,
        list(args["propagation_source_weights"]),
        list(args["propagation_target_weights"]),
        list(args["propagation_core_weights"]),
        list(args["propagation_biases"]),
        list(propagation_topks),
        propagation_pairwise_kind,
        propagation_compress_name,
        list(args["val_norm_weights"]),
        list(args["val_norm_biases"]),
        list(args["level_transition_source_weights"]),
        list(args["level_transition_target_weights"]),
        list(args["level_transition_core_weights"]),
        list(args["level_transition_biases"]),
        list(level_transition_topks),
        list(args["level_norm_weights"]),
        list(args["level_norm_biases"]),
        list(args["skip_source_weights"]),
        list(args["skip_target_weights"]),
        list(args["skip_core_weights"]),
        list(args["skip_biases"]),
        list(args["skip_gates"]),
        list(skip_topks),
        list(trace_tensors),
        grad_query_val,
        list(grad_next_memory),
    )
    if not isinstance(result, (list, tuple)) or len(result) != len(tensor_args):
        raise TypeError("causal_memory_scan_fused_backward_cuda must return one grad per saved tensor.")
    return tuple(None if grad is None else grad for grad in result)


def _accumulate_optional_grad(current: Tensor | None, update: Tensor | None) -> Tensor | None:
    if update is None:
        return current
    if current is None:
        return update
    return current + update


def _chunked_causal_memory_scan_backward(
    ctx: Any,
    tensor_args: tuple[Tensor, ...],
    checkpoint_tensors: tuple[Tensor, ...],
    grad_outputs: tuple[Tensor | None, ...],
) -> tuple[Any, ...]:
    detached_tensors = [tensor.detach().requires_grad_(tensor.requires_grad) for tensor in tensor_args]
    unpacked_args = _unpack_causal_memory_scan_tensor_args(tuple(detached_tensors), ctx.num_levels)
    grad_accum: list[Tensor | None] = [None] * len(detached_tensors)
    grad_query = grad_outputs[0]
    carry_memory_grads = tuple(
        grad if grad is not None else torch.zeros_like(tensor_args[1 + index])
        for index, grad in enumerate(grad_outputs[1: 1 + (2 * ctx.num_levels)])
    )
    num_chunks = checkpoint_tensors[0].shape[0] if checkpoint_tensors else 1
    shared_indices = [0, *range(1 + (2 * ctx.num_levels), len(detached_tensors))]

    for chunk_index in range(num_chunks - 1, -1, -1):
        start = chunk_index * ctx.checkpoint_stride
        end = min(start + ctx.checkpoint_stride, detached_tensors[0].shape[1])
        if chunk_index == 0:
            chunk_memory = tuple(detached_tensors[1: 1 + (2 * ctx.num_levels)])
            chunk_memory_specs = [
                ("orig", 1 + memory_index)
                for memory_index in range(2 * ctx.num_levels)
                if detached_tensors[1 + memory_index].requires_grad
            ]
        else:
            chunk_memory = tuple(
                checkpoint_tensors[memory_index][chunk_index].detach().requires_grad_(True)
                for memory_index in range(2 * ctx.num_levels)
            )
            chunk_memory_specs = [
                ("carry", memory_index) for memory_index in range(2 * ctx.num_levels)
            ]

        chunk_tensor_args = _repack_causal_memory_scan_chunk_args(
            unpacked_args,
            aligned_s=detached_tensors[0][:, start:end, :],
            flat_memory=chunk_memory,
            route_kind_name=ctx.route_kind_name,
            transition_compress_name=ctx.transition_compress_name,
            propagation_pairwise_kind=ctx.propagation_pairwise_kind,
            propagation_compress_name=ctx.propagation_compress_name,
        )

        local_inputs: list[Tensor] = []
        local_specs: list[tuple[str, int]] = []
        for original_index in shared_indices:
            leaf = detached_tensors[original_index]
            if not leaf.requires_grad:
                continue
            local_inputs.append(leaf)
            local_specs.append(("orig", original_index))
        for spec, memory_tensor in zip(chunk_memory_specs, chunk_memory, strict=False):
            local_inputs.append(memory_tensor)
            local_specs.append(spec)

        with torch.enable_grad():
            query_val, next_memory = _causal_memory_scan_fused_native_forward(
                *chunk_tensor_args,
                num_levels=ctx.num_levels,
                write_topks=ctx.write_topks,
                propagation_topks=ctx.propagation_topks,
                level_transition_topks=ctx.level_transition_topks,
                skip_topks=ctx.skip_topks,
                route_kind_name=ctx.route_kind_name,
                transition_compress_name=ctx.transition_compress_name,
                propagation_pairwise_kind=ctx.propagation_pairwise_kind,
                propagation_compress_name=ctx.propagation_compress_name,
            )
            output_tensors = (query_val, *next_memory)
            grad_tensors = [
                (
                    grad_query[:, start:end, :]
                    if grad_query is not None
                    else torch.zeros_like(query_val)
                )
            ]
            grad_tensors.extend(
                grad if grad is not None else torch.zeros_like(output)
                for grad, output in zip(carry_memory_grads, next_memory, strict=False)
            )
            grads = torch.autograd.grad(
                output_tensors,
                local_inputs,
                grad_outputs=grad_tensors,
                allow_unused=True,
            )

        next_carry: list[Tensor | None] = [None] * (2 * ctx.num_levels)
        for spec, grad in zip(local_specs, grads, strict=False):
            if spec[0] == "orig":
                grad_accum[spec[1]] = _accumulate_optional_grad(grad_accum[spec[1]], grad)
            else:
                next_carry[spec[1]] = grad
        if chunk_index > 0:
            carry_memory_grads = tuple(
                grad if grad is not None else torch.zeros_like(chunk_memory[memory_index])
                for memory_index, grad in enumerate(next_carry)
            )

    return (*grad_accum, None, None, None, None, None, None, None, None, None)


class _CausalMemoryScanFusedFunction(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any) -> tuple[Tensor, ...]:
        num_levels = int(args[-9])
        write_topks = tuple(int(v) for v in args[-8])
        propagation_topks = tuple(int(v) for v in args[-7])
        level_transition_topks = tuple(int(v) for v in args[-6])
        skip_topks = tuple(int(v) for v in args[-5])
        route_kind_name = str(args[-4])
        transition_compress_name = str(args[-3])
        propagation_pairwise_kind = str(args[-2])
        propagation_compress_name = str(args[-1])
        tensor_args = tuple(arg for arg in args[:-9])
        checkpoint_stride = _experimental_fused_training_checkpoint_stride(tensor_args[0].shape[1])
        checkpoint_tensors: tuple[Tensor, ...] = ()
        trace_tensors: tuple[Tensor, ...] = ()
        multihead_scan = (
            route_kind_name.startswith("multihead_max_") or propagation_pairwise_kind.startswith("multihead_max_")
        ) and not _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind)
        if not multihead_scan and checkpoint_stride is not None and checkpoint_stride < tensor_args[0].shape[1]:
            query_val, next_memory, checkpoint_tensors = _causal_memory_scan_fused_native_forward_with_checkpoints(
                *tensor_args,
                num_levels=num_levels,
                write_topks=write_topks,
                propagation_topks=propagation_topks,
                level_transition_topks=level_transition_topks,
                skip_topks=skip_topks,
                route_kind_name=route_kind_name,
                transition_compress_name=transition_compress_name,
                propagation_pairwise_kind=propagation_pairwise_kind,
                propagation_compress_name=propagation_compress_name,
                checkpoint_stride=checkpoint_stride,
            )
        elif (
            not multihead_scan
            and _experimental_scan_backward_cuda_enabled()
            and native_supports("causal_memory_scan_fused_backward_cuda")
            and tensor_args[0].is_cuda
        ):
            unpacked = _unpack_causal_memory_scan_tensor_args(tensor_args, num_levels)
            query_val, next_memory, trace_tensors = causal_memory_scan_fused_trace_native(
                aligned_s=unpacked["aligned_s"],
                flat_memory=tuple(unpacked["flat_memory"]),
                value_to_state_weight=unpacked["value_to_state_weight"],
                value_to_state_bias=_load_optional_tensor(unpacked["value_to_state_bias"]),
                s_prediction_weight=unpacked["s_prediction_weight"],
                prediction_input_norm_weight=unpacked["prediction_input_norm_weight"],
                prediction_input_norm_bias=_load_optional_tensor(unpacked["prediction_input_norm_bias"]),
                read_template_val=unpacked["read_template_val"],
                read_projection_weights=tuple(unpacked["read_projection_weights"]),
                read_gates=tuple(unpacked["read_gates"]),
                write_source_weights=tuple(unpacked["write_source_weights"]),
                write_target_weights=tuple(unpacked["write_target_weights"]),
                write_core_weights=tuple(unpacked["write_core_weights"]),
                write_biases=tuple(unpacked["write_biases"]),
                write_topks=write_topks,
                route_kind_name=route_kind_name,
                transition_compress_name=transition_compress_name,
                propagation_source_weights=tuple(unpacked["propagation_source_weights"]),
                propagation_target_weights=tuple(unpacked["propagation_target_weights"]),
                propagation_core_weights=tuple(unpacked["propagation_core_weights"]),
                propagation_biases=tuple(unpacked["propagation_biases"]),
                propagation_topks=propagation_topks,
                propagation_pairwise_kind=propagation_pairwise_kind,
                propagation_compress_name=propagation_compress_name,
                val_norm_weights=tuple(unpacked["val_norm_weights"]),
                val_norm_biases=tuple(unpacked["val_norm_biases"]),
                level_transition_source_weights=tuple(unpacked["level_transition_source_weights"]),
                level_transition_target_weights=tuple(unpacked["level_transition_target_weights"]),
                level_transition_core_weights=tuple(unpacked["level_transition_core_weights"]),
                level_transition_biases=tuple(unpacked["level_transition_biases"]),
                level_transition_topks=level_transition_topks,
                level_norm_weights=tuple(unpacked["level_norm_weights"]),
                level_norm_biases=tuple(unpacked["level_norm_biases"]),
                skip_source_weights=tuple(unpacked["skip_source_weights"]),
                skip_target_weights=tuple(unpacked["skip_target_weights"]),
                skip_core_weights=tuple(unpacked["skip_core_weights"]),
                skip_biases=tuple(unpacked["skip_biases"]),
                skip_gates=tuple(unpacked["skip_gates"]),
                skip_topks=skip_topks,
            )
        else:
            query_val, next_memory = _causal_memory_scan_fused_native_forward(
                *tensor_args,
                num_levels=num_levels,
                write_topks=write_topks,
                propagation_topks=propagation_topks,
                level_transition_topks=level_transition_topks,
                skip_topks=skip_topks,
                route_kind_name=route_kind_name,
                transition_compress_name=transition_compress_name,
                propagation_pairwise_kind=propagation_pairwise_kind,
                propagation_compress_name=propagation_compress_name,
            )
        ctx.num_levels = num_levels
        ctx.write_topks = write_topks
        ctx.propagation_topks = propagation_topks
        ctx.level_transition_topks = level_transition_topks
        ctx.skip_topks = skip_topks
        ctx.route_kind_name = route_kind_name
        ctx.transition_compress_name = transition_compress_name
        ctx.propagation_pairwise_kind = propagation_pairwise_kind
        ctx.propagation_compress_name = propagation_compress_name
        ctx.tensor_arg_count = len(tensor_args)
        ctx.checkpoint_tensor_count = len(checkpoint_tensors)
        ctx.trace_tensor_count = len(trace_tensors)
        ctx.checkpoint_stride = checkpoint_stride or 0
        ctx.save_for_backward(*tensor_args, *checkpoint_tensors, *trace_tensors)
        return (query_val, *next_memory)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor | None) -> tuple[Any, ...]:
        saved_tensors = tuple(ctx.saved_tensors)
        tensor_args = saved_tensors[: ctx.tensor_arg_count]
        checkpoint_end = ctx.tensor_arg_count + ctx.checkpoint_tensor_count
        checkpoint_tensors = saved_tensors[ctx.tensor_arg_count: checkpoint_end]
        trace_tensors = saved_tensors[checkpoint_end : checkpoint_end + ctx.trace_tensor_count]
        stream_capturing = False
        if tensor_args and isinstance(tensor_args[0], torch.Tensor) and tensor_args[0].is_cuda:
            try:
                stream_capturing = bool(torch.cuda.is_current_stream_capturing())
            except Exception:
                stream_capturing = False
        if checkpoint_tensors and ctx.checkpoint_stride > 0:
            return _chunked_causal_memory_scan_backward(
                ctx,
                tensor_args,
                checkpoint_tensors,
                grad_outputs,
            )

        if (
            _experimental_scan_backward_cuda_enabled()
            and native_supports("causal_memory_scan_fused_backward_cuda")
            and tensor_args[0].is_cuda
            and not stream_capturing
        ):
            if grad_outputs[0] is None:
                with torch.no_grad():
                    zero_query, _ = _causal_memory_scan_fused_native_forward(
                        *tensor_args,
                        num_levels=ctx.num_levels,
                        write_topks=ctx.write_topks,
                        propagation_topks=ctx.propagation_topks,
                        level_transition_topks=ctx.level_transition_topks,
                        skip_topks=ctx.skip_topks,
                        route_kind_name=ctx.route_kind_name,
                        transition_compress_name=ctx.transition_compress_name,
                        propagation_pairwise_kind=ctx.propagation_pairwise_kind,
                        propagation_compress_name=ctx.propagation_compress_name,
                    )
                grad_query_val = torch.zeros_like(zero_query)
            else:
                grad_query_val = grad_outputs[0]
            grad_next_memory = tuple(
                grad if grad is not None else torch.zeros_like(tensor)
                for grad, tensor in zip(
                    grad_outputs[1 : 1 + (2 * ctx.num_levels)],
                    tensor_args[1 : 1 + (2 * ctx.num_levels)],
                    strict=False,
                )
            )
            tensor_grads = _causal_memory_scan_fused_backward_cuda(
                tensor_args,
                num_levels=ctx.num_levels,
                write_topks=ctx.write_topks,
                propagation_topks=ctx.propagation_topks,
                level_transition_topks=ctx.level_transition_topks,
                skip_topks=ctx.skip_topks,
                route_kind_name=ctx.route_kind_name,
                transition_compress_name=ctx.transition_compress_name,
                propagation_pairwise_kind=ctx.propagation_pairwise_kind,
                propagation_compress_name=ctx.propagation_compress_name,
                trace_tensors=tuple(trace_tensors),
                grad_query_val=grad_query_val,
                grad_next_memory=grad_next_memory,
            )
            return (*tensor_grads, None, None, None, None, None, None, None, None, None)

        detached_tensors: list[Tensor] = []
        grad_inputs: list[Tensor] = []
        grad_index_map: list[int | None] = []
        for tensor in tensor_args:
            leaf = tensor.detach().requires_grad_(tensor.requires_grad)
            detached_tensors.append(leaf)
            if leaf.requires_grad:
                grad_index_map.append(len(grad_inputs))
                grad_inputs.append(leaf)
            else:
                grad_index_map.append(None)

        with torch.enable_grad():
            query_val, next_memory = _causal_memory_scan_fused_native_forward(
                *tuple(detached_tensors),
                num_levels=ctx.num_levels,
                write_topks=ctx.write_topks,
                propagation_topks=ctx.propagation_topks,
                level_transition_topks=ctx.level_transition_topks,
                skip_topks=ctx.skip_topks,
                route_kind_name=ctx.route_kind_name,
                transition_compress_name=ctx.transition_compress_name,
                propagation_pairwise_kind=ctx.propagation_pairwise_kind,
                propagation_compress_name=ctx.propagation_compress_name,
            )
            outputs = (query_val, *next_memory)
            grad_tensors = [
                grad if grad is not None else torch.zeros_like(output)
                for grad, output in zip(grad_outputs, outputs, strict=False)
            ]
            grads = torch.autograd.grad(
                outputs,
                grad_inputs,
                grad_outputs=grad_tensors,
                allow_unused=True,
            )

        tensor_grads: list[Tensor | None] = []
        for index in grad_index_map:
            tensor_grads.append(None if index is None else grads[index])
        return (*tensor_grads, None, None, None, None, None, None, None, None, None)



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


def _signed_abs_softmax_from_scores(scores: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    return torch.sign(clean_scores) * torch.softmax(clean_scores.abs(), dim=-1)


def _signed_abs_softmax_backward(scores: Tensor, grad_routes: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    signs = torch.sign(clean_scores)
    probs = torch.softmax(clean_scores.abs(), dim=-1)
    signed_routes = signs * probs
    dot = (grad_routes * signed_routes).sum(dim=-1, keepdim=True)
    return signs * probs * (signs * grad_routes - dot)


def _packed_true_mask(scores: Tensor) -> Tensor:
    return torch.ones_like(scores, dtype=torch.bool)


def _pairwise_topk_compress_kind(route_compress_name: str) -> int | None:
    if route_compress_name == "softmax":
        return 0
    if route_compress_name == "signed_abs_softmax":
        return 1
    if route_compress_name == "signed_entmax15":
        return 2
    return None


def _propagation_topk_compress_kind(edge_compress_name: str) -> int | None:
    if edge_compress_name == "softsign":
        return 0
    if edge_compress_name == "signed_abs_softmax":
        return 1
    if edge_compress_name == "signed_entmax15":
        return 2
    return None


def _pairwise_routes_from_scores(scores: Tensor, compress_kind: int) -> Tensor:
    if compress_kind == 1:
        return _signed_abs_softmax_from_scores(scores)
    if compress_kind == 2:
        return torch.ops.jakal_net.signed_entmax15(scores, _packed_true_mask(scores))
    return torch.softmax(scores, dim=-1)


def _pairwise_routes_backward(scores: Tensor, routes: Tensor, grad_routes: Tensor, compress_kind: int) -> Tensor:
    if compress_kind == 1:
        return _signed_abs_softmax_backward(scores, grad_routes)
    if compress_kind == 2:
        return torch.ops.jakal_net.signed_entmax15_backward(
            scores.contiguous(),
            routes.contiguous(),
            grad_routes.contiguous(),
            _packed_true_mask(scores),
        )
    return _native_module().softmax_backward_cuda(
        routes.contiguous(),
        grad_routes.contiguous(),
    )


def _propagation_edges_from_scores(scores: Tensor, compress_kind: int) -> Tensor:
    if compress_kind == 1:
        return _signed_abs_softmax_from_scores(scores)
    if compress_kind == 2:
        return torch.ops.jakal_net.signed_entmax15(scores, _packed_true_mask(scores))
    return torch.nn.functional.softsign(scores)


def _propagation_edges_backward(scores: Tensor, edges: Tensor, grad_edges: Tensor, compress_kind: int) -> Tensor:
    if compress_kind == 1:
        return _signed_abs_softmax_backward(scores, grad_edges)
    if compress_kind == 2:
        return torch.ops.jakal_net.signed_entmax15_backward(
            scores.contiguous(),
            edges.contiguous(),
            grad_edges.contiguous(),
            _packed_true_mask(scores),
        )
    return _native_module().softsign_backward_cuda(
        scores.contiguous(),
        grad_edges.contiguous(),
    )


def _flatten_dense_tensors(
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...], int, int]:
    batch_shape = tuple(layer_val.shape[:-2])
    nodes = layer_val.shape[-2]
    out_dim = projected_val.shape[-1]
    return (
        layer_val.reshape(-1, nodes, layer_val.shape[-1]).contiguous(),
        projected_state.reshape(-1, nodes).contiguous(),
        projected_val.reshape(-1, nodes, out_dim).contiguous(),
        batch_shape,
        nodes,
        out_dim,
    )


def _flatten_pairwise_transition_tensors(
    sender_strength: Tensor,
    src_val: Tensor,
    dst_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[int, ...], int, int, int]:
    batch_shape = tuple(src_val.shape[:-2])
    src_nodes = src_val.shape[-2]
    dst_nodes = dst_val.shape[-2]
    out_dim = projected_val.shape[-1]
    return (
        sender_strength.reshape(-1, src_nodes).contiguous(),
        src_val.reshape(-1, src_nodes, src_val.shape[-1]).contiguous(),
        dst_val.reshape(-1, dst_nodes, dst_val.shape[-1]).contiguous(),
        projected_state.reshape(-1, src_nodes).contiguous(),
        projected_val.reshape(-1, src_nodes, out_dim).contiguous(),
        batch_shape,
        src_nodes,
        dst_nodes,
        out_dim,
    )


def _save_optional_tensor(tensor: Tensor | None, reference: Tensor) -> Tensor:
    if tensor is not None:
        return tensor
    return torch.empty(0, device=reference.device, dtype=reference.dtype)


def _load_optional_tensor(tensor: Tensor) -> Tensor | None:
    return None if tensor.numel() == 0 else tensor


def _accumulator_dtype_for(tensor: Tensor) -> torch.dtype:
    if tensor.dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return tensor.dtype


def _gather_sequence_rows(source: Tensor, indices: Tensor) -> Tensor:
    batch = torch.arange(source.shape[0], device=source.device).view(-1, 1, 1)
    return source[batch, indices]


def _window_source_indices(
    *,
    target_nodes: int,
    source_nodes: int,
    window: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    width = min(window + 1, source_nodes)
    base = torch.arange(target_nodes, device=device).unsqueeze(-1)
    offsets = torch.arange(width, device=device)
    indices = base - (width - 1 - offsets)
    valid = indices >= 0
    return indices.clamp(min=0, max=source_nodes - 1).to(torch.long), valid


def _pairwise_transition_reduce_backward(
    routes: Tensor,
    indices: Tensor,
    weighted_state: Tensor,
    weighted_val: Tensor,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    selected_grad_state = _gather_sequence_rows(grad_delta_state, indices)
    selected_grad_val = _gather_sequence_rows(grad_delta_val, indices)
    grad_routes = selected_grad_state * weighted_state.unsqueeze(-1)
    grad_routes = grad_routes + (
        selected_grad_val * weighted_val.unsqueeze(-2)
    ).sum(dim=-1)
    grad_weighted_state = (routes * selected_grad_state).sum(dim=-1)
    grad_weighted_val = (routes.unsqueeze(-1) * selected_grad_val).sum(dim=-2)
    return grad_routes, grad_weighted_state, grad_weighted_val


def _hadamard_pairwise_scores(
    target_val: Tensor,
    source_val: Tensor,
    in_weight: Tensor,
    in_bias: Tensor | None,
    out_weight: Tensor,
    out_bias: Tensor | None,
) -> Tensor:
    hidden = torch.einsum("bid,hd,bjd->bijh", target_val, in_weight, source_val)
    if in_bias is not None:
        hidden = hidden + in_bias.view(1, 1, 1, -1)
    hidden = F.silu(hidden)
    scores = torch.matmul(hidden, out_weight.transpose(0, 1)).squeeze(-1)
    if out_bias is not None:
        scores = scores + out_bias
    return scores


def _hadamard_route_logits(
    src_val: Tensor,
    dst_val: Tensor,
    source_weight: Tensor,
    source_bias: Tensor | None,
    target_weight: Tensor,
    target_bias: Tensor | None,
    hidden_weight: Tensor,
    hidden_bias: Tensor | None,
    out_weight: Tensor,
    out_bias: Tensor | None,
    bias: Tensor | None,
    temperature: float,
) -> Tensor:
    projected_source = F.linear(src_val, source_weight, source_bias)
    projected_target = F.linear(dst_val, target_weight, target_bias)
    width = projected_source.shape[-1]
    source_linear_weight, target_linear_weight, hadamard_weight = torch.split(
        hidden_weight,
        width,
        dim=-1,
    )
    hidden = torch.einsum(
        "bid,hd,bkd->bikh",
        projected_source,
        hadamard_weight,
        projected_target,
    )
    hidden = hidden + F.linear(projected_source, source_linear_weight).unsqueeze(-2)
    hidden = hidden + F.linear(
        projected_target,
        target_linear_weight,
        hidden_bias,
    ).unsqueeze(-3)
    hidden = F.silu(hidden)
    logits = torch.matmul(hidden, out_weight.transpose(0, 1)).squeeze(-1)
    if out_bias is not None:
        logits = logits + out_bias
    if bias is not None:
        logits = logits + bias
    if temperature != 1.0:
        logits = logits / temperature
    return logits


def _hadamard_route_dense_chunk(
    sender_strength: Tensor,
    src_val: Tensor,
    dst_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    source_bias: Tensor | None,
    target_weight: Tensor,
    target_bias: Tensor | None,
    hidden_weight: Tensor,
    hidden_bias: Tensor | None,
    out_weight: Tensor,
    out_bias: Tensor | None,
    bias: Tensor | None,
    temperature: float,
    dst_block_size: int,
) -> tuple[Tensor, Tensor]:
    batch_flat = src_val.shape[0]
    dst_nodes = dst_val.shape[1]
    out_dim = projected_val.shape[-1]
    state_acc_dtype = _accumulator_dtype_for(projected_state)
    val_acc_dtype = _accumulator_dtype_for(projected_val)
    dst_step = dst_nodes if dst_block_size <= 0 else min(dst_block_size, dst_nodes)

    running_max: Tensor | None = None
    running_sum: Tensor | None = None
    for dst_start in range(0, dst_nodes, dst_step):
        dst_end = min(dst_start + dst_step, dst_nodes)
        logits = _hadamard_route_logits(
            src_val,
            dst_val[:, dst_start:dst_end, :],
            source_weight,
            source_bias,
            target_weight,
            target_bias,
            hidden_weight,
            hidden_bias,
            out_weight,
            out_bias,
            bias,
            temperature,
        )
        block_max = logits.amax(dim=-1)
        block_exp = torch.exp(logits - block_max.unsqueeze(-1))
        block_sum = block_exp.sum(dim=-1)
        if running_max is None or running_sum is None:
            running_max = block_max
            running_sum = block_sum
            continue
        next_max = torch.maximum(running_max, block_max)
        running_sum = running_sum * torch.exp(running_max - next_max) + block_sum * torch.exp(
            block_max - next_max
        )
        running_max = next_max

    if running_max is None or running_sum is None:
        raise RuntimeError("Dense hadamard route chunk requires at least one destination block.")

    state_sender = sender_strength.to(dtype=state_acc_dtype) * projected_state.to(
        dtype=state_acc_dtype
    )
    val_sender = sender_strength.to(dtype=val_acc_dtype).unsqueeze(-1) * projected_val.to(
        dtype=val_acc_dtype
    )
    state_blocks: list[Tensor] = []
    val_blocks: list[Tensor] = []

    for dst_start in range(0, dst_nodes, dst_step):
        dst_end = min(dst_start + dst_step, dst_nodes)
        logits = _hadamard_route_logits(
            src_val,
            dst_val[:, dst_start:dst_end, :],
            source_weight,
            source_bias,
            target_weight,
            target_bias,
            hidden_weight,
            hidden_bias,
            out_weight,
            out_bias,
            bias,
            temperature,
        )
        routes = torch.exp(logits - running_max.unsqueeze(-1)) / running_sum.unsqueeze(-1)
        transport = routes.transpose(1, 2).contiguous()
        state_blocks.append(
            torch.bmm(
                transport.to(dtype=state_acc_dtype),
                state_sender.unsqueeze(-1),
            ).squeeze(-1)
        )
        val_blocks.append(
            torch.bmm(
                transport.to(dtype=val_acc_dtype),
                val_sender,
            )
        )

    return torch.cat(state_blocks, dim=1), torch.cat(val_blocks, dim=1)


class _HadamardPropagationDense(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        in_weight: Tensor,
        in_bias: Tensor,
        out_weight: Tensor,
        out_bias: Tensor,
        target_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().propagation_dense(
            "hadamard_mlp",
            in_weight,
            _load_optional_tensor(out_bias),
            in_weight,
            _load_optional_tensor(in_bias),
            out_weight,
            _load_optional_tensor(out_bias),
            "softsign",
            layer_val,
            projected_state,
            projected_val,
            target_block_size,
            source_block_size,
        )
        ctx.target_block_size = int(target_block_size)
        ctx.source_block_size = int(source_block_size)
        ctx.save_for_backward(
            layer_val,
            projected_state,
            projected_val,
            in_weight,
            in_bias,
            out_weight,
            out_bias,
        )
        return result

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            layer_val,
            projected_state,
            projected_val,
            in_weight,
            in_bias_tensor,
            out_weight,
            out_bias_tensor,
        ) = ctx.saved_tensors
        in_bias = _load_optional_tensor(in_bias_tensor)
        out_bias = _load_optional_tensor(out_bias_tensor)
        flat_val, flat_projected_state, flat_projected_val, _batch_shape, num_nodes, _out_dim = (
            _flatten_dense_tensors(layer_val, projected_state, projected_val)
        )
        flat_grad_state = grad_delta_state.reshape(-1, num_nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, num_nodes, grad_delta_val.shape[-1]).contiguous()

        grad_flat_val = torch.zeros_like(flat_val)
        grad_flat_projected_state = torch.zeros_like(flat_projected_state)
        grad_flat_projected_val = torch.zeros_like(flat_projected_val)
        grad_in_weight = torch.zeros_like(in_weight)
        grad_in_bias = torch.zeros_like(in_bias) if in_bias is not None else None
        grad_out_weight = torch.zeros_like(out_weight)
        grad_out_bias = torch.zeros_like(out_bias) if out_bias is not None else None

        target_step = num_nodes if ctx.target_block_size <= 0 else min(ctx.target_block_size, num_nodes)
        source_step = num_nodes if ctx.source_block_size <= 0 else min(ctx.source_block_size, num_nodes)

        for target_start in range(0, num_nodes, target_step):
            target_end = min(target_start + target_step, num_nodes)
            grad_state_block = flat_grad_state[:, target_start:target_end]
            grad_val_block = flat_grad_val[:, target_start:target_end, :]
            for source_start in range(0, num_nodes, source_step):
                source_end = min(source_start + source_step, num_nodes)
                with torch.enable_grad():
                    target_block = (
                        flat_val[:, target_start:target_end, :].detach().requires_grad_(True)
                    )
                    source_block = (
                        flat_val[:, source_start:source_end, :].detach().requires_grad_(True)
                    )
                    source_state = (
                        flat_projected_state[:, source_start:source_end]
                        .detach()
                        .requires_grad_(True)
                    )
                    source_proj_val = (
                        flat_projected_val[:, source_start:source_end, :]
                        .detach()
                        .requires_grad_(True)
                    )
                    in_weight_leaf = in_weight.detach().requires_grad_(True)
                    inputs: list[Tensor] = [
                        target_block,
                        source_block,
                        source_state,
                        source_proj_val,
                        in_weight_leaf,
                    ]
                    in_bias_leaf: Tensor | None = None
                    if in_bias is not None:
                        in_bias_leaf = in_bias.detach().requires_grad_(True)
                        inputs.append(in_bias_leaf)
                    out_weight_leaf = out_weight.detach().requires_grad_(True)
                    inputs.append(out_weight_leaf)
                    out_bias_leaf: Tensor | None = None
                    if out_bias is not None:
                        out_bias_leaf = out_bias.detach().requires_grad_(True)
                        inputs.append(out_bias_leaf)

                    scores = _hadamard_pairwise_scores(
                        target_block,
                        source_block,
                        in_weight_leaf,
                        in_bias_leaf,
                        out_weight_leaf,
                        out_bias_leaf,
                    )
                    edges = F.softsign(scores)
                    local_delta_state = torch.bmm(
                        edges.to(dtype=source_state.dtype),
                        source_state.unsqueeze(-1),
                    ).squeeze(-1)
                    local_delta_val = torch.bmm(
                        edges.to(dtype=source_proj_val.dtype),
                        source_proj_val,
                    )
                    grads = torch.autograd.grad(
                        (local_delta_state, local_delta_val),
                        inputs,
                        grad_outputs=(
                            grad_state_block.to(dtype=local_delta_state.dtype),
                            grad_val_block.to(dtype=local_delta_val.dtype),
                        ),
                        allow_unused=True,
                    )

                grad_target, grad_source, grad_source_state, grad_source_proj_val, grad_in_weight_block = (
                    grads[:5]
                )
                next_index = 5
                grad_in_bias_block = None
                if in_bias is not None:
                    grad_in_bias_block = grads[next_index]
                    next_index += 1
                grad_out_weight_block = grads[next_index]
                next_index += 1
                grad_out_bias_block = grads[next_index] if out_bias is not None else None

                if grad_target is not None:
                    grad_flat_val[:, target_start:target_end, :] += grad_target
                if grad_source is not None:
                    grad_flat_val[:, source_start:source_end, :] += grad_source
                if grad_source_state is not None:
                    grad_flat_projected_state[:, source_start:source_end] += grad_source_state
                if grad_source_proj_val is not None:
                    grad_flat_projected_val[:, source_start:source_end, :] += grad_source_proj_val
                if grad_in_weight_block is not None:
                    grad_in_weight += grad_in_weight_block
                if grad_in_bias is not None and grad_in_bias_block is not None:
                    grad_in_bias += grad_in_bias_block
                if grad_out_weight_block is not None:
                    grad_out_weight += grad_out_weight_block
                if grad_out_bias is not None and grad_out_bias_block is not None:
                    grad_out_bias += grad_out_bias_block

        return (
            grad_flat_val.reshape_as(layer_val),
            grad_flat_projected_state.reshape_as(projected_state),
            grad_flat_projected_val.reshape_as(projected_val),
            grad_in_weight,
            grad_in_bias,
            grad_out_weight,
            grad_out_bias,
            None,
            None,
        )


class _HadamardPropagationQueryDense(Function):
    @staticmethod
    def forward(
        ctx: Any,
        query_val: Tensor,
        source_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        in_weight: Tensor,
        in_bias: Tensor,
        out_weight: Tensor,
        out_bias: Tensor,
        query_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().propagation_query_dense(
            "hadamard_mlp",
            in_weight,
            _load_optional_tensor(out_bias),
            in_weight,
            _load_optional_tensor(in_bias),
            out_weight,
            _load_optional_tensor(out_bias),
            "softsign",
            query_val,
            source_val,
            projected_state,
            projected_val,
            query_block_size,
            source_block_size,
        )
        ctx.query_block_size = int(query_block_size)
        ctx.source_block_size = int(source_block_size)
        ctx.save_for_backward(
            query_val,
            source_val,
            projected_state,
            projected_val,
            in_weight,
            in_bias,
            out_weight,
            out_bias,
        )
        return result

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            query_val,
            source_val,
            projected_state,
            projected_val,
            in_weight,
            in_bias_tensor,
            out_weight,
            out_bias_tensor,
        ) = ctx.saved_tensors
        in_bias = _load_optional_tensor(in_bias_tensor)
        out_bias = _load_optional_tensor(out_bias_tensor)
        (
            flat_query,
            flat_source,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            query_nodes,
            source_nodes,
            _out_dim,
        ) = _flatten_query_tensors(query_val, source_val, projected_state, projected_val)
        flat_grad_state = grad_delta_state.reshape(-1, query_nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, query_nodes, grad_delta_val.shape[-1]).contiguous()

        grad_flat_query = torch.zeros_like(flat_query)
        grad_flat_source = torch.zeros_like(flat_source)
        grad_flat_projected_state = torch.zeros_like(flat_projected_state)
        grad_flat_projected_val = torch.zeros_like(flat_projected_val)
        grad_in_weight = torch.zeros_like(in_weight)
        grad_in_bias = torch.zeros_like(in_bias) if in_bias is not None else None
        grad_out_weight = torch.zeros_like(out_weight)
        grad_out_bias = torch.zeros_like(out_bias) if out_bias is not None else None

        query_step = (
            query_nodes if ctx.query_block_size <= 0 else min(ctx.query_block_size, query_nodes)
        )
        source_step = (
            source_nodes if ctx.source_block_size <= 0 else min(ctx.source_block_size, source_nodes)
        )

        for query_start in range(0, query_nodes, query_step):
            query_end = min(query_start + query_step, query_nodes)
            grad_state_block = flat_grad_state[:, query_start:query_end]
            grad_val_block = flat_grad_val[:, query_start:query_end, :]
            for source_start in range(0, source_nodes, source_step):
                source_end = min(source_start + source_step, source_nodes)
                with torch.enable_grad():
                    query_block = (
                        flat_query[:, query_start:query_end, :].detach().requires_grad_(True)
                    )
                    source_block = (
                        flat_source[:, source_start:source_end, :].detach().requires_grad_(True)
                    )
                    source_state = (
                        flat_projected_state[:, source_start:source_end]
                        .detach()
                        .requires_grad_(True)
                    )
                    source_proj_val = (
                        flat_projected_val[:, source_start:source_end, :]
                        .detach()
                        .requires_grad_(True)
                    )
                    in_weight_leaf = in_weight.detach().requires_grad_(True)
                    inputs: list[Tensor] = [
                        query_block,
                        source_block,
                        source_state,
                        source_proj_val,
                        in_weight_leaf,
                    ]
                    in_bias_leaf: Tensor | None = None
                    if in_bias is not None:
                        in_bias_leaf = in_bias.detach().requires_grad_(True)
                        inputs.append(in_bias_leaf)
                    out_weight_leaf = out_weight.detach().requires_grad_(True)
                    inputs.append(out_weight_leaf)
                    out_bias_leaf: Tensor | None = None
                    if out_bias is not None:
                        out_bias_leaf = out_bias.detach().requires_grad_(True)
                        inputs.append(out_bias_leaf)

                    scores = _hadamard_pairwise_scores(
                        query_block,
                        source_block,
                        in_weight_leaf,
                        in_bias_leaf,
                        out_weight_leaf,
                        out_bias_leaf,
                    )
                    edges = F.softsign(scores)
                    local_delta_state = torch.bmm(
                        edges.to(dtype=source_state.dtype),
                        source_state.unsqueeze(-1),
                    ).squeeze(-1)
                    local_delta_val = torch.bmm(
                        edges.to(dtype=source_proj_val.dtype),
                        source_proj_val,
                    )
                    grads = torch.autograd.grad(
                        (local_delta_state, local_delta_val),
                        inputs,
                        grad_outputs=(
                            grad_state_block.to(dtype=local_delta_state.dtype),
                            grad_val_block.to(dtype=local_delta_val.dtype),
                        ),
                        allow_unused=True,
                    )

                grad_query_block, grad_source_block, grad_source_state, grad_source_proj_val, grad_in_weight_block = (
                    grads[:5]
                )
                next_index = 5
                grad_in_bias_block = None
                if in_bias is not None:
                    grad_in_bias_block = grads[next_index]
                    next_index += 1
                grad_out_weight_block = grads[next_index]
                next_index += 1
                grad_out_bias_block = grads[next_index] if out_bias is not None else None

                if grad_query_block is not None:
                    grad_flat_query[:, query_start:query_end, :] += grad_query_block
                if grad_source_block is not None:
                    grad_flat_source[:, source_start:source_end, :] += grad_source_block
                if grad_source_state is not None:
                    grad_flat_projected_state[:, source_start:source_end] += grad_source_state
                if grad_source_proj_val is not None:
                    grad_flat_projected_val[:, source_start:source_end, :] += grad_source_proj_val
                if grad_in_weight_block is not None:
                    grad_in_weight += grad_in_weight_block
                if grad_in_bias is not None and grad_in_bias_block is not None:
                    grad_in_bias += grad_in_bias_block
                if grad_out_weight_block is not None:
                    grad_out_weight += grad_out_weight_block
                if grad_out_bias is not None and grad_out_bias_block is not None:
                    grad_out_bias += grad_out_bias_block

        return (
            grad_flat_query.reshape_as(query_val),
            grad_flat_source.reshape_as(source_val),
            grad_flat_projected_state.reshape_as(projected_state),
            grad_flat_projected_val.reshape_as(projected_val),
            grad_in_weight,
            grad_in_bias,
            grad_out_weight,
            grad_out_bias,
            None,
            None,
        )


class _HadamardTransitionPairwiseDense(Function):
    @staticmethod
    def forward(
        ctx: Any,
        sender_strength: Tensor,
        src_val: Tensor,
        dst_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        source_weight: Tensor,
        source_bias: Tensor,
        target_weight: Tensor,
        target_bias: Tensor,
        core_weight: Tensor,
        bias: Tensor,
        hidden_weight: Tensor,
        hidden_bias: Tensor,
        out_weight: Tensor,
        out_bias: Tensor,
        temperature: float,
        src_block_size: int,
        dst_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        result = _native_module().transition_pairwise_dense(
            "source_target_hadamard_mlp_route",
            source_weight,
            _load_optional_tensor(source_bias),
            target_weight,
            _load_optional_tensor(target_bias),
            core_weight,
            _load_optional_tensor(bias),
            hidden_weight,
            _load_optional_tensor(hidden_bias),
            out_weight,
            _load_optional_tensor(out_bias),
            float(temperature),
            route_compress_name,
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            src_block_size,
            dst_block_size,
        )
        ctx.temperature = float(temperature)
        ctx.src_block_size = int(src_block_size)
        ctx.dst_block_size = int(dst_block_size)
        ctx.save_for_backward(
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            source_weight,
            source_bias,
            target_weight,
            target_bias,
            core_weight,
            bias,
            hidden_weight,
            hidden_bias,
            out_weight,
            out_bias,
        )
        return result

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            source_weight,
            source_bias_tensor,
            target_weight,
            target_bias_tensor,
            core_weight,
            bias_tensor,
            hidden_weight,
            hidden_bias_tensor,
            out_weight,
            out_bias_tensor,
        ) = ctx.saved_tensors
        source_bias = _load_optional_tensor(source_bias_tensor)
        target_bias = _load_optional_tensor(target_bias_tensor)
        bias = _load_optional_tensor(bias_tensor)
        hidden_bias = _load_optional_tensor(hidden_bias_tensor)
        out_bias = _load_optional_tensor(out_bias_tensor)
        flat_sender = sender_strength.reshape(-1, sender_strength.shape[-1]).contiguous()
        flat_src = src_val.reshape(-1, src_val.shape[-2], src_val.shape[-1]).contiguous()
        flat_dst = dst_val.reshape(-1, dst_val.shape[-2], dst_val.shape[-1]).contiguous()
        flat_projected_state = projected_state.reshape(-1, projected_state.shape[-1]).contiguous()
        flat_projected_val = projected_val.reshape(
            -1,
            projected_val.shape[-2],
            projected_val.shape[-1],
        ).contiguous()
        batch_flat = flat_src.shape[0]
        src_nodes = flat_src.shape[1]
        flat_grad_state = grad_delta_state.reshape(batch_flat, grad_delta_state.shape[-1]).contiguous()
        flat_grad_val = grad_delta_val.reshape(
            batch_flat,
            grad_delta_val.shape[-2],
            grad_delta_val.shape[-1],
        ).contiguous()

        grad_sender = torch.zeros_like(flat_sender)
        grad_src = torch.zeros_like(flat_src)
        grad_dst = torch.zeros_like(flat_dst)
        grad_projected_state = torch.zeros_like(flat_projected_state)
        grad_projected_val = torch.zeros_like(flat_projected_val)
        grad_source_weight = torch.zeros_like(source_weight)
        grad_source_bias = torch.zeros_like(source_bias) if source_bias is not None else None
        grad_target_weight = torch.zeros_like(target_weight)
        grad_target_bias = torch.zeros_like(target_bias) if target_bias is not None else None
        grad_bias = torch.zeros_like(bias) if bias is not None else None
        grad_hidden_weight = torch.zeros_like(hidden_weight)
        grad_hidden_bias = torch.zeros_like(hidden_bias) if hidden_bias is not None else None
        grad_out_weight = torch.zeros_like(out_weight)
        grad_out_bias = torch.zeros_like(out_bias) if out_bias is not None else None

        src_step = src_nodes if ctx.src_block_size <= 0 else min(ctx.src_block_size, src_nodes)
        for src_start in range(0, src_nodes, src_step):
            src_end = min(src_start + src_step, src_nodes)
            with torch.enable_grad():
                sender_block = flat_sender[:, src_start:src_end].detach().requires_grad_(True)
                src_block = flat_src[:, src_start:src_end, :].detach().requires_grad_(True)
                dst_full = flat_dst.detach().requires_grad_(True)
                projected_state_block = (
                    flat_projected_state[:, src_start:src_end].detach().requires_grad_(True)
                )
                projected_val_block = (
                    flat_projected_val[:, src_start:src_end, :].detach().requires_grad_(True)
                )
                source_weight_leaf = source_weight.detach().requires_grad_(True)
                inputs: list[Tensor] = [
                    sender_block,
                    src_block,
                    dst_full,
                    projected_state_block,
                    projected_val_block,
                    source_weight_leaf,
                ]
                source_bias_leaf: Tensor | None = None
                if source_bias is not None:
                    source_bias_leaf = source_bias.detach().requires_grad_(True)
                    inputs.append(source_bias_leaf)
                target_weight_leaf = target_weight.detach().requires_grad_(True)
                inputs.append(target_weight_leaf)
                target_bias_leaf: Tensor | None = None
                if target_bias is not None:
                    target_bias_leaf = target_bias.detach().requires_grad_(True)
                    inputs.append(target_bias_leaf)
                bias_leaf: Tensor | None = None
                if bias is not None:
                    bias_leaf = bias.detach().requires_grad_(True)
                    inputs.append(bias_leaf)
                hidden_weight_leaf = hidden_weight.detach().requires_grad_(True)
                inputs.append(hidden_weight_leaf)
                hidden_bias_leaf: Tensor | None = None
                if hidden_bias is not None:
                    hidden_bias_leaf = hidden_bias.detach().requires_grad_(True)
                    inputs.append(hidden_bias_leaf)
                out_weight_leaf = out_weight.detach().requires_grad_(True)
                inputs.append(out_weight_leaf)
                out_bias_leaf: Tensor | None = None
                if out_bias is not None:
                    out_bias_leaf = out_bias.detach().requires_grad_(True)
                    inputs.append(out_bias_leaf)

                local_delta_state, local_delta_val = _hadamard_route_dense_chunk(
                    sender_block,
                    src_block,
                    dst_full,
                    projected_state_block,
                    projected_val_block,
                    source_weight_leaf,
                    source_bias_leaf,
                    target_weight_leaf,
                    target_bias_leaf,
                    hidden_weight_leaf,
                    hidden_bias_leaf,
                    out_weight_leaf,
                    out_bias_leaf,
                    bias_leaf,
                    ctx.temperature,
                    ctx.dst_block_size,
                )
                grads = torch.autograd.grad(
                    (local_delta_state, local_delta_val),
                    inputs,
                    grad_outputs=(
                        flat_grad_state.to(dtype=local_delta_state.dtype),
                        flat_grad_val.to(dtype=local_delta_val.dtype),
                    ),
                    allow_unused=True,
                )

            grad_sender_block, grad_src_block, grad_dst_block, grad_projected_state_block, grad_projected_val_block, grad_source_weight_block = grads[
                :6
            ]
            next_index = 6
            grad_source_bias_block = None
            if source_bias is not None:
                grad_source_bias_block = grads[next_index]
                next_index += 1
            grad_target_weight_block = grads[next_index]
            next_index += 1
            grad_target_bias_block = None
            if target_bias is not None:
                grad_target_bias_block = grads[next_index]
                next_index += 1
            grad_bias_block = None
            if bias is not None:
                grad_bias_block = grads[next_index]
                next_index += 1
            grad_hidden_weight_block = grads[next_index]
            next_index += 1
            grad_hidden_bias_block = None
            if hidden_bias is not None:
                grad_hidden_bias_block = grads[next_index]
                next_index += 1
            grad_out_weight_block = grads[next_index]
            next_index += 1
            grad_out_bias_block = grads[next_index] if out_bias is not None else None

            if grad_sender_block is not None:
                grad_sender[:, src_start:src_end] += grad_sender_block
            if grad_src_block is not None:
                grad_src[:, src_start:src_end, :] += grad_src_block
            if grad_dst_block is not None:
                grad_dst += grad_dst_block
            if grad_projected_state_block is not None:
                grad_projected_state[:, src_start:src_end] += grad_projected_state_block
            if grad_projected_val_block is not None:
                grad_projected_val[:, src_start:src_end, :] += grad_projected_val_block
            if grad_source_weight_block is not None:
                grad_source_weight += grad_source_weight_block
            if grad_source_bias is not None and grad_source_bias_block is not None:
                grad_source_bias += grad_source_bias_block
            if grad_target_weight_block is not None:
                grad_target_weight += grad_target_weight_block
            if grad_target_bias is not None and grad_target_bias_block is not None:
                grad_target_bias += grad_target_bias_block
            if grad_bias is not None and grad_bias_block is not None:
                grad_bias += grad_bias_block
            if grad_hidden_weight_block is not None:
                grad_hidden_weight += grad_hidden_weight_block
            if grad_hidden_bias is not None and grad_hidden_bias_block is not None:
                grad_hidden_bias += grad_hidden_bias_block
            if grad_out_weight_block is not None:
                grad_out_weight += grad_out_weight_block
            if grad_out_bias is not None and grad_out_bias_block is not None:
                grad_out_bias += grad_out_bias_block

        return (
            grad_sender.reshape_as(sender_strength),
            grad_src.reshape_as(src_val),
            grad_dst.reshape_as(dst_val),
            grad_projected_state.reshape_as(projected_state),
            grad_projected_val.reshape_as(projected_val),
            grad_source_weight,
            grad_source_bias,
            grad_target_weight,
            grad_target_bias,
            None,
            grad_bias,
            grad_hidden_weight,
            grad_hidden_bias,
            grad_out_weight,
            grad_out_bias,
            None,
            None,
            None,
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


class _LowRankTransitionPairwiseTopK(Function):
    @staticmethod
    def forward(
        ctx: Any,
        sender_strength: Tensor,
        src_val: Tensor,
        dst_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        temperature: float,
        topk: int,
        src_block_size: int,
        dst_block_size: int,
        compress_kind: int,
    ) -> tuple[Tensor, Tensor]:
        (
            flat_sender,
            flat_src,
            flat_dst,
            flat_projected_state,
            flat_projected_val,
            batch_shape,
            src_nodes,
            dst_nodes,
            out_dim,
        ) = _flatten_pairwise_transition_tensors(
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
        )
        k = min(int(topk), dst_nodes)
        projected_source = torch.matmul(flat_src, source_weight.t()).contiguous()
        weighted_projected_source = projected_source * core_weight.view(1, 1, -1).to(
            projected_source.dtype
        )
        if temperature != 1.0:
            weighted_projected_source = weighted_projected_source / float(temperature)
        projected_target = torch.matmul(flat_dst, target_weight.t()).contiguous()
        weighted_projected_state = (
            flat_sender.to(dtype=torch.float32) * flat_projected_state.to(dtype=torch.float32)
        ).contiguous()
        weighted_projected_val = (
            flat_sender.to(dtype=torch.float32).unsqueeze(-1)
            * flat_projected_val.to(dtype=torch.float32)
        ).contiguous()
        score_bias = float(bias.item()) if bias is not None else 0.0
        delta_state, delta_val, scores, indices = _native_module().low_rank_pairwise_topk_forward_cuda(
            weighted_projected_source,
            projected_target,
            weighted_projected_state,
            weighted_projected_val,
            k,
            score_bias,
            int(compress_kind),
        )
        ctx.batch_shape = batch_shape
        ctx.src_nodes = src_nodes
        ctx.dst_nodes = dst_nodes
        ctx.out_dim = out_dim
        ctx.temperature = float(temperature)
        ctx.has_bias = bias is not None
        ctx.compress_kind = int(compress_kind)
        ctx.save_for_backward(
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            scores,
            indices,
        )
        return (
            delta_state.to(dtype=projected_state.dtype).reshape(*batch_shape, dst_nodes),
            delta_val.to(dtype=projected_val.dtype).reshape(*batch_shape, dst_nodes, out_dim),
        )

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            scores,
            indices,
        ) = ctx.saved_tensors
        (
            flat_sender,
            flat_src,
            flat_dst,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            src_nodes,
            _dst_nodes,
            out_dim,
        ) = _flatten_pairwise_transition_tensors(
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
        )
        flat_scores = scores.reshape(-1, src_nodes, scores.shape[-1]).contiguous()
        flat_indices = indices.reshape(-1, src_nodes, indices.shape[-1]).contiguous()
        flat_routes = _pairwise_routes_from_scores(flat_scores, ctx.compress_kind).contiguous()
        weighted_state = (
            flat_sender.to(dtype=torch.float32) * flat_projected_state.to(dtype=torch.float32)
        ).contiguous()
        weighted_val = (
            flat_sender.to(dtype=torch.float32).unsqueeze(-1)
            * flat_projected_val.to(dtype=torch.float32)
        ).contiguous()
        flat_grad_state = grad_delta_state.reshape(-1, grad_delta_state.shape[-1]).to(
            dtype=torch.float32
        )
        flat_grad_val = grad_delta_val.reshape(-1, grad_delta_state.shape[-1], out_dim).to(
            dtype=torch.float32
        )
        grad_routes, grad_weighted_state, grad_weighted_val = _pairwise_transition_reduce_backward(
            flat_routes,
            flat_indices,
            weighted_state,
            weighted_val,
            flat_grad_state.contiguous(),
            flat_grad_val.contiguous(),
        )
        module = _native_module()
        grad_scores = _pairwise_routes_backward(
            flat_scores.contiguous(),
            flat_routes.contiguous(),
            grad_routes.contiguous(),
            ctx.compress_kind,
        )
        projected_target = torch.matmul(flat_dst, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_src, source_weight.t()).contiguous()
        (
            grad_src_route,
            grad_dst,
            grad_target_weight_from_kernel,
            grad_source_weight_from_kernel,
            grad_core_weight,
            grad_bias,
        ) = module.low_rank_pairwise_topk_backward_cuda(
            flat_src,
            flat_dst,
            target_weight.contiguous(),
            source_weight.contiguous(),
            core_weight.contiguous(),
            projected_source,
            projected_target,
            flat_indices,
            grad_scores.contiguous(),
            ctx.temperature,
        )
        grad_sender = (
            grad_weighted_state * flat_projected_state.to(dtype=torch.float32)
            + (
                grad_weighted_val * flat_projected_val.to(dtype=torch.float32)
            ).sum(dim=-1)
        )
        grad_projected_state = grad_weighted_state * flat_sender.to(dtype=torch.float32)
        grad_projected_val = grad_weighted_val * flat_sender.to(dtype=torch.float32).unsqueeze(-1)
        return (
            grad_sender.reshape_as(sender_strength).to(dtype=sender_strength.dtype),
            grad_src_route.reshape_as(src_val).to(dtype=src_val.dtype),
            grad_dst.reshape_as(dst_val).to(dtype=dst_val.dtype),
            grad_projected_state.reshape_as(projected_state).to(dtype=projected_state.dtype),
            grad_projected_val.reshape_as(projected_val).to(dtype=projected_val.dtype),
            grad_source_weight_from_kernel,
            grad_target_weight_from_kernel,
            grad_core_weight,
            grad_bias if ctx.has_bias else None,
            None,
            None,
            None,
            None,
            None,
        )


class _LowRankPropagationTopK(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        compress_kind: int,
        topk: int,
        target_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        (
            flat_val,
            flat_projected_state,
            flat_projected_val,
            batch_shape,
            nodes,
            out_dim,
        ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
        k = min(int(topk), nodes)
        projected_target = torch.matmul(flat_val, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_val, source_weight.t()).contiguous()
        weighted_projected_source = projected_source * core_weight.view(1, 1, -1).to(
            projected_source.dtype
        )
        score_bias = float(bias.item()) if bias is not None else 0.0
        delta_state, delta_val = _native_module().low_rank_propagation_topk_forward_cuda(
            weighted_projected_source,
            projected_target,
            flat_projected_state.to(dtype=torch.float32).contiguous(),
            flat_projected_val.to(dtype=torch.float32).contiguous(),
            k,
            score_bias,
            int(compress_kind),
        )
        ctx.k = k
        ctx.compress_kind = int(compress_kind)
        ctx.has_bias = bias is not None
        ctx.save_for_backward(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            _save_optional_tensor(bias, core_weight),
        )
        return (
            delta_state.to(dtype=projected_state.dtype).reshape(*batch_shape, nodes),
            delta_val.to(dtype=projected_val.dtype).reshape(*batch_shape, nodes, out_dim),
        )

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            bias_tensor,
        ) = ctx.saved_tensors
        bias = _load_optional_tensor(bias_tensor)
        (
            flat_val,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            nodes,
            out_dim,
        ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
        projected_target = torch.matmul(flat_val, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_val, source_weight.t()).contiguous()
        weighted_projected_source = projected_source * core_weight.view(1, 1, -1).to(
            projected_source.dtype
        )
        scores = torch.bmm(projected_target, weighted_projected_source.transpose(1, 2))
        if bias is not None:
            scores = scores + bias
        best_scores, best_indices = scores.topk(ctx.k, dim=-1, largest=True, sorted=True)
        edges = _propagation_edges_from_scores(best_scores, ctx.compress_kind).contiguous()
        flat_grad_state = grad_delta_state.reshape(-1, nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).contiguous()
        (
            edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        ) = _coerce_query_reduce_backward_inputs(
            edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        module = _native_module()
        grad_edges, grad_projected_state, grad_projected_val = module.query_topk_reduce_backward_cuda(
            edges,
            best_indices.contiguous(),
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        grad_scores = _propagation_edges_backward(
            best_scores.contiguous(),
            edges.contiguous(),
            grad_edges.contiguous(),
            ctx.compress_kind,
        )
        (
            grad_target,
            grad_source,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias,
        ) = module.low_rank_pairwise_topk_backward_cuda(
            flat_val,
            flat_val,
            source_weight.contiguous(),
            target_weight.contiguous(),
            core_weight.contiguous(),
            projected_target,
            projected_source,
            best_indices.contiguous(),
            grad_scores.contiguous(),
            1.0,
        )
        grad_layer = grad_target + grad_source
        return (
            grad_layer.reshape_as(layer_val).to(dtype=layer_val.dtype),
            grad_projected_state.reshape_as(projected_state).to(dtype=projected_state.dtype),
            grad_projected_val.reshape_as(projected_val).to(dtype=projected_val.dtype),
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias if ctx.has_bias else None,
            None,
            None,
            None,
            None,
        )


class _LowRankPropagationWindow(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        compress_kind: int,
        window: int,
        target_block_size: int,
        source_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        (
            flat_val,
            flat_projected_state,
            flat_projected_val,
            batch_shape,
            nodes,
            out_dim,
        ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
        projected_target = torch.matmul(flat_val, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_val, source_weight.t()).contiguous()
        weighted_projected_source = projected_source * core_weight.view(1, 1, -1).to(
            projected_source.dtype
        )
        score_bias = float(bias.item()) if bias is not None else 0.0
        module = _native_module()
        if int(compress_kind) == 1:
            delta_state, delta_val = module.low_rank_propagation_window_entmax15_forward_cuda(
                weighted_projected_source,
                projected_target,
                flat_projected_state.to(dtype=torch.float32).contiguous(),
                flat_projected_val.to(dtype=torch.float32).contiguous(),
                int(window),
                score_bias,
            )
        else:
            delta_state, delta_val = module.low_rank_propagation_window_forward_cuda(
                weighted_projected_source,
                projected_target,
                flat_projected_state.to(dtype=torch.float32).contiguous(),
                flat_projected_val.to(dtype=torch.float32).contiguous(),
                int(window),
                score_bias,
            )
        ctx.compress_kind = int(compress_kind)
        ctx.window = int(window)
        ctx.has_bias = bias is not None
        ctx.save_for_backward(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            _save_optional_tensor(bias, core_weight),
        )
        return (
            delta_state.to(dtype=projected_state.dtype).reshape(*batch_shape, nodes),
            delta_val.to(dtype=projected_val.dtype).reshape(*batch_shape, nodes, out_dim),
        )

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        (
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            bias_tensor,
        ) = ctx.saved_tensors
        bias = _load_optional_tensor(bias_tensor)
        (
            flat_val,
            flat_projected_state,
            flat_projected_val,
            _batch_shape,
            nodes,
            out_dim,
        ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
        projected_target = torch.matmul(flat_val, target_weight.t()).contiguous()
        projected_source = torch.matmul(flat_val, source_weight.t()).contiguous()
        width = min(ctx.window + 1, nodes)
        index_2d, valid_2d = _window_source_indices(
            target_nodes=nodes,
            source_nodes=nodes,
            window=ctx.window,
            device=flat_val.device,
        )
        flat_indices = index_2d.view(1, nodes, width).expand(flat_val.shape[0], -1, -1).contiguous()
        valid = valid_2d.view(1, nodes, width).expand(flat_val.shape[0], -1, -1)
        selected_projected_source = _gather_sequence_rows(projected_source, flat_indices)
        weighted_selected_source = selected_projected_source * core_weight.view(1, 1, 1, -1).to(
            selected_projected_source.dtype
        )
        scores = (
            projected_target.unsqueeze(-2) * weighted_selected_source
        ).sum(dim=-1)
        if bias is not None:
            scores = scores + bias
        if ctx.compress_kind == 1:
            packed_mask = valid.contiguous()
            edges = torch.ops.jakal_net.signed_entmax15(scores, packed_mask)
        else:
            edges = torch.nn.functional.softsign(scores)
            edges = edges * valid.to(dtype=edges.dtype)
        flat_grad_state = grad_delta_state.reshape(-1, nodes).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).contiguous()
        (
            edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        ) = _coerce_query_reduce_backward_inputs(
            edges,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        module = _native_module()
        grad_edges, grad_projected_state, grad_projected_val = module.query_topk_reduce_backward_cuda(
            edges.contiguous(),
            flat_indices,
            flat_projected_state,
            flat_projected_val,
            flat_grad_state,
            flat_grad_val,
        )
        valid_f32 = valid.to(dtype=grad_edges.dtype)
        grad_edges = grad_edges * valid_f32
        if ctx.compress_kind == 1:
            grad_scores = torch.ops.jakal_net.signed_entmax15_backward(
                scores.contiguous(),
                edges.contiguous(),
                grad_edges.contiguous(),
                valid.contiguous(),
            )
            grad_scores = grad_scores * valid_f32
        else:
            grad_scores = module.softsign_backward_cuda(
                scores.contiguous(),
                grad_edges.contiguous(),
            )
            grad_scores = grad_scores * valid_f32
        (
            grad_target,
            grad_source,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias,
        ) = module.low_rank_pairwise_topk_backward_cuda(
            flat_val,
            flat_val,
            source_weight.contiguous(),
            target_weight.contiguous(),
            core_weight.contiguous(),
            projected_target,
            projected_source,
            flat_indices,
            grad_scores.contiguous(),
            1.0,
        )
        grad_layer = grad_target + grad_source
        return (
            grad_layer.reshape_as(layer_val).to(dtype=layer_val.dtype),
            grad_projected_state.reshape_as(projected_state).to(dtype=projected_state.dtype),
            grad_projected_val.reshape_as(projected_val).to(dtype=projected_val.dtype),
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
    if edge_compress_name == "softsign" and isinstance(pairwise_fn, HadamardMLPPairwise):
        delta_state, delta_val = _HadamardPropagationDense.apply(
            layer_val,
            projected_state,
            projected_val,
            pairwise_fn.proj_in.weight,
            _save_optional_tensor(pairwise_fn.proj_in.bias, pairwise_fn.proj_in.weight),
            pairwise_fn.proj_out.weight,
            _save_optional_tensor(pairwise_fn.proj_out.bias, pairwise_fn.proj_out.weight),
            target_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_dense(
        spec.kind,
        spec.weight,
        spec.bias,
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
        edge_compress_name,
        layer_val,
        projected_state,
        projected_val,
        target_block_size,
        source_block_size,
    ))


def propagation_query_dense_native(
    *,
    pairwise_fn: object,
    edge_compress_name: str,
    query_val: Tensor,
    source_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    query_block_size: int,
    source_block_size: int,
) -> Any:
    if not supports_pairwise_kernel(pairwise_fn):
        raise TypeError("Unsupported pairwise_fn for native dense query propagation.")
    if edge_compress_name == "softsign" and isinstance(pairwise_fn, HadamardMLPPairwise):
        delta_state, delta_val = _HadamardPropagationQueryDense.apply(
            query_val,
            source_val,
            projected_state,
            projected_val,
            pairwise_fn.proj_in.weight,
            _save_optional_tensor(pairwise_fn.proj_in.bias, pairwise_fn.proj_in.weight),
            pairwise_fn.proj_out.weight,
            _save_optional_tensor(pairwise_fn.proj_out.bias, pairwise_fn.proj_out.weight),
            query_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_query_dense(
        spec.kind,
        spec.weight,
        spec.bias,
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
        edge_compress_name,
        query_val,
        source_val,
        projected_state,
        projected_val,
        query_block_size,
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
    use_entmax15_cuda_autograd = (
        _experimental_fused_training_enabled()
        and edge_compress_name == "signed_entmax15"
        and _signed_entmax15_ops_available()
        and native_supports("query_topk_reduce_backward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and native_supports("low_rank_propagation_window_entmax15_forward_cuda")
        and isinstance(pairwise_fn, LowRankBilinearPairwise)
        and _cuda_float_tensor(layer_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    use_cuda_autograd = (
        _experimental_fused_training_enabled()
        and edge_compress_name == "softsign"
        and _query_backward_ops_available()
        and native_supports("low_rank_propagation_window_forward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and isinstance(pairwise_fn, LowRankBilinearPairwise)
        and _cuda_float_tensor(layer_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    if use_cuda_autograd or use_entmax15_cuda_autograd:
        delta_state, delta_val = _LowRankPropagationWindow.apply(
            layer_val,
            projected_state,
            projected_val,
            pairwise_fn.source_proj.weight,
            pairwise_fn.target_proj.weight,
            pairwise_fn.weight,
            pairwise_fn.bias,
            1 if edge_compress_name == "signed_entmax15" else 0,
            window,
            target_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_window(
        spec.kind,
        spec.weight,
        spec.bias,
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
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
    compress_kind = _propagation_topk_compress_kind(edge_compress_name)
    use_cuda_autograd = (
        _experimental_fused_training_enabled()
        and compress_kind is not None
        and compress_kind in {0, 1}
        and _query_backward_ops_available()
        and native_supports("low_rank_propagation_topk_forward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and isinstance(pairwise_fn, LowRankBilinearPairwise)
        and topk <= 32
        and _cuda_float_tensor(layer_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    if use_cuda_autograd:
        delta_state, delta_val = _LowRankPropagationTopK.apply(
            layer_val,
            projected_state,
            projected_val,
            pairwise_fn.source_proj.weight,
            pairwise_fn.target_proj.weight,
            pairwise_fn.weight,
            pairwise_fn.bias,
            int(compress_kind),
            topk,
            target_block_size,
            source_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_kernel_spec(pairwise_fn)
    return _to_layer_delta(_native_module().propagation_topk(
        spec.kind,
        spec.weight,
        spec.bias,
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
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
        _experimental_fused_training_enabled()
        and _query_backward_ops_available()
        and native_supports("propagation_query_topk_select")
        and native_supports("diagonal_pairwise_topk_backward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and not isinstance(pairwise_fn, HadamardMLPPairwise)
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
        spec.in_weight,
        spec.in_bias,
        spec.out_weight,
        spec.out_bias,
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
    route_compress_name: str,
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
        route_compress_name,
        sender_strength,
        src_val,
        projected_state,
        projected_val,
        dst_nodes,
        src_block_size,
        dst_block_size,
    ))


def transition_pairwise_dense_native(
    *,
    route_fn: object,
    route_compress_name: str = "softmax",
    sender_strength: Tensor,
    src_val: Tensor,
    dst_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    src_block_size: int,
    dst_block_size: int,
) -> Any:
    if not supports_pairwise_route_kernel(route_fn):
        raise TypeError("Unsupported pairwise route_fn for native dense transition.")
    inner = getattr(route_fn, "route_fn", route_fn)
    temperature = float(getattr(route_fn, "temperature", 1.0))
    if isinstance(inner, SourceTargetHadamardMLPRoute):
        if route_compress_name != "softmax":
            raise TypeError(
                "Native hadamard pairwise dense transition currently supports only softmax "
                "route compression."
            )
        src_block_size = 4 if src_block_size <= 0 else min(src_block_size, 4)
        dst_block_size = 4 if dst_block_size <= 0 else min(dst_block_size, 4)
        delta_state, delta_val = _HadamardTransitionPairwiseDense.apply(
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            inner.source_proj.weight,
            _save_optional_tensor(inner.source_proj.bias, inner.source_proj.weight),
            inner.target_proj.weight,
            _save_optional_tensor(inner.target_proj.bias, inner.target_proj.weight),
            inner.proj_in.weight,
            _save_optional_tensor(None, inner.proj_in.weight),
            inner.proj_in.weight,
            _save_optional_tensor(inner.proj_in.bias, inner.proj_in.weight),
            inner.proj_out.weight,
            _save_optional_tensor(inner.proj_out.bias, inner.proj_out.weight),
            temperature,
            src_block_size,
            dst_block_size,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    spec = pairwise_route_kernel_spec(route_fn)
    return _to_layer_delta(_native_module().transition_pairwise_dense(
        spec.kind,
        spec.source_weight,
        spec.source_bias,
        spec.target_weight,
        spec.target_bias,
        spec.core_weight,
        spec.bias,
        spec.hidden_weight,
        spec.hidden_bias,
        spec.out_weight,
        spec.out_bias,
        spec.temperature,
        route_compress_name,
        sender_strength,
        src_val,
        dst_val,
        projected_state,
        projected_val,
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
        _experimental_fused_training_enabled()
        and _query_backward_ops_available()
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
        spec.source_bias,
        spec.target_weight,
        spec.target_bias,
        spec.core_weight,
        spec.bias,
        spec.hidden_weight,
        spec.hidden_bias,
        spec.out_weight,
        spec.out_bias,
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
    route_compress_name: str = "softmax",
) -> Any:
    if not supports_pairwise_route_kernel(route_fn):
        raise TypeError("Unsupported pairwise route_fn for native sparse transition.")
    inner = getattr(route_fn, "route_fn", route_fn)
    temperature = float(getattr(route_fn, "temperature", 1.0))
    compress_kind = _pairwise_topk_compress_kind(route_compress_name)
    use_cuda_autograd = (
        _query_backward_ops_available()
        and native_supports("low_rank_pairwise_topk_forward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
        and isinstance(inner, LowRankBilinearRoute)
        and compress_kind is not None
        and (compress_kind != 2 or _signed_entmax15_ops_available())
        and topk <= 32
        and _cuda_float_tensor(sender_strength)
        and _cuda_float_tensor(src_val)
        and _cuda_float_tensor(dst_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    if use_cuda_autograd:
        delta_state, delta_val = _LowRankTransitionPairwiseTopK.apply(
            sender_strength,
            src_val,
            dst_val,
            projected_state,
            projected_val,
            inner.source_proj.weight,
            inner.target_proj.weight,
            inner.weight,
            inner.bias,
            temperature,
            topk,
            src_block_size,
            dst_block_size,
            int(compress_kind),
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    if route_compress_name != "softmax":
        raise TypeError(
            "Native sparse pairwise transition supports signed_abs_softmax and signed_entmax15 "
            "only through the CUDA low-rank autograd path."
        )
    spec = pairwise_route_kernel_spec(route_fn)
    return _to_layer_delta(_native_module().transition_pairwise_topk(
        spec.kind,
        spec.source_weight,
        spec.source_bias,
        spec.target_weight,
        spec.target_bias,
        spec.core_weight,
        spec.bias,
        spec.hidden_weight,
        spec.hidden_bias,
        spec.out_weight,
        spec.out_bias,
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
