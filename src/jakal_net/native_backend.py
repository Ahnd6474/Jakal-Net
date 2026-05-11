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

from jakal_net._architectural_common import signed_softmax_state
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
    MultiHeadPairwise,
    SourceTargetHadamardMLPRoute,
)
try:
    from jakal_net.triton_signed_smoothmax import (
        lowrank_signed_smoothmax_backward_owner,
        lowrank_signed_smoothmax_backward_owner_generic,
        multihead_signed_smoothmax_scores,
        multihead_signed_smoothmax_scores_and_head_grads_tile,
        multihead_signed_smoothmax_pass1_full,
        multihead_signed_smoothmax_tile_partials,
        multihead_signed_smoothmax_scores_tile,
        signed_abs_softmax_backward_tile_from_projections,
        signed_abs_softmax_tile_stats_from_projections,
        signed_abs_softmax_backward_tile,
        signed_abs_softmax_edge_dot_tile,
        signed_abs_softmax_tile_stats,
        triton_signed_smoothmax_available,
    )
except Exception:  # noqa: BLE001
    lowrank_signed_smoothmax_backward_owner = None
    lowrank_signed_smoothmax_backward_owner_generic = None
    multihead_signed_smoothmax_scores = None
    multihead_signed_smoothmax_scores_and_head_grads_tile = None
    multihead_signed_smoothmax_pass1_full = None
    multihead_signed_smoothmax_tile_partials = None
    multihead_signed_smoothmax_scores_tile = None
    signed_abs_softmax_backward_tile_from_projections = None
    signed_abs_softmax_tile_stats_from_projections = None
    signed_abs_softmax_backward_tile = None
    signed_abs_softmax_edge_dot_tile = None
    signed_abs_softmax_tile_stats = None

    def triton_signed_smoothmax_available() -> bool:
        return False

DEFAULT_NATIVE_MODULE = "jakal_net_native"
NATIVE_MODULE_ENV = "JAKAL_NET_NATIVE_MODULE"
DISABLE_NATIVE_ENV = "JAKAL_NET_DISABLE_NATIVE"
EXPERIMENTAL_FUSED_TRAINING_ENV = "JAKAL_NET_ENABLE_EXPERIMENTAL_FUSED_TRAINING"
EXPERIMENTAL_FUSED_TRAINING_CHECKPOINT_STRIDE_ENV = "JAKAL_NET_FUSED_TRAINING_CHECKPOINT_STRIDE"
EXPERIMENTAL_SCAN_BACKWARD_CUDA_ENV = "JAKAL_NET_ENABLE_EXPERIMENTAL_SCAN_BACKWARD_CUDA"
EXPERIMENTAL_CAUSAL_DENSE_PROP_FORWARD_CUDA_ENV = "JAKAL_NET_ENABLE_CAUSAL_DENSE_PROP_FORWARD_CUDA"
EXPERIMENTAL_DIAGONAL_DENSE_PROP_CUDA_ENV = "JAKAL_NET_ENABLE_DIAGONAL_DENSE_PROP_CUDA"
EXPERIMENTAL_DENSE_MH_SAVE_POLICY_ENV = "JAKAL_NET_EXPERIMENTAL_DENSE_MH_SAVE_POLICY"

_FULL_TOPK_INDEX_CACHE: dict[tuple[str, tuple[int, ...]], Tensor] = {}


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


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip() in {"1", "true", "TRUE", "yes"}


def _experimental_dense_mh_save_policy() -> str:
    policy = os.environ.get(EXPERIMENTAL_DENSE_MH_SAVE_POLICY_ENV, "").strip().lower()
    if policy in {"balanced", "speed"}:
        return policy
    return ""


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


def dense_apply_native_available(device_type: str) -> bool:
    return native_supports("apply_delta_to_layer") and native_supports_device(device_type)


def nomemory_causal_stack_fused_native_available(device_type: str) -> bool:
    return (
        native_supports("nomemory_causal_stack_fused")
        and native_supports("nomemory_causal_stack_fused_backward_cuda")
        and native_supports_device(device_type)
    )


def nomemory_causal_stack_ffn_fused_native_available(device_type: str) -> bool:
    return (
        native_supports("nomemory_causal_stack_ffn_fused")
        and native_supports("nomemory_causal_stack_ffn_fused_backward_cuda")
        and native_supports_device(device_type)
    )


def value_ffn_native_available(device_type: str) -> bool:
    return native_supports("value_ffn") and native_supports_device(device_type)


def bilinear_propagation_softsign_value_ffn_native_available(device_type: str) -> bool:
    return (
        native_supports("bilinear_propagation_softsign_value_ffn_forward_cuda")
        and native_supports_device(device_type)
    )


def bilinear_propagation_softsign_value_ffn_backward_native_available(device_type: str) -> bool:
    return (
        native_supports("bilinear_propagation_softsign_value_ffn_backward_cuda")
        and native_supports_device(device_type)
    )


def propagation_value_ffn_fused_native_available(device_type: str, *, dense: bool) -> bool:
    if not native_supports_device(device_type):
        return False
    if dense:
        return (
            native_supports("low_rank_propagation_causal_dense_value_ffn_forward_cuda")
            and native_supports("low_rank_propagation_causal_dense_value_ffn_backward_cuda")
        )
    return native_supports("low_rank_propagation_window_value_ffn_forward_cuda")


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
    level_ffn_norm_weights: tuple[Tensor, ...],
    level_ffn_norm_biases: tuple[Tensor, ...],
    level_ffn_in_weights: tuple[Tensor, ...],
    level_ffn_in_biases: tuple[Tensor, ...],
    level_ffn_out_weights: tuple[Tensor, ...],
    level_ffn_out_biases: tuple[Tensor, ...],
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
            list(level_ffn_norm_weights),
            list(level_ffn_norm_biases),
            list(level_ffn_in_weights),
            list(level_ffn_in_biases),
            list(level_ffn_out_weights),
            list(level_ffn_out_biases),
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
            list(level_ffn_norm_weights),
            list(level_ffn_norm_biases),
            list(level_ffn_in_weights),
            list(level_ffn_in_biases),
            list(level_ffn_out_weights),
            list(level_ffn_out_biases),
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
    level_ffn_norm_weights: tuple[Tensor, ...],
    level_ffn_norm_biases: tuple[Tensor, ...],
    level_ffn_in_weights: tuple[Tensor, ...],
    level_ffn_in_biases: tuple[Tensor, ...],
    level_ffn_out_weights: tuple[Tensor, ...],
    level_ffn_out_biases: tuple[Tensor, ...],
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
            list(level_ffn_norm_weights),
            list(level_ffn_norm_biases),
            list(level_ffn_in_weights),
            list(level_ffn_in_biases),
            list(level_ffn_out_weights),
            list(level_ffn_out_biases),
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
            list(level_ffn_norm_weights),
            list(level_ffn_norm_biases),
            list(level_ffn_in_weights),
            list(level_ffn_in_biases),
            list(level_ffn_out_weights),
            list(level_ffn_out_biases),
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
    level_ffn_norm_weights: tuple[Tensor, ...],
    level_ffn_norm_biases: tuple[Tensor, ...],
    level_ffn_in_weights: tuple[Tensor, ...],
    level_ffn_in_biases: tuple[Tensor, ...],
    level_ffn_out_weights: tuple[Tensor, ...],
    level_ffn_out_biases: tuple[Tensor, ...],
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
        level_ffn_norm_weights=level_ffn_norm_weights,
        level_ffn_norm_biases=level_ffn_norm_biases,
        level_ffn_in_weights=level_ffn_in_weights,
        level_ffn_in_biases=level_ffn_in_biases,
        level_ffn_out_weights=level_ffn_out_weights,
        level_ffn_out_biases=level_ffn_out_biases,
        skip_source_weights=skip_source_weights,
        skip_target_weights=skip_target_weights,
        skip_core_weights=skip_core_weights,
        skip_biases=skip_biases,
        skip_gates=skip_gates,
        skip_topks=skip_topks,
    )
    if (
        torch.is_grad_enabled()
        and _experimental_fused_training_enabled()
        and _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind)
    ):
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


def _experimental_causal_dense_prop_forward_cuda_enabled() -> bool:
    return _env_flag(EXPERIMENTAL_CAUSAL_DENSE_PROP_FORWARD_CUDA_ENV)


def _experimental_diagonal_dense_prop_cuda_enabled() -> bool:
    return _env_flag(EXPERIMENTAL_DIAGONAL_DENSE_PROP_CUDA_ENV)


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
    level_ffn_norm_weights: tuple[Tensor, ...],
    level_ffn_norm_biases: tuple[Tensor, ...],
    level_ffn_in_weights: tuple[Tensor, ...],
    level_ffn_in_biases: tuple[Tensor, ...],
    level_ffn_out_weights: tuple[Tensor, ...],
    level_ffn_out_biases: tuple[Tensor, ...],
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
        *level_ffn_norm_weights,
        *level_ffn_norm_biases,
        *level_ffn_in_weights,
        *level_ffn_in_biases,
        *level_ffn_out_weights,
        *level_ffn_out_biases,
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
        ("level_ffn_norm_weights", num_levels),
        ("level_ffn_norm_biases", num_levels),
        ("level_ffn_in_weights", num_levels),
        ("level_ffn_in_biases", num_levels),
        ("level_ffn_out_weights", num_levels),
        ("level_ffn_out_biases", num_levels),
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


def _native_scan_softsign_state(state: Tensor) -> Tensor:
    return F.softsign(torch.nan_to_num(state))


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
    key = (str(scores.device), tuple(scores.shape))
    cached = _FULL_TOPK_INDEX_CACHE.get(key)
    if cached is None or cached.device != scores.device:
        cached = (
            torch.arange(scores.shape[-1], device=scores.device, dtype=torch.long)
            .view(1, 1, -1)
            .expand(scores.shape[0], scores.shape[1], scores.shape[2])
            .contiguous()
        )
        _FULL_TOPK_INDEX_CACHE[key] = cached
    return cached


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
    nodes = layer_val.shape[1]
    k = nodes if int(topk) <= 0 else min(max(1, int(topk)), nodes)
    if k == nodes:
        edges = _native_scan_signed_abs_softmax(scores)
        delta_state = torch.bmm(edges.to(layer_state.dtype), layer_state.unsqueeze(-1)).squeeze(-1)
        delta_val = torch.bmm(edges.to(layer_val.dtype), layer_val)
        return delta_state, delta_val

    selected_scores, selected_indices = torch.topk(scores, k, dim=-1)
    edges = _native_scan_signed_abs_softmax(selected_scores)
    selected_state = _native_scan_gather_state(layer_state, selected_indices)
    selected_val = _native_scan_gather_val(layer_val, selected_indices)
    delta_state = (edges * selected_state).sum(dim=-1)
    delta_val = (edges.unsqueeze(-1) * selected_val).sum(dim=-2)
    return delta_state, delta_val

def _native_scan_apply_delta(
    layer_state: Tensor,
    layer_val: Tensor,
    delta_state: Tensor,
    delta_val: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor,
    *,
    state_activation_name: str = "signed_softmax",
) -> tuple[Tensor, Tensor]:
    if state_activation_name == "signed_softmax":
        next_state = _native_scan_signed_softmax_state(layer_state + delta_state)
    elif state_activation_name == "softsign":
        next_state = _native_scan_softsign_state(layer_state + delta_state)
    else:
        raise ValueError(f"Unsupported state_activation_name: {state_activation_name!r}.")
    next_val = _native_scan_layer_norm(layer_val + delta_val, val_norm_weight, val_norm_bias)
    return next_state, next_val


def dense_apply_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    delta_state: Tensor,
    delta_val: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor,
    state_activation_name: str = "signed_softmax",
) -> tuple[Tensor, Tensor]:
    if dense_apply_native_available(layer_state.device.type):
        result = _native_module().apply_delta_to_layer(
            layer_state,
            layer_val,
            delta_state,
            delta_val,
            val_norm_weight,
            val_norm_bias,
            state_activation_name,
            True,
        )
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], Tensor)
            and isinstance(result[1], Tensor)
        ):
            return result
        raise TypeError("apply_delta_to_layer must return (state, val).")
    return _native_scan_apply_delta(
        layer_state,
        layer_val,
        delta_state,
        delta_val,
        val_norm_weight,
        val_norm_bias,
        state_activation_name=state_activation_name,
    )


def _native_scan_read_memory(
    memory_state: list[tuple[Tensor, Tensor]],
    val_norm_weights: tuple[Tensor, ...],
    val_norm_biases: tuple[Tensor, ...],
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
    *,
    gates_are_sigmoid: bool = False,
) -> Tensor:
    read_sum: Tensor | None = None
    cached_template: Tensor | None = None
    cached_projection_weights: list[Tensor | None] = [None] * len(read_projection_weights)
    cached_gates: list[Tensor | None] = [None] * len(read_gates)
    for index, (state, val) in enumerate(memory_state):
        read_val = _native_scan_layer_norm(val, val_norm_weights[index], val_norm_biases[index])
        sender_strength = F.softplus(state).unsqueeze(-1)
        read_summary = (sender_strength * read_val).sum(dim=-2)
        if cached_template is None or cached_template.dtype != read_summary.dtype or cached_template.device != read_summary.device:
            cached_template = read_template_val.to(device=read_summary.device, dtype=read_summary.dtype).unsqueeze(0)
        projection_weight = cached_projection_weights[index]
        if (
            projection_weight is None
            or projection_weight.dtype != read_summary.dtype
            or projection_weight.device != read_summary.device
        ):
            projection_weight = read_projection_weights[index].to(device=read_summary.device, dtype=read_summary.dtype)
            cached_projection_weights[index] = projection_weight
        gate = cached_gates[index]
        if gate is None or gate.dtype != read_summary.dtype or gate.device != read_summary.device:
            gate = read_gates[index].to(device=read_summary.device, dtype=read_summary.dtype)
            if not gates_are_sigmoid:
                gate = torch.sigmoid(gate)
            cached_gates[index] = gate
        projected = F.linear(read_summary + cached_template, projection_weight, None)
        term = gate * projected
        read_sum = term if read_sum is None else read_sum + term
    if read_sum is None:
        raise RuntimeError("_native_scan_read_memory requires at least one memory level.")
    return read_sum


def _native_scan_apply_ffn(args: dict[str, Any], level_index: int, state: Tensor, val: Tensor) -> tuple[Tensor, Tensor]:
    in_weight = args["level_ffn_in_weights"][level_index]
    if in_weight.numel() == 0:
        return state, val
    normalized = _native_scan_layer_norm(
        val,
        args["level_ffn_norm_weights"][level_index],
        args["level_ffn_norm_biases"][level_index],
    )
    in_bias = args["level_ffn_in_biases"][level_index]
    out_bias = args["level_ffn_out_biases"][level_index]
    hidden = F.linear(
        normalized,
        in_weight.to(device=val.device, dtype=val.dtype),
        in_bias.to(device=val.device, dtype=val.dtype) if in_bias.numel() != 0 else None,
    )
    delta = F.linear(
        F.gelu(hidden),
        args["level_ffn_out_weights"][level_index].to(device=val.device, dtype=val.dtype),
        out_bias.to(device=val.device, dtype=val.dtype) if out_bias.numel() != 0 else None,
    )
    return state, val + delta


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
    aligned_s = args["aligned_s"].contiguous()
    projected_s = F.linear(aligned_s, args["s_prediction_weight"].to(dtype=aligned_s.dtype), None).contiguous()
    current_memory = [
        (args["flat_memory"][index * 2], args["flat_memory"][index * 2 + 1])
        for index in range(num_levels)
    ]
    query_val = torch.empty_like(projected_s)
    prediction_input_norm_bias = args["prediction_input_norm_bias"]

    read_template_cast = args["read_template_val"].to(device=aligned_s.device, dtype=aligned_s.dtype)
    read_projection_weights_cast = tuple(
        weight.to(device=aligned_s.device, dtype=aligned_s.dtype)
        for weight in args["read_projection_weights"]
    )
    read_gates_sigmoid = tuple(
        torch.sigmoid(gate.to(device=aligned_s.device, dtype=aligned_s.dtype))
        for gate in args["read_gates"]
    )

    for time_index in range(aligned_s.shape[1]):
        token_val = aligned_s.narrow(1, time_index, 1).contiguous()
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
        level_state, level_val = _native_scan_apply_ffn(args, 0, level_state, level_val)
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
        level_state, level_val = _native_scan_apply_ffn(args, 0, level_state, level_val)
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
            updated_state, updated_val = _native_scan_apply_ffn(args, level_index, updated_state, updated_val)

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
                updated_state, updated_val = _native_scan_apply_ffn(args, level_index, updated_state, updated_val)

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
                updated_state, updated_val = _native_scan_apply_ffn(args, level_index, updated_state, updated_val)

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
            updated_state, updated_val = _native_scan_apply_ffn(args, level_index, updated_state, updated_val)
            next_memory.append((updated_state, updated_val))

        current_memory = next_memory
        read_vector = _native_scan_read_memory(
            current_memory,
            args["val_norm_weights"],
            args["val_norm_biases"],
            read_template_cast,
            read_projection_weights_cast,
            read_gates_sigmoid,
            gates_are_sigmoid=True,
        )
        query_input = projected_s[:, time_index, :] + read_vector
        query_val[:, time_index, :].copy_(
            _native_scan_layer_norm(
                query_input,
                args["prediction_input_norm_weight"],
                prediction_input_norm_bias,
            )
        )

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
    if not _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind):
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
        list(args["level_ffn_norm_weights"]),
        list(args["level_ffn_norm_biases"]),
        list(args["level_ffn_in_weights"]),
        list(args["level_ffn_in_biases"]),
        list(args["level_ffn_out_weights"]),
        list(args["level_ffn_out_biases"]),
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
        level_ffn_norm_weights=tuple(args["level_ffn_norm_weights"]),
        level_ffn_norm_biases=tuple(args["level_ffn_norm_biases"]),
        level_ffn_in_weights=tuple(args["level_ffn_in_weights"]),
        level_ffn_in_biases=tuple(args["level_ffn_in_biases"]),
        level_ffn_out_weights=tuple(args["level_ffn_out_weights"]),
        level_ffn_out_biases=tuple(args["level_ffn_out_biases"]),
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
        level_ffn_norm_weights=tuple(args["level_ffn_norm_weights"]),
        level_ffn_norm_biases=tuple(args["level_ffn_norm_biases"]),
        level_ffn_in_weights=tuple(args["level_ffn_in_weights"]),
        level_ffn_in_biases=tuple(args["level_ffn_in_biases"]),
        level_ffn_out_weights=tuple(args["level_ffn_out_weights"]),
        level_ffn_out_biases=tuple(args["level_ffn_out_biases"]),
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
        list(args["level_ffn_norm_weights"]),
        list(args["level_ffn_norm_biases"]),
        list(args["level_ffn_in_weights"]),
        list(args["level_ffn_in_biases"]),
        list(args["level_ffn_out_weights"]),
        list(args["level_ffn_out_biases"]),
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
        legacy_scan = _native_scan_uses_legacy_low_rank_extension(route_kind_name, propagation_pairwise_kind)
        if legacy_scan and checkpoint_stride is not None and checkpoint_stride < tensor_args[0].shape[1]:
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
            legacy_scan
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
                level_ffn_norm_weights=tuple(unpacked["level_ffn_norm_weights"]),
                level_ffn_norm_biases=tuple(unpacked["level_ffn_norm_biases"]),
                level_ffn_in_weights=tuple(unpacked["level_ffn_in_weights"]),
                level_ffn_in_biases=tuple(unpacked["level_ffn_in_biases"]),
                level_ffn_out_weights=tuple(unpacked["level_ffn_out_weights"]),
                level_ffn_out_biases=tuple(unpacked["level_ffn_out_biases"]),
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
            and _native_scan_uses_legacy_low_rank_extension(ctx.route_kind_name, ctx.propagation_pairwise_kind)
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


def _masked_signed_abs_softmax_from_scores(scores: Tensor, mask: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    stats = clean_scores.abs().masked_fill(~mask, float("-inf"))
    probs = torch.softmax(stats, dim=-1)
    return torch.sign(clean_scores) * probs * mask.to(dtype=probs.dtype)


def _signed_abs_softmax_backward(scores: Tensor, grad_routes: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    signs = torch.sign(clean_scores)
    probs = torch.softmax(clean_scores.abs(), dim=-1)
    signed_routes = signs * probs
    dot = (grad_routes * signed_routes).sum(dim=-1, keepdim=True)
    return signs * probs * (signs * grad_routes - dot)


def _masked_signed_abs_softmax_backward(scores: Tensor, edges: Tensor, grad_routes: Tensor, mask: Tensor) -> Tensor:
    clean_scores = torch.nan_to_num(scores)
    signs = torch.sign(clean_scores)
    probs = edges * signs
    dot = (grad_routes * edges).sum(dim=-1, keepdim=True)
    grad_scores = signs * probs * (signs * grad_routes - dot)
    return grad_scores * mask.to(dtype=grad_scores.dtype)


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


def _is_triton_multihead_signed_smoothmax_lowrank_pairwise(pairwise_fn: object) -> bool:
    return (
        isinstance(pairwise_fn, MultiHeadPairwise)
        and pairwise_fn.aggregate == "signed_smoothmax"
        and 0 < len(pairwise_fn.heads) <= 4
        and all(isinstance(head, LowRankBilinearPairwise) for head in pairwise_fn.heads)
    )


def _stack_multihead_lowrank_weights(
    pairwise_fn: MultiHeadPairwise,
) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    source_weight = torch.stack(
        [head.source_proj.weight for head in pairwise_fn.heads],
        dim=0,
    )
    target_weight = torch.stack(
        [head.target_proj.weight for head in pairwise_fn.heads],
        dim=0,
    )
    core_weight = torch.stack(
        [head.normalized_weight() for head in pairwise_fn.heads],
        dim=0,
    )
    biases = [head.bias for head in pairwise_fn.heads]
    if all(bias is None for bias in biases):
        bias = None
    else:
        if any(bias is None for bias in biases):
            raise TypeError("All multihead low-rank heads must consistently define bias.")
        bias = torch.stack([bias for bias in biases if bias is not None], dim=0)
    return source_weight, target_weight, core_weight, bias


def _multihead_lowrank_signed_smoothmax_dense_forward(
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (
        flat_val,
        flat_projected_state,
        flat_projected_val,
        batch_shape,
        nodes,
        out_dim,
    ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
    num_heads = int(source_weight.shape[0])
    rank_dim = int(core_weight.shape[1])
    projected_target = torch.matmul(
        flat_val.unsqueeze(1),
        target_weight.transpose(1, 2).unsqueeze(0),
    ).contiguous()
    projected_source = torch.matmul(
        flat_val.unsqueeze(1),
        source_weight.transpose(1, 2).unsqueeze(0),
    ).contiguous()
    weighted_source = projected_source * core_weight.view(
        1,
        num_heads,
        1,
        rank_dim,
    ).to(dtype=projected_source.dtype, device=projected_source.device)
    head_scores = torch.bmm(
        projected_target.reshape(-1, nodes, rank_dim),
        weighted_source.reshape(-1, nodes, rank_dim).transpose(1, 2),
    ).reshape(flat_val.shape[0], num_heads, nodes, nodes).contiguous()
    if bias is not None:
        head_scores = head_scores + bias.view(1, num_heads, 1, 1).to(
            dtype=head_scores.dtype,
            device=head_scores.device,
        )
    head_weights = torch.softmax(torch.nan_to_num(head_scores).abs(), dim=1)
    scores = (head_scores * head_weights).sum(dim=1)
    mask = torch.tril(torch.ones((nodes, nodes), device=scores.device, dtype=torch.bool)).view(1, nodes, nodes)
    stats = torch.nan_to_num(scores).abs().masked_fill(~mask, float("-inf"))
    row_max = stats.max(dim=-1).values
    shifted = stats - row_max.unsqueeze(-1)
    exp_stats = torch.exp(shifted) * mask.to(dtype=shifted.dtype)
    row_denom = exp_stats.sum(dim=-1).clamp_min(torch.finfo(exp_stats.dtype).tiny)
    probs = exp_stats / row_denom.unsqueeze(-1)
    edges = torch.sign(torch.nan_to_num(scores)) * probs * mask.to(dtype=probs.dtype)
    delta_state = torch.bmm(edges.to(dtype=flat_projected_state.dtype), flat_projected_state.unsqueeze(-1)).squeeze(-1)
    delta_val = torch.bmm(edges.to(dtype=flat_projected_val.dtype), flat_projected_val)
    return (
        delta_state.reshape(*batch_shape, nodes),
        delta_val.reshape(*batch_shape, nodes, out_dim),
        scores,
        row_max,
        row_denom,
        projected_target,
        projected_source,
    )


def _multihead_low_rank_propagation_causal_dense_signed_abs_backward_native(
    *,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
) -> tuple[Tensor | None, ...]:
    result = _native_module().multihead_low_rank_propagation_causal_dense_signed_abs_backward_cuda(
        layer_val,
        projected_state,
        projected_val,
        source_weight,
        target_weight,
        core_weight,
        _save_optional_tensor(bias, core_weight),
        grad_delta_state,
        grad_delta_val,
    )
    expected = 7
    if not isinstance(result, (list, tuple)) or len(result) != expected:
        raise TypeError(
            "multihead_low_rank_propagation_causal_dense_signed_abs_backward_cuda must return one grad per tensor input."
    )
    return tuple(None if grad is None else grad for grad in result)


def _multihead_lowrank_signed_smoothmax_dense_backward_full_gemm(
    *,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
    saved_projected_target: Tensor | None = None,
    saved_projected_source: Tensor | None = None,
    saved_row_max: Tensor | None = None,
    saved_row_denom: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    (
        flat_val,
        flat_projected_state,
        flat_projected_val,
        batch_shape,
        nodes,
        out_dim,
    ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
    flat_grad_state = grad_delta_state.reshape(-1, nodes).contiguous()
    flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).contiguous()

    compute_dtype = flat_val.dtype
    stats_dtype = torch.float32
    flat_grad_state_compute = flat_grad_state.to(dtype=compute_dtype)
    flat_grad_val_compute = flat_grad_val.to(dtype=compute_dtype)
    flat_projected_state_compute = flat_projected_state.to(dtype=compute_dtype)
    flat_projected_val_compute = flat_projected_val.to(dtype=compute_dtype)
    batch = flat_val.shape[0]
    heads = int(source_weight.shape[0])
    rank_dim = int(core_weight.shape[1])
    projected_target = saved_projected_target
    if projected_target is None:
        projected_target = torch.matmul(
            flat_val.unsqueeze(1),
            target_weight.transpose(1, 2).unsqueeze(0),
        ).contiguous()
    else:
        projected_target = saved_projected_target.contiguous()
    projected_source = saved_projected_source
    if projected_source is None:
        projected_source = torch.matmul(
            flat_val.unsqueeze(1),
            source_weight.transpose(1, 2).unsqueeze(0),
        ).contiguous()
    else:
        projected_source = saved_projected_source.contiguous()
    projected_target = projected_target.to(dtype=compute_dtype)
    projected_source = projected_source.to(dtype=compute_dtype)
    weighted_source = projected_source * core_weight.view(
        1,
        heads,
        1,
        rank_dim,
    ).to(dtype=projected_source.dtype, device=projected_source.device)
    projected_target_flat = projected_target.reshape(batch * heads, nodes, rank_dim)
    weighted_source_flat = weighted_source.reshape(batch * heads, nodes, rank_dim)
    weighted_source_t = weighted_source_flat.transpose(1, 2).contiguous()
    source_weight_compute = source_weight.to(dtype=compute_dtype)
    target_weight_compute = target_weight.to(dtype=compute_dtype)
    flat_val_compute = flat_val.to(dtype=compute_dtype)
    flat_val_matrix = flat_val_compute.reshape(1, batch * nodes, flat_val_compute.shape[-1]).expand(
        heads,
        -1,
        -1,
    )
    grad_projected_state = torch.zeros_like(flat_projected_state_compute)
    grad_projected_val = torch.zeros_like(flat_projected_val_compute)
    grad_layer = torch.zeros(
        (batch, nodes, flat_val.shape[-1]),
        device=flat_val.device,
        dtype=compute_dtype,
    )
    grad_target_weight = torch.zeros_like(target_weight_compute)
    grad_weighted_source = torch.zeros_like(weighted_source, dtype=compute_dtype)
    grad_bias = None if bias is None else torch.zeros((heads,), device=flat_val.device, dtype=compute_dtype)
    row_tile = 256
    source_index = torch.arange(nodes, device=flat_val.device).view(1, 1, nodes)

    for row_start in range(0, nodes, row_tile):
        row_end = min(row_start + row_tile, nodes)
        tile_len = row_end - row_start
        projected_target_tile = projected_target[:, :, row_start:row_end, :].contiguous()
        projected_target_tile_flat = projected_target_tile.reshape(batch * heads, tile_len, rank_dim)
        head_scores_tile = torch.bmm(
            projected_target_tile_flat,
            weighted_source_t,
        ).reshape(batch, heads, tile_len, nodes).contiguous()
        if bias is not None:
            head_scores_tile = head_scores_tile + bias.view(1, bias.shape[0], 1, 1).to(
                dtype=head_scores_tile.dtype,
                device=head_scores_tile.device,
            )
        clean_head_scores_tile = torch.nan_to_num(head_scores_tile)
        head_weights_tile = torch.softmax(clean_head_scores_tile.abs(), dim=1)
        scores_tile = (clean_head_scores_tile * head_weights_tile).sum(dim=1)
        row_index = torch.arange(row_start, row_end, device=flat_val.device).view(1, tile_len, 1)
        mask_tile = source_index <= row_index
        if (
            saved_row_max is not None
            and saved_row_denom is not None
            and triton_signed_smoothmax_available()
            and signed_abs_softmax_edge_dot_tile is not None
            and signed_abs_softmax_backward_tile is not None
            and scores_tile.is_cuda
        ):
            grad_edges_tile = torch.bmm(
                flat_grad_val_compute[:, row_start:row_end, :],
                flat_projected_val_compute.transpose(1, 2),
            )
            grad_edges_tile = grad_edges_tile + (
                flat_grad_state_compute[:, row_start:row_end].unsqueeze(-1)
                * flat_projected_state_compute.unsqueeze(1)
            )
            row_max_tile = saved_row_max[:, row_start:row_end]
            row_denom_tile = saved_row_denom[:, row_start:row_end]
            edge_dot_tile = signed_abs_softmax_edge_dot_tile(
                scores_tile,
                grad_edges_tile,
                row_max_tile,
                row_denom_tile,
                0,
                target_start=row_start,
            )
            edges_tile, grad_scores_tile = signed_abs_softmax_backward_tile(
                scores_tile,
                grad_edges_tile,
                row_max_tile,
                row_denom_tile,
                edge_dot_tile,
                0,
                target_start=row_start,
            )
        elif (
            _env_flag("JAKAL_NET_ENABLE_EXPERIMENTAL_SIGNED_ABS_SOFTMAX_FROM_PROJ")
            and triton_signed_smoothmax_available()
            and signed_abs_softmax_tile_stats_from_projections is not None
            and signed_abs_softmax_backward_tile_from_projections is not None
            and scores_tile.is_cuda
        ):
            row_max_tile, row_denom_tile, row_numer_tile = signed_abs_softmax_tile_stats_from_projections(
                scores_tile,
                flat_projected_state_compute,
                flat_projected_val_compute,
                flat_grad_state_compute[:, row_start:row_end],
                flat_grad_val_compute[:, row_start:row_end, :],
                0,
                target_start=row_start,
            )
            edge_dot_tile = row_numer_tile / row_denom_tile.clamp_min(torch.finfo(row_denom_tile.dtype).tiny)
            edges_tile, grad_scores_tile = signed_abs_softmax_backward_tile_from_projections(
                scores_tile,
                flat_projected_state_compute,
                flat_projected_val_compute,
                flat_grad_state_compute[:, row_start:row_end],
                flat_grad_val_compute[:, row_start:row_end, :],
                row_max_tile,
                row_denom_tile,
                edge_dot_tile,
                0,
                target_start=row_start,
            )
        elif (
            triton_signed_smoothmax_available()
            and signed_abs_softmax_tile_stats_from_projections is not None
            and signed_abs_softmax_backward_tile_from_projections is not None
            and scores_tile.is_cuda
        ):
            grad_edges_tile = torch.bmm(
                flat_grad_val_compute[:, row_start:row_end, :],
                flat_projected_val_compute.transpose(1, 2),
            )
            grad_edges_tile = grad_edges_tile + (
                flat_grad_state_compute[:, row_start:row_end].unsqueeze(-1)
                * flat_projected_state_compute.unsqueeze(1)
            )
            row_max_tile, row_denom_tile, row_numer_tile = signed_abs_softmax_tile_stats(
                scores_tile,
                grad_edges_tile,
                0,
                target_start=row_start,
            )
            edge_dot_tile = row_numer_tile / row_denom_tile.clamp_min(torch.finfo(row_denom_tile.dtype).tiny)
            edges_tile, grad_scores_tile = signed_abs_softmax_backward_tile(
                scores_tile,
                grad_edges_tile,
                row_max_tile,
                row_denom_tile,
                edge_dot_tile,
                0,
                target_start=row_start,
            )
        else:
            grad_edges_tile = torch.bmm(
                flat_grad_val_compute[:, row_start:row_end, :],
                flat_projected_val_compute.transpose(1, 2),
            )
            grad_edges_tile = grad_edges_tile + (
                flat_grad_state_compute[:, row_start:row_end].unsqueeze(-1)
                * flat_projected_state_compute.unsqueeze(1)
            )
            edges_tile = _masked_signed_abs_softmax_from_scores(scores_tile, mask_tile)
            grad_scores_tile = _masked_signed_abs_softmax_backward(
                scores_tile.contiguous(),
                edges_tile.contiguous(),
                grad_edges_tile.contiguous(),
                mask_tile,
            )
        grad_projected_state = grad_projected_state + torch.bmm(
            edges_tile.transpose(1, 2).to(dtype=compute_dtype),
            flat_grad_state_compute[:, row_start:row_end].unsqueeze(-1),
        ).squeeze(-1)
        grad_projected_val = grad_projected_val + torch.bmm(
            edges_tile.transpose(1, 2).to(dtype=compute_dtype),
            flat_grad_val_compute[:, row_start:row_end, :],
        )
        head_factor_tile = 1.0 + torch.sign(clean_head_scores_tile) * (
            clean_head_scores_tile - scores_tile.unsqueeze(1)
        )
        grad_head_scores_tile = (
            grad_scores_tile.unsqueeze(1).to(dtype=compute_dtype)
            * head_weights_tile.to(dtype=compute_dtype)
            * head_factor_tile.to(dtype=compute_dtype)
        ).contiguous()
        if grad_bias is not None:
            grad_bias = grad_bias + grad_head_scores_tile.sum(dim=(0, 2, 3))
        grad_head_scores_tile_flat = grad_head_scores_tile.reshape(batch * heads, tile_len, nodes)
        grad_target_tile = torch.bmm(
            grad_head_scores_tile_flat,
            weighted_source_flat,
        ).reshape(batch, heads, tile_len, rank_dim).contiguous()
        grad_weighted_source = grad_weighted_source + torch.bmm(
            grad_head_scores_tile_flat.transpose(1, 2),
            projected_target_tile_flat.to(dtype=compute_dtype),
        ).reshape(batch, heads, nodes, rank_dim)
        grad_target_rows = grad_target_tile.permute(1, 0, 2, 3).reshape(heads, batch * tile_len, rank_dim)
        grad_layer[:, row_start:row_end, :] = grad_layer[:, row_start:row_end, :] + torch.bmm(
            grad_target_rows,
            target_weight_compute,
        ).sum(dim=0).reshape(batch, tile_len, flat_val.shape[-1])
        flat_val_tile_matrix = flat_val_compute[:, row_start:row_end, :].reshape(
            1,
            batch * tile_len,
            flat_val_compute.shape[-1],
        ).expand(heads, -1, -1)
        grad_target_weight = grad_target_weight + torch.bmm(
            grad_target_tile.permute(1, 3, 0, 2).reshape(heads, rank_dim, batch * tile_len),
            flat_val_tile_matrix,
        )

    grad_source = grad_weighted_source * core_weight.view(
        1,
        heads,
        1,
        rank_dim,
    ).to(dtype=compute_dtype, device=grad_weighted_source.device)
    grad_source_rows = grad_source.permute(1, 0, 2, 3).reshape(heads, batch * nodes, rank_dim)
    grad_layer = grad_layer + torch.bmm(
        grad_source_rows,
        source_weight_compute,
    ).sum(dim=0).reshape(batch, nodes, flat_val.shape[-1])
    grad_source_weight = torch.bmm(
        grad_source.permute(1, 3, 0, 2).reshape(heads, rank_dim, batch * nodes),
        flat_val_matrix,
    )
    grad_core_weight = (
        grad_weighted_source.to(dtype=stats_dtype)
        * projected_source.to(dtype=stats_dtype)
    ).sum(dim=(0, 2))
    if grad_bias is not None:
        grad_bias = grad_bias.to(dtype=bias.dtype)

    return (
        grad_layer.reshape(*batch_shape, nodes, flat_val.shape[-1]).to(dtype=layer_val.dtype),
        grad_projected_state.reshape_as(projected_state).to(dtype=projected_state.dtype),
        grad_projected_val.reshape_as(projected_val).to(dtype=projected_val.dtype),
        grad_source_weight.to(dtype=source_weight.dtype),
        grad_target_weight.to(dtype=target_weight.dtype),
        grad_core_weight.to(dtype=core_weight.dtype),
        grad_bias,
    )


def _multihead_lowrank_signed_smoothmax_dense_backward_triton_gemm(
    *,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    (
        flat_val,
        flat_projected_state,
        flat_projected_val,
        batch_shape,
        nodes,
        out_dim,
    ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
    flat_grad_state = grad_delta_state.reshape(-1, nodes).contiguous()
    flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).contiguous()

    compute_dtype = flat_val.dtype
    projected_target_bhnr = torch.matmul(
        flat_val.unsqueeze(1),
        target_weight.transpose(1, 2).unsqueeze(0),
    ).contiguous()
    projected_source_bhnr = torch.matmul(
        flat_val.unsqueeze(1),
        source_weight.transpose(1, 2).unsqueeze(0),
    ).contiguous()
    weighted_source_bhnr = projected_source_bhnr * core_weight.view(
        1,
        core_weight.shape[0],
        1,
        core_weight.shape[1],
    ).to(dtype=projected_source_bhnr.dtype, device=projected_source_bhnr.device)

    batch = flat_val.shape[0]
    stats_dtype = torch.float32
    flat_grad_state_compute = flat_grad_state.to(dtype=compute_dtype)
    flat_grad_val_compute = flat_grad_val.to(dtype=compute_dtype)
    flat_projected_state_compute = flat_projected_state.to(dtype=compute_dtype)
    flat_projected_val_compute = flat_projected_val.to(dtype=compute_dtype)
    row_max = torch.full((batch, nodes), float("-inf"), device=flat_val.device, dtype=stats_dtype)
    row_denom = torch.zeros((batch, nodes), device=flat_val.device, dtype=stats_dtype)
    edge_numer = torch.zeros((batch, nodes), device=flat_val.device, dtype=stats_dtype)
    tile_nodes = 128
    row_index = torch.arange(nodes, device=flat_val.device).view(1, nodes, 1)

    for source_start in range(0, nodes, tile_nodes):
        source_end = min(source_start + tile_nodes, nodes)
        tile_scores = multihead_signed_smoothmax_scores_tile(
            projected_target_bhnr,
            weighted_source_bhnr[:, :, source_start:source_end, :],
            bias,
        )
        grad_edges_tile = torch.bmm(
            flat_grad_val_compute,
            flat_projected_val_compute[:, source_start:source_end, :].transpose(1, 2),
        )
        grad_edges_tile = grad_edges_tile + (
            flat_grad_state_compute.unsqueeze(-1)
            * flat_projected_state_compute[:, None, source_start:source_end]
        )
        tile_max, tile_denom, tile_numer = signed_abs_softmax_tile_stats(
            tile_scores,
            grad_edges_tile,
            source_start,
        )
        new_row_max = torch.maximum(row_max, tile_max)
        prev_scale = torch.where(
            torch.isfinite(row_max),
            torch.exp(row_max - new_row_max),
            torch.zeros_like(row_max),
        )
        tile_scale = torch.where(
            torch.isfinite(tile_max),
            torch.exp(tile_max - new_row_max),
            torch.zeros_like(tile_max),
        )
        row_denom = row_denom * prev_scale + tile_denom * tile_scale
        edge_numer = edge_numer * prev_scale + tile_numer * tile_scale
        row_max = new_row_max

    row_denom = row_denom.clamp_min(torch.finfo(row_denom.dtype).tiny)
    edge_dot = edge_numer / row_denom
    grad_projected_state = torch.zeros_like(flat_projected_state_compute)
    grad_projected_val = torch.zeros_like(flat_projected_val_compute)

    grad_target_compute = torch.zeros_like(projected_target_bhnr, dtype=compute_dtype)
    grad_source_compute = torch.zeros_like(projected_source_bhnr, dtype=compute_dtype)
    flat_val_compute = flat_val.to(dtype=compute_dtype)
    grad_core_weight = torch.zeros_like(core_weight, dtype=torch.float32)
    grad_bias = None if bias is None else torch.zeros_like(bias, dtype=torch.float32)
    flat_val_matrix = flat_val_compute.reshape(1, batch * nodes, flat_val_compute.shape[-1]).expand(
        int(source_weight.shape[0]),
        -1,
        -1,
    )

    for source_start in range(0, nodes, tile_nodes):
        source_end = min(source_start + tile_nodes, nodes)
        tile_scores, head_grads = multihead_signed_smoothmax_scores_and_head_grads_tile(
            projected_target_bhnr,
            weighted_source_bhnr[:, :, source_start:source_end, :],
            source_start,
            bias,
        )
        tile_len = source_end - source_start
        source_index = torch.arange(source_start, source_end, device=flat_val.device).view(1, 1, tile_len)
        causal = source_index <= row_index
        tile_probs = torch.where(
            causal,
            torch.exp(tile_scores.abs() - row_max.unsqueeze(-1)) / row_denom.unsqueeze(-1),
            torch.zeros_like(tile_scores),
        )
        tile_signs = torch.sign(torch.nan_to_num(tile_scores))
        tile_edges = tile_signs * tile_probs
        grad_edges_tile = torch.bmm(
            flat_grad_val_compute,
            flat_projected_val_compute[:, source_start:source_end, :].transpose(1, 2),
        )
        grad_edges_tile = grad_edges_tile + (
            flat_grad_state_compute.unsqueeze(-1)
            * flat_projected_state_compute[:, None, source_start:source_end]
        )
        grad_scores_tile = tile_signs * tile_probs * (
            tile_signs * grad_edges_tile - edge_dot.unsqueeze(-1)
        )
        grad_scores_tile = torch.where(causal, grad_scores_tile, torch.zeros_like(grad_scores_tile))
        grad_projected_state[:, source_start:source_end] = grad_projected_state[:, source_start:source_end] + torch.bmm(
            tile_edges.transpose(1, 2).to(dtype=compute_dtype),
            flat_grad_state_compute.unsqueeze(-1),
        ).squeeze(-1)
        grad_projected_val[:, source_start:source_end, :] = grad_projected_val[:, source_start:source_end, :] + torch.bmm(
            tile_edges.transpose(1, 2).to(dtype=compute_dtype),
            flat_grad_val_compute,
        )
        grad_target_tile, grad_source_tile, grad_core_tile, grad_bias_tile = multihead_signed_smoothmax_tile_partials(
            projected_target_bhnr,
            projected_source_bhnr[:, :, source_start:source_end, :],
            weighted_source_bhnr[:, :, source_start:source_end, :],
            core_weight,
            grad_scores_tile,
            head_grads,
            bias,
        )
        grad_target_compute = grad_target_compute + grad_target_tile.to(dtype=compute_dtype)
        grad_source_compute[:, :, source_start:source_end, :] = (
            grad_source_compute[:, :, source_start:source_end, :]
            + grad_source_tile.to(dtype=compute_dtype)
        )
        grad_core_weight = grad_core_weight + grad_core_tile
        if grad_bias is not None and grad_bias_tile is not None:
            grad_bias = grad_bias + grad_bias_tile

    heads = int(source_weight.shape[0])
    rank_dim = int(core_weight.shape[1])
    grad_target_rows = grad_target_compute.permute(1, 0, 2, 3).reshape(heads, batch * nodes, rank_dim)
    grad_source_rows = grad_source_compute.permute(1, 0, 2, 3).reshape(heads, batch * nodes, rank_dim)
    grad_layer = torch.bmm(
        grad_target_rows,
        target_weight.to(dtype=compute_dtype),
    ).sum(dim=0).reshape(batch, nodes, flat_val.shape[-1])
    grad_layer = grad_layer + torch.bmm(
        grad_source_rows,
        source_weight.to(dtype=compute_dtype),
    ).sum(dim=0).reshape(batch, nodes, flat_val.shape[-1])
    grad_source_weight = torch.bmm(
        grad_source_compute.permute(1, 3, 0, 2).reshape(heads, rank_dim, batch * nodes),
        flat_val_matrix,
    )
    grad_target_weight = torch.bmm(
        grad_target_compute.permute(1, 3, 0, 2).reshape(heads, rank_dim, batch * nodes),
        flat_val_matrix,
    )

    return (
        grad_layer.reshape(*batch_shape, nodes, flat_val.shape[-1]).to(dtype=layer_val.dtype),
        grad_projected_state.reshape_as(projected_state).to(dtype=projected_state.dtype),
        grad_projected_val.reshape_as(projected_val).to(dtype=projected_val.dtype),
        grad_source_weight.to(dtype=source_weight.dtype),
        grad_target_weight.to(dtype=target_weight.dtype),
        grad_core_weight.to(dtype=core_weight.dtype),
        None if grad_bias is None else grad_bias.to(dtype=bias.dtype if bias is not None else grad_bias.dtype),
    )


def _multihead_lowrank_signed_smoothmax_dense_backward_triton_owner(
    *,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    (
        flat_val,
        flat_projected_state,
        flat_projected_val,
        batch_shape,
        nodes,
        out_dim,
    ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
    flat_grad_state = grad_delta_state.reshape(-1, nodes).contiguous()
    flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).contiguous()

    compute_dtype = flat_val.dtype
    projected_target_bhnr = torch.matmul(
        flat_val.unsqueeze(1),
        target_weight.transpose(1, 2).unsqueeze(0),
    ).contiguous()
    projected_source_bhnr = torch.matmul(
        flat_val.unsqueeze(1),
        source_weight.transpose(1, 2).unsqueeze(0),
    ).contiguous()
    weighted_source_bhnr = projected_source_bhnr * core_weight.view(
        1,
        core_weight.shape[0],
        1,
        core_weight.shape[1],
    ).to(dtype=projected_source_bhnr.dtype, device=projected_source_bhnr.device)

    projected_target_hbnr = projected_target_bhnr.permute(1, 0, 2, 3).contiguous()
    projected_source_hbnr = projected_source_bhnr.permute(1, 0, 2, 3).contiguous()
    weighted_source_hbnr = weighted_source_bhnr.permute(1, 0, 2, 3).contiguous()
    row_max, row_denom, edge_dot = multihead_signed_smoothmax_pass1_full(
        projected_target_hbnr,
        weighted_source_hbnr,
        flat_projected_state,
        flat_projected_val,
        flat_grad_state,
        flat_grad_val,
        bias,
    )
    if lowrank_signed_smoothmax_backward_owner_generic is None:
        raise RuntimeError("Generic Triton owner backward is unavailable.")
    (
        grad_target_hbnr,
        grad_source_hbnr,
        grad_projected_state_hbnr_unused,
        grad_projected_val_hbnr_unused,
        grad_core_partial,
        grad_bias_partial,
    ) = lowrank_signed_smoothmax_backward_owner_generic(
        projected_target_hbnr,
        projected_source_hbnr,
        weighted_source_hbnr,
        core_weight.contiguous(),
        flat_projected_state,
        flat_projected_val,
        flat_grad_state,
        flat_grad_val,
        row_max,
        row_denom,
        edge_dot,
        bias,
    )
    grad_target_bhnr = grad_target_hbnr.permute(1, 0, 2, 3).contiguous()
    grad_source_bhnr = grad_source_hbnr.permute(1, 0, 2, 3).contiguous()
    grad_target_compute = grad_target_bhnr.to(dtype=compute_dtype)
    grad_source_compute = grad_source_bhnr.to(dtype=compute_dtype)
    flat_val_compute = flat_val.to(dtype=compute_dtype)
    heads = int(source_weight.shape[0])
    rank_dim = int(core_weight.shape[1])
    flat_val_matrix = flat_val_compute.reshape(1, flat_val_compute.shape[0] * nodes, flat_val_compute.shape[-1]).expand(heads, -1, -1)
    grad_target_rows = grad_target_compute.permute(1, 0, 2, 3).reshape(heads, flat_val_compute.shape[0] * nodes, rank_dim)
    grad_source_rows = grad_source_compute.permute(1, 0, 2, 3).reshape(heads, flat_val_compute.shape[0] * nodes, rank_dim)
    grad_layer = torch.bmm(
        grad_target_rows,
        target_weight.to(dtype=compute_dtype),
    ).sum(dim=0).reshape(flat_val_compute.shape[0], nodes, flat_val_compute.shape[-1])
    grad_layer = grad_layer + torch.bmm(
        grad_source_rows,
        source_weight.to(dtype=compute_dtype),
    ).sum(dim=0).reshape(flat_val_compute.shape[0], nodes, flat_val_compute.shape[-1])
    grad_source_weight = torch.bmm(
        grad_source_compute.permute(1, 3, 0, 2).reshape(heads, rank_dim, flat_val_compute.shape[0] * nodes),
        flat_val_matrix,
    )
    grad_target_weight = torch.bmm(
        grad_target_compute.permute(1, 3, 0, 2).reshape(heads, rank_dim, flat_val_compute.shape[0] * nodes),
        flat_val_matrix,
    )
    grad_core_weight = grad_core_partial.sum(dim=(0, 1)).to(dtype=core_weight.dtype)
    grad_bias = None
    if grad_bias_partial is not None:
        grad_bias = grad_bias_partial.sum(dim=(0, 1)).to(dtype=bias.dtype if bias is not None else grad_bias_partial.dtype)
    grad_projected_state = grad_projected_state_hbnr_unused.to(dtype=projected_state.dtype)
    grad_projected_val = grad_projected_val_hbnr_unused.to(dtype=projected_val.dtype)

    return (
        grad_layer.reshape(*batch_shape, nodes, flat_val.shape[-1]).to(dtype=layer_val.dtype),
        grad_projected_state.reshape_as(projected_state),
        grad_projected_val.reshape_as(projected_val),
        grad_source_weight.to(dtype=source_weight.dtype),
        grad_target_weight.to(dtype=target_weight.dtype),
        grad_core_weight,
        grad_bias,
    )


def _multihead_lowrank_signed_smoothmax_dense_backward_analytic(
    *,
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    grad_delta_state: Tensor,
    grad_delta_val: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
    (
        flat_val,
        flat_projected_state,
        flat_projected_val,
        batch_shape,
        nodes,
        out_dim,
    ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
    flat_grad_state = grad_delta_state.reshape(-1, nodes).contiguous()
    flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).contiguous()
    flat_val_2d = flat_val.reshape(-1, flat_val.shape[-1]).contiguous()

    num_heads = int(source_weight.shape[0])
    target_heads: list[Tensor] = []
    source_heads: list[Tensor] = []
    weighted_source_heads: list[Tensor] = []
    score_heads: list[Tensor] = []
    for head_index in range(num_heads):
        target_head = torch.matmul(flat_val, target_weight[head_index].t()).contiguous()
        source_head = torch.matmul(flat_val, source_weight[head_index].t()).contiguous()
        weighted_source_head = source_head * core_weight[head_index].to(
            dtype=source_head.dtype,
            device=source_head.device,
        ).view(1, 1, -1)
        score_head = torch.bmm(target_head, weighted_source_head.transpose(1, 2))
        if bias is not None:
            score_head = score_head + bias[head_index].to(
                device=score_head.device,
                dtype=score_head.dtype,
            )
        target_heads.append(target_head)
        source_heads.append(source_head)
        weighted_source_heads.append(weighted_source_head)
        score_heads.append(score_head)

    head_scores = torch.stack(score_heads, dim=1)
    clean_head_scores = torch.nan_to_num(head_scores)
    head_weights = torch.softmax(clean_head_scores.abs(), dim=1)
    scores = (head_scores * head_weights).sum(dim=1)
    mask = torch.tril(torch.ones((nodes, nodes), device=scores.device, dtype=torch.bool)).view(
        1, nodes, nodes
    )
    stats = torch.nan_to_num(scores).abs().masked_fill(~mask, float("-inf"))
    row_max = stats.max(dim=-1).values
    shifted = stats - row_max.unsqueeze(-1)
    exp_stats = torch.exp(shifted) * mask.to(dtype=shifted.dtype)
    row_denom = exp_stats.sum(dim=-1).clamp_min(torch.finfo(exp_stats.dtype).tiny)
    probs = exp_stats / row_denom.unsqueeze(-1)
    edges = torch.sign(torch.nan_to_num(scores)) * probs * mask.to(dtype=probs.dtype)

    grad_edges_val = torch.bmm(flat_grad_val, flat_projected_val.transpose(1, 2))
    grad_scores_val = _masked_signed_abs_softmax_backward(
        scores.contiguous(),
        edges.contiguous(),
        grad_edges_val.contiguous(),
        mask,
    )
    grad_projected_val = torch.bmm(
        edges.transpose(1, 2).to(dtype=flat_grad_val.dtype),
        flat_grad_val,
    )
    grad_edges_state = flat_grad_state.unsqueeze(-1) * flat_projected_state.unsqueeze(1)
    grad_projected_state = torch.bmm(
        edges.transpose(1, 2).to(dtype=flat_grad_state.dtype),
        flat_grad_state.unsqueeze(-1),
    ).squeeze(-1)
    grad_scores_state = _masked_signed_abs_softmax_backward(
        scores.contiguous(),
        edges.contiguous(),
        grad_edges_state.contiguous(),
        mask,
    )
    grad_scores = grad_scores_val + grad_scores_state
    grad_head_scores = grad_scores.unsqueeze(1) * head_weights * (
        1.0
        + torch.sign(clean_head_scores)
        * (clean_head_scores - scores.unsqueeze(1))
    )

    grad_layer = torch.zeros_like(flat_val)
    grad_source_weight = torch.zeros_like(source_weight)
    grad_target_weight = torch.zeros_like(target_weight)
    grad_core_weight = torch.zeros_like(core_weight)
    grad_bias = None if bias is None else torch.zeros_like(bias)
    for head_index in range(num_heads):
        grad_score_head = grad_head_scores[:, head_index].contiguous()
        target_head = target_heads[head_index]
        source_head = source_heads[head_index]
        weighted_source_head = weighted_source_heads[head_index]
        grad_target_head = torch.bmm(grad_score_head, weighted_source_head)
        grad_weighted_source_head = torch.bmm(grad_score_head.transpose(1, 2), target_head)
        grad_source_head = grad_weighted_source_head * core_weight[head_index].to(
            dtype=grad_weighted_source_head.dtype,
            device=grad_weighted_source_head.device,
        ).view(1, 1, -1)
        grad_core_weight[head_index] = (
            grad_weighted_source_head * source_head
        ).sum(dim=(0, 1)).to(dtype=grad_core_weight.dtype)
        grad_target_weight[head_index] = (
            grad_target_head.reshape(-1, grad_target_head.shape[-1]).transpose(0, 1)
            @ flat_val_2d
        ).to(dtype=grad_target_weight.dtype)
        grad_source_weight[head_index] = (
            grad_source_head.reshape(-1, grad_source_head.shape[-1]).transpose(0, 1)
            @ flat_val_2d
        ).to(dtype=grad_source_weight.dtype)
        grad_layer = grad_layer + torch.matmul(
            grad_target_head.to(dtype=flat_val.dtype),
            target_weight[head_index].to(dtype=flat_val.dtype),
        )
        grad_layer = grad_layer + torch.matmul(
            grad_source_head.to(dtype=flat_val.dtype),
            source_weight[head_index].to(dtype=flat_val.dtype),
        )
        if grad_bias is not None:
            grad_bias[head_index] = grad_score_head.sum().to(dtype=grad_bias.dtype)

    return (
        grad_layer.reshape(*batch_shape, nodes, flat_val.shape[-1]).to(dtype=layer_val.dtype),
        grad_projected_state.reshape_as(projected_state).to(dtype=projected_state.dtype),
        grad_projected_val.reshape_as(projected_val).to(dtype=projected_val.dtype),
        grad_source_weight,
        grad_target_weight,
        grad_core_weight,
        grad_bias,
    )


class _MultiHeadLowRankPropagationDenseSignedAbsTriton(Function):
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
    ) -> tuple[Tensor, Tensor]:
        (
            delta_state,
            delta_val,
            _scores_unused,
            row_max,
            row_denom,
            projected_target,
            projected_source,
        ) = _multihead_lowrank_signed_smoothmax_dense_forward(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            bias,
        )
        ctx.has_bias = bias is not None
        save_policy = _experimental_dense_mh_save_policy()
        ctx.save_policy = save_policy
        ctx.save_for_backward(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            _save_optional_tensor(bias, core_weight),
            projected_target if save_policy == "speed" else _save_optional_tensor(None, layer_val),
            projected_source if save_policy in {"balanced", "speed"} else _save_optional_tensor(None, layer_val),
            row_max if save_policy in {"balanced", "speed"} else _save_optional_tensor(None, layer_val),
            row_denom if save_policy in {"balanced", "speed"} else _save_optional_tensor(None, layer_val),
        )
        return delta_state, delta_val

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
            saved_projected_target,
            saved_projected_source,
            saved_row_max,
            saved_row_denom,
        ) = ctx.saved_tensors
        bias = _load_optional_tensor(bias_tensor)
        projected_target = _load_optional_tensor(saved_projected_target)
        projected_source = _load_optional_tensor(saved_projected_source)
        row_max = _load_optional_tensor(saved_row_max)
        row_denom = _load_optional_tensor(saved_row_denom)
        if layer_val.is_cuda and int(source_weight.shape[0]) == 4:
            (
                grad_layer,
                grad_projected_state,
                grad_projected_val,
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias,
            ) = _multihead_lowrank_signed_smoothmax_dense_backward_full_gemm(
                layer_val=layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_weight=source_weight,
                target_weight=target_weight,
                core_weight=core_weight,
                bias=bias,
                grad_delta_state=grad_delta_state,
                grad_delta_val=grad_delta_val,
                saved_projected_target=projected_target,
                saved_projected_source=projected_source,
                saved_row_max=row_max,
                saved_row_denom=row_denom,
            )
            return (
                grad_layer.to(dtype=layer_val.dtype),
                grad_projected_state.to(dtype=projected_state.dtype),
                grad_projected_val.to(dtype=projected_val.dtype),
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias if ctx.has_bias else None,
            )
        if (
            _env_flag("JAKAL_NET_ENABLE_EXPERIMENTAL_TRITON_OWNER")
            and triton_signed_smoothmax_available()
            and lowrank_signed_smoothmax_backward_owner_generic is not None
            and layer_val.is_cuda
        ):
            (
                grad_layer,
                grad_projected_state,
                grad_projected_val,
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias,
            ) = _multihead_lowrank_signed_smoothmax_dense_backward_triton_owner(
                layer_val=layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_weight=source_weight,
                target_weight=target_weight,
                core_weight=core_weight,
                bias=bias,
                grad_delta_state=grad_delta_state,
                grad_delta_val=grad_delta_val,
            )
            return (
                grad_layer.to(dtype=layer_val.dtype),
                grad_projected_state.to(dtype=projected_state.dtype),
                grad_projected_val.to(dtype=projected_val.dtype),
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias if ctx.has_bias else None,
            )
        if (
            triton_signed_smoothmax_available()
            and multihead_signed_smoothmax_tile_partials is not None
            and multihead_signed_smoothmax_scores_and_head_grads_tile is not None
            and multihead_signed_smoothmax_scores_tile is not None
            and signed_abs_softmax_edge_dot_tile is not None
            and layer_val.is_cuda
        ):
            (
                grad_layer,
                grad_projected_state,
                grad_projected_val,
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias,
            ) = _multihead_lowrank_signed_smoothmax_dense_backward_triton_gemm(
                layer_val=layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_weight=source_weight,
                target_weight=target_weight,
                core_weight=core_weight,
                bias=bias,
                grad_delta_state=grad_delta_state,
                grad_delta_val=grad_delta_val,
            )
            return (
                grad_layer.to(dtype=layer_val.dtype),
                grad_projected_state.to(dtype=projected_state.dtype),
                grad_projected_val.to(dtype=projected_val.dtype),
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias if ctx.has_bias else None,
            )
        if native_supports("multihead_low_rank_propagation_causal_dense_signed_abs_backward_cuda"):
            (
                grad_layer,
                grad_projected_state,
                grad_projected_val,
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias,
            ) = _multihead_low_rank_propagation_causal_dense_signed_abs_backward_native(
                layer_val=layer_val,
                projected_state=projected_state,
                projected_val=projected_val,
                source_weight=source_weight,
                target_weight=target_weight,
                core_weight=core_weight,
                bias=bias,
                grad_delta_state=grad_delta_state,
                grad_delta_val=grad_delta_val,
            )
            return (
                grad_layer.to(dtype=layer_val.dtype),
                grad_projected_state.to(dtype=projected_state.dtype),
                grad_projected_val.to(dtype=projected_val.dtype),
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias if ctx.has_bias else None,
            )
        (
            grad_layer,
            grad_projected_state,
            grad_projected_val,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias,
        ) = _multihead_lowrank_signed_smoothmax_dense_backward_analytic(
            layer_val=layer_val,
            projected_state=projected_state,
            projected_val=projected_val,
            source_weight=source_weight,
            target_weight=target_weight,
            core_weight=core_weight,
            bias=bias,
            grad_delta_state=grad_delta_state,
            grad_delta_val=grad_delta_val,
        )
        return (
            grad_layer,
            grad_projected_state,
            grad_projected_val,
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias if ctx.has_bias else None,
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


def _split_nomemory_stack_layer_tensors(
    layer_tensors: tuple[Tensor, ...],
    num_layers: int,
) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]]:
    expected = num_layers * 6
    if len(layer_tensors) != expected:
        raise ValueError(f"Expected {expected} layer tensors, got {len(layer_tensors)}.")
    source_weights: list[Tensor] = []
    target_weights: list[Tensor] = []
    core_weights: list[Tensor] = []
    bias_tensors: list[Tensor] = []
    norm_weights: list[Tensor] = []
    norm_biases: list[Tensor] = []
    for offset in range(0, expected, 6):
        source_weights.append(layer_tensors[offset])
        target_weights.append(layer_tensors[offset + 1])
        core_weights.append(layer_tensors[offset + 2])
        bias_tensors.append(layer_tensors[offset + 3])
        norm_weights.append(layer_tensors[offset + 4])
        norm_biases.append(layer_tensors[offset + 5])
    return (
        tuple(source_weights),
        tuple(target_weights),
        tuple(core_weights),
        tuple(bias_tensors),
        tuple(norm_weights),
        tuple(norm_biases),
    )


def _split_nomemory_stack_ffn_layer_tensors(
    layer_tensors: tuple[Tensor, ...],
    num_layers: int,
) -> tuple[
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]:
    expected = num_layers * 6
    if len(layer_tensors) != expected:
        raise ValueError(f"Expected {expected} FFN layer tensors, got {len(layer_tensors)}.")
    norm_weights: list[Tensor] = []
    norm_biases: list[Tensor] = []
    in_weights: list[Tensor] = []
    in_biases: list[Tensor] = []
    out_weights: list[Tensor] = []
    out_biases: list[Tensor] = []
    for offset in range(0, expected, 6):
        norm_weights.append(layer_tensors[offset])
        norm_biases.append(layer_tensors[offset + 1])
        in_weights.append(layer_tensors[offset + 2])
        in_biases.append(layer_tensors[offset + 3])
        out_weights.append(layer_tensors[offset + 4])
        out_biases.append(layer_tensors[offset + 5])
    return (
        tuple(norm_weights),
        tuple(norm_biases),
        tuple(in_weights),
        tuple(in_biases),
        tuple(out_weights),
        tuple(out_biases),
    )


def _split_nomemory_stack_specs(
    specs: tuple[tuple[int, int, int, int], ...],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    compress_kinds: list[int] = []
    windows: list[int] = []
    target_block_sizes: list[int] = []
    source_block_sizes: list[int] = []
    for compress_kind, window, target_block_size, source_block_size in specs:
        compress_kinds.append(int(compress_kind))
        windows.append(int(window))
        target_block_sizes.append(int(target_block_size))
        source_block_sizes.append(int(source_block_size))
    return (
        tuple(compress_kinds),
        tuple(windows),
        tuple(target_block_sizes),
        tuple(source_block_sizes),
    )


def nomemory_causal_stack_fused_native(
    *,
    token_val: Tensor,
    anchor_state: Tensor,
    anchor_val: Tensor,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    sequence_tensors: tuple[Tensor, ...],
    prediction_tensors: tuple[Tensor, ...],
    sequence_specs: tuple[tuple[int, int, int, int], ...],
    prediction_specs: tuple[tuple[int, int, int, int], ...],
    state_activation_name: str,
) -> Tensor:
    num_sequence_layers = len(sequence_specs)
    num_prediction_layers = len(prediction_specs)
    (
        sequence_source_weights,
        sequence_target_weights,
        sequence_core_weights,
        sequence_biases,
        sequence_norm_weights,
        sequence_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(sequence_tensors, num_sequence_layers)
    (
        prediction_source_weights,
        prediction_target_weights,
        prediction_core_weights,
        prediction_biases,
        prediction_norm_weights,
        prediction_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(prediction_tensors, num_prediction_layers)
    (
        sequence_compress_kinds,
        sequence_windows,
        sequence_target_block_sizes,
        sequence_source_block_sizes,
    ) = _split_nomemory_stack_specs(sequence_specs)
    (
        prediction_compress_kinds,
        prediction_windows,
        prediction_target_block_sizes,
        prediction_source_block_sizes,
    ) = _split_nomemory_stack_specs(prediction_specs)
    result = _native_module().nomemory_causal_stack_fused(
        token_val,
        anchor_state,
        anchor_val,
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, token_val),
        list(sequence_source_weights),
        list(sequence_target_weights),
        list(sequence_core_weights),
        list(sequence_biases),
        list(sequence_norm_weights),
        list(sequence_norm_biases),
        list(sequence_compress_kinds),
        list(sequence_windows),
        list(sequence_target_block_sizes),
        list(sequence_source_block_sizes),
        list(prediction_source_weights),
        list(prediction_target_weights),
        list(prediction_core_weights),
        list(prediction_biases),
        list(prediction_norm_weights),
        list(prediction_norm_biases),
        list(prediction_compress_kinds),
        list(prediction_windows),
        list(prediction_target_block_sizes),
        list(prediction_source_block_sizes),
        state_activation_name,
    )
    if not isinstance(result, Tensor):
        raise TypeError("nomemory_causal_stack_fused must return a Tensor.")
    return result


def nomemory_causal_stack_fused_trace_native(
    *,
    token_val: Tensor,
    anchor_state: Tensor,
    anchor_val: Tensor,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    sequence_tensors: tuple[Tensor, ...],
    prediction_tensors: tuple[Tensor, ...],
    sequence_specs: tuple[tuple[int, int, int, int], ...],
    prediction_specs: tuple[tuple[int, int, int, int], ...],
    state_activation_name: str,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    num_sequence_layers = len(sequence_specs)
    num_prediction_layers = len(prediction_specs)
    (
        sequence_source_weights,
        sequence_target_weights,
        sequence_core_weights,
        sequence_biases,
        sequence_norm_weights,
        sequence_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(sequence_tensors, num_sequence_layers)
    (
        prediction_source_weights,
        prediction_target_weights,
        prediction_core_weights,
        prediction_biases,
        prediction_norm_weights,
        prediction_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(prediction_tensors, num_prediction_layers)
    (
        sequence_compress_kinds,
        sequence_windows,
        sequence_target_block_sizes,
        sequence_source_block_sizes,
    ) = _split_nomemory_stack_specs(sequence_specs)
    (
        prediction_compress_kinds,
        prediction_windows,
        prediction_target_block_sizes,
        prediction_source_block_sizes,
    ) = _split_nomemory_stack_specs(prediction_specs)
    result = _native_module().nomemory_causal_stack_fused_trace(
        token_val,
        anchor_state,
        anchor_val,
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, token_val),
        list(sequence_source_weights),
        list(sequence_target_weights),
        list(sequence_core_weights),
        list(sequence_biases),
        list(sequence_norm_weights),
        list(sequence_norm_biases),
        list(sequence_compress_kinds),
        list(sequence_windows),
        list(sequence_target_block_sizes),
        list(sequence_source_block_sizes),
        list(prediction_source_weights),
        list(prediction_target_weights),
        list(prediction_core_weights),
        list(prediction_biases),
        list(prediction_norm_weights),
        list(prediction_norm_biases),
        list(prediction_compress_kinds),
        list(prediction_windows),
        list(prediction_target_block_sizes),
        list(prediction_source_block_sizes),
        state_activation_name,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("nomemory_causal_stack_fused_trace must return (query_val, trace_tensors).")
    query_val, trace_tensors = result
    if not isinstance(query_val, Tensor):
        raise TypeError("nomemory_causal_stack_fused_trace query_val must be a Tensor.")
    if not isinstance(trace_tensors, (list, tuple)):
        raise TypeError("nomemory_causal_stack_fused_trace trace_tensors must be a sequence.")
    return query_val, tuple(trace_tensors)


def _nomemory_causal_stack_fused_backward_cuda(
    *,
    token_val: Tensor,
    anchor_state: Tensor,
    anchor_val: Tensor,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    sequence_tensors: tuple[Tensor, ...],
    prediction_tensors: tuple[Tensor, ...],
    sequence_specs: tuple[tuple[int, int, int, int], ...],
    prediction_specs: tuple[tuple[int, int, int, int], ...],
    state_activation_name: str,
    trace_tensors: tuple[Tensor, ...],
    grad_query_val: Tensor,
) -> tuple[Tensor | None, ...]:
    num_sequence_layers = len(sequence_specs)
    num_prediction_layers = len(prediction_specs)
    (
        sequence_source_weights,
        sequence_target_weights,
        sequence_core_weights,
        sequence_biases,
        sequence_norm_weights,
        sequence_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(sequence_tensors, num_sequence_layers)
    (
        prediction_source_weights,
        prediction_target_weights,
        prediction_core_weights,
        prediction_biases,
        prediction_norm_weights,
        prediction_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(prediction_tensors, num_prediction_layers)
    (
        sequence_compress_kinds,
        sequence_windows,
        sequence_target_block_sizes,
        sequence_source_block_sizes,
    ) = _split_nomemory_stack_specs(sequence_specs)
    (
        prediction_compress_kinds,
        prediction_windows,
        prediction_target_block_sizes,
        prediction_source_block_sizes,
    ) = _split_nomemory_stack_specs(prediction_specs)
    result = _native_module().nomemory_causal_stack_fused_backward_cuda(
        token_val,
        anchor_state,
        anchor_val,
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, token_val),
        list(sequence_source_weights),
        list(sequence_target_weights),
        list(sequence_core_weights),
        list(sequence_biases),
        list(sequence_norm_weights),
        list(sequence_norm_biases),
        list(sequence_compress_kinds),
        list(sequence_windows),
        list(sequence_target_block_sizes),
        list(sequence_source_block_sizes),
        list(prediction_source_weights),
        list(prediction_target_weights),
        list(prediction_core_weights),
        list(prediction_biases),
        list(prediction_norm_weights),
        list(prediction_norm_biases),
        list(prediction_compress_kinds),
        list(prediction_windows),
        list(prediction_target_block_sizes),
        list(prediction_source_block_sizes),
        state_activation_name,
        list(trace_tensors),
        grad_query_val,
    )
    expected = 6 + len(sequence_tensors) + len(prediction_tensors)
    if not isinstance(result, (list, tuple)) or len(result) != expected:
        raise TypeError("nomemory_causal_stack_fused_backward_cuda must return one grad per saved tensor.")
    result = tuple(None if grad is None else grad for grad in result)
    base_grads = list(result[:6])
    offset = 6
    grouped_sequence_grads = [result[offset + (index * num_sequence_layers): offset + ((index + 1) * num_sequence_layers)] for index in range(6)]
    offset += num_sequence_layers * 6
    grouped_prediction_grads = [result[offset + (index * num_prediction_layers): offset + ((index + 1) * num_prediction_layers)] for index in range(6)]
    flat_grads: list[Tensor | None] = base_grads
    for layer_index in range(num_sequence_layers):
        for group_index in range(6):
            flat_grads.append(grouped_sequence_grads[group_index][layer_index])
    for layer_index in range(num_prediction_layers):
        for group_index in range(6):
            flat_grads.append(grouped_prediction_grads[group_index][layer_index])
    return tuple(flat_grads)


def nomemory_causal_stack_ffn_fused_native(
    *,
    token_val: Tensor,
    anchor_state: Tensor,
    anchor_val: Tensor,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    sequence_tensors: tuple[Tensor, ...],
    prediction_tensors: tuple[Tensor, ...],
    sequence_ffn_tensors: tuple[Tensor, ...],
    prediction_ffn_tensors: tuple[Tensor, ...],
    sequence_specs: tuple[tuple[int, int, int, int], ...],
    prediction_specs: tuple[tuple[int, int, int, int], ...],
    state_activation_name: str,
) -> Tensor:
    num_sequence_layers = len(sequence_specs)
    num_prediction_layers = len(prediction_specs)
    (
        sequence_source_weights,
        sequence_target_weights,
        sequence_core_weights,
        sequence_biases,
        sequence_norm_weights,
        sequence_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(sequence_tensors, num_sequence_layers)
    (
        prediction_source_weights,
        prediction_target_weights,
        prediction_core_weights,
        prediction_biases,
        prediction_norm_weights,
        prediction_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(prediction_tensors, num_prediction_layers)
    (
        sequence_ffn_norm_weights,
        sequence_ffn_norm_biases,
        sequence_ffn_in_weights,
        sequence_ffn_in_biases,
        sequence_ffn_out_weights,
        sequence_ffn_out_biases,
    ) = _split_nomemory_stack_ffn_layer_tensors(sequence_ffn_tensors, num_sequence_layers)
    (
        prediction_ffn_norm_weights,
        prediction_ffn_norm_biases,
        prediction_ffn_in_weights,
        prediction_ffn_in_biases,
        prediction_ffn_out_weights,
        prediction_ffn_out_biases,
    ) = _split_nomemory_stack_ffn_layer_tensors(prediction_ffn_tensors, num_prediction_layers)
    (
        sequence_compress_kinds,
        sequence_windows,
        sequence_target_block_sizes,
        sequence_source_block_sizes,
    ) = _split_nomemory_stack_specs(sequence_specs)
    (
        prediction_compress_kinds,
        prediction_windows,
        prediction_target_block_sizes,
        prediction_source_block_sizes,
    ) = _split_nomemory_stack_specs(prediction_specs)
    result = _native_module().nomemory_causal_stack_ffn_fused(
        token_val,
        anchor_state,
        anchor_val,
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, token_val),
        list(sequence_source_weights),
        list(sequence_target_weights),
        list(sequence_core_weights),
        list(sequence_biases),
        list(sequence_norm_weights),
        list(sequence_norm_biases),
        list(sequence_ffn_norm_weights),
        list(sequence_ffn_norm_biases),
        list(sequence_ffn_in_weights),
        list(sequence_ffn_in_biases),
        list(sequence_ffn_out_weights),
        list(sequence_ffn_out_biases),
        list(sequence_compress_kinds),
        list(sequence_windows),
        list(sequence_target_block_sizes),
        list(sequence_source_block_sizes),
        list(prediction_source_weights),
        list(prediction_target_weights),
        list(prediction_core_weights),
        list(prediction_biases),
        list(prediction_norm_weights),
        list(prediction_norm_biases),
        list(prediction_ffn_norm_weights),
        list(prediction_ffn_norm_biases),
        list(prediction_ffn_in_weights),
        list(prediction_ffn_in_biases),
        list(prediction_ffn_out_weights),
        list(prediction_ffn_out_biases),
        list(prediction_compress_kinds),
        list(prediction_windows),
        list(prediction_target_block_sizes),
        list(prediction_source_block_sizes),
        state_activation_name,
    )
    if not isinstance(result, Tensor):
        raise TypeError("nomemory_causal_stack_ffn_fused must return a Tensor.")
    return result


def value_ffn_native(
    *,
    layer_val: Tensor,
    norm_weight: Tensor,
    norm_bias: Tensor | None,
    in_weight: Tensor,
    in_bias: Tensor | None,
    out_weight: Tensor,
    out_bias: Tensor | None,
    activation_name: str = "gelu",
) -> Tensor:
    result = _native_module().value_ffn(
        layer_val,
        norm_weight,
        _save_optional_tensor(norm_bias, layer_val),
        in_weight,
        _save_optional_tensor(in_bias, layer_val),
        out_weight,
        _save_optional_tensor(out_bias, layer_val),
        activation_name,
    )
    if not isinstance(result, Tensor):
        raise TypeError("value_ffn must return a Tensor.")
    return result


def bilinear_propagation_softsign_value_ffn_forward_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    activation_name: str = "gelu",
) -> tuple[Tensor, Tensor]:
    activation_kind = {"gelu": 0, "silu": 1, "relu": 2}.get(str(activation_name))
    if activation_kind is None:
        raise ValueError(f"Unsupported activation_name: {activation_name!r}.")
    result = _native_module().bilinear_propagation_softsign_value_ffn_forward_cuda(
        layer_state,
        layer_val,
        weight,
        _save_optional_tensor(bias, layer_val),
        int(window),
        residual_gate,
        val_norm_weight,
        _save_optional_tensor(val_norm_bias, layer_val),
        ffn_norm_weight,
        _save_optional_tensor(ffn_norm_bias, layer_val),
        ffn_in_weight,
        _save_optional_tensor(ffn_in_bias, layer_val),
        ffn_out_weight,
        _save_optional_tensor(ffn_out_bias, layer_val),
        int(activation_kind),
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], Tensor)
        or not isinstance(result[1], Tensor)
    ):
        raise TypeError(
            "bilinear_propagation_softsign_value_ffn_forward_cuda must return (state, val)."
        )
    return result


def _bilinear_activation_kind(activation_name: str) -> int:
    activation_kind = {"gelu": 0, "silu": 1, "relu": 2}.get(str(activation_name))
    if activation_kind is None:
        raise ValueError(f"Unsupported activation_name: {activation_name!r}.")
    return int(activation_kind)


def _bilinear_propagation_softsign_value_ffn_reference(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    activation_name: str,
) -> tuple[Tensor, Tensor]:
    target_proj = torch.matmul(layer_val, weight)
    scores = torch.matmul(target_proj, layer_val.transpose(-1, -2))
    if bias is not None:
        scores = scores + bias.to(device=scores.device, dtype=scores.dtype)
    nodes = int(layer_val.shape[1])
    target_index = torch.arange(nodes, device=scores.device).view(1, nodes, 1)
    source_index = torch.arange(nodes, device=scores.device).view(1, 1, nodes)
    mask = (source_index <= target_index) & (source_index >= (target_index - int(window)))
    clean_scores = torch.nan_to_num(scores)
    signs = torch.sign(clean_scores) * mask.to(dtype=clean_scores.dtype)
    magnitudes = clean_scores.abs().masked_fill(~mask, float("-inf"))
    edges = torch.nan_to_num(signs * torch.softmax(magnitudes, dim=-1))
    source_strength = F.softplus(layer_state)
    delta_state = torch.matmul(
        edges.to(dtype=source_strength.dtype),
        source_strength.unsqueeze(-1),
    ).squeeze(-1)
    projected_val = layer_val * source_strength.to(dtype=layer_val.dtype).unsqueeze(-1)
    delta_val = torch.matmul(edges.to(dtype=projected_val.dtype), projected_val)
    gate_state = residual_gate.to(device=delta_state.device, dtype=delta_state.dtype)
    gate_val = residual_gate.to(device=delta_val.device, dtype=delta_val.dtype)
    next_state = F.softsign(torch.nan_to_num(layer_state + delta_state * gate_state))
    next_val = F.layer_norm(
        layer_val + delta_val * gate_val,
        [layer_val.shape[-1]],
        val_norm_weight,
        val_norm_bias,
        1e-5,
    )
    del ffn_norm_weight, ffn_norm_bias
    hidden = F.linear(
        next_val,
        ffn_in_weight,
        ffn_in_bias,
    )
    if activation_name == "gelu":
        activated = F.gelu(hidden)
    elif activation_name == "silu":
        activated = F.silu(hidden)
    elif activation_name == "relu":
        activated = F.relu(hidden)
    else:
        raise ValueError(f"Unsupported activation_name: {activation_name!r}.")
    next_val = next_val + F.linear(activated, ffn_out_weight, ffn_out_bias)
    return next_state, next_val


def _bilinear_propagation_softsign_value_ffn_backward_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    activation_name: str,
    grad_next_state: Tensor,
    grad_next_val: Tensor,
) -> tuple[Tensor | None, ...]:
    result = _native_module().bilinear_propagation_softsign_value_ffn_backward_cuda(
        layer_state,
        layer_val,
        weight,
        _save_optional_tensor(bias, layer_val),
        int(window),
        residual_gate,
        val_norm_weight,
        _save_optional_tensor(val_norm_bias, layer_val),
        ffn_norm_weight,
        _save_optional_tensor(ffn_norm_bias, layer_val),
        ffn_in_weight,
        _save_optional_tensor(ffn_in_bias, layer_val),
        ffn_out_weight,
        _save_optional_tensor(ffn_out_bias, layer_val),
        _bilinear_activation_kind(activation_name),
        grad_next_state,
        grad_next_val,
    )
    expected = 13
    if not isinstance(result, (list, tuple)) or len(result) != expected:
        raise TypeError(
            "bilinear_propagation_softsign_value_ffn_backward_cuda must return one grad per tensor input."
        )
    return tuple(None if grad is None else grad for grad in result)


class _BilinearPropagationSoftsignValueFFN(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_state: Tensor,
        layer_val: Tensor,
        weight: Tensor,
        bias: Tensor | None,
        window: int,
        residual_gate: Tensor,
        val_norm_weight: Tensor,
        val_norm_bias: Tensor | None,
        ffn_norm_weight: Tensor,
        ffn_norm_bias: Tensor | None,
        ffn_in_weight: Tensor,
        ffn_in_bias: Tensor | None,
        ffn_out_weight: Tensor,
        ffn_out_bias: Tensor | None,
        activation_name: str,
    ) -> tuple[Tensor, Tensor]:
        ctx.window = int(window)
        ctx.activation_name = str(activation_name)
        ctx.save_for_backward(
            layer_state,
            layer_val,
            weight,
            _save_optional_tensor(bias, layer_val),
            residual_gate,
            val_norm_weight,
            _save_optional_tensor(val_norm_bias, layer_val),
            ffn_norm_weight,
            _save_optional_tensor(ffn_norm_bias, layer_val),
            ffn_in_weight,
            _save_optional_tensor(ffn_in_bias, layer_val),
            ffn_out_weight,
            _save_optional_tensor(ffn_out_bias, layer_val),
        )
        return bilinear_propagation_softsign_value_ffn_forward_native(
            layer_state=layer_state,
            layer_val=layer_val,
            weight=weight,
            bias=bias,
            window=int(window),
            residual_gate=residual_gate,
            val_norm_weight=val_norm_weight,
            val_norm_bias=val_norm_bias,
            ffn_norm_weight=ffn_norm_weight,
            ffn_norm_bias=ffn_norm_bias,
            ffn_in_weight=ffn_in_weight,
            ffn_in_bias=ffn_in_bias,
            ffn_out_weight=ffn_out_weight,
            ffn_out_bias=ffn_out_bias,
            activation_name=activation_name,
        )

    @staticmethod
    def backward(ctx: Any, grad_next_state: Tensor, grad_next_val: Tensor) -> tuple[Any, ...]:
        (
            layer_state,
            layer_val,
            weight,
            bias_tensor,
            residual_gate,
            val_norm_weight,
            val_norm_bias_tensor,
            ffn_norm_weight,
            ffn_norm_bias_tensor,
            ffn_in_weight,
            ffn_in_bias_tensor,
            ffn_out_weight,
            ffn_out_bias_tensor,
        ) = ctx.saved_tensors
        if bilinear_propagation_softsign_value_ffn_backward_native_available(layer_val.device.type):
            grads = _bilinear_propagation_softsign_value_ffn_backward_native(
                layer_state=layer_state,
                layer_val=layer_val,
                weight=weight,
                bias=_load_optional_tensor(bias_tensor),
                window=ctx.window,
                residual_gate=residual_gate,
                val_norm_weight=val_norm_weight,
                val_norm_bias=_load_optional_tensor(val_norm_bias_tensor),
                ffn_norm_weight=ffn_norm_weight,
                ffn_norm_bias=_load_optional_tensor(ffn_norm_bias_tensor),
                ffn_in_weight=ffn_in_weight,
                ffn_in_bias=_load_optional_tensor(ffn_in_bias_tensor),
                ffn_out_weight=ffn_out_weight,
                ffn_out_bias=_load_optional_tensor(ffn_out_bias_tensor),
                activation_name=ctx.activation_name,
                grad_next_state=grad_next_state.contiguous(),
                grad_next_val=grad_next_val.contiguous(),
            )
        else:
            def _make_leaf(tensor: Tensor) -> Tensor:
                return tensor.detach().requires_grad_(tensor.requires_grad)

            base_inputs = [
                _make_leaf(layer_state),
                _make_leaf(layer_val),
                _make_leaf(weight),
                _make_leaf(residual_gate),
                _make_leaf(val_norm_weight),
                _make_leaf(ffn_norm_weight),
                _make_leaf(ffn_in_weight),
                _make_leaf(ffn_out_weight),
            ]
            detached_bias = (
                None if bias_tensor.numel() == 0 else bias_tensor.detach().requires_grad_(bias_tensor.requires_grad)
            )
            detached_val_norm_bias = (
                None
                if val_norm_bias_tensor.numel() == 0
                else val_norm_bias_tensor.detach().requires_grad_(val_norm_bias_tensor.requires_grad)
            )
            detached_ffn_norm_bias = (
                None
                if ffn_norm_bias_tensor.numel() == 0
                else ffn_norm_bias_tensor.detach().requires_grad_(ffn_norm_bias_tensor.requires_grad)
            )
            detached_ffn_in_bias = (
                None
                if ffn_in_bias_tensor.numel() == 0
                else ffn_in_bias_tensor.detach().requires_grad_(ffn_in_bias_tensor.requires_grad)
            )
            detached_ffn_out_bias = (
                None
                if ffn_out_bias_tensor.numel() == 0
                else ffn_out_bias_tensor.detach().requires_grad_(ffn_out_bias_tensor.requires_grad)
            )
            with torch.enable_grad():
                next_state, next_val = _bilinear_propagation_softsign_value_ffn_reference(
                    layer_state=base_inputs[0],
                    layer_val=base_inputs[1],
                    weight=base_inputs[2],
                    bias=detached_bias,
                    window=ctx.window,
                    residual_gate=base_inputs[3],
                    val_norm_weight=base_inputs[4],
                    val_norm_bias=detached_val_norm_bias,
                    ffn_norm_weight=base_inputs[5],
                    ffn_norm_bias=detached_ffn_norm_bias,
                    ffn_in_weight=base_inputs[6],
                    ffn_in_bias=detached_ffn_in_bias,
                    ffn_out_weight=base_inputs[7],
                    ffn_out_bias=detached_ffn_out_bias,
                    activation_name=ctx.activation_name,
                )
                all_inputs = base_inputs + [
                    detached_bias,
                    detached_val_norm_bias,
                    detached_ffn_norm_bias,
                    detached_ffn_in_bias,
                    detached_ffn_out_bias,
                ]
                grad_targets = [tensor for tensor in all_inputs if tensor is not None and tensor.requires_grad]
                grad_values = torch.autograd.grad(
                    (next_state, next_val),
                    grad_targets,
                    (grad_next_state, grad_next_val),
                    allow_unused=False,
                )
            grad_iter = iter(grad_values)
            grads_list: list[Tensor | None] = []
            for tensor in all_inputs:
                if tensor is None or not tensor.requires_grad:
                    grads_list.append(None)
                else:
                    grads_list.append(next(grad_iter))
            grads = (
                grads_list[0],
                grads_list[1],
                grads_list[2],
                grads_list[8],
                grads_list[3],
                grads_list[4],
                grads_list[9],
                grads_list[5],
                grads_list[10],
                grads_list[6],
                grads_list[11],
                grads_list[7],
                grads_list[12],
            )
        grad_bias = None if bias_tensor.numel() == 0 else grads[3]
        grad_val_norm_bias = None if val_norm_bias_tensor.numel() == 0 else grads[6]
        grad_ffn_norm_bias = None if ffn_norm_bias_tensor.numel() == 0 else grads[8]
        grad_ffn_in_bias = None if ffn_in_bias_tensor.numel() == 0 else grads[10]
        grad_ffn_out_bias = None if ffn_out_bias_tensor.numel() == 0 else grads[12]
        return (
            grads[0],
            grads[1],
            grads[2],
            grad_bias,
            None,
            grads[4],
            grads[5],
            grad_val_norm_bias,
            grads[7],
            grad_ffn_norm_bias,
            grads[9],
            grad_ffn_in_bias,
            grads[11],
            grad_ffn_out_bias,
            None,
        )


def bilinear_propagation_softsign_value_ffn_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    activation_name: str = "gelu",
) -> tuple[Tensor, Tensor]:
    # Keep the public helper aligned with the Python mixed-norm stack until the
    # fused CUDA kernel path is updated to match the same semantics.
    return _bilinear_propagation_softsign_value_ffn_reference(
        layer_state=layer_state,
        layer_val=layer_val,
        weight=weight,
        bias=bias,
        window=int(window),
        residual_gate=residual_gate,
        val_norm_weight=val_norm_weight,
        val_norm_bias=val_norm_bias,
        ffn_norm_weight=ffn_norm_weight,
        ffn_norm_bias=ffn_norm_bias,
        ffn_in_weight=ffn_in_weight,
        ffn_in_bias=ffn_in_bias,
        ffn_out_weight=ffn_out_weight,
        ffn_out_bias=ffn_out_bias,
        activation_name=activation_name,
    )


def _save_optional_bias(tensor: Tensor | None, reference: Tensor) -> Tensor:
    return _save_optional_tensor(tensor, reference)


def _propagation_value_ffn_reference(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    ffn_residual_scale: Tensor,
    state_activation_name: str,
    ffn_activation_name: str,
) -> tuple[Tensor, Tensor]:
    normalized_core = core_weight / core_weight.norm(p=2).clamp_min(1e-6)
    projected_source = F.linear(layer_val, source_weight, bias=None)
    projected_target = F.linear(layer_val, target_weight, bias=None)
    weighted_projected_source = projected_source * normalized_core.to(
        device=projected_source.device,
        dtype=projected_source.dtype,
    ).view(1, 1, -1)
    scores = torch.matmul(projected_target, weighted_projected_source.transpose(-1, -2))
    if bias is not None:
        scores = scores + bias.to(device=scores.device, dtype=scores.dtype)
    nodes = int(layer_val.shape[1])
    target_index = torch.arange(nodes, device=scores.device).view(1, nodes, 1)
    source_index = torch.arange(nodes, device=scores.device).view(1, 1, nodes)
    mask = (source_index <= target_index) & (source_index >= (target_index - int(window)))
    clean_scores = torch.nan_to_num(scores)
    signs = torch.sign(clean_scores) * mask.to(dtype=clean_scores.dtype)
    magnitudes = clean_scores.abs().masked_fill(~mask, float("-inf"))
    edges = torch.nan_to_num(signs * torch.softmax(magnitudes, dim=-1))
    source_strength = F.softplus(layer_state)
    delta_state = torch.matmul(
        edges.to(dtype=source_strength.dtype),
        source_strength.unsqueeze(-1),
    ).squeeze(-1)
    projected_val = layer_val * source_strength.to(dtype=layer_val.dtype).unsqueeze(-1)
    delta_val = torch.matmul(edges.to(dtype=projected_val.dtype), projected_val)
    gate_state = residual_gate.to(device=delta_state.device, dtype=delta_state.dtype)
    gate_val = residual_gate.to(device=delta_val.device, dtype=delta_val.dtype)
    if state_activation_name == "signed_softmax":
        next_state = signed_softmax_state(layer_state + delta_state * gate_state)
    elif state_activation_name == "softsign":
        next_state = F.softsign(torch.nan_to_num(layer_state + delta_state * gate_state))
    else:
        raise ValueError(f"Unsupported state_activation_name: {state_activation_name!r}.")
    next_val = F.layer_norm(
        layer_val + delta_val * gate_val,
        [layer_val.shape[-1]],
        val_norm_weight,
        val_norm_bias,
        1e-5,
    )
    del ffn_norm_weight, ffn_norm_bias
    hidden = F.linear(
        next_val,
        ffn_in_weight,
        ffn_in_bias,
    )
    if ffn_activation_name == "gelu":
        activated = F.gelu(hidden)
    elif ffn_activation_name == "silu":
        activated = F.silu(hidden)
    elif ffn_activation_name == "relu":
        activated = F.relu(hidden)
    else:
        raise ValueError(f"Unsupported ffn_activation_name: {ffn_activation_name!r}.")
    scale = ffn_residual_scale.to(device=ffn_out_weight.device, dtype=ffn_out_weight.dtype)
    out_weight = ffn_out_weight * scale
    out_bias = None if ffn_out_bias is None else ffn_out_bias * scale.to(
        device=ffn_out_bias.device,
        dtype=ffn_out_bias.dtype,
    )
    next_val = next_val + F.linear(activated, out_weight, out_bias)
    return next_state, next_val


def low_rank_propagation_causal_dense_value_ffn_forward_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    ffn_residual_scale: Tensor,
    state_activation_name: str,
    ffn_activation_name: str,
) -> tuple[Tensor, Tensor]:
    result = _native_module().low_rank_propagation_causal_dense_value_ffn_forward_cuda(
        layer_state,
        layer_val,
        source_weight,
        target_weight,
        core_weight,
        _save_optional_bias(bias, layer_val),
        residual_gate,
        val_norm_weight,
        _save_optional_bias(val_norm_bias, layer_val),
        ffn_norm_weight,
        _save_optional_bias(ffn_norm_bias, layer_val),
        ffn_in_weight,
        _save_optional_bias(ffn_in_bias, layer_val),
        ffn_out_weight,
        _save_optional_bias(ffn_out_bias, layer_val),
        ffn_residual_scale,
        state_activation_name,
        ffn_activation_name,
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], Tensor)
        or not isinstance(result[1], Tensor)
    ):
        raise TypeError(
            "low_rank_propagation_causal_dense_value_ffn_forward_cuda must return (state, val)."
        )
    return result


def low_rank_propagation_window_value_ffn_forward_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    ffn_residual_scale: Tensor,
    state_activation_name: str,
    ffn_activation_name: str,
) -> tuple[Tensor, Tensor]:
    result = _native_module().low_rank_propagation_window_value_ffn_forward_cuda(
        layer_state,
        layer_val,
        source_weight,
        target_weight,
        core_weight,
        _save_optional_bias(bias, layer_val),
        int(window),
        residual_gate,
        val_norm_weight,
        _save_optional_bias(val_norm_bias, layer_val),
        ffn_norm_weight,
        _save_optional_bias(ffn_norm_bias, layer_val),
        ffn_in_weight,
        _save_optional_bias(ffn_in_bias, layer_val),
        ffn_out_weight,
        _save_optional_bias(ffn_out_bias, layer_val),
        ffn_residual_scale,
        state_activation_name,
        ffn_activation_name,
    )
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], Tensor)
        or not isinstance(result[1], Tensor)
    ):
        raise TypeError(
            "low_rank_propagation_window_value_ffn_forward_cuda must return (state, val)."
        )
    return result


def _low_rank_propagation_causal_dense_value_ffn_backward_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    ffn_residual_scale: Tensor,
    state_activation_name: str,
    ffn_activation_name: str,
    grad_next_state: Tensor,
    grad_next_val: Tensor,
) -> tuple[Tensor | None, ...]:
    result = _native_module().low_rank_propagation_causal_dense_value_ffn_backward_cuda(
        layer_state,
        layer_val,
        source_weight,
        target_weight,
        core_weight,
        _save_optional_bias(bias, layer_val),
        residual_gate,
        val_norm_weight,
        _save_optional_bias(val_norm_bias, layer_val),
        ffn_norm_weight,
        _save_optional_bias(ffn_norm_bias, layer_val),
        ffn_in_weight,
        _save_optional_bias(ffn_in_bias, layer_val),
        ffn_out_weight,
        _save_optional_bias(ffn_out_bias, layer_val),
        ffn_residual_scale,
        state_activation_name,
        ffn_activation_name,
        grad_next_state,
        grad_next_val,
    )
    expected = 16
    if not isinstance(result, (list, tuple)) or len(result) != expected:
        raise TypeError(
            "low_rank_propagation_causal_dense_value_ffn_backward_cuda must return one grad per tensor input."
        )
    return tuple(None if grad is None else grad for grad in result)


class _LowRankPropagationCausalDenseValueFFN(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_state: Tensor,
        layer_val: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        residual_gate: Tensor,
        val_norm_weight: Tensor,
        val_norm_bias: Tensor | None,
        ffn_norm_weight: Tensor,
        ffn_norm_bias: Tensor | None,
        ffn_in_weight: Tensor,
        ffn_in_bias: Tensor | None,
        ffn_out_weight: Tensor,
        ffn_out_bias: Tensor | None,
        ffn_residual_scale: Tensor,
        state_activation_name: str,
        ffn_activation_name: str,
    ) -> tuple[Tensor, Tensor]:
        ctx.state_activation_name = state_activation_name
        ctx.ffn_activation_name = ffn_activation_name
        ctx.save_for_backward(
            layer_state,
            layer_val,
            source_weight,
            target_weight,
            core_weight,
            _save_optional_bias(bias, layer_val),
            residual_gate,
            val_norm_weight,
            _save_optional_bias(val_norm_bias, layer_val),
            ffn_norm_weight,
            _save_optional_bias(ffn_norm_bias, layer_val),
            ffn_in_weight,
            _save_optional_bias(ffn_in_bias, layer_val),
            ffn_out_weight,
            _save_optional_bias(ffn_out_bias, layer_val),
            ffn_residual_scale,
        )
        return low_rank_propagation_causal_dense_value_ffn_forward_native(
            layer_state=layer_state,
            layer_val=layer_val,
            source_weight=source_weight,
            target_weight=target_weight,
            core_weight=core_weight,
            bias=bias,
            residual_gate=residual_gate,
            val_norm_weight=val_norm_weight,
            val_norm_bias=val_norm_bias,
            ffn_norm_weight=ffn_norm_weight,
            ffn_norm_bias=ffn_norm_bias,
            ffn_in_weight=ffn_in_weight,
            ffn_in_bias=ffn_in_bias,
            ffn_out_weight=ffn_out_weight,
            ffn_out_bias=ffn_out_bias,
            ffn_residual_scale=ffn_residual_scale,
            state_activation_name=state_activation_name,
            ffn_activation_name=ffn_activation_name,
        )

    @staticmethod
    def backward(ctx: Any, grad_next_state: Tensor, grad_next_val: Tensor) -> tuple[Any, ...]:
        (
            layer_state,
            layer_val,
            source_weight,
            target_weight,
            core_weight,
            bias_tensor,
            residual_gate,
            val_norm_weight,
            val_norm_bias_tensor,
            ffn_norm_weight,
            ffn_norm_bias_tensor,
            ffn_in_weight,
            ffn_in_bias_tensor,
            ffn_out_weight,
            ffn_out_bias_tensor,
            ffn_residual_scale,
        ) = ctx.saved_tensors
        grads = _low_rank_propagation_causal_dense_value_ffn_backward_native(
            layer_state=layer_state,
            layer_val=layer_val,
            source_weight=source_weight,
            target_weight=target_weight,
            core_weight=core_weight,
            bias=_load_optional_tensor(bias_tensor),
            residual_gate=residual_gate,
            val_norm_weight=val_norm_weight,
            val_norm_bias=_load_optional_tensor(val_norm_bias_tensor),
            ffn_norm_weight=ffn_norm_weight,
            ffn_norm_bias=_load_optional_tensor(ffn_norm_bias_tensor),
            ffn_in_weight=ffn_in_weight,
            ffn_in_bias=_load_optional_tensor(ffn_in_bias_tensor),
            ffn_out_weight=ffn_out_weight,
            ffn_out_bias=_load_optional_tensor(ffn_out_bias_tensor),
            ffn_residual_scale=ffn_residual_scale,
            state_activation_name=ctx.state_activation_name,
            ffn_activation_name=ctx.ffn_activation_name,
            grad_next_state=grad_next_state,
            grad_next_val=grad_next_val,
        )
        grad_bias = None if bias_tensor.numel() == 0 else grads[5]
        grad_val_norm_bias = None if val_norm_bias_tensor.numel() == 0 else grads[8]
        grad_ffn_norm_bias = None if ffn_norm_bias_tensor.numel() == 0 else grads[10]
        grad_ffn_in_bias = None if ffn_in_bias_tensor.numel() == 0 else grads[12]
        grad_ffn_out_bias = None if ffn_out_bias_tensor.numel() == 0 else grads[14]
        return (
            grads[0],
            grads[1],
            grads[2],
            grads[3],
            grads[4],
            grad_bias,
            grads[6],
            grads[7],
            grad_val_norm_bias,
            grads[9],
            grad_ffn_norm_bias,
            grads[11],
            grad_ffn_in_bias,
            grads[13],
            grad_ffn_out_bias,
            grads[15],
            None,
            None,
        )


class _LowRankPropagationWindowValueFFN(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_state: Tensor,
        layer_val: Tensor,
        source_weight: Tensor,
        target_weight: Tensor,
        core_weight: Tensor,
        bias: Tensor | None,
        window: int,
        residual_gate: Tensor,
        val_norm_weight: Tensor,
        val_norm_bias: Tensor | None,
        ffn_norm_weight: Tensor,
        ffn_norm_bias: Tensor | None,
        ffn_in_weight: Tensor,
        ffn_in_bias: Tensor | None,
        ffn_out_weight: Tensor,
        ffn_out_bias: Tensor | None,
        ffn_residual_scale: Tensor,
        state_activation_name: str,
        ffn_activation_name: str,
    ) -> tuple[Tensor, Tensor]:
        ctx.window = int(window)
        ctx.state_activation_name = state_activation_name
        ctx.ffn_activation_name = ffn_activation_name
        ctx.save_for_backward(
            layer_state,
            layer_val,
            source_weight,
            target_weight,
            core_weight,
            _save_optional_bias(bias, layer_val),
            residual_gate,
            val_norm_weight,
            _save_optional_bias(val_norm_bias, layer_val),
            ffn_norm_weight,
            _save_optional_bias(ffn_norm_bias, layer_val),
            ffn_in_weight,
            _save_optional_bias(ffn_in_bias, layer_val),
            ffn_out_weight,
            _save_optional_bias(ffn_out_bias, layer_val),
            ffn_residual_scale,
        )
        return low_rank_propagation_window_value_ffn_forward_native(
            layer_state=layer_state,
            layer_val=layer_val,
            source_weight=source_weight,
            target_weight=target_weight,
            core_weight=core_weight,
            bias=bias,
            window=window,
            residual_gate=residual_gate,
            val_norm_weight=val_norm_weight,
            val_norm_bias=val_norm_bias,
            ffn_norm_weight=ffn_norm_weight,
            ffn_norm_bias=ffn_norm_bias,
            ffn_in_weight=ffn_in_weight,
            ffn_in_bias=ffn_in_bias,
            ffn_out_weight=ffn_out_weight,
            ffn_out_bias=ffn_out_bias,
            ffn_residual_scale=ffn_residual_scale,
            state_activation_name=state_activation_name,
            ffn_activation_name=ffn_activation_name,
        )

    @staticmethod
    def backward(ctx: Any, grad_next_state: Tensor, grad_next_val: Tensor) -> tuple[Any, ...]:
        (
            layer_state,
            layer_val,
            source_weight,
            target_weight,
            core_weight,
            bias_tensor,
            residual_gate,
            val_norm_weight,
            val_norm_bias_tensor,
            ffn_norm_weight,
            ffn_norm_bias_tensor,
            ffn_in_weight,
            ffn_in_bias_tensor,
            ffn_out_weight,
            ffn_out_bias_tensor,
            ffn_residual_scale,
        ) = ctx.saved_tensors
        detached_inputs = [
            layer_state.detach().requires_grad_(True),
            layer_val.detach().requires_grad_(True),
            source_weight.detach().requires_grad_(True),
            target_weight.detach().requires_grad_(True),
            core_weight.detach().requires_grad_(True),
            residual_gate.detach().requires_grad_(True),
            val_norm_weight.detach().requires_grad_(True),
            ffn_norm_weight.detach().requires_grad_(True),
            ffn_in_weight.detach().requires_grad_(True),
            ffn_out_weight.detach().requires_grad_(True),
            ffn_residual_scale.detach().requires_grad_(True),
        ]
        detached_bias = (
            None
            if bias_tensor.numel() == 0
            else bias_tensor.detach().requires_grad_(True)
        )
        detached_val_norm_bias = (
            None
            if val_norm_bias_tensor.numel() == 0
            else val_norm_bias_tensor.detach().requires_grad_(True)
        )
        detached_ffn_norm_bias = (
            None
            if ffn_norm_bias_tensor.numel() == 0
            else ffn_norm_bias_tensor.detach().requires_grad_(True)
        )
        detached_ffn_in_bias = (
            None
            if ffn_in_bias_tensor.numel() == 0
            else ffn_in_bias_tensor.detach().requires_grad_(True)
        )
        detached_ffn_out_bias = (
            None
            if ffn_out_bias_tensor.numel() == 0
            else ffn_out_bias_tensor.detach().requires_grad_(True)
        )
        with torch.enable_grad():
            next_state, next_val = _propagation_value_ffn_reference(
                layer_state=detached_inputs[0],
                layer_val=detached_inputs[1],
                source_weight=detached_inputs[2],
                target_weight=detached_inputs[3],
                core_weight=detached_inputs[4],
                bias=detached_bias,
                window=ctx.window,
                residual_gate=detached_inputs[5],
                val_norm_weight=detached_inputs[6],
                val_norm_bias=detached_val_norm_bias,
                ffn_norm_weight=detached_inputs[7],
                ffn_norm_bias=detached_ffn_norm_bias,
                ffn_in_weight=detached_inputs[8],
                ffn_in_bias=detached_ffn_in_bias,
                ffn_out_weight=detached_inputs[9],
                ffn_out_bias=detached_ffn_out_bias,
                ffn_residual_scale=detached_inputs[10],
                state_activation_name=ctx.state_activation_name,
                ffn_activation_name=ctx.ffn_activation_name,
            )
            grad_inputs = list(detached_inputs)
            optional_inputs = [
                detached_bias,
                detached_val_norm_bias,
                detached_ffn_norm_bias,
                detached_ffn_in_bias,
                detached_ffn_out_bias,
            ]
            grad_targets = grad_inputs + [t for t in optional_inputs if t is not None]
            grads = torch.autograd.grad(
                (next_state, next_val),
                grad_targets,
                (grad_next_state, grad_next_val),
                allow_unused=False,
            )
        base_count = len(detached_inputs)
        base_grads = grads[:base_count]
        optional_grads_iter = iter(grads[base_count:])

        def _next_optional_grad(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return next(optional_grads_iter)

        grad_bias = _next_optional_grad(detached_bias)
        grad_val_norm_bias = _next_optional_grad(detached_val_norm_bias)
        grad_ffn_norm_bias = _next_optional_grad(detached_ffn_norm_bias)
        grad_ffn_in_bias = _next_optional_grad(detached_ffn_in_bias)
        grad_ffn_out_bias = _next_optional_grad(detached_ffn_out_bias)
        return (
            base_grads[0],
            base_grads[1],
            base_grads[2],
            base_grads[3],
            base_grads[4],
            grad_bias,
            None,
            base_grads[5],
            base_grads[6],
            grad_val_norm_bias,
            base_grads[7],
            grad_ffn_norm_bias,
            base_grads[8],
            grad_ffn_in_bias,
            base_grads[9],
            grad_ffn_out_bias,
            base_grads[10],
            None,
            None,
        )


def low_rank_propagation_value_ffn_fused_native(
    *,
    layer_state: Tensor,
    layer_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    window: int,
    residual_gate: Tensor,
    val_norm_weight: Tensor,
    val_norm_bias: Tensor | None,
    ffn_norm_weight: Tensor,
    ffn_norm_bias: Tensor | None,
    ffn_in_weight: Tensor,
    ffn_in_bias: Tensor | None,
    ffn_out_weight: Tensor,
    ffn_out_bias: Tensor | None,
    ffn_residual_scale: Tensor,
    state_activation_name: str,
    ffn_activation_name: str,
) -> tuple[Tensor, Tensor]:
    # Keep the public helper aligned with the Python mixed-norm stack until the
    # fused CUDA kernel path is updated to match the same semantics.
    return _propagation_value_ffn_reference(
        layer_state=layer_state,
        layer_val=layer_val,
        source_weight=source_weight,
        target_weight=target_weight,
        core_weight=core_weight,
        bias=bias,
        window=window,
        residual_gate=residual_gate,
        val_norm_weight=val_norm_weight,
        val_norm_bias=val_norm_bias,
        ffn_norm_weight=ffn_norm_weight,
        ffn_norm_bias=ffn_norm_bias,
        ffn_in_weight=ffn_in_weight,
        ffn_in_bias=ffn_in_bias,
        ffn_out_weight=ffn_out_weight,
        ffn_out_bias=ffn_out_bias,
        ffn_residual_scale=ffn_residual_scale,
        state_activation_name=state_activation_name,
        ffn_activation_name=ffn_activation_name,
    )


def nomemory_causal_stack_ffn_fused_trace_native(
    *,
    token_val: Tensor,
    anchor_state: Tensor,
    anchor_val: Tensor,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    sequence_tensors: tuple[Tensor, ...],
    prediction_tensors: tuple[Tensor, ...],
    sequence_ffn_tensors: tuple[Tensor, ...],
    prediction_ffn_tensors: tuple[Tensor, ...],
    sequence_specs: tuple[tuple[int, int, int, int], ...],
    prediction_specs: tuple[tuple[int, int, int, int], ...],
    state_activation_name: str,
) -> tuple[Tensor, tuple[Tensor, ...]]:
    num_sequence_layers = len(sequence_specs)
    num_prediction_layers = len(prediction_specs)
    (
        sequence_source_weights,
        sequence_target_weights,
        sequence_core_weights,
        sequence_biases,
        sequence_norm_weights,
        sequence_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(sequence_tensors, num_sequence_layers)
    (
        prediction_source_weights,
        prediction_target_weights,
        prediction_core_weights,
        prediction_biases,
        prediction_norm_weights,
        prediction_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(prediction_tensors, num_prediction_layers)
    (
        sequence_ffn_norm_weights,
        sequence_ffn_norm_biases,
        sequence_ffn_in_weights,
        sequence_ffn_in_biases,
        sequence_ffn_out_weights,
        sequence_ffn_out_biases,
    ) = _split_nomemory_stack_ffn_layer_tensors(sequence_ffn_tensors, num_sequence_layers)
    (
        prediction_ffn_norm_weights,
        prediction_ffn_norm_biases,
        prediction_ffn_in_weights,
        prediction_ffn_in_biases,
        prediction_ffn_out_weights,
        prediction_ffn_out_biases,
    ) = _split_nomemory_stack_ffn_layer_tensors(prediction_ffn_tensors, num_prediction_layers)
    (
        sequence_compress_kinds,
        sequence_windows,
        sequence_target_block_sizes,
        sequence_source_block_sizes,
    ) = _split_nomemory_stack_specs(sequence_specs)
    (
        prediction_compress_kinds,
        prediction_windows,
        prediction_target_block_sizes,
        prediction_source_block_sizes,
    ) = _split_nomemory_stack_specs(prediction_specs)
    result = _native_module().nomemory_causal_stack_ffn_fused_trace(
        token_val,
        anchor_state,
        anchor_val,
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, token_val),
        list(sequence_source_weights),
        list(sequence_target_weights),
        list(sequence_core_weights),
        list(sequence_biases),
        list(sequence_norm_weights),
        list(sequence_norm_biases),
        list(sequence_ffn_norm_weights),
        list(sequence_ffn_norm_biases),
        list(sequence_ffn_in_weights),
        list(sequence_ffn_in_biases),
        list(sequence_ffn_out_weights),
        list(sequence_ffn_out_biases),
        list(sequence_compress_kinds),
        list(sequence_windows),
        list(sequence_target_block_sizes),
        list(sequence_source_block_sizes),
        list(prediction_source_weights),
        list(prediction_target_weights),
        list(prediction_core_weights),
        list(prediction_biases),
        list(prediction_norm_weights),
        list(prediction_norm_biases),
        list(prediction_ffn_norm_weights),
        list(prediction_ffn_norm_biases),
        list(prediction_ffn_in_weights),
        list(prediction_ffn_in_biases),
        list(prediction_ffn_out_weights),
        list(prediction_ffn_out_biases),
        list(prediction_compress_kinds),
        list(prediction_windows),
        list(prediction_target_block_sizes),
        list(prediction_source_block_sizes),
        state_activation_name,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("nomemory_causal_stack_ffn_fused_trace must return (query_val, trace_tensors).")
    query_val, trace_tensors = result
    if not isinstance(query_val, Tensor):
        raise TypeError("nomemory_causal_stack_ffn_fused_trace query_val must be a Tensor.")
    if not isinstance(trace_tensors, (list, tuple)):
        raise TypeError("nomemory_causal_stack_ffn_fused_trace trace_tensors must be a sequence.")
    return query_val, tuple(trace_tensors)


def _nomemory_causal_stack_ffn_fused_backward_cuda(
    *,
    token_val: Tensor,
    anchor_state: Tensor,
    anchor_val: Tensor,
    s_prediction_weight: Tensor,
    prediction_input_norm_weight: Tensor,
    prediction_input_norm_bias: Tensor | None,
    sequence_tensors: tuple[Tensor, ...],
    prediction_tensors: tuple[Tensor, ...],
    sequence_ffn_tensors: tuple[Tensor, ...],
    prediction_ffn_tensors: tuple[Tensor, ...],
    sequence_specs: tuple[tuple[int, int, int, int], ...],
    prediction_specs: tuple[tuple[int, int, int, int], ...],
    state_activation_name: str,
    trace_tensors: tuple[Tensor, ...],
    grad_query_val: Tensor,
) -> tuple[Tensor | None, ...]:
    num_sequence_layers = len(sequence_specs)
    num_prediction_layers = len(prediction_specs)
    (
        sequence_source_weights,
        sequence_target_weights,
        sequence_core_weights,
        sequence_biases,
        sequence_norm_weights,
        sequence_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(sequence_tensors, num_sequence_layers)
    (
        prediction_source_weights,
        prediction_target_weights,
        prediction_core_weights,
        prediction_biases,
        prediction_norm_weights,
        prediction_norm_biases,
    ) = _split_nomemory_stack_layer_tensors(prediction_tensors, num_prediction_layers)
    (
        sequence_ffn_norm_weights,
        sequence_ffn_norm_biases,
        sequence_ffn_in_weights,
        sequence_ffn_in_biases,
        sequence_ffn_out_weights,
        sequence_ffn_out_biases,
    ) = _split_nomemory_stack_ffn_layer_tensors(sequence_ffn_tensors, num_sequence_layers)
    (
        prediction_ffn_norm_weights,
        prediction_ffn_norm_biases,
        prediction_ffn_in_weights,
        prediction_ffn_in_biases,
        prediction_ffn_out_weights,
        prediction_ffn_out_biases,
    ) = _split_nomemory_stack_ffn_layer_tensors(prediction_ffn_tensors, num_prediction_layers)
    (
        sequence_compress_kinds,
        sequence_windows,
        sequence_target_block_sizes,
        sequence_source_block_sizes,
    ) = _split_nomemory_stack_specs(sequence_specs)
    (
        prediction_compress_kinds,
        prediction_windows,
        prediction_target_block_sizes,
        prediction_source_block_sizes,
    ) = _split_nomemory_stack_specs(prediction_specs)
    result = _native_module().nomemory_causal_stack_ffn_fused_backward_cuda(
        token_val,
        anchor_state,
        anchor_val,
        s_prediction_weight,
        prediction_input_norm_weight,
        _save_optional_tensor(prediction_input_norm_bias, token_val),
        list(sequence_source_weights),
        list(sequence_target_weights),
        list(sequence_core_weights),
        list(sequence_biases),
        list(sequence_norm_weights),
        list(sequence_norm_biases),
        list(sequence_ffn_norm_weights),
        list(sequence_ffn_norm_biases),
        list(sequence_ffn_in_weights),
        list(sequence_ffn_in_biases),
        list(sequence_ffn_out_weights),
        list(sequence_ffn_out_biases),
        list(sequence_compress_kinds),
        list(sequence_windows),
        list(sequence_target_block_sizes),
        list(sequence_source_block_sizes),
        list(prediction_source_weights),
        list(prediction_target_weights),
        list(prediction_core_weights),
        list(prediction_biases),
        list(prediction_norm_weights),
        list(prediction_norm_biases),
        list(prediction_ffn_norm_weights),
        list(prediction_ffn_norm_biases),
        list(prediction_ffn_in_weights),
        list(prediction_ffn_in_biases),
        list(prediction_ffn_out_weights),
        list(prediction_ffn_out_biases),
        list(prediction_compress_kinds),
        list(prediction_windows),
        list(prediction_target_block_sizes),
        list(prediction_source_block_sizes),
        state_activation_name,
        list(trace_tensors),
        grad_query_val,
    )
    expected = 6 + len(sequence_tensors) + len(prediction_tensors) + len(sequence_ffn_tensors) + len(prediction_ffn_tensors)
    if not isinstance(result, (list, tuple)) or len(result) != expected:
        raise TypeError("nomemory_causal_stack_ffn_fused_backward_cuda must return one grad per saved tensor.")
    result = tuple(None if grad is None else grad for grad in result)
    base_grads = list(result[:6])
    offset = 6
    grouped_sequence_grads = [result[offset + (index * num_sequence_layers): offset + ((index + 1) * num_sequence_layers)] for index in range(6)]
    offset += num_sequence_layers * 6
    grouped_prediction_grads = [result[offset + (index * num_prediction_layers): offset + ((index + 1) * num_prediction_layers)] for index in range(6)]
    offset += num_prediction_layers * 6
    grouped_sequence_ffn_grads = [result[offset + (index * num_sequence_layers): offset + ((index + 1) * num_sequence_layers)] for index in range(6)]
    offset += num_sequence_layers * 6
    grouped_prediction_ffn_grads = [result[offset + (index * num_prediction_layers): offset + ((index + 1) * num_prediction_layers)] for index in range(6)]
    flat_grads: list[Tensor | None] = base_grads
    for layer_index in range(num_sequence_layers):
        for group_index in range(6):
            flat_grads.append(grouped_sequence_grads[group_index][layer_index])
    for layer_index in range(num_prediction_layers):
        for group_index in range(6):
            flat_grads.append(grouped_prediction_grads[group_index][layer_index])
    for layer_index in range(num_sequence_layers):
        for group_index in range(6):
            flat_grads.append(grouped_sequence_ffn_grads[group_index][layer_index])
    for layer_index in range(num_prediction_layers):
        for group_index in range(6):
            flat_grads.append(grouped_prediction_ffn_grads[group_index][layer_index])
    return tuple(flat_grads)


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
            None,
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
            "softmax",
            float(temperature),
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


def _low_rank_dense_signed_abs_forward(
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
    target_block_size: int,
) -> tuple[Tensor, Tensor]:
    (
        flat_val,
        flat_projected_state,
        flat_projected_val,
        batch_shape,
        nodes,
        out_dim,
    ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
    target_step = nodes if target_block_size <= 0 else min(int(target_block_size), nodes)
    projected_target = torch.matmul(flat_val, target_weight.t()).contiguous()
    weighted_source = (
        torch.matmul(flat_val, source_weight.t())
        * core_weight.to(dtype=flat_val.dtype).view(1, 1, -1)
    ).contiguous()
    state_blocks: list[Tensor] = []
    val_blocks: list[Tensor] = []
    source_state = flat_projected_state.to(dtype=torch.float32)
    source_val = flat_projected_val.to(dtype=torch.float32)
    weighted_source_t = weighted_source.transpose(1, 2).contiguous()
    for target_start in range(0, nodes, target_step):
        target_end = min(target_start + target_step, nodes)
        scores = torch.bmm(
            projected_target[:, target_start:target_end, :],
            weighted_source_t,
        )
        if bias is not None:
            scores = scores + bias.to(dtype=scores.dtype)
        clean_scores = torch.nan_to_num(scores).to(dtype=torch.float32)
        edges = torch.nan_to_num(
            torch.sign(clean_scores) * torch.softmax(clean_scores.abs(), dim=-1)
        )
        state_blocks.append(
            torch.bmm(edges, source_state.unsqueeze(-1)).squeeze(-1).to(dtype=projected_state.dtype)
        )
        val_blocks.append(torch.bmm(edges, source_val).to(dtype=projected_val.dtype))
    return (
        torch.cat(state_blocks, dim=1).reshape(*batch_shape, nodes),
        torch.cat(val_blocks, dim=1).reshape(*batch_shape, nodes, out_dim),
    )


def _low_rank_dense_signed_abs_forward_native(
    layer_val: Tensor,
    projected_state: Tensor,
    projected_val: Tensor,
    source_weight: Tensor,
    target_weight: Tensor,
    core_weight: Tensor,
    bias: Tensor | None,
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
    weighted_source = (
        torch.matmul(flat_val, source_weight.t())
        * core_weight.to(dtype=flat_val.dtype).view(1, 1, -1)
    ).contiguous()
    if bias is not None:
        bias_column = bias.to(dtype=weighted_source.dtype).reshape(1, 1, 1).expand(
            weighted_source.size(0),
            weighted_source.size(1),
            1,
        )
        target_ones = torch.ones(
            projected_target.size(0),
            projected_target.size(1),
            1,
            dtype=projected_target.dtype,
            device=projected_target.device,
        )
        weighted_source = torch.cat((weighted_source, bias_column), dim=-1).contiguous()
        projected_target = torch.cat((projected_target, target_ones), dim=-1).contiguous()
    delta_state, delta_val = _native_module().low_rank_propagation_dense_forward_cuda(
        weighted_source,
        projected_target,
        flat_projected_state.to(dtype=torch.float32).contiguous(),
        flat_projected_val.to(dtype=torch.float32).contiguous(),
        1,
    )
    return (
        delta_state.to(dtype=projected_state.dtype).reshape(*batch_shape, nodes),
        delta_val.to(dtype=projected_val.dtype).reshape(*batch_shape, nodes, out_dim),
    )


class _LowRankPropagationDenseSignedAbsRecompute(Function):
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
        target_block_size: int,
    ) -> tuple[Tensor, Tensor]:
        ctx.has_bias = bias is not None
        ctx.target_block_size = int(target_block_size)
        ctx.save_for_backward(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            _save_optional_tensor(bias, core_weight),
        )
        if (
            os.environ.get("JAKAL_NET_ENABLE_SCALAR_DENSE_CUDA", "").strip()
            in {"1", "true", "TRUE", "yes"}
            and hasattr(_native_module(), "low_rank_propagation_dense_forward_cuda")
        ):
            return _low_rank_dense_signed_abs_forward_native(
                layer_val,
                projected_state,
                projected_val,
                source_weight,
                target_weight,
                core_weight,
                bias,
            )
        return _low_rank_dense_signed_abs_forward(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            bias,
            int(target_block_size),
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
        detached_inputs = [
            layer_val.detach().requires_grad_(True),
            projected_state.detach().requires_grad_(True),
            projected_val.detach().requires_grad_(True),
            source_weight.detach().requires_grad_(True),
            target_weight.detach().requires_grad_(True),
            core_weight.detach().requires_grad_(True),
        ]
        detached_bias = None if bias is None else bias.detach().requires_grad_(True)
        with torch.enable_grad():
            delta_state, delta_val = _low_rank_dense_signed_abs_forward(
                detached_inputs[0],
                detached_inputs[1],
                detached_inputs[2],
                detached_inputs[3],
                detached_inputs[4],
                detached_inputs[5],
                detached_bias,
                ctx.target_block_size,
            )
            grads = torch.autograd.grad(
                (delta_state, delta_val),
                (*detached_inputs, detached_bias) if detached_bias is not None else tuple(detached_inputs),
                (grad_delta_state, grad_delta_val),
                allow_unused=False,
            )
        if detached_bias is None:
            grad_layer, grad_projected_state, grad_projected_val, grad_source_weight, grad_target_weight, grad_core_weight = grads
            grad_bias = None
        else:
            (
                grad_layer,
                grad_projected_state,
                grad_projected_val,
                grad_source_weight,
                grad_target_weight,
                grad_core_weight,
                grad_bias,
            ) = grads
        return (
            grad_layer.to(dtype=layer_val.dtype),
            grad_projected_state.to(dtype=projected_state.dtype),
            grad_projected_val.to(dtype=projected_val.dtype),
            grad_source_weight,
            grad_target_weight,
            grad_core_weight,
            grad_bias if ctx.has_bias else None,
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


class _DiagonalPropagationCausalDenseSignedAbs(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        normalized_weight: Tensor,
        bias: Tensor,
        has_bias: bool,
    ) -> tuple[Tensor, Tensor]:
        (
            flat_val,
            flat_projected_state,
            flat_projected_val,
            batch_shape,
            nodes,
            out_dim,
        ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
        if out_dim != flat_val.shape[-1]:
            raise ValueError("diagonal causal dense propagation requires projected_val dim == layer dim.")
        flat_val = flat_val.contiguous()
        state_f32 = flat_projected_state.to(dtype=torch.float32).contiguous()
        val_f32 = flat_projected_val.to(dtype=torch.float32).contiguous()
        norm_weight = normalized_weight.to(dtype=flat_val.dtype).contiguous()
        bias_arg = bias.to(dtype=flat_val.dtype).contiguous() if has_bias else torch.empty(
            0,
            dtype=flat_val.dtype,
            device=flat_val.device,
        )
        delta_state, delta_val = _native_module().diagonal_propagation_causal_dense_signed_abs_forward_cuda(
            flat_val,
            state_f32,
            val_f32,
            norm_weight,
            bias_arg,
        )
        ctx.has_bias = bool(has_bias)
        ctx.batch_shape = batch_shape
        ctx.nodes = nodes
        ctx.out_dim = out_dim
        ctx.layer_dtype = layer_val.dtype
        ctx.projected_state_dtype = projected_state.dtype
        ctx.projected_val_dtype = projected_val.dtype
        ctx.normalized_weight_dtype = normalized_weight.dtype
        ctx.save_for_backward(flat_val, state_f32, val_f32, norm_weight, bias_arg)
        return (
            delta_state.to(dtype=projected_state.dtype).reshape(*batch_shape, nodes),
            delta_val.to(dtype=projected_val.dtype).reshape(*batch_shape, nodes, out_dim),
        )

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        flat_val, state_f32, val_f32, norm_weight, bias = ctx.saved_tensors
        flat_grad_state = grad_delta_state.reshape(-1, ctx.nodes).to(dtype=torch.float32).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, ctx.nodes, ctx.out_dim).to(dtype=torch.float32).contiguous()
        (
            grad_layer,
            grad_projected_state,
            grad_projected_val,
            grad_norm_weight,
            grad_bias,
        ) = _native_module().diagonal_propagation_causal_dense_signed_abs_backward_cuda(
            flat_val,
            state_f32,
            val_f32,
            norm_weight,
            bias,
            flat_grad_state,
            flat_grad_val,
        )
        grad_bias_out = grad_bias.reshape(()) if ctx.has_bias else None
        return (
            grad_layer.reshape(*ctx.batch_shape, ctx.nodes, ctx.out_dim).to(dtype=ctx.layer_dtype),
            grad_projected_state.reshape(*ctx.batch_shape, ctx.nodes).to(dtype=ctx.projected_state_dtype),
            grad_projected_val.reshape(*ctx.batch_shape, ctx.nodes, ctx.out_dim).to(dtype=ctx.projected_val_dtype),
            grad_norm_weight.to(dtype=ctx.normalized_weight_dtype),
            grad_bias_out,
            None,
        )


class _DiagonalPropagationCausalDenseSignedAbsMatmul(Function):
    @staticmethod
    def forward(
        ctx: Any,
        layer_val: Tensor,
        projected_state: Tensor,
        projected_val: Tensor,
        normalized_weight: Tensor,
        bias: Tensor,
        has_bias: bool,
    ) -> tuple[Tensor, Tensor]:
        (
            flat_val,
            flat_projected_state,
            flat_projected_val,
            batch_shape,
            nodes,
            out_dim,
        ) = _flatten_dense_tensors(layer_val, projected_state, projected_val)
        if out_dim != flat_val.shape[-1]:
            raise ValueError("diagonal causal dense propagation requires projected_val dim == layer dim.")
        flat_val = flat_val.contiguous()
        state_f32 = flat_projected_state.to(dtype=torch.float32).contiguous()
        val_f32 = flat_projected_val.to(dtype=torch.float32).contiguous()
        weight = normalized_weight.to(dtype=flat_val.dtype).view(1, 1, -1)
        q = flat_val * weight
        scores = torch.bmm(q, flat_val.transpose(1, 2)).to(dtype=torch.float32)
        if has_bias:
            scores = scores + bias.to(dtype=torch.float32)
        mask = torch.ones((nodes, nodes), dtype=torch.bool, device=flat_val.device).tril()
        scores = scores.masked_fill(~mask.view(1, nodes, nodes), 0.0)
        stats = scores.abs().masked_fill(~mask.view(1, nodes, nodes), float("-inf"))
        probs = torch.softmax(stats, dim=-1)
        edges = torch.sign(scores) * probs * mask.view(1, nodes, nodes).to(dtype=probs.dtype)
        delta_state = torch.bmm(edges, state_f32.unsqueeze(-1)).squeeze(-1)
        delta_val = torch.bmm(edges, val_f32)
        ctx.has_bias = bool(has_bias)
        ctx.batch_shape = batch_shape
        ctx.nodes = nodes
        ctx.out_dim = out_dim
        ctx.layer_dtype = layer_val.dtype
        ctx.projected_state_dtype = projected_state.dtype
        ctx.projected_val_dtype = projected_val.dtype
        ctx.normalized_weight_dtype = normalized_weight.dtype
        ctx.save_for_backward(flat_val, state_f32, val_f32, normalized_weight, bias)
        return (
            delta_state.to(dtype=projected_state.dtype).reshape(*batch_shape, nodes),
            delta_val.to(dtype=projected_val.dtype).reshape(*batch_shape, nodes, out_dim),
        )

    @staticmethod
    def backward(ctx: Any, grad_delta_state: Tensor, grad_delta_val: Tensor) -> tuple[Any, ...]:
        flat_val, state_f32, val_f32, normalized_weight, bias = ctx.saved_tensors
        nodes = ctx.nodes
        out_dim = ctx.out_dim
        flat_grad_state = grad_delta_state.reshape(-1, nodes).to(dtype=torch.float32).contiguous()
        flat_grad_val = grad_delta_val.reshape(-1, nodes, out_dim).to(dtype=torch.float32).contiguous()
        weight = normalized_weight.to(dtype=flat_val.dtype).view(1, 1, -1)
        q = flat_val * weight
        scores = torch.bmm(q, flat_val.transpose(1, 2)).to(dtype=torch.float32)
        if ctx.has_bias:
            scores = scores + bias.to(dtype=torch.float32)
        mask = torch.ones((nodes, nodes), dtype=torch.bool, device=flat_val.device).tril()
        mask_3d = mask.view(1, nodes, nodes)
        scores = scores.masked_fill(~mask_3d, 0.0)
        stats = scores.abs().masked_fill(~mask_3d, float("-inf"))
        probs = torch.softmax(stats, dim=-1)
        signs = torch.sign(scores)
        edges = signs * probs * mask_3d.to(dtype=probs.dtype)

        grad_edges = flat_grad_state.unsqueeze(-1) * state_f32.unsqueeze(1)
        grad_edges = grad_edges + torch.bmm(flat_grad_val, val_f32.transpose(1, 2))
        grad_edges = grad_edges * mask_3d.to(dtype=grad_edges.dtype)
        dot = (grad_edges * edges).sum(dim=-1, keepdim=True)
        grad_scores = signs * probs * (signs * grad_edges - dot)
        grad_scores = grad_scores * mask_3d.to(dtype=grad_scores.dtype)

        grad_q = torch.bmm(grad_scores.to(dtype=flat_val.dtype), flat_val)
        grad_key_val = torch.bmm(grad_scores.transpose(1, 2).to(dtype=flat_val.dtype), q)
        grad_layer = grad_key_val + grad_q * weight
        grad_norm_weight = (grad_q * flat_val).sum(dim=(0, 1)).to(dtype=normalized_weight.dtype)
        grad_projected_state = torch.bmm(edges.transpose(1, 2), flat_grad_state.unsqueeze(-1)).squeeze(-1)
        grad_projected_val = torch.bmm(edges.transpose(1, 2), flat_grad_val)
        grad_bias = grad_scores.sum().reshape(()) if ctx.has_bias else None
        return (
            grad_layer.reshape(*ctx.batch_shape, nodes, out_dim).to(dtype=ctx.layer_dtype),
            grad_projected_state.reshape(*ctx.batch_shape, nodes).to(dtype=ctx.projected_state_dtype),
            grad_projected_val.reshape(*ctx.batch_shape, nodes, out_dim).to(dtype=ctx.projected_val_dtype),
            grad_norm_weight.to(dtype=ctx.normalized_weight_dtype),
            grad_bias,
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
        if int(compress_kind) == 2:
            delta_state, delta_val = module.low_rank_propagation_window_entmax15_forward_cuda(
                weighted_projected_source,
                projected_target,
                flat_projected_state.to(dtype=torch.float32).contiguous(),
                flat_projected_val.to(dtype=torch.float32).contiguous(),
                int(window),
                score_bias,
            )
        elif int(compress_kind) == 1:
            state_f32 = flat_projected_state.to(dtype=torch.float32).contiguous()
            val_f32 = flat_projected_val.to(dtype=torch.float32).contiguous()
            if int(window) + 1 >= int(nodes):
                delta_state, delta_val = module.low_rank_propagation_causal_dense_signed_abs_forward_cuda(
                    weighted_projected_source,
                    projected_target,
                    state_f32,
                    val_f32,
                    score_bias,
                )
            else:
                delta_state, delta_val = module.low_rank_propagation_window_signed_abs_forward_cuda(
                    weighted_projected_source,
                    projected_target,
                    state_f32,
                    val_f32,
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
        if ctx.compress_kind == 2:
            packed_mask = valid.contiguous()
            edges = torch.ops.jakal_net.signed_entmax15(scores, packed_mask)
        elif ctx.compress_kind == 1:
            edges = _masked_signed_abs_softmax_from_scores(scores, valid)
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
        if ctx.compress_kind == 2:
            grad_scores = torch.ops.jakal_net.signed_entmax15_backward(
                scores.contiguous(),
                edges.contiguous(),
                grad_edges.contiguous(),
                valid.contiguous(),
            )
            grad_scores = grad_scores * valid_f32
        elif ctx.compress_kind == 1:
            grad_scores = _masked_signed_abs_softmax_backward(
                scores.contiguous(),
                edges.contiguous(),
                grad_edges.contiguous(),
                valid.contiguous(),
            )
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
    source_state: Tensor | None = None,
    target_block_size: int,
    source_block_size: int,
) -> Any:
    if not supports_pairwise_kernel(pairwise_fn):
        raise TypeError("Unsupported pairwise_fn for native propagation.")
    if source_state is None and edge_compress_name == "softsign" and isinstance(pairwise_fn, HadamardMLPPairwise):
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
    use_low_rank_dense_recompute = (
        source_state is None
        and torch.is_grad_enabled()
        and _experimental_fused_training_enabled()
        and edge_compress_name == "signed_abs_softmax"
        and isinstance(pairwise_fn, LowRankBilinearPairwise)
        and _cuda_float_tensor(layer_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
    )
    if use_low_rank_dense_recompute:
        delta_state, delta_val = _LowRankPropagationDenseSignedAbsRecompute.apply(
            layer_val,
            projected_state,
            projected_val,
            pairwise_fn.source_proj.weight,
            pairwise_fn.target_proj.weight,
            pairwise_fn.weight,
            pairwise_fn.bias,
            target_block_size,
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
        source_state,
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
    use_diagonal_causal_dense_cuda = (
        _experimental_diagonal_dense_prop_cuda_enabled()
        and edge_compress_name == "signed_abs_softmax"
        and isinstance(pairwise_fn, DiagonalBilinearPairwise)
        and native_supports("diagonal_propagation_causal_dense_signed_abs_forward_cuda")
        and native_supports("diagonal_propagation_causal_dense_signed_abs_backward_cuda")
        and _cuda_float_tensor(layer_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
        and int(window) + 1 >= int(layer_val.shape[-2])
        and int(projected_val.shape[-1]) == int(layer_val.shape[-1])
    )
    if use_diagonal_causal_dense_cuda:
        bias = (
            pairwise_fn.bias
            if getattr(pairwise_fn, "bias", None) is not None
            else torch.empty(0, dtype=layer_val.dtype, device=layer_val.device)
        )
        delta_state, delta_val = _DiagonalPropagationCausalDenseSignedAbsMatmul.apply(
            layer_val,
            projected_state,
            projected_val,
            pairwise_fn.normalized_weight(),
            bias,
            getattr(pairwise_fn, "bias", None) is not None,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
    use_multihead_signed_abs_cuda_autograd = (
        _experimental_fused_training_enabled()
        and edge_compress_name == "signed_abs_softmax"
        and _is_triton_multihead_signed_smoothmax_lowrank_pairwise(pairwise_fn)
        and native_supports("multihead_low_rank_propagation_causal_dense_signed_abs_forward_cuda")
        and _cuda_float_tensor(layer_val)
        and _cuda_float_tensor(projected_state)
        and _cuda_float_tensor(projected_val)
        and int(window) + 1 >= int(layer_val.shape[-2])
    )
    if use_multihead_signed_abs_cuda_autograd:
        assert isinstance(pairwise_fn, MultiHeadPairwise)
        source_weight, target_weight, core_weight, bias = _stack_multihead_lowrank_weights(pairwise_fn)
        delta_state, delta_val = _MultiHeadLowRankPropagationDenseSignedAbsTriton.apply(
            layer_val,
            projected_state,
            projected_val,
            source_weight,
            target_weight,
            core_weight,
            bias,
        )
        return LayerDelta(delta_state=delta_state, delta_val=delta_val)
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
    use_signed_abs_cuda_autograd = (
        _experimental_fused_training_enabled()
        and _experimental_causal_dense_prop_forward_cuda_enabled()
        and edge_compress_name == "signed_abs_softmax"
        and _query_backward_ops_available()
        and native_supports("low_rank_propagation_causal_dense_signed_abs_forward_cuda")
        and native_supports("low_rank_pairwise_topk_backward_cuda")
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
    if use_cuda_autograd or use_signed_abs_cuda_autograd or use_entmax15_cuda_autograd:
        delta_state, delta_val = _LowRankPropagationWindow.apply(
            layer_val,
            projected_state,
            projected_val,
            pairwise_fn.source_proj.weight,
            pairwise_fn.target_proj.weight,
            pairwise_fn.weight,
            pairwise_fn.bias,
            2
            if edge_compress_name == "signed_entmax15"
            else 1
            if edge_compress_name == "signed_abs_softmax"
            else 0,
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
        and topk <= 64
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
        route_compress_name,
        spec.temperature,
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
        and topk <= 64
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


def read_memory_vector_native(
    *,
    flat_memory: tuple[Tensor, ...],
    read_template_val: Tensor,
    read_projection_weights: tuple[Tensor, ...],
    read_gates: tuple[Tensor, ...],
) -> Tensor:
    if not native_supports("read_memory_vector"):
        raise RuntimeError("Native read_memory_vector is not available.")
    return _native_module().read_memory_vector(
        list(flat_memory),
        read_template_val,
        list(read_projection_weights),
        list(read_gates),
    )
