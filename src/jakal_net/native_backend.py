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
        _query_backward_ops_available()
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


def transition_pairwise_dense_native(
    *,
    route_fn: object,
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
) -> Any:
    if not supports_pairwise_route_kernel(route_fn):
        raise TypeError("Unsupported pairwise route_fn for native sparse transition.")
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
