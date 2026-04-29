from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput  # noqa: E402
from jakal_net.core import Layer  # noqa: E402
from jakal_net.native_backend import (  # noqa: E402
    _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs,
    _diagonal_signed_smoothmax_row_stats_from_flat,
    _diagonal_signed_smoothmax_tile_from_flat,
    _low_rank_multihead_max_parts,
    _multihead_signed_smoothmax_tile,
)
from jakal_net.propagation import _compress_edges, causal_window_mask  # noqa: E402
from train_causal_memory_lm import (  # noqa: E402
    DocumentBatch,
    Lion,
    compute_masked_lm_head_loss,
    resolve_autocast_dtype,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def _temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _pairwise_kind(kind: str) -> str:
    if kind == "diagonal":
        return "diagonal_bilinear"
    if kind == "lowrank":
        return "low_rank_bilinear"
    raise ValueError(kind)


def _bool_env(value: bool) -> str:
    return "1" if value else "0"


def _build_model(
    *,
    implementation: str,
    kind: str,
    vocab_size: int,
    seq_len: int,
    dim: int,
    s_layers: int,
    prediction_layers: int,
    s_window: int,
    pairwise_rank: int,
    route_rank: int,
    pairwise_heads: int,
    aggregate: str,
    device: torch.device,
) -> CausalHierarchicalMemoryLM:
    return CausalHierarchicalMemoryLM(
        vocab_size=vocab_size,
        dim=dim,
        max_seq_len=seq_len,
        s_layers=s_layers,
        memory_slots=(384, 96, 24),
        memory_update_intervals=(1, 2, 4),
        prediction_layers=prediction_layers,
        s_window=s_window,
        prediction_window=64,
        memory_topk=16,
        memory_train_mode="dense",
        memory_eval_mode="dense",
        eval_topk=16,
        pairwise_kind=_pairwise_kind(kind),
        route_kind="low_rank_bilinear",
        pairwise_rank=pairwise_rank,
        route_rank=route_rank,
        pairwise_heads=pairwise_heads,
        route_heads=4,
        pairwise_head_aggregate=aggregate,
        implementation=implementation,
        feed_forward_layers=False,
        disable_memory=True,
    ).to(device)


def _make_batch(*, batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> DocumentBatch:
    return DocumentBatch(
        context=torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long),
        target=torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long),
        loss_mask=torch.ones((batch_size, seq_len), device=device, dtype=torch.float32),
        reset_mask=torch.ones((batch_size,), device=device, dtype=torch.bool),
    )


def _named_grads(module: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    grads: dict[str, torch.Tensor | None] = {}
    for name, parameter in module.named_parameters():
        grads[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
    return grads


def _max_grad_diff(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> tuple[str | None, float]:
    worst_name: str | None = None
    worst_value = 0.0
    for name, left_grad in left.items():
        right_grad = right.get(name)
        if left_grad is None and right_grad is None:
            continue
        if left_grad is None or right_grad is None:
            return name, float("inf")
        diff = (left_grad - right_grad).abs().max().item()
        if diff > worst_value:
            worst_name = name
            worst_value = float(diff)
    return worst_name, worst_value


def _forward_loss(
    model: CausalHierarchicalMemoryLM,
    batch: DocumentBatch,
    *,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    autocast_dtype = resolve_autocast_dtype(precision)
    autocast_context = (
        torch.autocast(device_type=batch.context.device.type, dtype=autocast_dtype)
        if autocast_dtype is not None
        else nullcontext()
    )
    with autocast_context:
        output = model(
            batch.context,
            reset_mask=batch.reset_mask,
            return_memory_state=True,
            return_layers=True,
            return_logits=False,
        )
        assert isinstance(output, MemoryScanOutput)
        if output.query_layer is None:
            raise RuntimeError("Model did not return query_layer.")
        loss = compute_masked_lm_head_loss(
            output.query_layer.val,
            model.lm_head.weight,
            batch.target,
            batch.loss_mask,
        )
        return output.query_layer.val, loss


def run_exactness_compare(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    env = {
        "JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_EDGE_DOT": "1",
    }
    if args.triton_backward is not None:
        env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD"] = _bool_env(args.triton_backward)
        if args.kind == "diagonal":
            env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD_DIAGONAL"] = _bool_env(args.triton_backward)
        else:
            env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD_LOWRANK"] = _bool_env(args.triton_backward)
    if args.triton_forward is not None:
        env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_FORWARD_LOWRANK"] = _bool_env(args.triton_forward)
    if args.diag_tile_mode != "auto":
        env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_DIAGONAL_TILE"] = "1" if args.diag_tile_mode == "triton" else "0"
    if args.diag_blocks:
        env["JAKAL_NET_TRITON_SIGNED_SMOOTHMAX_DIAG_PASS2_BLOCKS"] = args.diag_blocks
    if args.lowrank_blocks:
        env["JAKAL_NET_TRITON_SIGNED_SMOOTHMAX_LOWRANK_PASS2_BLOCKS"] = args.lowrank_blocks

    with _temporary_env(env):
        _set_seed(args.seed)
        reference = _build_model(
            implementation="reference",
            kind=args.kind,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            s_layers=args.s_layers,
            prediction_layers=args.prediction_layers,
            s_window=args.s_window,
            pairwise_rank=args.pairwise_rank,
            route_rank=args.route_rank,
            pairwise_heads=args.pairwise_heads,
            aggregate="signed_smoothmax",
            device=device,
        )
        _set_seed(args.seed)
        native = _build_model(
            implementation="native",
            kind=args.kind,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            s_layers=args.s_layers,
            prediction_layers=args.prediction_layers,
            s_window=args.s_window,
            pairwise_rank=args.pairwise_rank,
            route_rank=args.route_rank,
            pairwise_heads=args.pairwise_heads,
            aggregate="signed_smoothmax",
            device=device,
        )
        native.load_state_dict(reference.state_dict())

        batch = _make_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
        )

        reference.zero_grad(set_to_none=True)
        ref_val, ref_loss = _forward_loss(reference, batch, precision=args.precision)
        ref_loss.backward()
        ref_grads = _named_grads(reference)

        native.zero_grad(set_to_none=True)
        native_val, native_loss = _forward_loss(native, batch, precision=args.precision)
        native_loss.backward()
        native_grads = _named_grads(native)

        grad_name, grad_diff = _max_grad_diff(ref_grads, native_grads)
        return {
            "kind": args.kind,
            "triton_backward": args.triton_backward,
            "triton_forward": args.triton_forward,
            "diag_blocks": args.diag_blocks,
            "lowrank_blocks": args.lowrank_blocks,
            "loss_diff": float((ref_loss.detach() - native_loss.detach()).abs().item()),
            "forward_max_diff": float((ref_val.detach() - native_val.detach()).abs().max().item()),
            "param_grad_name": grad_name,
            "param_grad_max_diff": float(grad_diff),
        }


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_block(fn: Any, *, device: torch.device) -> float:
    if device.type != "cuda":
        start = time.perf_counter()
        fn()
        return (time.perf_counter() - start) * 1000.0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def _benchmark_one_step_once(
    *,
    model: CausalHierarchicalMemoryLM,
    optimizer: torch.optim.Optimizer,
    batch: DocumentBatch,
    precision: str,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    holder: dict[str, torch.Tensor] = {}

    def do_forward() -> None:
        autocast_dtype = resolve_autocast_dtype(precision)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            output = model(
                batch.context,
                reset_mask=batch.reset_mask,
                return_memory_state=True,
                return_layers=True,
                return_logits=False,
            )
            assert isinstance(output, MemoryScanOutput)
            if output.query_layer is None:
                raise RuntimeError("Model did not return query_layer.")
            holder["query_val"] = output.query_layer.val

    def do_loss() -> None:
        holder["loss"] = compute_masked_lm_head_loss(
            holder["query_val"],
            model.lm_head.weight,
            batch.target,
            batch.loss_mask,
        )

    def do_backward() -> None:
        holder["loss"].backward()

    def do_opt() -> None:
        optimizer.step()

    forward_ms = _time_block(do_forward, device=device)
    loss_ms = _time_block(do_loss, device=device)
    backward_ms = _time_block(do_backward, device=device)
    opt_ms = _time_block(do_opt, device=device)
    _cuda_sync(device)
    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = float(torch.cuda.max_memory_allocated(device)) / (1024.0 ** 3)
    return {
        "forward_ms": forward_ms,
        "loss_ms": loss_ms,
        "backward_ms": backward_ms,
        "opt_ms": opt_ms,
        "total_ms": forward_ms + loss_ms + backward_ms + opt_ms,
        "peak_gb": peak_gb,
        "loss": float(holder["loss"].detach().item()),
    }


def run_one_step_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    env = {
        "JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_EDGE_DOT": "1",
        "JAKAL_NET_LM_HEAD_CE_CHUNK": str(args.lm_head_chunk),
    }
    if args.triton_backward is not None:
        env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD"] = _bool_env(args.triton_backward)
        if args.kind == "diagonal":
            env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD_DIAGONAL"] = _bool_env(args.triton_backward)
        else:
            env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD_LOWRANK"] = _bool_env(args.triton_backward)
    if args.triton_forward is not None:
        env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_FORWARD_LOWRANK"] = _bool_env(args.triton_forward)
    if args.diag_tile_mode != "auto":
        env["JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_DIAGONAL_TILE"] = "1" if args.diag_tile_mode == "triton" else "0"
    if args.diag_blocks:
        env["JAKAL_NET_TRITON_SIGNED_SMOOTHMAX_DIAG_PASS2_BLOCKS"] = args.diag_blocks
    if args.lowrank_blocks:
        env["JAKAL_NET_TRITON_SIGNED_SMOOTHMAX_LOWRANK_PASS2_BLOCKS"] = args.lowrank_blocks

    with _temporary_env(env):
        _set_seed(args.seed)
        model = _build_model(
            implementation="native",
            kind=args.kind,
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            s_layers=args.s_layers,
            prediction_layers=args.prediction_layers,
            s_window=args.s_window,
            pairwise_rank=args.pairwise_rank,
            route_rank=args.route_rank,
            pairwise_heads=args.pairwise_heads,
            aggregate="signed_smoothmax",
            device=device,
        )
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        batch = _make_batch(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            device=device,
        )

        for _ in range(args.warmup):
            _benchmark_one_step_once(
                model=model,
                optimizer=optimizer,
                batch=batch,
                precision=args.precision,
                device=device,
            )

        samples = [
            _benchmark_one_step_once(
                model=model,
                optimizer=optimizer,
                batch=batch,
                precision=args.precision,
                device=device,
            )
            for _ in range(args.iters)
        ]
        keys = ("forward_ms", "loss_ms", "backward_ms", "opt_ms", "total_ms", "peak_gb", "loss")
        averaged = {key: float(sum(sample[key] for sample in samples) / len(samples)) for key in keys}
        averaged.update(
            {
                "kind": args.kind,
                "triton_backward": args.triton_backward,
                "triton_forward": args.triton_forward,
                "diag_blocks": args.diag_blocks,
                "lowrank_blocks": args.lowrank_blocks,
                "warmup": args.warmup,
                "iters": args.iters,
            }
        )
        return averaged


def run_sweep(args: argparse.Namespace) -> dict[str, object]:
    if args.kind not in {"diagonal", "lowrank"}:
        raise ValueError("sweep only supports diagonal or lowrank.")
    candidates = [candidate.strip() for candidate in args.candidates.split(";") if candidate.strip()]
    results: list[dict[str, object]] = []
    for candidate in candidates:
        sweep_args = argparse.Namespace(**vars(args))
        sweep_args.triton_backward = True
        if args.kind == "diagonal":
            sweep_args.diag_blocks = candidate
            sweep_args.lowrank_blocks = None
        else:
            sweep_args.lowrank_blocks = candidate
            sweep_args.diag_blocks = None
        results.append(run_one_step_benchmark(sweep_args))
    return {"kind": args.kind, "results": results}


def _diagonal_forward_delta(
    *,
    flat_val: torch.Tensor,
    projected_state: torch.Tensor,
    projected_val: torch.Tensor,
    core: torch.Tensor,
    bias_arg: torch.Tensor,
    has_bias: bool,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    row_max, row_denom = _diagonal_signed_smoothmax_row_stats_from_flat(
        flat_val=flat_val,
        core=core,
        bias_arg=bias_arg,
        has_bias=has_bias,
        tile_size=tile_size,
    )
    batch, nodes, out_dim = projected_val.shape
    delta_state = torch.zeros((batch, nodes), dtype=torch.float32, device=flat_val.device)
    delta_val = torch.zeros((batch, nodes, out_dim), dtype=torch.float32, device=flat_val.device)
    for source_start in range(0, nodes, tile_size):
        source_end = min(source_start + tile_size, nodes)
        scores, _, valid_mask = _diagonal_signed_smoothmax_tile_from_flat(
            flat_val=flat_val,
            core=core,
            bias_arg=bias_arg,
            has_bias=has_bias,
            source_start=source_start,
            source_end=source_end,
            return_head_grads=False,
        )
        probs = torch.exp(scores.abs() - row_max.unsqueeze(-1)).masked_fill(~valid_mask, 0.0) / row_denom.unsqueeze(-1)
        edges = torch.sign(scores) * probs
        delta_state = delta_state + torch.bmm(
            edges,
            projected_state[:, source_start:source_end].unsqueeze(-1),
        ).squeeze(-1)
        delta_val = delta_val + torch.bmm(edges, projected_val[:, source_start:source_end, :])
    return row_max, row_denom, delta_state, delta_val


def run_diagonal_diagnose(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    _set_seed(args.seed)
    batch = args.batch_size
    nodes = args.seq_len
    dim = args.dim
    heads = args.pairwise_heads
    tile_nodes = min(args.tile_size, nodes)
    source_start = 0
    source_end = tile_nodes
    flat_val = torch.randn(batch, nodes, dim, device=device, dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32)
    projected_state = torch.randn(batch, nodes, device=device, dtype=torch.float32)
    projected_val = torch.randn(batch, nodes, dim, device=device, dtype=torch.float32)
    core = torch.randn(heads, dim, device=device, dtype=flat_val.dtype)
    bias_arg = torch.randn(heads, device=device, dtype=flat_val.dtype) if args.has_bias else torch.empty(0, device=device, dtype=flat_val.dtype)

    def collect(tile_mode: str) -> dict[str, torch.Tensor]:
        env = {"JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_DIAGONAL_TILE": "1" if tile_mode == "triton" else "0"}
        with _temporary_env(env):
            scores, head_grads, valid_mask = _diagonal_signed_smoothmax_tile_from_flat(
                flat_val=flat_val,
                core=core,
                bias_arg=bias_arg,
                has_bias=args.has_bias,
                source_start=source_start,
                source_end=source_end,
                return_head_grads=True,
            )
            row_max, row_denom, delta_state, delta_val = _diagonal_forward_delta(
                flat_val=flat_val,
                projected_state=projected_state,
                projected_val=projected_val,
                core=core,
                bias_arg=bias_arg,
                has_bias=args.has_bias,
                tile_size=args.tile_size,
            )
            return {
                "scores": scores.detach(),
                "head_grads": head_grads.detach() if head_grads is not None else torch.empty(0, device=device),
                "valid_mask": valid_mask.detach(),
                "row_max": row_max.detach(),
                "row_denom": row_denom.detach(),
                "delta_state": delta_state.detach(),
                "delta_val": delta_val.detach(),
            }

    fallback = collect("fallback")
    triton = collect("triton")
    valid_mask = fallback["valid_mask"]
    head_valid_mask = valid_mask.unsqueeze(0)
    score_diff = (fallback["scores"] - triton["scores"]).abs().masked_fill(~valid_mask, 0.0).max().item()
    head_grad_diff = (fallback["head_grads"] - triton["head_grads"]).abs().masked_fill(~head_valid_mask, 0.0).max().item()
    return {
        "score_max_diff": float(score_diff),
        "head_grad_max_diff": float(head_grad_diff),
        "row_max_diff": float((fallback["row_max"] - triton["row_max"]).abs().max().item()),
        "row_denom_diff": float((fallback["row_denom"] - triton["row_denom"]).abs().max().item()),
        "delta_state_diff": float((fallback["delta_state"] - triton["delta_state"]).abs().max().item()),
        "delta_val_diff": float((fallback["delta_val"] - triton["delta_val"]).abs().max().item()),
    }


def _get_diag_prop(model: CausalHierarchicalMemoryLM, stage: str):
    if stage == "sequence":
        return model.s_module.sequence_layers[0]
    if stage == "prediction":
        return model.prediction_layers[0]
    raise ValueError(stage)


def run_diagonal_propagation_diagnose(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    env = {
        "JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_BACKWARD_DIAGONAL": _bool_env(args.triton_backward),
        "JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_DIAGONAL_TILE": "1" if args.diag_tile_mode == "triton" else "0",
        "JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_EDGE_DOT": "1",
    }
    with _temporary_env(env):
        _set_seed(args.seed)
        reference = _build_model(
            implementation="reference",
            kind="diagonal",
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            s_layers=args.s_layers,
            prediction_layers=args.prediction_layers,
            s_window=args.s_window,
            pairwise_rank=args.pairwise_rank,
            route_rank=args.route_rank,
            pairwise_heads=args.pairwise_heads,
            aggregate="signed_smoothmax",
            device=device,
        )
        _set_seed(args.seed)
        native = _build_model(
            implementation="native",
            kind="diagonal",
            vocab_size=args.vocab_size,
            seq_len=args.seq_len,
            dim=args.dim,
            s_layers=args.s_layers,
            prediction_layers=args.prediction_layers,
            s_window=args.s_window,
            pairwise_rank=args.pairwise_rank,
            route_rank=args.route_rank,
            pairwise_heads=args.pairwise_heads,
            aggregate="signed_smoothmax",
            device=device,
        )
        native.load_state_dict(reference.state_dict())

        value_dtype = torch.float32
        layer = Layer(
            dim=args.dim,
            num_nodes=args.seq_len,
            state=torch.randn(args.batch_size, args.seq_len, device=device, dtype=value_dtype),
            val=torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=value_dtype),
        )

        results: dict[str, object] = {}
        stages = ["sequence", "prediction"] if args.stage == "both" else [args.stage]
        for stage in stages:
            ref_prop = _get_diag_prop(reference, stage)
            native_prop = _get_diag_prop(native, stage)
            directional_val = ref_prop._directional_val(layer.val)
            ref_scores = ref_prop.pairwise_fn(directional_val, directional_val)
            native_scores = native_prop.pairwise_fn(directional_val, directional_val)
            ref_projected_state, ref_projected_val = ref_prop._project_inputs(layer, directional_val=directional_val)
            ref_projected_state, ref_projected_val = ref_prop._fold_state_weight_into_projected_inputs(
                ref_projected_state,
                ref_projected_val,
                layer.state,
            )
            native_projected_state, native_projected_val = native_prop._project_inputs(layer, directional_val=directional_val)
            native_projected_state, native_projected_val = native_prop._fold_state_weight_into_projected_inputs(
                native_projected_state,
                native_projected_val,
                layer.state,
            )
            mask_2d = causal_window_mask(
                0,
                layer.num_nodes,
                0,
                layer.num_nodes,
                native_prop.window or 0,
                device=ref_scores.device,
            )
            view_shape = (1,) * (ref_scores.ndim - 2) + mask_2d.shape
            mask = mask_2d.view(view_shape)
            ref_edges = _compress_edges(ref_scores, ref_prop.edge_compress_fn, mask=mask)
            ref_edges = ref_prop._weight_edges(ref_edges, layer.state)
            row_max, row_denom = _diagonal_signed_smoothmax_row_stats_from_flat(
                flat_val=directional_val,
                core=torch.stack([head.normalized_weight() for head in native_prop.pairwise_fn.heads]),
                bias_arg=(
                    torch.stack([head.bias for head in native_prop.pairwise_fn.heads])
                    if all(getattr(head, "bias", None) is not None for head in native_prop.pairwise_fn.heads)
                    else torch.empty(0, device=directional_val.device, dtype=directional_val.dtype)
                ),
                has_bias=all(getattr(head, "bias", None) is not None for head in native_prop.pairwise_fn.heads),
                tile_size=layer.num_nodes,
            )
            native_edges = torch.zeros_like(ref_edges, dtype=torch.float32)
            for source_start in range(0, layer.num_nodes, layer.num_nodes):
                source_end = min(source_start + layer.num_nodes, layer.num_nodes)
                scores_tile, _, valid_mask = _diagonal_signed_smoothmax_tile_from_flat(
                    flat_val=directional_val,
                    core=torch.stack([head.normalized_weight() for head in native_prop.pairwise_fn.heads]),
                    bias_arg=(
                        torch.stack([head.bias for head in native_prop.pairwise_fn.heads])
                        if all(getattr(head, "bias", None) is not None for head in native_prop.pairwise_fn.heads)
                        else torch.empty(0, device=directional_val.device, dtype=directional_val.dtype)
                    ),
                    has_bias=all(getattr(head, "bias", None) is not None for head in native_prop.pairwise_fn.heads),
                    source_start=source_start,
                    source_end=source_end,
                    return_head_grads=False,
                )
                probs = torch.exp(scores_tile.abs() - row_max.unsqueeze(-1)).masked_fill(~valid_mask, 0.0) / row_denom.unsqueeze(-1)
                native_edges[:, :, source_start:source_end] = torch.sign(scores_tile) * probs
            native_edges = native_prop._weight_edges(native_edges.to(dtype=ref_edges.dtype), layer.state)
            ref_delta = ref_prop._compute_delta_reference(layer)
            native_delta = native_prop.compute_delta(layer)
            results[stage] = {
                "score_max_diff": float((ref_scores - native_scores).abs().max().item()),
                "projected_state_max_diff": float((ref_projected_state - native_projected_state).abs().max().item()),
                "projected_val_max_diff": float((ref_projected_val - native_projected_val).abs().max().item()),
                "edge_max_diff": float((ref_edges - native_edges.to(dtype=ref_edges.dtype)).abs().max().item()),
                "delta_state_max_diff": float((ref_delta.delta_state - native_delta.delta_state).abs().max().item()),
                "delta_val_max_diff": float((ref_delta.delta_val - native_delta.delta_val).abs().max().item()),
            }
        return results


def _get_lowrank_prop(model: CausalHierarchicalMemoryLM, stage: str):
    if stage == "sequence":
        return model.s_module.sequence_layers[0]
    if stage == "prediction":
        return model.prediction_layers[0]
    raise ValueError(stage)


def run_lowrank_backward_diagnose(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    _set_seed(args.seed)
    model = _build_model(
        implementation="native",
        kind="lowrank",
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        s_layers=args.s_layers,
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        aggregate="smoothmax",
        device=device,
    )

    value_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    layer = Layer(
        dim=args.dim,
        num_nodes=args.seq_len,
        state=torch.randn(args.batch_size, args.seq_len, device=device, dtype=torch.float32),
        val=torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=value_dtype),
    )

    stages = ["sequence", "prediction"] if args.stage == "both" else [args.stage]
    results: dict[str, object] = {}
    for stage in stages:
        prop = _get_lowrank_prop(model, stage)
        directional_val = prop._directional_val(layer.val)
        projected_state, projected_val = prop._project_inputs(layer, directional_val=directional_val)
        projected_state, projected_val = prop._fold_state_weight_into_projected_inputs(
            projected_state,
            projected_val,
            layer.state,
        )
        lowrank_parts = _low_rank_multihead_max_parts(prop.pairwise_fn)
        if lowrank_parts is None:
            raise RuntimeError("Expected lowrank multihead parts.")
        source_weights, target_weights, core_weights, biases, has_bias, aggregate = lowrank_parts
        if aggregate != "smoothmax":
            raise RuntimeError(f"Expected smoothmax aggregate, got {aggregate!r}")

        projected_source, projected_target, _weighted_source, core_cast, _raw_source, _raw_target = (
            _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs._project(
                directional_val,
                source_weights,
                target_weights,
                core_weights,
                exact_per_head=False,
            )
        )
        heads = int(projected_source.shape[0])
        batch = int(projected_source.shape[1])
        nodes = int(projected_source.shape[2])
        rank_dim = int(projected_source.shape[3])
        bias_arg = biases.to(dtype=projected_target.dtype).contiguous() if has_bias else biases

        grad_state = torch.randn((batch, nodes), device=device, dtype=torch.float32)
        grad_val = torch.randn((batch, nodes, args.dim), device=device, dtype=torch.float32)

        source_var = projected_source.detach().clone().to(torch.float32).requires_grad_(True)
        target_var = projected_target.detach().clone().to(torch.float32).requires_grad_(True)
        core_var = core_cast.detach().clone().to(torch.float32).requires_grad_(True)
        weighted_source_var = source_var * core_var.view(heads, 1, 1, rank_dim)
        scores = torch.einsum("bhkr,bhir->bhki", target_var, weighted_source_var)
        if has_bias:
            scores = scores + bias_arg.to(dtype=scores.dtype).view(heads, 1, 1, 1)
        scores.retain_grad()
        mask = causal_window_mask(
            0,
            nodes,
            0,
            nodes,
            prop.window or 0,
            device=device,
        ).view(1, 1, nodes, nodes)
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        combined = torch.logsumexp(masked_scores, dim=0) - torch.log(
            torch.tensor(float(heads), device=device, dtype=masked_scores.dtype)
        )
        combined.retain_grad()
        stats = combined.abs().masked_fill(~mask.squeeze(0).squeeze(0).unsqueeze(0), float("-inf"))
        edge_probs = torch.softmax(stats, dim=-1).masked_fill(~mask.squeeze(0).squeeze(0).unsqueeze(0), 0.0)
        edges = torch.sign(combined) * edge_probs
        delta_state = torch.bmm(edges.to(torch.float32), projected_state.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        delta_val = torch.bmm(edges.to(torch.float32), projected_val.to(torch.float32))
        loss = (delta_state * grad_state).sum() + (delta_val * grad_val).sum()
        loss.backward()

        with torch.no_grad():
            valid_mask = mask.squeeze(0).squeeze(0).unsqueeze(0)
            row_max = stats.amax(dim=-1)
            row_denom = torch.exp(stats - row_max.unsqueeze(-1)).masked_fill(~valid_mask, 0.0).sum(dim=-1).clamp_min(1.0e-20)
            probs = torch.exp(combined.abs() - row_max.unsqueeze(-1)).masked_fill(~valid_mask, 0.0) / row_denom.unsqueeze(-1)
            signs = torch.sign(combined).masked_fill(~valid_mask, 0.0)
            grad_edges = grad_state.unsqueeze(-1) * projected_state.unsqueeze(1).to(torch.float32)
            grad_edges = grad_edges + torch.bmm(grad_val, projected_val.to(torch.float32).transpose(1, 2))
            edge_dot = (grad_edges * edges).sum(dim=-1)
            grad_scores = signs * probs * (signs * grad_edges - edge_dot.unsqueeze(-1))
            grad_scores = grad_scores.masked_fill(~valid_mask, 0.0)
            head_weights = torch.softmax(masked_scores, dim=0).masked_fill(~mask, 0.0)
            grad_scores_h = head_weights * grad_scores.unsqueeze(0)
            grad_scores_flat = grad_scores_h.reshape(heads * batch, nodes, nodes).to(torch.float32)
            weighted_source_flat = weighted_source_var.detach().reshape(heads * batch, nodes, rank_dim).to(torch.float32)
            target_flat = target_var.detach().reshape(heads * batch, nodes, rank_dim).to(torch.float32)
            manual_grad_target = torch.bmm(grad_scores_flat, weighted_source_flat).reshape(heads, batch, nodes, rank_dim)
            manual_grad_weighted_source = torch.bmm(grad_scores_flat.transpose(1, 2), target_flat).reshape(
                heads, batch, nodes, rank_dim
            )
            manual_grad_source = manual_grad_weighted_source * core_var.detach().view(heads, 1, 1, rank_dim)
            manual_grad_core = (manual_grad_weighted_source * source_var.detach()).sum(dim=(1, 2))

        results[stage] = {
            "combined_grad_max_diff": float((combined.grad - grad_scores).abs().max().item()),
            "head_score_grad_max_diff": float((scores.grad - grad_scores_h).abs().masked_fill(~mask, 0.0).max().item()),
            "projected_target_grad_max_diff": float((target_var.grad - manual_grad_target).abs().max().item()),
            "projected_source_grad_max_diff": float((source_var.grad - manual_grad_source).abs().max().item()),
            "core_grad_max_diff": float((core_var.grad - manual_grad_core).abs().max().item()),
            "combined_grad_ref_max_abs": float(combined.grad.abs().max().item()),
            "combined_grad_manual_max_abs": float(grad_scores.abs().max().item()),
            "head_score_grad_ref_max_abs": float(scores.grad.abs().masked_fill(~mask, 0.0).max().item()),
            "head_score_grad_manual_max_abs": float(grad_scores_h.abs().masked_fill(~mask, 0.0).max().item()),
        }
    return results


def run_lowrank_custom_backward_diagnose(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    _set_seed(args.seed)
    model = _build_model(
        implementation="native",
        kind="lowrank",
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        s_layers=args.s_layers,
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        aggregate="smoothmax",
        device=device,
    )

    value_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    layer = Layer(
        dim=args.dim,
        num_nodes=args.seq_len,
        state=torch.randn(args.batch_size, args.seq_len, device=device, dtype=torch.float32),
        val=torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=value_dtype),
    )

    stages = ["sequence", "prediction"] if args.stage == "both" else [args.stage]
    results: dict[str, object] = {}
    for stage in stages:
        prop = _get_lowrank_prop(model, stage)
        directional_val = prop._directional_val(layer.val)
        projected_state, projected_val = prop._project_inputs(layer, directional_val=directional_val)
        projected_state, projected_val = prop._fold_state_weight_into_projected_inputs(
            projected_state,
            projected_val,
            layer.state,
        )
        lowrank_parts = _low_rank_multihead_max_parts(prop.pairwise_fn)
        if lowrank_parts is None:
            raise RuntimeError("Expected lowrank multihead parts.")
        source_weights, target_weights, core_weights, biases, has_bias, aggregate = lowrank_parts
        if aggregate != "smoothmax":
            raise RuntimeError(f"Expected smoothmax aggregate, got {aggregate!r}")

        grad_state = torch.randn((args.batch_size, args.seq_len), device=device, dtype=torch.float32)
        grad_val = torch.randn((args.batch_size, args.seq_len, args.dim), device=device, dtype=torch.float32)

        work_dtype = directional_val.dtype

        def clone_param(t: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
            return t.detach().clone().to(dtype=dtype or t.dtype).requires_grad_(True)

        ref_layer_val = directional_val.detach().clone().to(work_dtype).requires_grad_(True)
        ref_proj_state = projected_state.detach().clone().to(torch.float32).requires_grad_(True)
        ref_proj_val = projected_val.detach().clone().to(work_dtype).requires_grad_(True)
        ref_source_weights = clone_param(source_weights)
        ref_target_weights = clone_param(target_weights)
        ref_core_weights = clone_param(core_weights)
        ref_biases = clone_param(biases, dtype=work_dtype) if has_bias else biases.detach().clone().to(work_dtype)

        projected_source, projected_target, weighted_source, _core_cast, _raw_source, _raw_target = (
            _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs._project(
                ref_layer_val,
                ref_source_weights,
                ref_target_weights,
                ref_core_weights,
                exact_per_head=False,
            )
        )
        scores = torch.einsum("bhkr,bhir->bhki", projected_target, weighted_source)
        if has_bias:
            scores = scores + ref_biases.to(dtype=scores.dtype).view(scores.shape[0], 1, 1, 1)
        mask = causal_window_mask(
            0,
            args.seq_len,
            0,
            args.seq_len,
            prop.window or 0,
            device=device,
        ).view(1, 1, args.seq_len, args.seq_len)
        masked_scores = scores.masked_fill(~mask, float("-inf"))
        combined = torch.logsumexp(masked_scores, dim=0) - torch.log(
            torch.tensor(float(scores.shape[0]), device=device, dtype=masked_scores.dtype)
        )
        valid_mask = mask.squeeze(0).squeeze(0).unsqueeze(0)
        stats = combined.abs().masked_fill(~valid_mask, float("-inf"))
        edge_probs = torch.softmax(stats, dim=-1).masked_fill(~valid_mask, 0.0)
        edges = torch.sign(combined) * edge_probs
        ref_delta_state = torch.bmm(edges.to(torch.float32), ref_proj_state.unsqueeze(-1)).squeeze(-1)
        ref_delta_val = torch.bmm(edges.to(work_dtype), ref_proj_val).to(torch.float32)
        ref_loss = (ref_delta_state * grad_state).sum() + (ref_delta_val * grad_val).sum()
        ref_loss.backward()

        native_layer_val = directional_val.detach().clone().to(work_dtype).requires_grad_(True)
        native_proj_state = projected_state.detach().clone().to(torch.float32).requires_grad_(True)
        native_proj_val = projected_val.detach().clone().to(work_dtype).requires_grad_(True)
        native_source_weights = clone_param(source_weights)
        native_target_weights = clone_param(target_weights)
        native_core_weights = clone_param(core_weights)
        native_biases = clone_param(biases, dtype=work_dtype) if has_bias else biases.detach().clone().to(work_dtype)
        native_delta_state, native_delta_val = _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs.apply(
            native_layer_val,
            native_proj_state,
            native_proj_val,
            native_source_weights,
            native_target_weights,
            native_core_weights,
            native_biases,
            has_bias,
            "smoothmax",
        )
        native_loss = (native_delta_state.to(torch.float32) * grad_state).sum() + (
            native_delta_val.to(torch.float32) * grad_val
        ).sum()
        native_loss.backward()

        def max_diff(left: torch.Tensor | None, right: torch.Tensor | None) -> float:
            if left is None or right is None:
                return float("inf")
            return float((left - right).abs().max().item())

        results[stage] = {
            "delta_state_forward_max_diff": float((ref_delta_state - native_delta_state.to(torch.float32)).abs().max().item()),
            "delta_val_forward_max_diff": float((ref_delta_val - native_delta_val.to(torch.float32)).abs().max().item()),
            "layer_val_grad_max_diff": max_diff(ref_layer_val.grad, native_layer_val.grad),
            "projected_state_grad_max_diff": max_diff(ref_proj_state.grad, native_proj_state.grad),
            "projected_val_grad_max_diff": max_diff(ref_proj_val.grad, native_proj_val.grad),
            "source_weights_grad_max_diff": max_diff(ref_source_weights.grad, native_source_weights.grad),
            "target_weights_grad_max_diff": max_diff(ref_target_weights.grad, native_target_weights.grad),
            "core_weights_grad_max_diff": max_diff(ref_core_weights.grad, native_core_weights.grad),
            "bias_grad_max_diff": max_diff(ref_biases.grad if has_bias else None, native_biases.grad if has_bias else None),
            "source_weights_grad_ref_max_abs": float(ref_source_weights.grad.abs().max().item()),
            "source_weights_grad_native_max_abs": float(native_source_weights.grad.abs().max().item()),
            "target_weights_grad_ref_max_abs": float(ref_target_weights.grad.abs().max().item()),
            "target_weights_grad_native_max_abs": float(native_target_weights.grad.abs().max().item()),
            "core_weights_grad_ref_max_abs": float(ref_core_weights.grad.abs().max().item()),
            "core_weights_grad_native_max_abs": float(native_core_weights.grad.abs().max().item()),
        }
    return results


def run_lowrank_forward_diagnose(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable.")

    _set_seed(args.seed)
    reference = _build_model(
        implementation="native",
        kind="lowrank",
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        s_layers=args.s_layers,
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        aggregate="signed_smoothmax",
        device=device,
    )
    trial = _build_model(
        implementation="native",
        kind="lowrank",
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        dim=args.dim,
        s_layers=args.s_layers,
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        aggregate="signed_smoothmax",
        device=device,
    )
    trial.load_state_dict(reference.state_dict())

    value_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    layer = Layer(
        dim=args.dim,
        num_nodes=args.seq_len,
        state=torch.randn(args.batch_size, args.seq_len, device=device, dtype=torch.float32),
        val=torch.randn(args.batch_size, args.seq_len, args.dim, device=device, dtype=value_dtype),
    )
    stages = ["sequence", "prediction"] if args.stage == "both" else [args.stage]
    results: dict[str, object] = {}
    for stage in stages:
        ref_prop = _get_lowrank_prop(reference, stage)
        trial_prop = _get_lowrank_prop(trial, stage)
        directional_val = ref_prop._directional_val(layer.val)
        projected_state, projected_val = ref_prop._project_inputs(layer, directional_val=directional_val)
        lowrank_parts = _low_rank_multihead_max_parts(trial_prop.pairwise_fn)
        if lowrank_parts is None:
            raise RuntimeError("Expected lowrank multihead parts.")
        source_weights, target_weights, core_weights, biases, has_bias, _aggregate = lowrank_parts
        projected_source, projected_target, weighted_source, _core_cast = _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs._project(
            directional_val,
            source_weights,
            target_weights,
            core_weights,
            exact_per_head=True,
        )
        bias_arg = biases.to(dtype=projected_target.dtype).contiguous() if has_bias else biases
        source_strength = ref_prop._edge_source_strength(layer.state)
        projected_state_exact = source_strength
        projected_val_exact = projected_val * source_strength.unsqueeze(-1)

        def collect(use_triton_forward: bool) -> dict[str, torch.Tensor]:
            env = {"JAKAL_NET_MULTIHEAD_SIGNED_SMOOTHMAX_TRITON_FORWARD_LOWRANK": "1" if use_triton_forward else "0"}
            with _temporary_env(env):
                row_max, row_denom = _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs._row_stats(
                    weighted_source,
                    projected_target,
                    bias_arg,
                    has_bias,
                    args.seq_len,
                    "signed_smoothmax",
                    use_triton_forward,
                )
                delta_state = torch.zeros((layer.batch_shape[0], layer.num_nodes), dtype=torch.float32, device=device)
                delta_val = torch.zeros((layer.batch_shape[0], layer.num_nodes, args.dim), dtype=torch.float32, device=device)
                for source_start in range(0, layer.num_nodes, args.seq_len):
                    source_end = min(source_start + args.seq_len, layer.num_nodes)
                    scores, _head = _LowRankMultiHeadMaxPropagationCausalDenseSignedAbs._best_score_tile(
                        weighted_source,
                        projected_target,
                        bias_arg,
                        has_bias,
                        source_start,
                        source_end,
                        "signed_smoothmax",
                        use_triton_forward,
                    )
                    probs = torch.exp(scores.abs() - row_max.unsqueeze(-1)).masked_fill(~torch.isfinite(scores), 0.0)
                    edges = torch.sign(scores) * (probs / row_denom.unsqueeze(-1))
                    weighted_edges = edges * projected_state_exact[:, source_start:source_end].unsqueeze(1)
                    delta_state = delta_state + weighted_edges.sum(dim=-1)
                    delta_val = delta_val + torch.bmm(weighted_edges, projected_val[:, source_start:source_end, :].to(dtype=torch.float32))
                scores, head_grads, valid_mask = _multihead_signed_smoothmax_tile(
                    projected_target=projected_target,
                    weighted_source=weighted_source,
                    bias_arg=bias_arg,
                    has_bias=has_bias,
                    source_start=0,
                    source_end=layer.num_nodes,
                    allow_triton=use_triton_forward,
                    return_head_grads=True,
                )
                return {
                    "row_max": row_max.detach(),
                    "row_denom": row_denom.detach(),
                    "delta_state": delta_state.detach(),
                    "delta_val": delta_val.detach(),
                    "scores": scores.detach(),
                    "head_grads": head_grads.detach() if head_grads is not None else torch.empty(0, device=device),
                    "valid_mask": valid_mask.detach(),
                }

        fallback = collect(False)
        triton = collect(True)
        valid_mask = fallback["valid_mask"]
        head_valid_mask = valid_mask.unsqueeze(0)
        results[stage] = {
            "score_max_diff": float((fallback["scores"] - triton["scores"]).abs().masked_fill(~valid_mask, 0.0).max().item()),
            "head_grad_max_diff": float((fallback["head_grads"] - triton["head_grads"]).abs().masked_fill(~head_valid_mask, 0.0).max().item()),
            "row_max_diff": float((fallback["row_max"] - triton["row_max"]).abs().max().item()),
            "row_denom_diff": float((fallback["row_denom"] - triton["row_denom"]).abs().max().item()),
            "delta_state_max_diff": float((fallback["delta_state"] - triton["delta_state"]).abs().max().item()),
            "delta_val_max_diff": float((fallback["delta_val"] - triton["delta_val"]).abs().max().item()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Exactness and one-step benchmark for signed_smoothmax Triton backward.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--kind", choices=["diagonal", "lowrank"], required=True)
        subparser.add_argument("--vocab-size", type=int, default=16384)
        subparser.add_argument("--seq-len", type=int, default=512)
        subparser.add_argument("--dim", type=int, default=384)
        subparser.add_argument("--batch-size", type=int, default=384)
        subparser.add_argument("--s-layers", type=int, default=6)
        subparser.add_argument("--prediction-layers", type=int, default=3)
        subparser.add_argument("--s-window", type=int, default=256)
        subparser.add_argument("--pairwise-rank", type=int, default=128)
        subparser.add_argument("--route-rank", type=int, default=96)
        subparser.add_argument("--pairwise-heads", type=int, default=4)
        subparser.add_argument("--precision", default="bf16")
        subparser.add_argument("--seed", type=int, default=1337)
        subparser.add_argument("--triton-backward", action=argparse.BooleanOptionalAction, default=None)
        subparser.add_argument("--triton-forward", action=argparse.BooleanOptionalAction, default=None)
        subparser.add_argument("--diag-blocks")
        subparser.add_argument("--lowrank-blocks")
        subparser.add_argument("--diag-tile-mode", choices=["auto", "triton", "fallback"], default="auto")

    compare_parser = subparsers.add_parser("compare")
    add_common(compare_parser)
    compare_parser.set_defaults(
        batch_size=4,
        seq_len=32,
        dim=64,
        s_layers=2,
        prediction_layers=1,
        s_window=32,
        pairwise_rank=16,
        route_rank=16,
    )
    compare_parser.set_defaults(func=run_exactness_compare)

    bench_parser = subparsers.add_parser("benchmark")
    add_common(bench_parser)
    bench_parser.add_argument("--warmup", type=int, default=1)
    bench_parser.add_argument("--iters", type=int, default=3)
    bench_parser.add_argument("--lr", type=float, default=2e-5)
    bench_parser.add_argument("--weight-decay", type=float, default=0.1)
    bench_parser.add_argument("--lm-head-chunk", type=int, default=8192)
    bench_parser.set_defaults(func=run_one_step_benchmark)

    sweep_parser = subparsers.add_parser("sweep")
    add_common(sweep_parser)
    sweep_parser.add_argument("--warmup", type=int, default=1)
    sweep_parser.add_argument("--iters", type=int, default=2)
    sweep_parser.add_argument("--lr", type=float, default=2e-5)
    sweep_parser.add_argument("--weight-decay", type=float, default=0.1)
    sweep_parser.add_argument("--lm-head-chunk", type=int, default=8192)
    sweep_parser.add_argument(
        "--candidates",
        default="32,32,32,32;64,32,32,32;32,64,32,32;32,32,64,32",
        help="semicolon-separated BLOCK tuples",
    )
    sweep_parser.set_defaults(func=run_sweep)

    diagnose_parser = subparsers.add_parser("diagnose-diagonal")
    add_common(diagnose_parser)
    for action in diagnose_parser._actions:
        if action.dest == "kind":
            action.required = False
            break
    diagnose_parser.add_argument("--tile-size", type=int, default=16)
    diagnose_parser.add_argument("--has-bias", action="store_true")
    diagnose_parser.set_defaults(
        kind="diagonal",
        batch_size=4,
        seq_len=32,
        dim=64,
        pairwise_rank=16,
        route_rank=16,
        func=run_diagonal_diagnose,
    )

    diagnose_prop_parser = subparsers.add_parser("diagnose-diagonal-propagation")
    add_common(diagnose_prop_parser)
    for action in diagnose_prop_parser._actions:
        if action.dest == "kind":
            action.required = False
            break
    diagnose_prop_parser.add_argument("--stage", choices=["sequence", "prediction", "both"], default="both")
    diagnose_prop_parser.set_defaults(
        kind="diagonal",
        batch_size=4,
        seq_len=32,
        dim=64,
        s_layers=2,
        prediction_layers=1,
        s_window=32,
        pairwise_rank=16,
        route_rank=16,
        func=run_diagonal_propagation_diagnose,
    )

    diagnose_lowrank_parser = subparsers.add_parser("diagnose-lowrank-forward")
    add_common(diagnose_lowrank_parser)
    for action in diagnose_lowrank_parser._actions:
        if action.dest == "kind":
            action.required = False
            break
    diagnose_lowrank_parser.add_argument("--stage", choices=["sequence", "prediction", "both"], default="both")
    diagnose_lowrank_parser.set_defaults(
        kind="lowrank",
        batch_size=4,
        seq_len=32,
        dim=64,
        s_layers=2,
        prediction_layers=1,
        s_window=32,
        pairwise_rank=16,
        route_rank=16,
        func=run_lowrank_forward_diagnose,
    )

    diagnose_lowrank_backward_parser = subparsers.add_parser("diagnose-lowrank-backward")
    add_common(diagnose_lowrank_backward_parser)
    for action in diagnose_lowrank_backward_parser._actions:
        if action.dest == "kind":
            action.required = False
            break
    diagnose_lowrank_backward_parser.add_argument("--stage", choices=["sequence", "prediction", "both"], default="both")
    diagnose_lowrank_backward_parser.set_defaults(
        kind="lowrank",
        batch_size=4,
        seq_len=32,
        dim=64,
        s_layers=2,
        prediction_layers=1,
        s_window=32,
        pairwise_rank=16,
        route_rank=16,
        func=run_lowrank_backward_diagnose,
    )

    diagnose_lowrank_custom_backward_parser = subparsers.add_parser("diagnose-lowrank-custom-backward")
    add_common(diagnose_lowrank_custom_backward_parser)
    for action in diagnose_lowrank_custom_backward_parser._actions:
        if action.dest == "kind":
            action.required = False
            break
    diagnose_lowrank_custom_backward_parser.add_argument("--stage", choices=["sequence", "prediction", "both"], default="both")
    diagnose_lowrank_custom_backward_parser.set_defaults(
        kind="lowrank",
        batch_size=4,
        seq_len=32,
        dim=64,
        s_layers=2,
        prediction_layers=1,
        s_window=32,
        pairwise_rank=16,
        route_rank=16,
        func=run_lowrank_custom_backward_diagnose,
    )

    args = parser.parse_args()
    result = args.func(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
