#!/usr/bin/env python3
"""Benchmark the single-head low-rank TensorCore dense propagation prototype.

This calls the existing native TF32 dense forward directly. The kernel is full
dense and non-causal, so use it to measure the TensorCore score/softmax/reduce
path before wiring a causal/window masked variant into training.
"""
from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from jakal_net.native_backend import _native_module, native_status


@dataclass(frozen=True)
class BenchResult:
    median_ms: float
    mean_ms: float
    peak_gb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--nodes", type=int, default=512)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--score-bias", type=float, default=0.0)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--include-projection", action="store_true")
    parser.add_argument("--causal-baseline", action="store_true")
    return parser.parse_args()


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def time_call(fn, *, device: torch.device, warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        fn()
    cuda_sync(device)
    reset_peak(device)
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        cuda_sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return BenchResult(
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        peak_gb=peak_gb(device),
    )


def signed_abs_dense_reference(
    weighted_source: torch.Tensor,
    target: torch.Tensor,
    projected_state: torch.Tensor,
    projected_val: torch.Tensor,
    score_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.bmm(target, weighted_source.transpose(1, 2)) + float(score_bias)
    edges = torch.sign(scores) * torch.softmax(scores.abs(), dim=-1)
    state = torch.bmm(edges, projected_state.unsqueeze(-1)).squeeze(-1)
    val = torch.bmm(edges, projected_val)
    return state, val


def causal_reference(
    weighted_source: torch.Tensor,
    target: torch.Tensor,
    projected_state: torch.Tensor,
    projected_val: torch.Tensor,
    score_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.bmm(target, weighted_source.transpose(1, 2)) + float(score_bias)
    n = scores.shape[-1]
    mask = torch.ones((n, n), dtype=torch.bool, device=scores.device).triu(1)
    signed = torch.sign(scores)
    abs_scores = scores.abs().masked_fill(mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(abs_scores, dim=-1)
    edges = signed * weights
    edges = edges.masked_fill(mask.unsqueeze(0), 0.0)
    state = torch.bmm(edges, projected_state.unsqueeze(-1)).squeeze(-1)
    val = torch.bmm(edges, projected_val)
    return state, val


def main() -> None:
    args = parse_args()
    if args.nodes % 16 != 0:
        raise SystemExit("--nodes must be divisible by 16 for the current TF32 kernel")
    if args.rank % 8 != 0:
        raise SystemExit("--rank must be divisible by 8 for the current TF32 kernel")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    status = native_status(force_reload=True)
    print(
        "native_status | "
        f"available={status.available} backend={status.backend_name} "
        f"ops={','.join(status.supported_ops)}"
    )
    native = _native_module()
    if "low_rank_propagation_dense_tf32_forward_cuda" not in status.supported_ops:
        raise SystemExit("native extension does not expose low_rank_propagation_dense_tf32_forward_cuda")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    storage_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    b, n, d, r = args.batch_size, args.nodes, args.dim, args.rank
    layer_val = F.normalize(torch.randn(b, n, d, device=device, dtype=storage_dtype), dim=-1)
    source_weight = torch.randn(r, d, device=device, dtype=storage_dtype) * (d ** -0.5)
    target_weight = torch.randn(r, d, device=device, dtype=storage_dtype) * (d ** -0.5)
    core_weight = torch.randn(r, device=device, dtype=storage_dtype) * (r ** -0.5)
    projected_state = torch.randn(b, n, device=device, dtype=torch.float32)
    projected_val = F.normalize(torch.randn(b, n, d, device=device, dtype=torch.float32), dim=-1)

    def project_inputs() -> tuple[torch.Tensor, torch.Tensor]:
        source = F.linear(layer_val, source_weight).float().contiguous()
        target = F.linear(layer_val, target_weight).float().contiguous()
        weighted_source = (source * core_weight.float().view(1, 1, r)).contiguous()
        return weighted_source, target

    weighted_source, target = project_inputs()
    cuda_sync(device)

    print(
        "config | "
        f"B={b} N={n} D={d} R={r} dtype={args.dtype} "
        f"include_projection={args.include_projection} score_bias={args.score_bias}"
    )
    print("note | native TF32 path is full dense/non-causal; causal/window masking is not in this kernel")

    def native_forward() -> tuple[torch.Tensor, torch.Tensor]:
        ws, tgt = project_inputs() if args.include_projection else (weighted_source, target)
        return native.low_rank_propagation_dense_tf32_forward_cuda(
            ws,
            tgt,
            projected_state.contiguous(),
            projected_val.contiguous(),
            float(args.score_bias),
        )

    def torch_forward() -> tuple[torch.Tensor, torch.Tensor]:
        ws, tgt = project_inputs() if args.include_projection else (weighted_source, target)
        return signed_abs_dense_reference(ws, tgt, projected_state, projected_val, args.score_bias)

    native_out = native_forward()
    cuda_sync(device)

    native_result = time_call(native_forward, device=device, warmup=args.warmup, iters=args.iters)
    torch_result = time_call(torch_forward, device=device, warmup=args.warmup, iters=args.iters)
    print(
        "native_tf32 | "
        f"median_ms={native_result.median_ms:.3f} mean_ms={native_result.mean_ms:.3f} "
        f"peak_gb={native_result.peak_gb:.3f}"
    )
    print(
        "torch_bmm | "
        f"median_ms={torch_result.median_ms:.3f} mean_ms={torch_result.mean_ms:.3f} "
        f"peak_gb={torch_result.peak_gb:.3f}"
    )

    if args.compare:
        ref_state, ref_val = torch_forward()
        cuda_sync(device)
        state_diff = (native_out[0] - ref_state).abs().max().item()
        val_diff = (native_out[1] - ref_val).abs().max().item()
        state_rel = (native_out[0] - ref_state).norm().div(ref_state.norm().clamp_min(1e-12)).item()
        val_rel = (native_out[1] - ref_val).norm().div(ref_val.norm().clamp_min(1e-12)).item()
        print(
            "compare_full_dense | "
            f"state_max={state_diff:.6g} val_max={val_diff:.6g} "
            f"state_rel={state_rel:.6g} val_rel={val_rel:.6g}"
        )

    if args.causal_baseline:
        def causal_forward() -> tuple[torch.Tensor, torch.Tensor]:
            ws, tgt = project_inputs() if args.include_projection else (weighted_source, target)
            return causal_reference(ws, tgt, projected_state, projected_val, args.score_bias)

        causal_result = time_call(causal_forward, device=device, warmup=args.warmup, iters=args.iters)
        print(
            "torch_causal_baseline | "
            f"median_ms={causal_result.median_ms:.3f} mean_ms={causal_result.mean_ms:.3f} "
            f"peak_gb={causal_result.peak_gb:.3f}"
        )


if __name__ == "__main__":
    main()
