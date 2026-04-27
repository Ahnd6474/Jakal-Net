#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time

import torch
from torch.nn import functional as F

from jakal_net.core import Layer
from jakal_net.modules import BilinearPairwise
from jakal_net.propagation import SparsePropagation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--nodes", type=int, default=512)
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--implementation", choices=("native", "reference"), default="native")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile-forward", action="store_true")
    return p.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**3


def median_ms(samples: list[float]) -> tuple[float, float]:
    return statistics.median(samples), statistics.fmean(samples)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    B, N, D = args.batch_size, args.nodes, args.dim

    pairwise = BilinearPairwise(D).to(device=device, dtype=dtype)
    propagation = SparsePropagation(
        pairwise_fn=pairwise,
        sparse_type="window",
        window=N,
        implementation=args.implementation,
    )

    def make_inputs(requires_grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.randn(B, N, device=device, dtype=dtype, requires_grad=requires_grad)
        val = F.normalize(torch.randn(B, N, D, device=device, dtype=dtype), dim=-1)
        val.requires_grad_(requires_grad)
        return state, val

    def forward_once(state: torch.Tensor, val: torch.Tensor):
        return propagation.compute_delta(Layer(dim=D, num_nodes=N, state=state, val=val))

    if args.compile_forward:
        compiler = getattr(torch, "compile", None)
        if compiler is None:
            raise SystemExit("torch.compile unavailable")
        forward_once = compiler(forward_once, mode="reduce-overhead", fullgraph=False)

    # Forward only.
    state_fwd, val_fwd = make_inputs(False)
    with torch.no_grad():
        for _ in range(args.warmup):
            forward_once(state_fwd, val_fwd)
        sync(device)
        reset_peak(device)
        fwd_samples = []
        for _ in range(args.iters):
            start = time.perf_counter()
            forward_once(state_fwd, val_fwd)
            sync(device)
            fwd_samples.append((time.perf_counter() - start) * 1000.0)
    fwd_median, fwd_mean = median_ms(fwd_samples)
    fwd_peak = peak_gb(device)

    # Forward + backward. Reuse leaf tensors so random generation/allocation is not timed.
    state, val = make_inputs(True)
    for _ in range(args.warmup):
        pairwise.zero_grad(set_to_none=True)
        state.grad = None
        val.grad = None
        out = forward_once(state, val)
        loss = out.delta_state.float().square().mean() + out.delta_val.float().square().mean()
        loss.backward()
    sync(device)
    reset_peak(device)
    fb_samples = []
    for _ in range(args.iters):
        pairwise.zero_grad(set_to_none=True)
        state.grad = None
        val.grad = None
        start = time.perf_counter()
        out = forward_once(state, val)
        loss = out.delta_state.float().square().mean() + out.delta_val.float().square().mean()
        loss.backward()
        sync(device)
        fb_samples.append((time.perf_counter() - start) * 1000.0)
    fb_median, fb_mean = median_ms(fb_samples)
    fb_peak = peak_gb(device)
    bwd_median = fb_median - fwd_median
    bwd_mean = fb_mean - fwd_mean

    print(
        "config | "
        f"B={B} N={N} D={D} dtype={args.dtype} impl={args.implementation} "
        f"compile_forward={args.compile_forward}"
    )
    print(f"forward | median_ms={fwd_median:.3f} mean_ms={fwd_mean:.3f} peak_gb={fwd_peak:.3f}")
    print(f"fwd_bwd | median_ms={fb_median:.3f} mean_ms={fb_mean:.3f} peak_gb={fb_peak:.3f}")
    print(f"backward_est | median_ms={bwd_median:.3f} mean_ms={bwd_mean:.3f}")


if __name__ == "__main__":
    main()
