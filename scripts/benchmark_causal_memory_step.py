from __future__ import annotations

import argparse
import contextlib
import statistics
import time
from typing import Sequence

import torch
import torch.nn.functional as F

from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM


class Lion(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients.")
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                state = self.state[parameter]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(parameter)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(gradient, alpha=1.0 - beta1)
                parameter.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add(gradient, alpha=1.0 - beta2)
        return loss


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(args: argparse.Namespace, device: torch.device) -> CausalHierarchicalMemoryLM:
    return CausalHierarchicalMemoryLM(
        vocab_size=args.vocab_size,
        dim=args.dim,
        max_seq_len=args.seq_len,
        s_layers=args.s_layers,
        memory_slots=tuple(args.memory_slots),
        memory_update_intervals=tuple(args.memory_update_intervals),
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        prediction_window=args.prediction_window,
        memory_topk=args.memory_topk,
        memory_train_mode=args.memory_train_mode,
        memory_eval_mode=args.memory_eval_mode,
        eval_topk=args.eval_topk,
        scan_backend=args.scan_backend,
        pairwise_kind=args.pairwise_kind,
        route_kind=args.route_kind,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        route_heads=args.route_heads,
        pairwise_anchor_heads=args.pairwise_anchor_heads,
        route_anchor_heads=args.route_anchor_heads,
        implementation=args.implementation,
        direction_only_values=args.direction_only_values,
        feed_forward_layers=not args.disable_feed_forward_layers,
        memory_feed_forward_layers=not (
            args.disable_feed_forward_layers or args.disable_memory_feed_forward_layers
        ),
        feed_forward_hidden_mult=args.feed_forward_hidden_mult,
        disable_memory=args.disable_memory,
        disable_memory_read=args.disable_memory_read,
        disable_memory_propagation=args.disable_memory_propagation,
    ).to(device)


def build_optimizer(
    args: argparse.Namespace,
    parameters: Sequence[torch.nn.Parameter],
) -> torch.optim.Optimizer:
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if args.optimizer == "lion":
        return Lion(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        kwargs = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        }
        if str(args.device).startswith("cuda"):
            kwargs["fused"] = True
        try:
            return torch.optim.AdamW(trainable, **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            return torch.optim.AdamW(trainable, **kwargs)
    raise ValueError(f"Unsupported optimizer: {args.optimizer!r}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark model-only causal-memory LM step time.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--s-layers", type=int, default=6)
    parser.add_argument("--memory-slots", type=int, nargs="+", default=[384, 96, 24])
    parser.add_argument("--memory-update-intervals", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prediction-layers", type=int, default=3)
    parser.add_argument("--s-window", type=int, default=512)
    parser.add_argument("--prediction-window", type=int, default=512)
    parser.add_argument("--memory-topk", type=int, default=16)
    parser.add_argument("--memory-train-mode", choices=("dense", "topk"), default="dense")
    parser.add_argument("--memory-eval-mode", choices=("dense", "topk"), default="topk")
    parser.add_argument("--eval-topk", type=int, default=16)
    parser.add_argument(
        "--pairwise-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "bilinear", "scaled_cosine"),
        default="low_rank_bilinear",
    )
    parser.add_argument(
        "--route-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "bilinear", "fixed_projection", "query_norm_dot"),
        default="low_rank_bilinear",
    )
    parser.add_argument("--pairwise-rank", type=int, default=256)
    parser.add_argument("--route-rank", type=int, default=256)
    parser.add_argument("--pairwise-heads", type=int, default=4)
    parser.add_argument("--route-heads", type=int, default=4)
    parser.add_argument("--pairwise-anchor-heads", type=int, default=0)
    parser.add_argument("--route-anchor-heads", type=int, default=0)
    parser.add_argument("--implementation", choices=("reference", "streaming", "kernel", "native"), default="native")
    parser.add_argument("--scan-backend", choices=("auto", "python", "native"), default="auto")
    parser.add_argument("--disable-feed-forward-layers", action="store_true")
    parser.add_argument("--disable-memory-feed-forward-layers", action="store_true")
    parser.add_argument("--disable-memory", action="store_true")
    parser.add_argument("--disable-memory-read", action="store_true")
    parser.add_argument("--disable-memory-propagation", action="store_true")
    parser.add_argument("--direction-only-values", action="store_true")
    parser.add_argument("--feed-forward-hidden-mult", type=float, default=2.0)
    parser.add_argument("--optimizer", choices=("lion", "adamw"), default="lion")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable.")
    if len(args.memory_slots) != len(args.memory_update_intervals):
        raise ValueError("memory_slots and memory_update_intervals must have the same length.")

    torch.manual_seed(args.seed)
    model = build_model(args, device)
    model.train()
    optimizer = build_optimizer(args, list(model.parameters()))
    autocast_dtype = torch.bfloat16 if args.bf16 else None

    input_ids = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)
    target = torch.randint(args.vocab_size, (args.batch_size, args.seq_len), device=device)

    print(
        "benchmark_config | "
        f"device={device.type} | batch={args.batch_size} | seq={args.seq_len} | dim={args.dim} | "
        f"s_layers={args.s_layers} | prediction_layers={args.prediction_layers} | "
        f"s_window={args.s_window} | prediction_window={args.prediction_window} | "
        f"disable_memory={args.disable_memory} | direction_only_values={args.direction_only_values} | "
        f"implementation={args.implementation} | optimizer={args.optimizer} | bf16={args.bf16}",
        flush=True,
    )

    timings: list[float] = []
    last_loss = float("nan")
    model.reset_dense_apply_stats()
    total_iters = args.warmup_iters + args.iters

    for step in range(total_iters):
        optimizer.zero_grad(set_to_none=True)
        _sync(device)
        start = time.perf_counter()
        if device.type == "cuda" and autocast_dtype is not None:
            context = torch.autocast(device_type="cuda", dtype=autocast_dtype)
        else:
            context = contextlib.nullcontext()
        with context:
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1))
        loss.backward()
        optimizer.step()
        _sync(device)
        elapsed = time.perf_counter() - start
        last_loss = float(loss.detach())
        if step >= args.warmup_iters:
            timings.append(elapsed)

    stats = model.dense_apply_stats()
    print(f"benchmark_loss | {last_loss:.6f}", flush=True)
    print(f"benchmark_mean_s | {statistics.fmean(timings):.6f}", flush=True)
    print(f"benchmark_median_s | {statistics.median(timings):.6f}", flush=True)
    print(f"benchmark_min_s | {min(timings):.6f}", flush=True)
    print(f"benchmark_max_s | {max(timings):.6f}", flush=True)
    print(f"benchmark_dense_apply | {stats}", flush=True)


if __name__ == "__main__":
    main()
