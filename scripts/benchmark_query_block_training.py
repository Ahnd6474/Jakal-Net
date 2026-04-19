from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from unittest import mock

import torch

from jakal_net.native_backend import DISABLE_NATIVE_ENV, native_status
from progressive_b_example import (
    ProgressiveBExampleLM,
    build_progressive_b_stage_specs,
    compute_next_token_loss,
    sample_next_token_batch,
)


def run_impl(
    *,
    implementation: str,
    vocab_size: int,
    dim: int,
    seq_len: int,
    target_len: int,
    batch_size: int,
    route_topk: int,
    warmup_layers: int,
    lite_layers: int,
    mid_layers: int,
    full_layers: int,
    final_refine_layers: int,
    query_refine_layers: int,
    block_residual: bool,
    query_residual: bool,
    share_route_families: bool,
    steps: int,
    warmup_steps: int,
) -> dict[str, object]:
    device = "cuda"
    actual_implementation = "kernel" if implementation in {"kernel_legacy", "kernel_native"} else implementation
    stage_specs = build_progressive_b_stage_specs(
        seq_nodes=seq_len,
        lite_layers=lite_layers,
        mid_layers=mid_layers,
        full_layers=full_layers,
    )
    tokens = torch.randint(0, vocab_size - 8, (200_000,), dtype=torch.long)
    query_block_start_token_id = vocab_size - 1

    timings: list[float] = []
    status = "ok"
    error = None
    env_override = (
        mock.patch.dict(os.environ, {DISABLE_NATIVE_ENV: "1"}, clear=False)
        if implementation == "kernel_legacy"
        else nullcontext()
    )
    with env_override:
        native_status(force_reload=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = ProgressiveBExampleLM(
            vocab_size=vocab_size,
            dim=dim,
            seq_nodes=seq_len,
            warmup_layers=warmup_layers,
            stage_specs=stage_specs,
            final_refine_layers=final_refine_layers,
            query_refine_layers=query_refine_layers,
            s_window=32,
            route_topk=route_topk,
            expanded_topk=route_topk,
            compressed_topk=2,
            sequence_sparse_type="window",
            expanded_sparse_type="topk",
            compressed_sparse_type="topk",
            route_mode="topk",
            implementation=actual_implementation,
            route_kind="low_rank_bilinear",
            pairwise_kind="low_rank_bilinear",
            query_topk=route_topk,
            edge_dropout_p=0.1,
            include_query_head=True,
            block_residual=block_residual,
            query_residual=query_residual,
            share_route_families=share_route_families,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        try:
            for step in range(warmup_steps + steps):
                optimizer.zero_grad(set_to_none=True)
                batch = sample_next_token_batch(
                    tokens,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    device=device,
                    target_len=target_len,
                    query_block_start_token_id=query_block_start_token_id,
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                loss, _ = compute_next_token_loss(
                    model,
                    batch,
                    query_block=True,
                    autocast_device_type="cuda",
                    autocast_dtype=torch.bfloat16,
                )
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                if step >= warmup_steps:
                    timings.append(time.perf_counter() - started)
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        result = {
            "implementation": implementation,
            "model_implementation": actual_implementation,
            "status": status,
            "error": error,
            "timings_s": timings,
            "median_s": statistics.median(timings) if timings else None,
            "mean_s": statistics.fmean(timings) if timings else None,
            "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }
        del optimizer
        del model
        torch.cuda.empty_cache()
        native_status(force_reload=True)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=16_384)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--target-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--route-topk", type=int, default=32)
    parser.add_argument("--warmup-layers", type=int, default=2)
    parser.add_argument("--lite-layers", type=int, default=5)
    parser.add_argument("--mid-layers", type=int, default=5)
    parser.add_argument("--full-layers", type=int, default=0)
    parser.add_argument("--final-refine-layers", type=int, default=3)
    parser.add_argument("--query-refine-layers", type=int, default=3)
    parser.add_argument("--block-residual", action="store_true")
    parser.add_argument("--query-residual", action="store_true")
    parser.add_argument("--share-route-families", action="store_true")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument(
        "--implementations",
        nargs="+",
        default=("kernel_legacy", "kernel_native"),
        choices=("reference", "kernel_legacy", "kernel_native", "native"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    results = []
    for implementation in args.implementations:
        results.append(
            run_impl(
                implementation=implementation,
                vocab_size=args.vocab_size,
                dim=args.dim,
                seq_len=args.seq_len,
                target_len=args.target_len,
                batch_size=args.batch_size,
                route_topk=args.route_topk,
                warmup_layers=args.warmup_layers,
                lite_layers=args.lite_layers,
                mid_layers=args.mid_layers,
                full_layers=args.full_layers,
                final_refine_layers=args.final_refine_layers,
                query_refine_layers=args.query_refine_layers,
                block_residual=args.block_residual,
                query_residual=args.query_residual,
                share_route_families=args.share_route_families,
                steps=args.steps,
                warmup_steps=args.warmup_steps,
            )
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
