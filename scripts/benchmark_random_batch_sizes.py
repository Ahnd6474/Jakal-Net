from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM  # noqa: E402
from train_causal_memory_lm import (  # noqa: E402
    DocumentBatch,
    TransformerBaselineLM,
    compute_masked_loss,
    count_parameters,
    resolve_autocast_dtype,
)


def make_model(kind: str, *, vocab_size: int, seq_len: int, dim: int) -> torch.nn.Module:
    if kind == "transformer":
        return TransformerBaselineLM(
            vocab_size=vocab_size,
            dim=dim,
            max_seq_len=seq_len,
            layers=5,
            heads=6,
            ff_mult=3.7005,
            dropout=0.0,
        )
    kwargs: dict[str, Any] = dict(
        vocab_size=vocab_size,
        dim=dim,
        max_seq_len=seq_len,
        s_layers=6,
        memory_slots=(384, 96, 24),
        memory_update_intervals=(1, 2, 4),
        prediction_layers=3,
        s_window=256,
        prediction_window=64,
        feed_forward_hidden_mult=2.0,
        memory_topk=16,
        memory_train_mode="dense",
        memory_eval_mode="topk",
        eval_topk=16,
        scan_backend="auto",
        pairwise_kind="low_rank_bilinear",
        route_kind="low_rank_bilinear",
        pairwise_rank=128,
        route_rank=96,
        pairwise_heads=4,
        route_heads=4,
        implementation="native",
        feed_forward_layers=False,
    )
    if kind == "nomemory":
        kwargs["disable_memory"] = True
    elif kind == "noread":
        kwargs["disable_memory_read"] = True
    else:
        raise ValueError(kind)
    return CausalHierarchicalMemoryLM(**kwargs)


def try_batch(model: torch.nn.Module, *, batch_size: int, vocab_size: int, seq_len: int, precision: str, device: torch.device) -> None:
    model.train()
    context = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)
    loss_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.float32)
    reset_mask = torch.ones((batch_size,), device=device, dtype=torch.bool)
    autocast_dtype = resolve_autocast_dtype(precision)
    ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if autocast_dtype is not None else torch.enable_grad()
    with ctx:
        out = model(context, reset_mask=reset_mask, return_memory_state=True)
        logits = out.logits if hasattr(out, "logits") else out
        loss = compute_masked_loss(logits, target, loss_mask)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Find random-token batch sizes that fit on GPU.")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    results = {}
    for kind in ("transformer", "nomemory", "noread"):
        torch.cuda.empty_cache()
        gc.collect()
        model = make_model(kind, vocab_size=args.vocab_size, seq_len=args.seq_len, dim=args.dim).to(device)
        params = count_parameters(model)
        best = 0
        tested = []
        batch = 1
        while batch <= args.max_batch:
            try:
                try_batch(model, batch_size=batch, vocab_size=args.vocab_size, seq_len=args.seq_len, precision=args.precision, device=device)
                best = batch
                tested.append({"batch": batch, "ok": True})
                batch *= 2
            except torch.cuda.OutOfMemoryError:
                tested.append({"batch": batch, "ok": False, "error": "OOM"})
                torch.cuda.empty_cache()
                break
        lo, hi = best + 1, min(batch - 1, args.max_batch)
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                try_batch(model, batch_size=mid, vocab_size=args.vocab_size, seq_len=args.seq_len, precision=args.precision, device=device)
                best = mid
                tested.append({"batch": mid, "ok": True})
                lo = mid + 1
            except torch.cuda.OutOfMemoryError:
                tested.append({"batch": mid, "ok": False, "error": "OOM"})
                torch.cuda.empty_cache()
                hi = mid - 1
        safe = max(1, int(best * 0.8))
        results[kind] = {"parameter_count": params, "max_fit_batch": best, "recommended_batch": safe, "tested": tested}
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print(kind, results[kind], flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
