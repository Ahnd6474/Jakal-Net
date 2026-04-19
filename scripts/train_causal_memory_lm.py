from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from lm_experiment_utils import (
    append_jsonl,
    create_run_directory,
    ensure_directory,
    write_csv_rows,
    write_json,
)
from train_progressive_b_lm import (
    DIALOGUE_SPECIAL_TOKENS,
    build_tokenizer,
    count_parameters,
    encode_text_in_chunks,
    load_token_stream_corpus,
    resolve_autocast_dtype,
)

from jakal_net import describe_device, resolve_device
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


@dataclass(frozen=True, slots=True)
class StreamingBatch:
    context: torch.Tensor
    target: torch.Tensor
    reset_mask: torch.Tensor


class StreamingTokenBatcher:
    def __init__(
        self,
        tokens: torch.Tensor,
        *,
        seq_len: int,
        batch_size: int,
        device: torch.device,
        random_starts: bool = True,
    ) -> None:
        if tokens.ndim != 1:
            raise ValueError("tokens must be a flat tensor.")
        if tokens.numel() <= seq_len:
            raise ValueError("tokens must be longer than seq_len.")
        self.tokens = tokens.detach().cpu().to(dtype=torch.long).contiguous()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.random_starts = random_starts
        self.max_start = int(self.tokens.numel() - seq_len - 1)
        if self.max_start < 0:
            raise ValueError("The corpus must be longer than seq_len + 1.")
        self.positions = torch.zeros(batch_size, dtype=torch.long)
        self.needs_reset = torch.ones(batch_size, dtype=torch.bool)

    def _sample_start(self) -> int:
        if self.max_start == 0:
            return 0
        return random.randint(0, self.max_start)

    def next_batch(self) -> StreamingBatch:
        contexts: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        reset_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        for item_index in range(self.batch_size):
            if bool(self.needs_reset[item_index]):
                start = self._sample_start() if self.random_starts else 0
                self.positions[item_index] = start
                reset_mask[item_index] = True
            start = int(self.positions[item_index].item())
            stop = start + self.seq_len
            contexts.append(self.tokens[start:stop])
            targets.append(self.tokens[start + 1 : stop + 1])
            next_start = stop
            if next_start > self.max_start:
                self.needs_reset[item_index] = True
            else:
                self.positions[item_index] = next_start
                self.needs_reset[item_index] = False
        return StreamingBatch(
            context=torch.stack(contexts, dim=0).to(self.device),
            target=torch.stack(targets, dim=0).to(self.device),
            reset_mask=reset_mask.to(self.device),
        )


def sample_random_batch(
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> StreamingBatch:
    max_start = int(tokens.numel() - seq_len - 1)
    if max_start < 0:
        raise ValueError("The corpus must be longer than seq_len + 1.")
    starts = torch.randint(0, max_start + 1, (batch_size,))
    context = torch.stack([tokens[start : start + seq_len] for start in starts.tolist()], dim=0)
    target = torch.stack(
        [tokens[start + 1 : start + seq_len + 1] for start in starts.tolist()],
        dim=0,
    )
    return StreamingBatch(
        context=context.to(device),
        target=target.to(device),
        reset_mask=torch.ones(batch_size, dtype=torch.bool, device=device),
    )


def save_pretokenized_bundle(
    path: Path,
    *,
    token_ids: torch.Tensor,
    vocab_size: int,
    tokenizer_label: str,
    tokenizer_model_path: str | None,
    corpus_info: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "token_ids": token_ids.detach().cpu().to(dtype=torch.long),
            "tokenizer_label": tokenizer_label,
            "tokenizer_model_path": tokenizer_model_path,
            "corpus_info": corpus_info,
            "vocab_size": vocab_size,
        },
        path,
    )


def load_pretokenized_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "token_ids" not in bundle:
        raise ValueError(f"Invalid pretokenized bundle: {path}")
    return bundle


def build_run_name(args: argparse.Namespace) -> str:
    memory_slug = "-".join(str(slot) for slot in args.memory_slots)
    return (
        f"causal-memory-s{args.s_layers}-b{memory_slug}-p{args.prediction_layers}"
        f"-dim{args.dim}-seq{args.seq_len}"
    )


def split_train_val(tokens: torch.Tensor, *, train_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 1:
        raise ValueError("tokens must be a flat tensor.")
    if tokens.numel() < 4:
        raise ValueError("tokens must contain at least four items to split train/val.")
    split_index = int(tokens.numel() * train_fraction)
    split_index = max(2, min(split_index, tokens.numel() - 2))
    return tokens[:split_index].contiguous(), tokens[split_index:].contiguous()


def estimate_steps_per_epoch(*, token_count: int, seq_len: int, batch_size: int) -> int:
    usable = max(1, token_count - seq_len)
    return max(1, math.ceil(usable / max(1, batch_size * seq_len)))


def perplexity_from_loss(loss_value: float) -> float:
    clamped = min(loss_value, 20.0)
    return float(math.exp(clamped))


def compute_learning_rate(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if total_steps <= 1:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    progress = 0.0
    denom = max(1, total_steps - warmup_steps)
    progress = max(0.0, min(1.0, (step - warmup_steps) / denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def detach_memory_state(memory_state: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if memory_state is None:
        return None
    detached = []
    for layer in memory_state:
        detached.append(layer.with_tensors(state=layer.state.detach(), val=layer.val.detach()))
    return tuple(detached)


def run_model(
    model: CausalHierarchicalMemoryLM,
    batch: StreamingBatch,
    *,
    memory_state: Sequence[Any] | None,
    precision: str,
    grad_enabled: bool,
) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    autocast_dtype = resolve_autocast_dtype(precision)
    autocast_supported = (
        autocast_dtype is not None
        and (
            batch.context.device.type == "cuda"
            or (batch.context.device.type == "cpu" and autocast_dtype == torch.bfloat16)
        )
    )
    autocast_context = (
        torch.autocast(device_type=batch.context.device.type, dtype=autocast_dtype)
        if autocast_supported
        else nullcontext()
    )
    grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with grad_context:
        with autocast_context:
            output = model(
                batch.context,
                memory_state=memory_state,
                reset_mask=batch.reset_mask,
                return_memory_state=True,
            )
            assert isinstance(output, MemoryScanOutput)
            logits = output.logits
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                batch.target.reshape(-1),
            )
    return loss, output.memory_state


@torch.no_grad()
def estimate_eval_loss(
    model: CausalHierarchicalMemoryLM,
    tokens: torch.Tensor,
    *,
    seq_len: int,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
    precision: str,
) -> float:
    model_was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(eval_steps):
        batch = sample_random_batch(tokens, seq_len=seq_len, batch_size=batch_size, device=device)
        loss, _ = run_model(model, batch, memory_state=None, precision=precision, grad_enabled=False)
        losses.append(float(loss.item()))
    if model_was_training:
        model.train()
    return float(sum(losses) / max(1, len(losses)))


def save_checkpoint(
    path: Path,
    *,
    model: CausalHierarchicalMemoryLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    train_loss: float,
    val_loss: float | None,
    tokenizer_label: str,
    tokenizer_model_path: str | None,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "tokenizer_label": tokenizer_label,
        "tokenizer_model_path": tokenizer_model_path,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the causal hierarchical memory LM.")
    parser.add_argument("--text-file")
    parser.add_argument("--text-source", action="append", default=[])
    parser.add_argument("--jsonl-source", action="append", default=[])
    parser.add_argument("--jsonl-text-key", action="append", default=["text", "content", "body"])
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--separator", default="\n\n")

    parser.add_argument("--tokenizer", choices=("char", "subword", "byte_bpe"), default="byte_bpe")
    parser.add_argument("--subword-vocab-size", type=int, default=16384)
    parser.add_argument("--subword-model-type", default="bpe")
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--subword-character-coverage", type=float, default=1.0)
    parser.add_argument("--subword-input-sentence-size", type=int, default=0)
    parser.add_argument("--subword-num-threads", type=int, default=0)
    parser.add_argument("--pretokenize-workers", type=int, default=8)
    parser.add_argument("--pretokenized-path")
    parser.add_argument("--save-pretokenized", action="store_true")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--s-layers", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, nargs="+", default=[512, 128, 32])
    parser.add_argument("--prediction-layers", type=int, default=2)
    parser.add_argument("--s-window", type=int, default=2048)
    parser.add_argument("--prediction-window", type=int, default=64)
    parser.add_argument("--memory-topk", type=int, default=16)
    parser.add_argument(
        "--pairwise-kind",
        choices=("low_rank_bilinear", "diagonal_bilinear", "bilinear", "additive_low_rank"),
        default="low_rank_bilinear",
    )
    parser.add_argument(
        "--route-kind",
        choices=("low_rank_bilinear", "diagonal_bilinear", "bilinear", "additive_low_rank"),
        default="low_rank_bilinear",
    )
    parser.add_argument("--pairwise-rank", type=int, default=64)
    parser.add_argument("--route-rank", type=int, default=64)
    parser.add_argument("--implementation", choices=("reference", "streaming", "kernel", "native"), default="streaming")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--carry-memory", action="store_true")

    parser.add_argument("--output-root", default="artifacts/training_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pretokenize_workers < 0:
        raise ValueError("pretokenize-workers must be non-negative.")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.seq_len <= 0:
        raise ValueError("seq-len must be positive.")
    if args.epochs <= 0.0:
        raise ValueError("epochs must be positive.")
    if args.eval_interval <= 0:
        raise ValueError("eval-interval must be positive.")
    if args.eval_steps <= 0:
        raise ValueError("eval-steps must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    print(f"using device: {describe_device(device)}", flush=True)

    tokenizer_model_path: str | None = None
    tokenizer_label: str
    tokens: torch.Tensor
    corpus_metadata: dict[str, Any]
    vocab_size: int

    if args.pretokenized_path and Path(args.pretokenized_path).exists():
        bundle = load_pretokenized_bundle(Path(args.pretokenized_path))
        tokens = bundle["token_ids"].detach().cpu().to(dtype=torch.long)
        tokenizer_label = str(bundle.get("tokenizer_label") or "unknown")
        tokenizer_model_path = bundle.get("tokenizer_model_path")
        corpus_metadata = dict(bundle.get("corpus_info") or {})
        vocab_size = int(bundle.get("vocab_size") or (int(tokens.max().item()) + 1))
        print(
            f"loaded_pretokenized | path={args.pretokenized_path} | tokens={tokens.numel():,} | tokenizer={tokenizer_label}",
            flush=True,
        )
    else:
        corpus = load_token_stream_corpus(
            default_text="Jakal-Net causal memory training sample.",
            text_file=args.text_file,
            text_sources=tuple(args.text_source),
            jsonl_sources=tuple(args.jsonl_source),
            jsonl_text_keys=tuple(args.jsonl_text_key),
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_text_key=args.hf_text_key,
            hf_streaming=args.hf_streaming,
            max_samples=args.max_samples,
            max_chars=args.max_chars,
            separator=args.separator,
        )
        print(
            f"corpus=token_stream | chars={corpus.char_count:,} | samples={corpus.sample_count:,} | files={corpus.file_count} | truncated={corpus.truncated}",
            flush=True,
        )
        vocab, tokenizer_label, tokenizer_path = build_tokenizer(
            corpus.text,
            text_path=corpus.text_path.as_posix() if corpus.text_path is not None else None,
            tokenizer=args.tokenizer,
            subword_vocab_size=args.subword_vocab_size,
            subword_model_type=args.subword_model_type,
            tokenizer_prefix=args.tokenizer_prefix,
            subword_character_coverage=args.subword_character_coverage,
            subword_input_sentence_size=args.subword_input_sentence_size,
            subword_num_threads=args.subword_num_threads,
            user_defined_symbols=DIALOGUE_SPECIAL_TOKENS,
        )
        tokenizer_model_path = None if tokenizer_path is None else str(tokenizer_path)
        workers = min(args.pretokenize_workers, max(1, torch.get_num_threads()))
        tokens = encode_text_in_chunks(vocab, corpus.text, workers=workers)
        vocab_size = int(getattr(vocab, "size", int(tokens.max().item()) + 1))
        corpus_metadata = {
            "source_label": corpus.source_label,
            "sample_count": corpus.sample_count,
            "file_count": corpus.file_count,
            "char_count": corpus.char_count,
            "truncated": corpus.truncated,
            "metadata": corpus.metadata,
        }
        print(
            f"tokenizer={tokenizer_label} | tokenizer_model={tokenizer_model_path} | token_count={tokens.numel():,}",
            flush=True,
        )
        if args.save_pretokenized and args.pretokenized_path:
            save_pretokenized_bundle(
                Path(args.pretokenized_path),
                token_ids=tokens,
                tokenizer_label=tokenizer_label,
                tokenizer_model_path=tokenizer_model_path,
                corpus_info=corpus_metadata,
                vocab_size=vocab_size,
            )
            print(f"saved_pretokenized | path={args.pretokenized_path}", flush=True)

    train_tokens, val_tokens = split_train_val(tokens, train_fraction=args.train_fraction)
    steps_per_epoch = estimate_steps_per_epoch(
        token_count=train_tokens.numel(),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))

    model = CausalHierarchicalMemoryLM(
        vocab_size=vocab_size,
        dim=args.dim,
        max_seq_len=args.seq_len,
        s_layers=args.s_layers,
        memory_slots=tuple(args.memory_slots),
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        prediction_window=args.prediction_window,
        memory_topk=args.memory_topk,
        pairwise_kind=args.pairwise_kind,
        route_kind=args.route_kind,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        implementation=args.implementation,
    )
    model = model.to(device)
    parameter_count = count_parameters(model)
    print(
        f"model=causal_memory | params={parameter_count:,} | dim={args.dim} | seq_len={args.seq_len} | memory_slots={args.memory_slots}",
        flush=True,
    )

    run_name = args.run_name or build_run_name(args)
    run_dir = create_run_directory(args.output_root, run_name)
    checkpoints_dir = ensure_directory(run_dir / "checkpoints")
    write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "tokenizer_label": tokenizer_label,
            "tokenizer_model_path": tokenizer_model_path,
            "parameter_count": parameter_count,
            "corpus_metadata": corpus_metadata,
        },
    )

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("tensorboard is not installed.")
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = None
    if args.precision == "fp16" and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    train_batcher = StreamingTokenBatcher(
        train_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        random_starts=True,
    )

    history_rows: list[dict[str, Any]] = []
    train_memory_state: tuple[Any, ...] | None = None
    best_val_loss = float("inf")
    optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    for step in range(1, total_steps + 1):
        lr = compute_learning_rate(
            step=step,
            total_steps=total_steps,
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_ratio=args.lr_min_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        batch = train_batcher.next_batch()
        if not args.carry_memory:
            batch = StreamingBatch(
                context=batch.context,
                target=batch.target,
                reset_mask=torch.ones_like(batch.reset_mask, dtype=torch.bool),
            )
            train_memory_state = None

        if scaler is not None:
            autocast_dtype = resolve_autocast_dtype(args.precision)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                output = model(
                    batch.context,
                    memory_state=train_memory_state,
                    reset_mask=batch.reset_mask,
                    return_memory_state=True,
                )
                assert isinstance(output, MemoryScanOutput)
                loss = F.cross_entropy(
                    output.logits.reshape(-1, output.logits.shape[-1]).float(),
                    batch.target.reshape(-1),
                )
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            current_memory_state = output.memory_state
        else:
            loss, current_memory_state = run_model(
                model,
                batch,
                memory_state=train_memory_state,
                precision=args.precision,
                grad_enabled=True,
            )
            (loss / args.grad_accum_steps).backward()

        train_memory_state = detach_memory_state(current_memory_state)

        should_step_optimizer = step % args.grad_accum_steps == 0 or step == total_steps
        if should_step_optimizer:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = float("nan")

        train_loss = float(loss.item())
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, step)
            writer.add_scalar("train/lr", lr, step)
            if not math.isnan(grad_norm):
                writer.add_scalar("train/grad_norm", grad_norm, step)

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"progress | step={step:5d}/{total_steps} | train_loss={train_loss:.4f} | lr={lr:.6g} | elapsed={elapsed:.1f}s",
                flush=True,
            )

        val_loss = None
        if step == 1 or step % args.eval_interval == 0 or step == total_steps:
            val_loss = estimate_eval_loss(
                model,
                val_tokens,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                eval_steps=args.eval_steps,
                device=device,
                precision=args.precision,
            )
            val_ppl = perplexity_from_loss(val_loss)
            print(
                f"eval | step={step} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("eval/val_loss", val_loss, step)
                writer.add_scalar("eval/val_ppl", val_ppl, step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    tokenizer_label=tokenizer_label,
                    tokenizer_model_path=tokenizer_model_path,
                )
                write_json(
                    checkpoints_dir / "best.json",
                    {
                        "step": step,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                )
        if step % args.checkpoint_interval == 0 or step == total_steps:
            save_checkpoint(
                checkpoints_dir / f"step_{step:06d}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
                train_loss=train_loss,
                val_loss=val_loss,
                tokenizer_label=tokenizer_label,
                tokenizer_model_path=tokenizer_model_path,
            )
            save_checkpoint(
                checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
                train_loss=train_loss,
                val_loss=val_loss,
                tokenizer_label=tokenizer_label,
                tokenizer_model_path=tokenizer_model_path,
            )
            write_json(
                checkpoints_dir / "last.json",
                {
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
            )

        row = {
            "step": step,
            "train_loss": train_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "val_loss": val_loss,
            "val_ppl": None if val_loss is None else perplexity_from_loss(val_loss),
        }
        history_rows.append(row)
        append_jsonl(run_dir / "history.jsonl", row)

    write_csv_rows(run_dir / "history.csv", history_rows)
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
