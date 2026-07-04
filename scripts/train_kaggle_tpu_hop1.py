from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")
# Kaggle TPU VMs expose legacy topology variables that conflict with PJRT's
# local v3-8 discovery. PJRT must discover the eight local devices itself.
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TPU_TASK_ID", None)

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from jakal_net import CausalMemoryLM


class TokenBlockDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, path: Path, *, seq_len: int) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        self.path = path
        self.seq_len = int(seq_len)
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16)
        self.block_count = max(0, (int(self.tokens.size) - 1) // self.seq_len)
        if self.block_count == 0:
            raise ValueError(f"{path} does not contain a complete token block.")

    def __len__(self) -> int:
        return self.block_count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if not 0 <= index < self.block_count:
            raise IndexError(index)
        start = index * self.seq_len
        token_block = np.asarray(
            self.tokens[start : start + self.seq_len + 1],
            dtype=np.int64,
        )
        return (
            torch.from_numpy(token_block[:-1].copy()),
            torch.from_numpy(token_block[1:].copy()),
        )

    def close(self) -> None:
        memory_map = getattr(self.tokens, "_mmap", None)
        if memory_map is not None:
            memory_map.close()

    def __del__(self) -> None:
        self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hop-1 knowledge LM on a Kaggle TPU VM.")
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/working/wikitext103"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/jakal_hop1_tpu"))
    parser.add_argument("--dataset-name", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-103-raw-v1")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ff-mult", type=float, default=3.0)
    parser.add_argument("--knowledge-size", type=int, default=4096)
    parser.add_argument("--batch-size-per-core", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-runtime-seconds", type=int, default=34_200)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--min-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--decay-steps", type=int, default=80_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def _batched(items: list[str], batch_size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _write_token_split(
    *,
    split: Any,
    tokenizer: Any,
    output_path: Path,
    batch_size: int = 512,
) -> int:
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    token_count = 0
    with temporary_path.open("wb") as output_file:
        texts = [str(value) for value in split["text"] if str(value)]
        for text_batch in _batched(texts, batch_size):
            encoded_batch = tokenizer(
                text_batch,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
            flattened: list[int] = []
            for token_ids in encoded_batch:
                flattened.extend(token_ids)
                flattened.append(int(tokenizer.eos_token_id))
            token_array = np.asarray(flattened, dtype=np.uint16)
            token_array.tofile(output_file)
            token_count += int(token_array.size)

    temporary_path.replace(output_path)
    return token_count


def prepare_wikitext(args: argparse.Namespace) -> dict[str, Any]:
    args.data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.data_dir / "metadata.json"
    train_path = args.data_dir / "train.bin"
    validation_path = args.data_dir / "validation.bin"
    if metadata_path.exists() and train_path.exists() and validation_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id.")
    if len(tokenizer) > np.iinfo(np.uint16).max:
        raise ValueError("The tokenizer vocabulary does not fit in uint16.")

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_tokens = _write_token_split(
        split=dataset["train"],
        tokenizer=tokenizer,
        output_path=train_path,
    )
    validation_tokens = _write_token_split(
        split=dataset["validation"],
        tokenizer=tokenizer,
        output_path=validation_path,
    )
    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "tokenizer": args.tokenizer,
        "vocab_size": len(tokenizer),
        "train_tokens": train_tokens,
        "validation_tokens": validation_tokens,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if step < args.warmup_steps:
        progress = float(step + 1) / float(max(1, args.warmup_steps))
        return args.min_learning_rate + progress * (args.learning_rate - args.min_learning_rate)
    decay_progress = min(
        1.0,
        float(step - args.warmup_steps) / float(max(1, args.decay_steps)),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return args.min_learning_rate + cosine * (args.learning_rate - args.min_learning_rate)


def _make_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    *,
    batch_size: int,
    world_size: int,
    rank: int,
    shuffle: bool,
    seed: int,
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DistributedSampler[tuple[Tensor, Tensor]]]:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
        num_workers=0,
    )
    return loader, sampler


def _evaluate(
    *,
    model: CausalMemoryLM,
    loader: DataLoader[tuple[Tensor, Tensor]],
    device: torch.device,
    step: int,
    max_batches: int,
) -> tuple[float, float]:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    model.eval()
    total_loss = 0.0
    total_batches = 0
    device_loader = pl.MpDeviceLoader(loader, device)
    with torch.no_grad():
        for input_ids, target_ids in device_loader:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
            )
            total_loss += float(loss.detach().cpu())
            total_batches += 1
            if total_batches >= max_batches:
                break

    reduced = xm.mesh_reduce(
        f"eval_{step}",
        (total_loss, total_batches),
        lambda values: (
            sum(value[0] for value in values),
            sum(value[1] for value in values),
        ),
    )
    mean_loss = float(reduced[0]) / float(max(1, reduced[1]))
    model.train()
    return mean_loss, math.exp(min(20.0, mean_loss))


def _save_checkpoint(
    *,
    model: CausalMemoryLM,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    step: int,
    validation_loss: float | None,
) -> None:
    import torch_xla.core.xla_model as xm

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": int(step),
        "validation_loss": validation_loss,
        "config": vars(args),
        "data": metadata,
    }
    xm.save(checkpoint, args.output_dir / "last.pt", master_only=True)


def _train_process(index: int, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    del index
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    device = xm.xla_device()
    rank = xr.global_ordinal()
    world_size = xr.world_size()
    is_master = rank == 0
    torch.manual_seed(args.seed + rank)

    train_dataset = TokenBlockDataset(args.data_dir / "train.bin", seq_len=args.seq_len)
    validation_dataset = TokenBlockDataset(args.data_dir / "validation.bin", seq_len=args.seq_len)
    train_loader, train_sampler = _make_loader(
        train_dataset,
        batch_size=args.batch_size_per_core,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )
    validation_loader, _ = _make_loader(
        validation_dataset,
        batch_size=args.batch_size_per_core,
        world_size=world_size,
        rank=rank,
        shuffle=False,
        seed=args.seed,
    )

    model = CausalMemoryLM(
        vocab_size=int(metadata["vocab_size"]),
        dim=args.dim,
        max_seq_len=args.seq_len,
        transformer_layers=args.layers,
        transformer_heads=args.heads,
        feed_forward_hidden_mult=args.ff_mult,
        transformer_dropout=0.0,
        knowledge_memory_size=args.knowledge_size,
        knowledge_hops=1,
        knowledge_activation="relu",
    ).to(device)
    xm.broadcast_master_param(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=False,
        foreach=False,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.jsonl"
    if is_master:
        config = {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "data": metadata,
            "world_size": world_size,
            "parameters": parameter_count,
        }
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(
            "startup"
            f" | device={device}"
            f" | world_size={world_size}"
            f" | params={parameter_count:,}"
            f" | global_batch={args.batch_size_per_core * world_size}"
            f" | scalar_gate_params={model.knowledge_block.residual_gate_logit.numel()}",
            flush=True,
        )

    model.train()
    start_time = time.monotonic()
    step = 0
    epoch = 0
    last_validation_loss: float | None = None
    stop_training = False
    while step < args.max_steps and not stop_training:
        train_sampler.set_epoch(epoch)
        device_loader = pl.MpDeviceLoader(train_loader, device)
        for input_ids, target_ids in device_loader:
            learning_rate = learning_rate_for_step(args, step)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(optimizer, barrier=False)
            xm.mark_step()
            step += 1

            if step == 1 or step % args.log_interval == 0:
                loss_value = float(loss.detach().cpu())
                grad_norm_value = float(grad_norm.detach().cpu())
                elapsed = time.monotonic() - start_time
                record = {
                    "step": step,
                    "train_loss": loss_value,
                    "grad_norm": grad_norm_value,
                    "learning_rate": learning_rate,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": step / max(elapsed, 1.0e-9),
                }
                if is_master:
                    with history_path.open("a", encoding="utf-8") as history_file:
                        history_file.write(json.dumps(record) + "\n")
                    print(
                        "train"
                        f" | step={step}"
                        f" | loss={loss_value:.4f}"
                        f" | grad_norm={grad_norm_value:.4f}"
                        f" | lr={learning_rate:.8f}"
                        f" | steps_per_second={record['steps_per_second']:.4f}",
                        flush=True,
                    )

            if step % args.eval_interval == 0:
                validation_loss, validation_ppl = _evaluate(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    step=step,
                    max_batches=args.eval_batches,
                )
                last_validation_loss = validation_loss
                if is_master:
                    with history_path.open("a", encoding="utf-8") as history_file:
                        history_file.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "validation_loss": validation_loss,
                                    "validation_ppl": validation_ppl,
                                }
                            )
                            + "\n"
                        )
                    print(
                        f"eval | step={step} | val_loss={validation_loss:.4f} | val_ppl={validation_ppl:.2f}",
                        flush=True,
                    )

            if step % args.checkpoint_interval == 0:
                _save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    metadata=metadata,
                    step=step,
                    validation_loss=last_validation_loss,
                )

            elapsed = time.monotonic() - start_time
            if step >= args.max_steps or elapsed >= args.max_runtime_seconds:
                stop_training = True
                break
        epoch += 1

    _save_checkpoint(
        model=model,
        optimizer=optimizer,
        args=args,
        metadata=metadata,
        step=step,
        validation_loss=last_validation_loss,
    )
    train_dataset.close()
    validation_dataset.close()
    if is_master:
        summary = {
            "step": step,
            "validation_loss": last_validation_loss,
            "elapsed_seconds": time.monotonic() - start_time,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"complete | {json.dumps(summary)}", flush=True)


def main() -> None:
    args = parse_args()
    metadata = prepare_wikitext(args)
    import torch_xla.distributed.xla_multiprocessing as xmp

    xmp.spawn(_train_process, args=(args, metadata), nprocs=None)


if __name__ == "__main__":
    main()
