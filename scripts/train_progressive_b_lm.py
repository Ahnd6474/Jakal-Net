from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from jakal_net import describe_device, resolve_device
from lm_experiment_utils import (
    MODEL_SIZE_PRESETS,
    CorpusLoadResult,
    append_jsonl,
    create_run_directory,
    ensure_directory,
    load_text_corpus,
    parse_preset_names,
    resolve_model_scale_preset,
    slugify_run_name,
    to_jsonable,
    write_csv_rows,
    write_json,
    write_jsonl,
)
from progressive_b_example import (
    ProgressiveBExampleLM,
    TrainingHistory,
    build_char_vocab,
    build_progressive_b_stage_specs,
    perplexity_from_loss,
    split_train_val,
    train_next_token_model,
)

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


DEFAULT_TEXT = """
Jakal-Net explores propagation and transition as separate operators.
Progressive B activation starts with stable sequence structure in S.
Then the B path grows from light compression to stronger hierarchical bottlenecks.
The final readout returns to S so token-level prediction remains anchored.
""".strip()
DEFAULT_TOKENIZER_CACHE_DIR = Path(tempfile.gettempdir()) / "jakal_net_tokenizers"


@dataclass(frozen=True, slots=True)
class SubwordVocab:
    processor: object
    model_path: Path
    model_type: str

    @property
    def size(self) -> int:
        return int(self.processor.get_piece_size())

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.processor.encode(text, out_type=int)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.processor.decode(list(map(int, token_ids))))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def estimate_steps_per_epoch(
    *,
    token_count: int,
    seq_len: int,
    batch_size: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    contexts_per_epoch = token_count - seq_len
    if contexts_per_epoch <= 0:
        raise ValueError("The corpus must be longer than seq_len.")
    return max(1, math.ceil(contexts_per_epoch / batch_size))


def resolve_tokenizer_prefix(
    *,
    text: str,
    tokenizer: str,
    model_type: str,
    vocab_size: int,
    prefix: str | None,
) -> Path:
    if prefix is not None:
        prefix_path = Path(prefix)
    else:
        corpus_digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        prefix_path = DEFAULT_TOKENIZER_CACHE_DIR / (
            f"progressive_b_{tokenizer}_{model_type}_{vocab_size}_{corpus_digest}"
        )
    if prefix_path.suffix:
        prefix_path = prefix_path.with_suffix("")
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    return prefix_path


def ensure_tokenizer_training_text(
    *,
    text: str,
    text_path: str | None,
    tokenizer_prefix: Path,
) -> Path:
    if text_path is not None:
        return Path(text_path)
    training_text_path = tokenizer_prefix.parent / f"{tokenizer_prefix.name}_corpus.txt"
    if not training_text_path.exists():
        training_text_path.write_text(text, encoding="utf-8")
    return training_text_path


def build_subword_vocab(
    text: str,
    *,
    text_path: str | None,
    vocab_size: int,
    model_type: str,
    tokenizer_prefix: str | None,
) -> SubwordVocab:
    if spm is None:
        raise ImportError(
            "sentencepiece is required for --tokenizer subword. "
            "Install dependencies from requirements-base.txt first."
        )
    if vocab_size <= 0:
        raise ValueError("subword-vocab-size must be positive.")

    prefix_path = resolve_tokenizer_prefix(
        text=text,
        tokenizer="subword",
        model_type=model_type,
        vocab_size=vocab_size,
        prefix=tokenizer_prefix,
    )
    model_path = prefix_path.with_suffix(".model")
    vocab_path = prefix_path.with_suffix(".vocab")

    if not model_path.exists() or not vocab_path.exists():
        training_text_path = ensure_tokenizer_training_text(
            text=text,
            text_path=text_path,
            tokenizer_prefix=prefix_path,
        )
        spm.SentencePieceTrainer.train(
            input=str(training_text_path),
            model_prefix=str(prefix_path),
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=1.0,
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            hard_vocab_limit=False,
            split_digits=True,
        )

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return SubwordVocab(
        processor=processor,
        model_path=model_path,
        model_type=model_type,
    )


def build_tokenizer(
    text: str,
    *,
    text_path: str | None,
    tokenizer: str,
    subword_vocab_size: int,
    subword_model_type: str,
    tokenizer_prefix: str | None,
) -> tuple[object, str, Path | None]:
    if tokenizer == "char":
        return build_char_vocab(text), "char", None
    if tokenizer == "subword":
        subword_vocab = build_subword_vocab(
            text,
            text_path=text_path,
            vocab_size=subword_vocab_size,
            model_type=subword_model_type,
            tokenizer_prefix=tokenizer_prefix,
        )
        return (
            subword_vocab,
            f"subword/{subword_vocab.model_type}",
            subword_vocab.model_path,
        )
    raise ValueError(f"Unsupported tokenizer: {tokenizer!r}.")


@torch.no_grad()
def generate_next_tokens_with_sampling(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    seq_len: int,
    device: torch.device | str,
    temperature: float | None,
    sample_topk: int | None,
    training_objective: str = "last_token",
) -> torch.Tensor:
    if temperature is not None and temperature <= 0.0:
        raise ValueError("temperature must be positive when sampling is enabled.")
    if sample_topk is not None and sample_topk <= 0:
        raise ValueError("sample-topk must be positive.")

    was_training = model.training
    model.eval()
    generated = prompt.to(device).clone()
    for _ in range(max_new_tokens):
        context = generated[-seq_len:].unsqueeze(0)
        if training_objective == "full_sequence_causal":
            logits = model(context, full_sequence_causal=True)[..., -1, :]
        else:
            logits = model(context)
        if temperature is None:
            next_token = torch.argmax(logits, dim=-1)
        else:
            scaled_logits = logits / temperature
            if sample_topk is not None:
                k = min(sample_topk, scaled_logits.shape[-1])
                topk_logits, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
                probs = torch.softmax(topk_logits, dim=-1)
                sampled_offset = torch.multinomial(probs, num_samples=1)
                next_token = topk_indices.gather(-1, sampled_offset).reshape(-1)
            else:
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
        generated = torch.cat((generated, next_token), dim=0)
    if was_training:
        model.train()
    return generated


def resolve_effective_model_config(
    args: argparse.Namespace,
    preset_name: str,
) -> dict[str, int | str]:
    if preset_name == "custom":
        return {
            "model_preset": "custom",
            "dim": args.dim,
            "warmup_layers": args.warmup_layers,
            "final_refine_layers": args.final_refine_layers,
            "lite_layers": args.lite_layers,
            "mid_layers": args.mid_layers,
            "full_layers": args.full_layers,
            "route_topk": args.route_topk,
        }
    preset = resolve_model_scale_preset(preset_name)
    return {
        "model_preset": preset.name,
        "dim": preset.dim,
        "warmup_layers": preset.warmup_layers,
        "final_refine_layers": preset.final_refine_layers,
        "lite_layers": preset.lite_layers,
        "mid_layers": preset.mid_layers,
        "full_layers": preset.full_layers,
        "route_topk": preset.route_topk,
    }


def build_metrics_rows(history: TrainingHistory, *, steps_per_epoch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, loss in enumerate(history.train_step_losses, start=1):
        rows.append(
            {
                "record_type": "train_step",
                "step": step,
                "epoch": step / steps_per_epoch,
                "minibatch_loss": loss,
            }
        )
    for step, train_loss, val_loss in zip(
        history.eval_steps,
        history.train_losses,
        history.val_losses,
        strict=True,
    ):
        rows.append(
            {
                "record_type": "eval",
                "step": step,
                "epoch": step / steps_per_epoch,
                "train_loss": train_loss,
                "train_ppl": perplexity_from_loss(train_loss),
                "val_loss": val_loss,
                "val_ppl": perplexity_from_loss(val_loss),
            }
        )
    return rows


def maybe_create_summary_writer(
    *,
    enabled: bool,
    tensorboard_dir: Path,
) -> SummaryWriter | None:
    if not enabled:
        return None
    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard logging requested, but torch.utils.tensorboard is unavailable. "
            "Install dependencies from requirements-base.txt first."
        )
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tensorboard_dir))


def log_history_to_tensorboard(
    writer: SummaryWriter | None,
    *,
    history: TrainingHistory,
    steps_per_epoch: int,
    summary: dict[str, Any],
    prompt_text: str,
    generated_text: str,
) -> None:
    if writer is None:
        return
    for step, loss in enumerate(history.train_step_losses, start=1):
        writer.add_scalar("train/minibatch_loss", loss, step)
        writer.add_scalar("train/epoch", step / steps_per_epoch, step)
    for step, train_loss, val_loss in zip(
        history.eval_steps,
        history.train_losses,
        history.val_losses,
        strict=True,
    ):
        writer.add_scalar("eval/train_loss", train_loss, step)
        writer.add_scalar("eval/val_loss", val_loss, step)
        writer.add_scalar("eval/train_ppl", perplexity_from_loss(train_loss), step)
        writer.add_scalar("eval/val_ppl", perplexity_from_loss(val_loss), step)
    writer.add_text("sample/prompt", prompt_text)
    writer.add_text("sample/generated", generated_text)
    writer.add_hparams(
        {
            "seq_len": int(summary["seq_len"]),
            "batch_size": int(summary["batch_size"]),
            "dim": int(summary["dim"]),
            "warmup_layers": int(summary["warmup_layers"]),
            "final_refine_layers": int(summary["final_refine_layers"]),
            "lite_layers": int(summary["lite_layers"]),
            "mid_layers": int(summary["mid_layers"]),
            "full_layers": int(summary["full_layers"]),
            "route_topk": int(summary["route_topk"]),
            "learning_rate": float(summary["learning_rate"]),
            "weight_decay": float(summary["weight_decay"]),
        },
        {
            "hparam/final_train_loss": float(summary["final_train_loss"]),
            "hparam/final_val_loss": float(summary["final_val_loss"]),
            "hparam/best_val_loss": float(summary["best_val_loss"]),
            "hparam/runtime_seconds": float(summary["runtime_seconds"]),
        },
        run_name="hparams",
    )
    writer.flush()


def print_eval_table(history: TrainingHistory, *, steps_per_epoch: int) -> None:
    for index, (step, train_loss, val_loss) in enumerate(
        zip(
            history.eval_steps,
            history.train_losses,
            history.val_losses,
            strict=True,
        ),
        start=1,
    ):
        print(
            f"eval {index:02d} | step={step:>5d} | epoch={step / steps_per_epoch:.3f} | "
            f"train_loss={train_loss:.4f} | train_ppl={perplexity_from_loss(train_loss):.2f} | "
            f"val_loss={val_loss:.4f} | val_ppl={perplexity_from_loss(val_loss):.2f}"
        )


def select_prompt_tokens(
    *,
    args: argparse.Namespace,
    vocab: object,
    train_tokens: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    if args.prompt_text:
        prompt = args.prompt_text
        prompt_ids = vocab.encode(prompt)
        if prompt_ids.numel() == 0:
            raise ValueError("--prompt-text must encode to at least one token.")
        return prompt_ids, prompt
    prompt_ids = train_tokens[: args.seq_len]
    return prompt_ids, vocab.decode(prompt_ids.tolist())


def save_run_artifacts(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    corpus: CorpusLoadResult,
    tokenizer_label: str,
    tokenizer_model_path: Path | None,
    effective_config: dict[str, int | str],
    parameter_count: int,
    steps_per_epoch: int,
    history: TrainingHistory,
    metrics_rows: Sequence[dict[str, Any]],
    prompt_text: str,
    generated_text: str,
    summary: dict[str, Any],
    checkpoint_payload: dict[str, Any] | None,
) -> None:
    write_json(run_dir / "corpus.json", corpus)
    write_json(
        run_dir / "config.json",
        {
            "cli_args": vars(args),
            "corpus": corpus,
            "tokenizer_label": tokenizer_label,
            "tokenizer_model_path": tokenizer_model_path,
            "model": effective_config,
            "parameter_count": parameter_count,
            "steps_per_epoch": steps_per_epoch,
        },
    )
    write_json(
        run_dir / "history.json",
        {
            "eval_steps": history.eval_steps,
            "train_step_losses": history.train_step_losses,
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
        },
    )
    write_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    write_csv_rows(run_dir / "metrics.csv", metrics_rows)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "sample_prompt.txt").write_text(prompt_text, encoding="utf-8")
    (run_dir / "sample_generated.txt").write_text(generated_text, encoding="utf-8")
    if checkpoint_payload is not None:
        torch.save(checkpoint_payload, run_dir / "checkpoint.pt")


def run_single_experiment(
    *,
    args: argparse.Namespace,
    session_dir: Path,
    experiment_name: str,
    corpus: CorpusLoadResult,
    vocab: object,
    tokenizer_label: str,
    tokenizer_model_path: Path | None,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device | str,
    teacher_forcing: bool,
    full_sequence_causal: bool,
) -> dict[str, Any]:
    effective_config = resolve_effective_model_config(args, experiment_name)
    steps_per_epoch = estimate_steps_per_epoch(
        token_count=int(train_tokens.numel()),
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("epochs must be positive.")
        train_steps = max(1, math.ceil(args.epochs * steps_per_epoch))
        requested_epochs = args.epochs
    else:
        train_steps = args.steps
        requested_epochs = train_steps / steps_per_epoch

    stage_specs = build_progressive_b_stage_specs(
        seq_nodes=args.seq_len,
        lite_layers=int(effective_config["lite_layers"]),
        mid_layers=int(effective_config["mid_layers"]),
        full_layers=int(effective_config["full_layers"]),
    )
    model = ProgressiveBExampleLM(
        vocab_size=vocab.size,
        dim=int(effective_config["dim"]),
        seq_nodes=args.seq_len,
        warmup_layers=int(effective_config["warmup_layers"]),
        stage_specs=stage_specs,
        final_refine_layers=int(effective_config["final_refine_layers"]),
        s_window=min(8, max(1, args.seq_len // 4)),
        route_topk=int(effective_config["route_topk"]),
        expanded_topk=int(effective_config["route_topk"]),
        compressed_topk=max(1, min(2, int(effective_config["route_topk"]))),
        expanded_sparse_type=args.expanded_propagation,
        compressed_sparse_type=args.compressed_propagation,
        expanded_window=args.expanded_window,
        compressed_window=args.compressed_window,
        implementation=args.implementation,
    )
    parameter_count = count_parameters(model)

    run_dir = session_dir / slugify_run_name(experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = (
        Path(args.tensorboard_dir) / slugify_run_name(experiment_name)
        if args.tensorboard_dir
        else run_dir / "tensorboard"
    )

    print(
        f"experiment={experiment_name} | params={parameter_count:,} | "
        f"dim={effective_config['dim']} | warmup={effective_config['warmup_layers']} | "
        f"stages={effective_config['lite_layers']}/{effective_config['mid_layers']}/{effective_config['full_layers']} | "
        f"refine={effective_config['final_refine_layers']} | route_topk={effective_config['route_topk']}"
    )
    print(
        f"schedule_steps={train_steps:,} | approx_epochs={requested_epochs:.3f} | "
        f"steps_per_epoch={steps_per_epoch:,} | batch_size={args.batch_size} | "
        f"objective={args.training_objective}"
    )

    start_time = time.perf_counter()

    def progress_callback(step: int, total_steps: int, minibatch_loss: float) -> None:
        if args.log_interval <= 0:
            return
        if step != 1 and step != total_steps and step % args.log_interval != 0:
            return
        elapsed = time.perf_counter() - start_time
        print(
            f"progress | experiment={experiment_name} | step={step:>5d}/{total_steps:<5d} | "
            f"epoch={step / steps_per_epoch:.3f} | {100.0 * step / total_steps:5.1f}% | "
            f"minibatch_loss={minibatch_loss:.4f} | elapsed={elapsed:.1f}s"
        )

    history = train_next_token_model(
        model,
        train_tokens,
        val_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        steps=train_steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        teacher_forcing=teacher_forcing,
        full_sequence_causal=full_sequence_causal,
        teacher_forcing_chunk_size=args.teacher_forcing_chunk_size,
        progress_callback=progress_callback,
    )
    runtime_seconds = time.perf_counter() - start_time
    print_eval_table(history, steps_per_epoch=steps_per_epoch)

    prompt_ids, prompt_text = select_prompt_tokens(args=args, vocab=vocab, train_tokens=train_tokens)
    generated = generate_next_tokens_with_sampling(
        model,
        prompt_ids,
        max_new_tokens=args.sample_tokens,
        seq_len=args.seq_len,
        device=device,
        temperature=args.temperature,
        sample_topk=args.sample_topk,
        training_objective=args.training_objective,
    )
    generated_text = vocab.decode(generated.tolist())

    best_index = min(range(len(history.val_losses)), key=history.val_losses.__getitem__)
    best_val_loss = history.val_losses[best_index]
    best_train_loss = history.train_losses[best_index]
    best_step = history.eval_steps[best_index]
    final_train_loss = history.train_losses[-1]
    final_val_loss = history.val_losses[-1]

    summary: dict[str, Any] = {
        "experiment": experiment_name,
        "model_preset": effective_config["model_preset"],
        "tokenizer": tokenizer_label,
        "tokenizer_vocab_size": vocab.size,
        "tokenizer_model_path": tokenizer_model_path,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "dim": effective_config["dim"],
        "warmup_layers": effective_config["warmup_layers"],
        "final_refine_layers": effective_config["final_refine_layers"],
        "lite_layers": effective_config["lite_layers"],
        "mid_layers": effective_config["mid_layers"],
        "full_layers": effective_config["full_layers"],
        "route_topk": effective_config["route_topk"],
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "steps": train_steps,
        "approx_epochs": requested_epochs,
        "steps_per_epoch": steps_per_epoch,
        "parameter_count": parameter_count,
        "runtime_seconds": runtime_seconds,
        "final_train_loss": final_train_loss,
        "final_train_ppl": perplexity_from_loss(final_train_loss),
        "final_val_loss": final_val_loss,
        "final_val_ppl": perplexity_from_loss(final_val_loss),
        "best_train_loss": best_train_loss,
        "best_train_ppl": perplexity_from_loss(best_train_loss),
        "best_val_loss": best_val_loss,
        "best_val_ppl": perplexity_from_loss(best_val_loss),
        "best_step": best_step,
        "corpus_source_label": corpus.source_label,
        "corpus_char_count": corpus.char_count,
        "corpus_sample_count": corpus.sample_count,
        "corpus_file_count": corpus.file_count,
        "corpus_truncated": corpus.truncated,
        "run_dir": run_dir,
        "tensorboard_dir": tensorboard_dir if args.tensorboard else None,
    }
    metrics_rows = build_metrics_rows(history, steps_per_epoch=steps_per_epoch)

    checkpoint_payload = None
    if args.save_checkpoint:
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab.size,
                "tokenizer": tokenizer_label,
                "tokenizer_model_path": None if tokenizer_model_path is None else str(tokenizer_model_path),
                "seq_nodes": args.seq_len,
                "implementation": args.implementation,
                "stage_specs": [asdict(spec) for spec in stage_specs],
                **{key: int(value) if isinstance(value, int) else value for key, value in effective_config.items()},
            },
            "history": {
                "eval_steps": history.eval_steps,
                "train_step_losses": history.train_step_losses,
                "train_losses": history.train_losses,
                "val_losses": history.val_losses,
            },
            "summary": summary,
            "corpus": to_jsonable(corpus),
        }

    save_run_artifacts(
        run_dir=run_dir,
        args=args,
        corpus=corpus,
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
        effective_config=effective_config,
        parameter_count=parameter_count,
        steps_per_epoch=steps_per_epoch,
        history=history,
        metrics_rows=metrics_rows,
        prompt_text=prompt_text,
        generated_text=generated_text,
        summary=summary,
        checkpoint_payload=checkpoint_payload,
    )

    writer = maybe_create_summary_writer(enabled=args.tensorboard, tensorboard_dir=tensorboard_dir)
    try:
        log_history_to_tensorboard(
            writer,
            history=history,
            steps_per_epoch=steps_per_epoch,
            summary=summary,
            prompt_text=prompt_text,
            generated_text=generated_text,
        )
    finally:
        if writer is not None:
            writer.close()

    print("\n--- prompt ---")
    print(prompt_text)
    print("\n--- sample ---")
    print(generated_text)
    print(f"\nartifacts={run_dir}")
    if args.tensorboard:
        print(f"tensorboard={tensorboard_dir}")

    return summary


def main() -> None:
    preset_choices = ("custom", *sorted(MODEL_SIZE_PRESETS))

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-file")
    parser.add_argument(
        "--text-source",
        action="append",
        default=[],
        help="Additional UTF-8 text file, directory, or glob. Can be repeated.",
    )
    parser.add_argument(
        "--jsonl-source",
        action="append",
        default=[],
        help="JSONL file, directory, or glob containing text records. Can be repeated.",
    )
    parser.add_argument(
        "--jsonl-text-key",
        action="append",
        default=["text"],
        help="JSONL field to read text from. Can be repeated.",
    )
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--corpus-max-samples", type=int)
    parser.add_argument("--corpus-max-chars", type=int)
    parser.add_argument("--corpus-separator", default="\n\n")
    parser.add_argument(
        "--training-objective",
        choices=("last_token", "teacher_forcing", "full_sequence_causal"),
        default="last_token",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("char", "subword"),
        default="char",
    )
    parser.add_argument("--subword-vocab-size", type=int, default=256)
    parser.add_argument(
        "--subword-model-type",
        choices=("bpe", "unigram"),
        default="bpe",
    )
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--model-preset",
        choices=preset_choices,
        default="custom",
    )
    parser.add_argument(
        "--sweep-presets",
        help="Comma-separated preset list such as tiny,small,base.",
    )
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--warmup-layers", type=int, default=2)
    parser.add_argument("--final-refine-layers", type=int, default=2)
    parser.add_argument("--lite-layers", type=int, default=2)
    parser.add_argument("--mid-layers", type=int, default=2)
    parser.add_argument("--full-layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--route-topk", type=int, default=4)
    parser.add_argument(
        "--implementation",
        choices=("reference", "streaming", "kernel", "native"),
        default="streaming",
    )
    parser.add_argument(
        "--expanded-propagation",
        choices=("topk", "window"),
        default="topk",
    )
    parser.add_argument(
        "--compressed-propagation",
        choices=("topk", "window"),
        default="topk",
    )
    parser.add_argument("--expanded-window", type=int)
    parser.add_argument("--compressed-window", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--prompt-text")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--sample-topk", type=int)
    parser.add_argument("--teacher-forcing-chunk-size", type=int)
    parser.add_argument("--run-name", default="progressive_b")
    parser.add_argument("--output-dir", default="artifacts/training_runs")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(args.device)}")

    teacher_forcing = args.training_objective == "teacher_forcing"
    full_sequence_causal = args.training_objective == "full_sequence_causal"
    if args.teacher_forcing_chunk_size is not None and args.teacher_forcing_chunk_size <= 0:
        raise ValueError("teacher-forcing-chunk-size must be positive.")
    if args.teacher_forcing_chunk_size is not None and not teacher_forcing:
        raise ValueError(
            "--teacher-forcing-chunk-size is only supported with "
            "--training-objective teacher_forcing."
        )

    corpus = load_text_corpus(
        default_text=DEFAULT_TEXT,
        text_file=args.text_file,
        text_sources=args.text_source,
        jsonl_sources=args.jsonl_source,
        jsonl_text_keys=tuple(dict.fromkeys(args.jsonl_text_key)),
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_text_key=args.hf_text_key,
        hf_streaming=args.hf_streaming,
        max_samples=args.corpus_max_samples,
        max_chars=args.corpus_max_chars,
        separator=args.corpus_separator,
    )
    print(
        f"corpus={corpus.source_label} | chars={corpus.char_count:,} | "
        f"samples={corpus.sample_count:,} | files={corpus.file_count:,} | "
        f"truncated={corpus.truncated}"
    )

    tokenizer_text_path = None if corpus.text_path is None else str(corpus.text_path)
    vocab, tokenizer_label, tokenizer_model_path = build_tokenizer(
        corpus.text,
        text_path=tokenizer_text_path,
        tokenizer=args.tokenizer,
        subword_vocab_size=args.subword_vocab_size,
        subword_model_type=args.subword_model_type,
        tokenizer_prefix=args.tokenizer_prefix,
    )
    tokens = vocab.encode(corpus.text)
    if tokens.numel() <= args.seq_len + 1:
        raise ValueError("The tokenized corpus must be longer than seq_len + 1.")
    train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.9)
    print(
        f"tokenizer={tokenizer_label} | tokenizer_vocab={vocab.size:,} | "
        f"train_tokens={train_tokens.numel():,} | val_tokens={val_tokens.numel():,}"
    )
    if tokenizer_model_path is not None:
        print(f"tokenizer_model={tokenizer_model_path}")

    sweep_presets = parse_preset_names(args.sweep_presets)
    if sweep_presets:
        experiment_names = sweep_presets
    else:
        experiment_names = (args.model_preset,)
    for name in experiment_names:
        if name != "custom":
            resolve_model_scale_preset(name)

    session_dir = create_run_directory(args.output_dir, args.run_name)
    ensure_directory(session_dir)
    print(f"session_dir={session_dir}")

    summaries: list[dict[str, Any]] = []
    for experiment_name in experiment_names:
        summary = run_single_experiment(
            args=args,
            session_dir=session_dir,
            experiment_name=experiment_name,
            corpus=corpus,
            vocab=vocab,
            tokenizer_label=tokenizer_label,
            tokenizer_model_path=tokenizer_model_path,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            device=device,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
        )
        summaries.append(summary)
        append_jsonl(session_dir / "experiments.jsonl", summary)

    write_json(session_dir / "session_summary.json", summaries)
    write_csv_rows(session_dir / "session_summary.csv", summaries)
    if len(summaries) > 1:
        best_summary = min(summaries, key=lambda row: row["best_val_loss"])
        write_json(
            session_dir / "best_run.json",
            {
                "best_experiment": best_summary["experiment"],
                "best_val_loss": best_summary["best_val_loss"],
                "best_val_ppl": best_summary["best_val_ppl"],
                "run_dir": best_summary["run_dir"],
            },
        )
        print(
            "best_experiment="
            f"{best_summary['experiment']} | best_val_loss={best_summary['best_val_loss']:.4f} | "
            f"run_dir={best_summary['run_dir']}"
        )


if __name__ == "__main__":
    main()
