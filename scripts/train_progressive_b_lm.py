from __future__ import annotations

import argparse
import hashlib
import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from jakal_net import (
    describe_device,
    resolve_device,
)
from progressive_b_example import (
    ProgressiveBExampleLM,
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


DEFAULT_TEXT = """
Jakal-Net explores propagation and transition as separate operators.
Progressive B activation starts with stable sequence structure in S.
Then the B path grows from light compression to stronger hierarchical bottlenecks.
The final readout returns to S so token-level prediction remains anchored.
""".strip()
DEFAULT_TOKENIZER_CACHE_DIR = Path(tempfile.gettempdir()) / "jakal_net_tokenizers"


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input text file must not be empty.")
    return text


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


def build_eval_schedule(*, steps: int, eval_interval: int) -> tuple[int, ...]:
    scheduled_steps: set[int] = {1, steps}
    for step in range(eval_interval, steps + 1, eval_interval):
        scheduled_steps.add(step)
    return tuple(sorted(scheduled_steps))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-file")
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
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--sample-topk", type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(args.device)}")

    text = load_text(args.text_file)
    vocab, tokenizer_label, tokenizer_model_path = build_tokenizer(
        text,
        text_path=args.text_file,
        tokenizer=args.tokenizer,
        subword_vocab_size=args.subword_vocab_size,
        subword_model_type=args.subword_model_type,
        tokenizer_prefix=args.tokenizer_prefix,
    )
    tokens = vocab.encode(text)
    if tokens.numel() <= args.seq_len + 1:
        raise ValueError("The tokenized corpus must be longer than seq_len + 1.")
    train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.9)
    steps_per_epoch = estimate_steps_per_epoch(
        token_count=train_tokens.numel(),
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
        lite_layers=args.lite_layers,
        mid_layers=args.mid_layers,
        full_layers=args.full_layers,
    )

    model = ProgressiveBExampleLM(
        vocab_size=vocab.size,
        dim=args.dim,
        seq_nodes=args.seq_len,
        warmup_layers=args.warmup_layers,
        stage_specs=stage_specs,
        final_refine_layers=args.final_refine_layers,
        s_window=min(8, max(1, args.seq_len // 4)),
        route_topk=args.route_topk,
        expanded_topk=args.route_topk,
        compressed_topk=max(1, min(2, args.route_topk)),
        expanded_sparse_type=args.expanded_propagation,
        compressed_sparse_type=args.compressed_propagation,
        expanded_window=args.expanded_window,
        compressed_window=args.compressed_window,
        implementation=args.implementation,
    )
    parameter_count = count_parameters(model)
    teacher_forcing = args.training_objective == "teacher_forcing"
    full_sequence_causal = args.training_objective == "full_sequence_causal"

    print(
        f"tokenizer={tokenizer_label} | tokenizer_vocab={vocab.size:,} | "
        f"corpus_chars={len(text):,} | "
        f"train_tokens={train_tokens.numel():,} | val_tokens={val_tokens.numel():,}"
    )
    if tokenizer_model_path is not None:
        print(f"tokenizer_model={tokenizer_model_path}")
    print(
        f"model_params={parameter_count:,} | dim={args.dim} | seq_len={args.seq_len} | "
        f"warmup={args.warmup_layers} | stages={args.lite_layers}/{args.mid_layers}/{args.full_layers} | "
        f"refine={args.final_refine_layers}"
    )
    print(
        f"schedule_steps={train_steps:,} | approx_epochs={requested_epochs:.3f} | "
        f"steps_per_epoch={steps_per_epoch:,} | batch_size={args.batch_size} | "
        f"objective={args.training_objective}"
    )
    if full_sequence_causal:
        print("causal_path=sequence_only_joint_blocks")

    start_time = time.perf_counter()

    def progress_callback(step: int, total_steps: int, minibatch_loss: float) -> None:
        if args.log_interval <= 0:
            return
        if step != 1 and step != total_steps and step % args.log_interval != 0:
            return
        elapsed = time.perf_counter() - start_time
        print(
            f"progress | step={step:>5d}/{total_steps:<5d} | "
            f"epoch={step / steps_per_epoch:.3f} | "
            f"{100.0 * step / total_steps:5.1f}% | "
            f"minibatch_loss={minibatch_loss:.4f} | "
            f"elapsed={elapsed:.1f}s"
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
        progress_callback=progress_callback,
    )

    eval_schedule = build_eval_schedule(steps=train_steps, eval_interval=args.eval_interval)
    for index, (step, (train_loss, val_loss)) in enumerate(
        zip(
            eval_schedule,
            zip(history.train_losses, history.val_losses, strict=True),
            strict=True,
        ),
        start=1,
    ):
        print(
            f"eval {index:02d} | step={step:>5d} | epoch={step / steps_per_epoch:.3f} | "
            f"train_loss={train_loss:.4f} | train_ppl={perplexity_from_loss(train_loss):.2f} | "
            f"val_loss={val_loss:.4f} | val_ppl={perplexity_from_loss(val_loss):.2f}"
        )

    prompt = train_tokens[: args.seq_len].to(device)
    generated = generate_next_tokens_with_sampling(
        model,
        prompt,
        max_new_tokens=args.sample_tokens,
        seq_len=args.seq_len,
        device=device,
        temperature=args.temperature,
        sample_topk=args.sample_topk,
        training_objective=args.training_objective,
    )
    prompt_text = vocab.decode(prompt.tolist())
    generated_text = vocab.decode(generated.tolist())

    print("\n--- prompt ---")
    print(prompt_text)
    print("\n--- sample ---")
    print(generated_text)


if __name__ == "__main__":
    main()
