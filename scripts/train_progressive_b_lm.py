from __future__ import annotations

import argparse
from pathlib import Path

import torch

from jakal_net import (
    describe_device,
    resolve_device,
)
from progressive_b_example import (
    ProgressiveBExampleLM,
    build_char_vocab,
    build_progressive_b_stage_specs,
    generate_next_tokens,
    perplexity_from_loss,
    split_train_val,
    train_next_token_model,
)


DEFAULT_TEXT = """
Jakal-Net explores propagation and transition as separate operators.
Progressive B activation starts with stable sequence structure in S.
Then the B path grows from light compression to stronger hierarchical bottlenecks.
The final readout returns to S so token-level prediction remains anchored.
""".strip()


def load_text(path: str | None) -> str:
    if path is None:
        return DEFAULT_TEXT
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input text file must not be empty.")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-file")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--warmup-layers", type=int, default=2)
    parser.add_argument("--final-refine-layers", type=int, default=2)
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
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(args.device)}")

    text = load_text(args.text_file)
    if len(text) <= args.seq_len + 1:
        raise ValueError("The corpus must be longer than seq_len + 1.")

    vocab = build_char_vocab(text)
    tokens = vocab.encode(text)
    train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.9)

    model = ProgressiveBExampleLM(
        vocab_size=vocab.size,
        dim=args.dim,
        seq_nodes=args.seq_len,
        warmup_layers=args.warmup_layers,
        stage_specs=build_progressive_b_stage_specs(seq_nodes=args.seq_len),
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

    history = train_next_token_model(
        model,
        train_tokens,
        val_tokens,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for index, (train_loss, val_loss) in enumerate(
        zip(history.train_losses, history.val_losses, strict=True),
        start=1,
    ):
        print(
            f"eval {index:02d} | "
            f"train_loss={train_loss:.4f} | train_ppl={perplexity_from_loss(train_loss):.2f} | "
            f"val_loss={val_loss:.4f} | val_ppl={perplexity_from_loss(val_loss):.2f}"
        )

    prompt = train_tokens[: args.seq_len].to(device)
    generated = generate_next_tokens(
        model,
        prompt,
        max_new_tokens=args.sample_tokens,
        seq_len=args.seq_len,
        device=device,
    )
    prompt_text = vocab.decode(prompt.tolist())
    generated_text = vocab.decode(generated.tolist())

    print("\n--- prompt ---")
    print(prompt_text)
    print("\n--- sample ---")
    print(generated_text)


if __name__ == "__main__":
    main()
