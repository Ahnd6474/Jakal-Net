from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(args: list[str]) -> None:
    print(" ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-interval", type=int, default=1_000)
    parser.add_argument("--data-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pretokenize-workers", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--implementation", choices=("reference", "native"), default="reference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dialogue-jsonl",
        default="artifacts/data/dialogue/en_dialogue_mix.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        default="artifacts/data/query_block_instruction_arxiv_pubmed_code_wiki_1_2_2_2_3.jsonl",
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_script = repo_root / "scripts" / "train_progressive_b_lm.py"
    build_script = repo_root / "scripts" / "build_mixed_next_sentence_corpus.py"
    output_jsonl = repo_root / args.output_jsonl

    build_cmd = [
        sys.executable,
        str(build_script),
        "--output",
        str(output_jsonl),
        "--dialogue-jsonl",
        str(repo_root / args.dialogue_jsonl),
        "--hf-source",
        "arxiv_1=ccdv/arxiv-summarization||train|article",
        "--hf-source",
        "arxiv_2=ccdv/arxiv-summarization||train|article",
        "--hf-source",
        "pubmed_1=ccdv/pubmed-summarization||train|article",
        "--hf-source",
        "pubmed_2=ccdv/pubmed-summarization||train|article",
        "--hf-source",
        "code_1=codeparrot/codeparrot-clean||train|content",
        "--hf-source",
        "code_2=codeparrot/codeparrot-clean||train|content",
        "--hf-source",
        "wiki_1=wikimedia/wikipedia|20231101.en|train|text",
        "--hf-source",
        "wiki_2=wikimedia/wikipedia|20231101.en|train|text",
        "--hf-source",
        "wiki_3=wikimedia/wikipedia|20231101.en|train|text",
        "--no-default-hf-sources",
        "--hf-streaming",
        "--max-records-per-source",
        "20000",
        "--max-pairs-per-source",
        "100000",
        "--max-dialogue-pairs",
        "100000",
        "--skip-failed-sources",
    ]

    train_cmd = [
        sys.executable,
        str(train_script),
        "--device",
        args.device,
        "--training-objective",
        "query_block",
        "--jsonl-source",
        str(output_jsonl),
        "--balance-batch-by-source",
        "--tokenizer",
        "byte_bpe",
        "--subword-vocab-size",
        "16384",
        "--tokenizer-prefix",
        str(repo_root / "artifacts" / "tokenizers" / "query_block_mix_byte_bpe_16384"),
        "--steps",
        str(args.steps),
        "--eval-interval",
        str(args.eval_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--batch-size",
        str(args.batch_size),
        "--data-workers",
        str(args.data_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--pretokenize-workers",
        str(args.pretokenize_workers),
        "--seq-len",
        "512",
        "--target-len",
        "128",
        "--dim",
        "512",
        "--warmup-layers",
        "2",
        "--final-refine-layers",
        "3",
        "--lite-layers",
        "5",
        "--mid-layers",
        "5",
        "--full-layers",
        "0",
        "--route-topk",
        "32",
        "--query-topk",
        "32",
        "--route-kind",
        "low_rank_bilinear",
        "--pairwise-kind",
        "low_rank_bilinear",
        "--route-mode",
        "topk",
        "--expanded-propagation",
        "topk",
        "--compressed-propagation",
        "topk",
        "--sequence-propagation",
        "window",
        "--precision",
        "bf16",
        "--implementation",
        args.implementation,
        "--s-window",
        "32",
        "--edge-dropout-p",
        "0.1",
        "--b-diversity-loss-weight",
        "0.0",
        "--route-concentration-loss-weight",
        "0.0",
        "--learning-rate",
        str(args.learning_rate),
    ]

    if args.dry_run:
        print("Build command:")
        print(" ".join(build_cmd))
        print("\nTrain command:")
        print(" ".join(train_cmd))
        return

    if not args.skip_build or not output_jsonl.exists():
        run_command(build_cmd)
    run_command(train_cmd)


if __name__ == "__main__":
    main()
