from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_causal_memory_lm import (  # noqa: E402
    build_special_token_id_map,
    build_tokenizer,
    load_serialized_documents,
    save_pretokenized_bundle,
    summarize_documents,
    summarize_tokenized_documents,
    tokenize_documents,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable pretokenized cache for causal memory LM training.")
    parser.add_argument("--jsonl-source", required=True, action="append")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    docs = load_serialized_documents(
        text_file=None,
        text_sources=(),
        jsonl_sources=tuple(args.jsonl_source),
        hf_dataset=None,
        hf_config=None,
        hf_split="train",
        hf_text_key="text",
        hf_streaming=False,
        max_samples=None,
    )
    summary = summarize_documents(docs)
    print(f"documents={len(docs):,} | kinds={summary}", flush=True)
    vocab, tokenizer_label, tokenizer_path = build_tokenizer(
        "",
        text_path=None,
        tokenizer="hf_auto",
        subword_vocab_size=16384,
        subword_model_type="bpe",
        tokenizer_prefix=None,
        subword_character_coverage=1.0,
        subword_input_sentence_size=0,
        subword_num_threads=0,
        user_defined_symbols=(),
        hf_tokenizer_model=args.tokenizer_dir,
        hf_trust_remote_code=False,
    )
    special_ids = build_special_token_id_map(vocab, tokenizer_label=tokenizer_label)
    workers = min(max(1, args.workers), max(1, torch.get_num_threads()))
    tokenized = tokenize_documents(
        docs,
        vocab=vocab,
        seq_len=args.seq_len,
        special_token_ids=special_ids,
        workers=workers,
        tokenizer_label=tokenizer_label,
    )
    corpus_info = {
        "document_summary": summary,
        "tokenized_summary": summarize_tokenized_documents(tokenized),
        "special_tokens": special_ids,
        "tokenizer_dir": args.tokenizer_dir,
    }
    print("tokenized=" + json.dumps(corpus_info["tokenized_summary"], ensure_ascii=False), flush=True)
    save_pretokenized_bundle(
        Path(args.output),
        documents=tokenized,
        vocab_size=int(getattr(vocab, "size")),
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=None if tokenizer_path is None else str(tokenizer_path),
        corpus_info=corpus_info,
    )
    print(f"saved_pretokenized | path={args.output}", flush=True)


if __name__ == "__main__":
    main()
