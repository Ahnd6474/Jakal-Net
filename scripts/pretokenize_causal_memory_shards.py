from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import torch

from train_causal_memory_lm import (
    CAUSAL_DOC_SPECIAL_TOKENS,
    _record_to_document,
    build_special_token_id_map,
    load_serialized_documents,
    save_pretokenized_bundle,
    summarize_documents,
    summarize_tokenized_documents,
    tokenize_documents,
    validate_flat_pretokenized_shard,
)
from train_progressive_b_lm import build_tokenizer
from lm_experiment_utils import _expand_sources


def split_jsonl_sources_round_robin(*, sources: Sequence[str], raw_shard_dir: Path, num_shards: int) -> list[Path]:
    raw_shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [raw_shard_dir / f"raw_shard_{index:04d}.jsonl" for index in range(num_shards)]
    handles = [path.open("w", encoding="utf-8") for path in shard_paths]
    try:
        shard_index = 0
        total_lines = 0
        for source in sources:
            source_path = Path(source)
            with source_path.open("r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip().lstrip("\ufeff")
                    if not line:
                        continue
                    handles[shard_index].write(line)
                    handles[shard_index].write("\n")
                    total_lines += 1
                    shard_index = (shard_index + 1) % num_shards
        print(
            f"split_complete | sources={len(sources)} | shards={num_shards} | documents={total_lines:,}",
            flush=True,
        )
    finally:
        for handle in handles:
            handle.close()
    return shard_paths


def build_tokenizer_from_cache(
    *,
    tokenizer_prefix: str,
    subword_vocab_size: int,
):
    vocab, tokenizer_label, tokenizer_path = build_tokenizer(
        "",
        text_path=None,
        tokenizer="byte_bpe",
        subword_vocab_size=subword_vocab_size,
        subword_model_type="bpe",
        tokenizer_prefix=tokenizer_prefix,
        subword_character_coverage=1.0,
        subword_input_sentence_size=0,
        subword_num_threads=0,
        user_defined_symbols=CAUSAL_DOC_SPECIAL_TOKENS,
    )
    return vocab, tokenizer_label, tokenizer_path


def build_single_shard(
    *,
    raw_shard_path: Path | None,
    bundle_path: Path,
    tokenizer_prefix: str,
    subword_vocab_size: int,
    seq_len: int,
    pretokenize_workers: int,
    processing_device: str | None,
    jsonl_sources: Sequence[str] = (),
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> None:
    start_time = time.time()
    if raw_shard_path is not None:
        documents = load_serialized_documents(
            text_file=None,
            text_sources=(),
            jsonl_sources=(str(raw_shard_path),),
            hf_dataset=None,
            hf_config=None,
            hf_split="train",
            hf_text_key="text",
            hf_streaming=False,
            max_samples=None,
        )
    else:
        if shard_index is None or num_shards is None:
            raise ValueError("shard_index and num_shards are required when raw_shard_path is not provided.")
        documents: list = []
        global_index = 0
        for raw_source in jsonl_sources:
            for path in _expand_sources((raw_source,), directory_suffixes=(".jsonl", ".json")):
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, raw_line in enumerate(handle, start=1):
                        line = raw_line.strip().lstrip("\ufeff")
                        if not line:
                            continue
                        if global_index % num_shards != shard_index:
                            global_index += 1
                            continue
                        document = _record_to_document(
                            json.loads(line),
                            source=f"{path}:{line_number}",
                        )
                        global_index += 1
                        if document is not None:
                            documents.append(document)
    print(
        f"shard_documents | path={raw_shard_path or f'virtual:{shard_index}/{num_shards}'} | documents={len(documents):,} | kinds={summarize_documents(documents)}",
        flush=True,
    )
    vocab, tokenizer_label, tokenizer_path = build_tokenizer_from_cache(
        tokenizer_prefix=tokenizer_prefix,
        subword_vocab_size=subword_vocab_size,
    )
    special_token_ids = build_special_token_id_map(vocab)
    tokenized_documents = tokenize_documents(
        documents,
        vocab=vocab,
        seq_len=seq_len,
        special_token_ids=special_token_ids,
        workers=pretokenize_workers,
        processing_device=processing_device,
    )
    corpus_info = {
        "document_summary": summarize_documents(documents),
        "tokenized_summary": summarize_tokenized_documents(tokenized_documents),
        "special_tokens": special_token_ids,
        "raw_shard_path": None if raw_shard_path is None else str(raw_shard_path),
        "virtual_shard_index": shard_index,
        "virtual_num_shards": num_shards,
    }
    save_pretokenized_bundle(
        bundle_path,
        documents=tokenized_documents,
        vocab_size=int(getattr(vocab, "size")),
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=None if tokenizer_path is None else str(tokenizer_path),
        corpus_info=corpus_info,
    )
    elapsed = time.time() - start_time
    print(
        f"shard_complete | path={bundle_path} | documents={len(tokenized_documents):,} | elapsed_sec={elapsed:.1f}",
        flush=True,
    )


def launch_parallel_shards(
    *,
    raw_shard_dir: Path | None,
    bundle_dir: Path,
    tokenizer_prefix: str,
    subword_vocab_size: int,
    seq_len: int,
    pretokenize_workers: int,
    processing_device: str | None,
    parallel_shards: int,
    skip_existing: bool,
    jsonl_sources: Sequence[str],
    num_shards: int,
) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    queue: list[int] = []
    for shard_index in range(num_shards):
        bundle_path = bundle_dir / f"raw_shard_{shard_index:04d}.pt"
        if skip_existing and bundle_path.exists():
            try:
                validate_flat_pretokenized_shard(bundle_path)
            except Exception as exc:
                print(f"rebuild_corrupt_existing | path={bundle_path} | error={exc}", flush=True)
            else:
                print(f"skip_existing | path={bundle_path}", flush=True)
                continue
        queue.append(shard_index)
    running: list[tuple[int, subprocess.Popen[str]]] = []
    while queue or running:
        while queue and len(running) < parallel_shards:
            shard_index = queue.pop(0)
            bundle_path = bundle_dir / f"raw_shard_{shard_index:04d}.pt"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--mode",
                "build-shard",
                "--bundle-path",
                str(bundle_path),
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(num_shards),
                "--tokenizer-prefix",
                tokenizer_prefix,
                "--subword-vocab-size",
                str(subword_vocab_size),
                "--seq-len",
                str(seq_len),
                "--pretokenize-workers",
                str(pretokenize_workers),
            ]
            for source in jsonl_sources:
                cmd.extend(["--jsonl-source", source])
            if raw_shard_dir is not None:
                cmd.extend(["--raw-shard-path", str(raw_shard_dir / f"raw_shard_{shard_index:04d}.jsonl")])
            if processing_device:
                cmd.extend(["--processing-device", processing_device])
            print(f"launch_shard | shard={shard_index} | bundle={bundle_path}", flush=True)
            running.append((shard_index, subprocess.Popen(cmd)))
        time.sleep(2)
        next_running: list[tuple[int, subprocess.Popen[str]]] = []
        for shard_index, proc in running:
            return_code = proc.poll()
            if return_code is None:
                next_running.append((shard_index, proc))
                continue
            if return_code != 0:
                raise RuntimeError(f"Shard build failed: shard={shard_index} exit={return_code}")
            print(f"finished_shard | shard={shard_index}", flush=True)
        running = next_running


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline sharded pretokenization for causal memory training.")
    parser.add_argument("--mode", choices=("split", "build-shard", "orchestrate"), default="orchestrate")
    parser.add_argument("--jsonl-source", action="append", default=[])
    parser.add_argument("--raw-shard-dir")
    parser.add_argument("--bundle-dir")
    parser.add_argument("--raw-shard-path")
    parser.add_argument("--bundle-path")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=32)
    parser.add_argument("--parallel-shards", type=int, default=8)
    parser.add_argument("--pretokenize-workers", type=int, default=8)
    parser.add_argument("--tokenizer-prefix", required=True)
    parser.add_argument("--subword-vocab-size", type=int, default=16384)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--processing-device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "split":
        if not args.raw_shard_dir:
            raise ValueError("--raw-shard-dir is required for split mode.")
        split_jsonl_sources_round_robin(
            sources=tuple(args.jsonl_source),
            raw_shard_dir=Path(args.raw_shard_dir),
            num_shards=args.num_shards,
        )
        return
    if args.mode == "build-shard":
        if not args.bundle_path:
            raise ValueError("--bundle-path is required for build-shard mode.")
        if not args.raw_shard_path and (args.shard_index is None or not args.jsonl_source):
            raise ValueError(
                "build-shard mode requires either --raw-shard-path or (--shard-index plus --jsonl-source)."
            )
        build_single_shard(
            raw_shard_path=None if not args.raw_shard_path else Path(args.raw_shard_path),
            bundle_path=Path(args.bundle_path),
            tokenizer_prefix=args.tokenizer_prefix,
            subword_vocab_size=args.subword_vocab_size,
            seq_len=args.seq_len,
            pretokenize_workers=args.pretokenize_workers,
            processing_device=args.processing_device,
            jsonl_sources=tuple(args.jsonl_source),
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
        return
    if not args.bundle_dir:
        raise ValueError("--bundle-dir is required for orchestrate mode.")
    raw_shard_dir = None if not args.raw_shard_dir else Path(args.raw_shard_dir)
    if raw_shard_dir is not None and not raw_shard_dir.exists():
        split_jsonl_sources_round_robin(
            sources=tuple(args.jsonl_source),
            raw_shard_dir=raw_shard_dir,
            num_shards=args.num_shards,
        )
    launch_parallel_shards(
        raw_shard_dir=raw_shard_dir,
        bundle_dir=Path(args.bundle_dir),
        tokenizer_prefix=args.tokenizer_prefix,
        subword_vocab_size=args.subword_vocab_size,
        seq_len=args.seq_len,
        pretokenize_workers=args.pretokenize_workers,
        processing_device=args.processing_device,
        parallel_shards=args.parallel_shards,
        skip_existing=args.skip_existing,
        jsonl_sources=tuple(args.jsonl_source),
        num_shards=args.num_shards,
    )


if __name__ == "__main__":
    main()
