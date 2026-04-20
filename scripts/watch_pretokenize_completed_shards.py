from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from train_causal_memory_lm import (
    CAUSAL_DOC_SPECIAL_TOKENS,
    build_training_text,
    byte_bpe_tokenizer_cache_exists,
    load_serialized_documents,
)
from train_progressive_b_lm import build_tokenizer


BUILDER_SCRIPTS = {
    "build_plain_dialogue_corpus.py",
    "build_segmented_dialogue_corpus.py",
    "build_math_qa_corpus.py",
    "build_reasoning_dialogue_corpus.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretokenize completed shard outputs while corpus build is still running.")
    parser.add_argument("--shard-root", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--tokenizer-prefix", required=True)
    parser.add_argument("--subword-vocab-size", type=int, default=16384)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--pretokenize-workers", type=int, default=8)
    parser.add_argument("--parallel-jobs", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--sample-docs-for-tokenizer", type=int, default=50000)
    parser.add_argument("--min-completed-shards-for-tokenizer", type=int, default=4)
    parser.add_argument("--processing-device")
    parser.add_argument("--log-prefix", default="shard-watch")
    return parser.parse_args()


def log(prefix: str, message: str) -> None:
    print(f"{prefix} | {message}", flush=True)


def _cmdline(process: psutil.Process) -> list[str]:
    try:
        return process.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return []


def _is_builder_process(cmd: list[str]) -> bool:
    for part in cmd:
        for script in BUILDER_SCRIPTS:
            if script in part:
                return True
    return False


def builder_is_running_for_path(path: Path) -> bool:
    target = str(path)
    for process in psutil.process_iter(["pid"]):
        cmd = _cmdline(process)
        if not cmd or not _is_builder_process(cmd):
            continue
        for index, token in enumerate(cmd[:-1]):
            if token == "--output" and cmd[index + 1] == target:
                return True
    return False


def any_builders_running(shard_root: Path) -> bool:
    shard_root_str = str(shard_root)
    for process in psutil.process_iter(["pid"]):
        cmd = _cmdline(process)
        if not cmd or not _is_builder_process(cmd):
            continue
        if any(shard_root_str in part for part in cmd):
            return True
    return False


def completed_shard_paths(shard_root: Path) -> list[Path]:
    completed: list[Path] = []
    for path in sorted(shard_root.rglob("*.jsonl")):
        if path.stat().st_size <= 0:
            continue
        if builder_is_running_for_path(path):
            continue
        completed.append(path)
    return completed


def ensure_tokenizer_cache(
    *,
    shard_paths: list[Path],
    tokenizer_prefix: str,
    subword_vocab_size: int,
    sample_docs_for_tokenizer: int,
    min_completed_shards: int,
    log_prefix: str,
) -> bool:
    if byte_bpe_tokenizer_cache_exists(tokenizer_prefix=tokenizer_prefix, vocab_size=subword_vocab_size):
        return True
    if len(shard_paths) < min_completed_shards:
        return False

    selected = shard_paths[: min(len(shard_paths), max(min_completed_shards, 8))]
    documents = load_serialized_documents(
        text_file=None,
        text_sources=(),
        jsonl_sources=tuple(str(path) for path in selected),
        hf_dataset=None,
        hf_config=None,
        hf_split="train",
        hf_text_key="text",
        hf_streaming=False,
        max_samples=sample_docs_for_tokenizer,
    )
    if not documents:
        return False

    log(log_prefix, f"building_tokenizer | shards={len(selected)} | sampled_documents={len(documents):,}")
    training_text = build_training_text(documents)
    build_tokenizer(
        training_text,
        text_path=None,
        tokenizer="byte_bpe",
        subword_vocab_size=subword_vocab_size,
        subword_model_type="bpe",
        tokenizer_prefix=tokenizer_prefix,
        subword_character_coverage=1.0,
        subword_input_sentence_size=0,
        subword_num_threads=max(1, os.cpu_count() or 1),
        user_defined_symbols=CAUSAL_DOC_SPECIAL_TOKENS,
    )
    log(log_prefix, f"tokenizer_ready | prefix={tokenizer_prefix}")
    return True


def launch_pretokenize_job(
    *,
    python_exe: str,
    shard_path: Path,
    bundle_path: Path,
    tokenizer_prefix: str,
    subword_vocab_size: int,
    seq_len: int,
    pretokenize_workers: int,
    processing_device: str | None,
) -> subprocess.Popen[str]:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(Path(__file__).resolve().parent / "pretokenize_causal_memory_shards.py"),
        "--mode",
        "build-shard",
        "--raw-shard-path",
        str(shard_path),
        "--bundle-path",
        str(bundle_path),
        "--tokenizer-prefix",
        tokenizer_prefix,
        "--subword-vocab-size",
        str(subword_vocab_size),
        "--seq-len",
        str(seq_len),
        "--pretokenize-workers",
        str(pretokenize_workers),
    ]
    if processing_device:
        cmd.extend(["--processing-device", processing_device])
    return subprocess.Popen(cmd, text=True)


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)
    python_exe = sys.executable

    running: dict[Path, subprocess.Popen[str]] = {}
    scheduled: set[Path] = set()

    while True:
        shard_paths = completed_shard_paths(shard_root)
        tokenizer_ready = ensure_tokenizer_cache(
            shard_paths=shard_paths,
            tokenizer_prefix=args.tokenizer_prefix,
            subword_vocab_size=args.subword_vocab_size,
            sample_docs_for_tokenizer=args.sample_docs_for_tokenizer,
            min_completed_shards=args.min_completed_shards_for_tokenizer,
            log_prefix=args.log_prefix,
        )

        next_running: dict[Path, subprocess.Popen[str]] = {}
        for shard_path, proc in running.items():
            return_code = proc.poll()
            if return_code is None:
                next_running[shard_path] = proc
                continue
            if return_code != 0:
                raise RuntimeError(f"Pretokenize failed for {shard_path} with exit code {return_code}")
            log(args.log_prefix, f"bundle_complete | shard={shard_path.name}")
        running = next_running

        if tokenizer_ready:
            for shard_path in shard_paths:
                if shard_path in scheduled or shard_path in running:
                    continue
                bundle_path = bundle_dir / shard_path.relative_to(shard_root)
                bundle_path = bundle_path.with_suffix(".pt")
                if bundle_path.exists():
                    scheduled.add(shard_path)
                    continue
                if len(running) >= args.parallel_jobs:
                    break
                proc = launch_pretokenize_job(
                    python_exe=python_exe,
                    shard_path=shard_path,
                    bundle_path=bundle_path,
                    tokenizer_prefix=args.tokenizer_prefix,
                    subword_vocab_size=args.subword_vocab_size,
                    seq_len=args.seq_len,
                    pretokenize_workers=args.pretokenize_workers,
                    processing_device=args.processing_device,
                )
                running[shard_path] = proc
                scheduled.add(shard_path)
                log(args.log_prefix, f"bundle_launch | shard={shard_path.name} | bundle={bundle_path}")

        builders_running = any_builders_running(shard_root)
        if not builders_running and not running and tokenizer_ready:
            remaining = [
                path
                for path in completed_shard_paths(shard_root)
                if not (bundle_dir / path.relative_to(shard_root)).with_suffix(".pt").exists()
            ]
            if not remaining:
                log(args.log_prefix, "all_done")
                return

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
