from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TaskSpec:
    corpus_key: str
    name: str
    command: list[str]
    output: Path


def _merge_jsonl_files(*, inputs: Sequence[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    with output.open("w", encoding="utf-8") as out_handle:
        for path in inputs:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as in_handle:
                for line in in_handle:
                    if not line.strip():
                        continue
                    out_handle.write(line)
                    total_lines += 1
    return total_lines


def _task_output(base_dir: Path, corpus_key: str, task_name: str) -> Path:
    return base_dir / "shards" / corpus_key / f"{task_name}.jsonl"


def build_task_specs(*, python_exe: str, scripts_dir: Path, output_dir: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []

    for shard_index in range(4):
        name = f"plain_ultrachat_s{shard_index:02d}"
        output = _task_output(output_dir, "plain_dialogue", name)
        tasks.append(
            TaskSpec(
                corpus_key="plain_dialogue",
                name=name,
                output=output,
                command=[
                    python_exe,
                    str(scripts_dir / "build_plain_dialogue_corpus.py"),
                    "--output",
                    str(output),
                    "--source-label",
                    "ultrachat",
                    "--max-records-per-source",
                    "200000",
                    "--num-shards",
                    "4",
                    "--shard-index",
                    str(shard_index),
                    "--skip-failed-sources",
                ],
            )
        )

    for label in ("python_code", "codeact", "metamath", "code", "math", "wiki"):
        for shard_index in range(2):
            name = f"segmented_{label}_s{shard_index:02d}"
            output = _task_output(output_dir, "segmented_mixed_docs", name)
            tasks.append(
                TaskSpec(
                    corpus_key="segmented_mixed_docs",
                    name=name,
                    output=output,
                    command=[
                        python_exe,
                        str(scripts_dir / "build_segmented_dialogue_corpus.py"),
                        "--output",
                        str(output),
                        "--source-label",
                        label,
                        "--max-records-per-source",
                        "500000",
                        "--num-shards",
                        "2",
                        "--shard-index",
                        str(shard_index),
                        "--skip-failed-sources",
                    ],
                )
            )

    for shard_index in range(2):
        name = f"mathqa_s{shard_index:02d}"
        output = _task_output(output_dir, "math_qa", name)
        tasks.append(
            TaskSpec(
                corpus_key="math_qa",
                name=name,
                output=output,
                command=[
                    python_exe,
                    str(scripts_dir / "build_math_qa_corpus.py"),
                    "--output",
                    str(output),
                    "--max-records-per-source",
                    "100000",
                    "--num-shards",
                    "2",
                    "--shard-index",
                    str(shard_index),
                    "--skip-failed-sources",
                ],
            )
        )

    reasoning_labels = (
        ("anli_r1", 1),
        ("anli_r2", 1),
        ("anli_r3", 1),
        ("proofwriter", 2),
        ("logicnli", 2),
        ("boardgameqa", 2),
        ("defeasible_atomic", 1),
        ("defeasible_snli", 1),
        ("defeasible_social", 1),
    )
    for label, num_shards in reasoning_labels:
        for shard_index in range(num_shards):
            name = f"reasoning_{label}_s{shard_index:02d}"
            output = _task_output(output_dir, "reasoning_dialogue", name)
            tasks.append(
                TaskSpec(
                    corpus_key="reasoning_dialogue",
                    name=name,
                    output=output,
                    command=[
                        python_exe,
                        str(scripts_dir / "build_reasoning_dialogue_corpus.py"),
                        "--output",
                        str(output),
                        "--source-label",
                        label,
                        "--max-records-per-source",
                        "100000",
                        "--num-shards",
                        str(num_shards),
                        "--shard-index",
                        str(shard_index),
                        "--skip-failed-sources",
                    ],
                )
            )

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel shard-aware corpus builder for causal memory training.")
    parser.add_argument("--output-dir", default="artifacts/data_parallel")
    parser.add_argument("--max-parallel", type=int, default=16)
    parser.add_argument("--python-exe", default=sys.executable)
    args = parser.parse_args()
    if args.max_parallel <= 0:
        raise ValueError("--max-parallel must be positive.")

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_task_specs(python_exe=args.python_exe, scripts_dir=scripts_dir, output_dir=output_dir)

    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    env.setdefault("PYTHONPATH", f"{repo_root / 'src'}:{scripts_dir}")

    started = time.perf_counter()
    pending = list(tasks)
    running: dict[Future[int], tuple[TaskSpec, subprocess.Popen[str], object]] = {}
    results: list[dict[str, object]] = []

    while pending or running:
        while pending and len(running) < args.max_parallel:
            task = pending.pop(0)
            task.output.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{task.name}.log"
            log_handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                task.command,
                cwd=repo_root,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            future: Future[int] = Future()
            future.set_running_or_notify_cancel()
            running[future] = (task, process, log_handle)
        if not running:
            continue
        time.sleep(1.0)
        completed: list[Future[int]] = []
        for future, (task, process, log_handle) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            log_handle.close()
            future.set_result(returncode)
            completed.append(future)
            results.append(
                {
                    "task": task.name,
                    "corpus_key": task.corpus_key,
                    "output": str(task.output),
                    "returncode": returncode,
                }
            )
        for future in completed:
            running.pop(future, None)

    corpus_outputs: dict[str, list[Path]] = {
        "plain_dialogue": [],
        "segmented_mixed_docs": [],
        "math_qa": [],
        "reasoning_dialogue": [],
    }
    for task in tasks:
        corpus_outputs[task.corpus_key].append(task.output)

    merged_info: dict[str, dict[str, object]] = {}
    for corpus_key, paths in corpus_outputs.items():
        final_output = output_dir / f"{corpus_key}.jsonl"
        line_count = _merge_jsonl_files(inputs=paths, output=final_output)
        merged_info[corpus_key] = {
            "output": str(final_output),
            "lines": line_count,
            "shards": [str(path) for path in paths],
        }

    meta = {
        "output_dir": str(output_dir),
        "max_parallel": args.max_parallel,
        "elapsed_seconds": time.perf_counter() - started,
        "tasks": results,
        "merged": merged_info,
    }
    meta_path = output_dir / "build_parallel.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
