from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


@dataclass(frozen=True, slots=True)
class TaskSpec:
    corpus_key: str
    name: str
    command: list[str]
    output: Path


def _nonempty_jsonl_lines(path: Path) -> Iterator[str]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield line


def _merge_jsonl_files(*, inputs: Sequence[Path], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    total_lines = 0
    with output.open("w", encoding="utf-8") as out_handle:
        for path in inputs:
            for line in _nonempty_jsonl_lines(path):
                out_handle.write(line)
                out_handle.write("\n")
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

    for shard_index in range(4):
        name = f"code_docs_s{shard_index:02d}"
        output = _task_output(output_dir, "code_docs", name)
        tasks.append(
            TaskSpec(
                corpus_key="code_docs",
                name=name,
                output=output,
                command=[
                    python_exe,
                    str(scripts_dir / "build_segmented_dialogue_corpus.py"),
                    "--output",
                    str(output),
                    "--no-default-mixed-dialogue-sources",
                    "--no-default-document-sources",
                    "--hf-document-source",
                    "code=codeparrot/codeparrot-clean||train|content|code",
                    "--source-label",
                    "code",
                    "--max-records-per-source",
                    "500000",
                    "--num-shards",
                    "4",
                    "--shard-index",
                    str(shard_index),
                    "--skip-failed-sources",
                ],
            )
        )

    mixed_labels = (
        ("python_code", 1, "250000"),
        ("codeact", 1, "250000"),
    )
    for label, num_shards, max_records in mixed_labels:
        for shard_index in range(num_shards):
            name = f"mixed_{label}_s{shard_index:02d}"
            output = _task_output(output_dir, "code_mixed_dialogue", name)
            tasks.append(
                TaskSpec(
                    corpus_key="code_mixed_dialogue",
                    name=name,
                    output=output,
                    command=[
                        python_exe,
                        str(scripts_dir / "build_segmented_dialogue_corpus.py"),
                        "--output",
                        str(output),
                        "--no-default-document-sources",
                        "--source-label",
                        label,
                        "--max-records-per-source",
                        max_records,
                        "--num-shards",
                        str(num_shards),
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
        ("proofwriter", 1),
        ("logicnli", 1),
        ("boardgameqa", 1),
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
                        "150000",
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
    parser = argparse.ArgumentParser(
        description="Download and shard raw dialogue/reasoning/code corpora without pretokenization."
    )
    parser.add_argument("--output-dir", default="artifacts/data_mix_raw_v1")
    parser.add_argument("--max-parallel", type=int, default=10)
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
    running: list[tuple[TaskSpec, subprocess.Popen[str], object]] = []
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
            running.append((task, process, log_handle))
            print(
                json.dumps(
                    {
                        "event": "task_started",
                        "task": task.name,
                        "corpus_key": task.corpus_key,
                        "output": str(task.output),
                        "log": str(log_path),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if not running:
            continue
        time.sleep(1.0)
        remaining: list[tuple[TaskSpec, subprocess.Popen[str], object]] = []
        for task, process, log_handle in running:
            returncode = process.poll()
            if returncode is None:
                remaining.append((task, process, log_handle))
                continue
            log_handle.close()
            result = {
                "task": task.name,
                "corpus_key": task.corpus_key,
                "output": str(task.output),
                "returncode": returncode,
            }
            results.append(result)
            print(json.dumps({"event": "task_finished", **result}, ensure_ascii=False), flush=True)
        running = remaining

    corpus_outputs: dict[str, list[Path]] = {
        "plain_dialogue": [],
        "code_docs": [],
        "code_mixed_dialogue": [],
        "reasoning_dialogue": [],
    }
    for task in tasks:
        corpus_outputs[task.corpus_key].append(task.output)

    merged_info: dict[str, dict[str, object]] = {}
    merged_paths: dict[str, Path] = {}
    for corpus_key, paths in corpus_outputs.items():
        final_output = output_dir / f"{corpus_key}.jsonl"
        line_count = _merge_jsonl_files(inputs=paths, output=final_output)
        merged_paths[corpus_key] = final_output
        merged_info[corpus_key] = {
            "output": str(final_output),
            "lines": line_count,
            "shards": [str(path) for path in paths],
        }

    code_corpus_path = output_dir / "code_corpus.jsonl"
    code_corpus_count = _merge_jsonl_files(
        inputs=(merged_paths["code_docs"], merged_paths["code_mixed_dialogue"]),
        output=code_corpus_path,
    )
    all_corpus_path = output_dir / "all_corpus.jsonl"
    all_corpus_count = _merge_jsonl_files(
        inputs=(
            merged_paths["plain_dialogue"],
            merged_paths["code_docs"],
            merged_paths["code_mixed_dialogue"],
            merged_paths["reasoning_dialogue"],
        ),
        output=all_corpus_path,
    )

    merged_info["code_corpus"] = {
        "output": str(code_corpus_path),
        "lines": code_corpus_count,
        "components": ["code_docs", "code_mixed_dialogue"],
    }
    merged_info["all_corpus"] = {
        "output": str(all_corpus_path),
        "lines": all_corpus_count,
        "components": ["plain_dialogue", "code_docs", "code_mixed_dialogue", "reasoning_dialogue"],
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
