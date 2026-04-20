from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lm_experiment_utils import write_json


DEFAULT_MATH_QA_SOURCES = (
    "math_lighteval=DigitalLearningGmbH/MATH-lighteval||train|problem|solution",
)


@dataclass(frozen=True, slots=True)
class MathQaSource:
    label: str
    dataset: str
    config: str | None
    split: str
    problem_key: str
    solution_key: str


def parse_math_qa_source(spec: str) -> MathQaSource:
    if "=" not in spec:
        raise ValueError("HF math QA source spec must be label=dataset|config|split|problem_key|solution_key.")
    label, payload = spec.split("=", 1)
    parts = payload.split("|")
    if len(parts) != 5:
        raise ValueError("HF math QA source spec must be label=dataset|config|split|problem_key|solution_key.")
    dataset, config, split, problem_key, solution_key = parts
    return MathQaSource(
        label=label.strip(),
        dataset=dataset.strip(),
        config=config.strip() or None,
        split=split.strip(),
        problem_key=problem_key.strip(),
        solution_key=solution_key.strip(),
    )


def _nested_get(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def write_math_qa_document(handle: Any, *, source: str, problem: str, solution: str) -> None:
    handle.write(
        json.dumps(
            {
                "kind": "dialogue",
                "source": f"math_qa:{source}",
                "messages": [
                    {"role": "user", "segments": [{"kind": "math", "text": problem}]},
                    {"role": "assistant", "segments": [{"kind": "math", "text": solution}]},
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def add_hf_math_qa_source(
    *,
    output_handle: Any,
    source: MathQaSource,
    streaming: bool,
    max_records: int | None,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to build the math QA corpus.") from exc

    dataset = load_dataset(source.dataset, source.config, split=source.split, streaming=streaming)
    written = 0
    for record_index, row in enumerate(dataset, start=1):
        if max_records is not None and written >= max_records:
            break
        problem = _nested_get(row, source.problem_key)
        solution = _nested_get(row, source.solution_key)
        if not isinstance(problem, str) or not problem.strip():
            continue
        if not isinstance(solution, str) or not solution.strip():
            continue
        write_math_qa_document(
            output_handle,
            source=f"{source.label}:{record_index}",
            problem=problem.strip(),
            solution=solution.strip(),
        )
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a math QA corpus as user/assistant math dialogue.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-math-qa-source", action="append", default=[])
    parser.add_argument("--no-default-sources", action="store_true")
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--no-hf-streaming", dest="hf_streaming", action="store_false")
    parser.add_argument("--max-records-per-source", type=int, default=50000)
    parser.add_argument("--skip-failed-sources", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_specs = ([] if args.no_default_sources else list(DEFAULT_MATH_QA_SOURCES)) + args.hf_math_qa_source
    sources = [parse_math_qa_source(spec) for spec in source_specs]
    source_counts: dict[str, int] = {}
    started = time.perf_counter()

    with output.open("w", encoding="utf-8") as handle:
        for source in sources:
            try:
                count = add_hf_math_qa_source(
                    output_handle=handle,
                    source=source,
                    streaming=args.hf_streaming,
                    max_records=args.max_records_per_source,
                )
            except Exception as exc:
                if not args.skip_failed_sources:
                    raise
                print(f"source={source.label} failed={type(exc).__name__}: {exc}", flush=True)
                source_counts[source.label] = 0
                continue
            source_counts[source.label] = count
            print(f"source={source.label} documents={count:,}", flush=True)

    meta = {
        "output": str(output),
        "total_documents": sum(source_counts.values()),
        "source_counts": source_counts,
        "math_qa_sources": [asdict(source) for source in sources],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(output.suffix + ".meta.json"), meta)
    print(f"output={output} total_documents={meta['total_documents']:,}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
