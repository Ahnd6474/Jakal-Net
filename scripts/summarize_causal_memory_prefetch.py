from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:  # pragma: no cover - depends on optional dependency
    event_accumulator = None

from lm_experiment_utils import write_json


def _resolve_event_path(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.is_file():
        return path
    candidates = sorted(path.glob("events.out.tfevents.*"))
    if not candidates:
        return None
    return candidates[-1]


def _scalar_by_step(path: Path) -> dict[str, dict[int, float]]:
    if event_accumulator is None:
        return {}
    event_path = _resolve_event_path(path)
    if event_path is None:
        return {}
    acc = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    return {
        tag: {event.step: event.value for event in acc.Scalars(tag)}
        for tag in acc.Tags().get("scalars", [])
    }


def _step_row(progress_rows: list[dict[str, Any]], target_step: int) -> dict[str, Any] | None:
    for row in progress_rows:
        if int(row["step"]) >= target_step:
            return row
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def summarize_result(
    result: dict[str, Any],
    *,
    steady_start_step: int,
    steady_end_step: int,
) -> dict[str, Any]:
    progress_rows = list(result.get("progress_rows") or [])
    progress_rows.sort(key=lambda row: int(row["step"]))
    step1_row = _step_row(progress_rows, 1)
    start_row = _step_row(progress_rows, steady_start_step)
    end_row = _step_row(progress_rows, steady_end_step)
    throughput = None
    if start_row is not None and end_row is not None:
        elapsed_delta = float(end_row["elapsed_s"]) - float(start_row["elapsed_s"])
        step_delta = int(end_row["step"]) - int(start_row["step"])
        if elapsed_delta > 0.0 and step_delta > 0:
            throughput = step_delta / elapsed_delta
    steady_rows = [
        row for row in progress_rows if steady_start_step <= int(row["step"]) <= steady_end_step
    ]
    cpu_batch_values = [float(row["cpu_batch_ms"]) for row in steady_rows]
    prefetch_q_values = [float(row["prefetch_q"]) for row in steady_rows]
    summary = {
        "variant": str(result["variant"]["name"]),
        "mode": str(result["variant"]["mode"]),
        "max_loaded_shards": int(result["variant"]["max_loaded_shards"]),
        "preload_flat_shards": bool(result["variant"]["preload_flat_shards"]),
        "startup_seconds_to_step1": None if step1_row is None else float(step1_row["wall_time_s"]),
        "steady_window_start_step": steady_start_step,
        "steady_window_end_step": steady_end_step,
        "steady_window_throughput_steps_per_s": throughput,
        "stdout_cpu_batch_ms_mean": _mean(cpu_batch_values),
        "stdout_prefetch_q_median": _median(prefetch_q_values),
        "stdout_prefetch_q_zero_count": sum(1 for value in prefetch_q_values if value <= 0.0),
        "log_path": result.get("log_path"),
        "run_dir": result.get("run_dir"),
        "tensorboard_dir": result.get("tensorboard_dir"),
        "stopped_at_target_step": bool(result.get("stopped_at_target_step")),
    }
    tensorboard_dir = result.get("tensorboard_dir")
    if tensorboard_dir:
        scalar_map = _scalar_by_step(Path(tensorboard_dir))
        cpu_map = scalar_map.get("train/cpu_batch_ms", {})
        q_map = scalar_map.get("train/prefetch_queue_size", {})
        tb_cpu_values = [float(cpu_map[step]) for step in sorted(cpu_map) if steady_start_step <= step <= steady_end_step]
        tb_q_values = [float(q_map[step]) for step in sorted(q_map) if steady_start_step <= step <= steady_end_step]
        summary["tb_cpu_batch_ms_mean"] = _mean(tb_cpu_values)
        summary["tb_prefetch_q_median"] = _median(tb_q_values)
    else:
        summary["tb_cpu_batch_ms_mean"] = None
        summary["tb_prefetch_q_median"] = None
    return summary


def _load_results(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark manifest must be a JSON object.")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Benchmark manifest must contain a results list.")
    return manifest_path.parent, results


def _print_summary_table(rows: list[dict[str, Any]]) -> None:
    headers = (
        "variant",
        "startup_s",
        "throughput_steps_per_s",
        "stdout_cpu_batch_ms_mean",
        "stdout_prefetch_q_median",
        "stdout_prefetch_q_zero_count",
    )
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["variant"]),
                    _format_number(row["startup_seconds_to_step1"]),
                    _format_number(row["steady_window_throughput_steps_per_s"]),
                    _format_number(row["stdout_cpu_batch_ms_mean"]),
                    _format_number(row["stdout_prefetch_q_median"]),
                    str(row["stdout_prefetch_q_zero_count"]),
                ]
            )
        )


def _format_number(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize causal-memory prefetch benchmark results.")
    parser.add_argument("benchmark_path", help="Benchmark manifest.json or its parent directory.")
    parser.add_argument("--steady-start-step", type=int, default=25)
    parser.add_argument("--steady-end-step", type=int, default=125)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root, results = _load_results(Path(args.benchmark_path))
    summaries = [
        summarize_result(
            result,
            steady_start_step=max(1, int(args.steady_start_step)),
            steady_end_step=max(1, int(args.steady_end_step)),
        )
        for result in results
    ]
    summaries.sort(
        key=lambda row: (
            row["steady_window_throughput_steps_per_s"] is None,
            -(row["steady_window_throughput_steps_per_s"] or 0.0),
            row["startup_seconds_to_step1"] is None,
            row["startup_seconds_to_step1"] or 0.0,
        )
    )
    write_json(benchmark_root / "summary.json", {"summaries": summaries})
    with (benchmark_root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summaries[0].keys()) if summaries else ["variant"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)
    _print_summary_table(summaries)


if __name__ == "__main__":
    main()
