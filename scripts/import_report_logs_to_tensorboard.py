from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from lm_experiment_utils import create_run_directory, write_csv_rows, write_json

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tensorboard is required to import report logs.") from exc


PROGRESS_RE = re.compile(
    r"progress \| step=\s*(?P<step>\d+)/(?P<total>\d+) \| stage=(?P<stage>[^|]+) \| span=(?P<span>\d+)"
    r" \| train_loss=(?P<train_loss>[-+0-9.eE]+) \| lr=(?P<lr>[-+0-9.eE]+)"
    r" \| cpu_batch_ms=(?P<cpu_batch_ms>[-+0-9.eE]+) \| prefetch_q=(?P<prefetch_q>\d+)"
    r" \| elapsed=(?P<elapsed>[-+0-9.eE]+)s"
)
EVAL_RE = re.compile(
    r"eval \| step=(?P<step>\d+) \| train_loss=(?P<train_loss>[-+0-9.eE]+)"
    r" \| val_loss=(?P<val_loss>[-+0-9.eE]+) \| val_ppl=(?P<val_ppl>[-+0-9.eE]+)"
)
STARTUP_PROGRESS_RE = re.compile(
    r"progress \| step=\s*(?P<step>\d+)/(?P<total>\d+) \| stage=(?P<stage>[^|]+) \| span=(?P<span>\d+)"
    r" \| train_loss=(?P<train_loss>[-+0-9.eE]+) \| lr=(?P<lr>[-+0-9.eE]+)"
    r" \| cpu_batch_ms=(?P<cpu_batch_ms>[-+0-9.eE]+) \| prefetch_q=(?P<prefetch_q>\d+)"
    r" \| elapsed=(?P<elapsed>[-+0-9.eE]+)s"
)
KV_RE = re.compile(r"^(?P<key>[A-Za-z0-9_]+)=(?P<value>.+)$")


def _coerce_value(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_progress_lines(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+:", "", line)
        progress_match = PROGRESS_RE.search(line) or STARTUP_PROGRESS_RE.search(line)
        if progress_match is not None:
            rows.append(
                {
                    "kind": "progress",
                    "step": int(progress_match.group("step")),
                    "total_steps": int(progress_match.group("total")),
                    "stage": progress_match.group("stage").strip(),
                    "span": int(progress_match.group("span")),
                    "train_loss": float(progress_match.group("train_loss")),
                    "lr": float(progress_match.group("lr")),
                    "cpu_batch_ms": float(progress_match.group("cpu_batch_ms")),
                    "prefetch_q": int(progress_match.group("prefetch_q")),
                    "elapsed_s": float(progress_match.group("elapsed")),
                }
            )
            continue
        eval_match = EVAL_RE.search(line)
        if eval_match is not None:
            rows.append(
                {
                    "kind": "eval",
                    "step": int(eval_match.group("step")),
                    "train_loss": float(eval_match.group("train_loss")),
                    "val_loss": float(eval_match.group("val_loss")),
                    "val_ppl": float(eval_match.group("val_ppl")),
                }
            )
    return rows


def parse_config_lines(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = KV_RE.match(line)
        if match is None:
            continue
        payload[match.group("key")] = _coerce_value(match.group("value"))
    return payload


def import_log(report_log: Path, *, output_root: Path) -> Path | None:
    text = report_log.read_text(encoding="utf-8")
    rows = parse_progress_lines(text)
    config = parse_config_lines(text)
    if not rows and not config:
        return None

    run_dir = create_run_directory(output_root, f"report-{report_log.stem}")
    writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    try:
        for row in rows:
            step = int(row["step"])
            if row["kind"] == "progress":
                writer.add_scalar("train/loss", float(row["train_loss"]), step)
                writer.add_scalar("train/lr", float(row["lr"]), step)
                writer.add_scalar("train/document_span", int(row["span"]), step)
                writer.add_scalar("train/cpu_batch_ms", float(row["cpu_batch_ms"]), step)
                writer.add_scalar("train/prefetch_queue_size", int(row["prefetch_q"]), step)
                writer.add_scalar("train/elapsed_s", float(row["elapsed_s"]), step)
            elif row["kind"] == "eval":
                writer.add_scalar("eval/train_loss", float(row["train_loss"]), step)
                writer.add_scalar("eval/val_loss", float(row["val_loss"]), step)
                writer.add_scalar("eval/val_ppl", float(row["val_ppl"]), step)
    finally:
        writer.flush()
        writer.close()

    metadata = {
        "source_report_log": str(report_log),
        "imported_rows": len(rows),
        "parsed_config": config,
    }
    write_json(run_dir / "import_metadata.json", metadata)
    if config:
        write_json(run_dir / "config.json", config)
    if rows:
        write_csv_rows(run_dir / "history.csv", rows)
        (run_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Import saved report logs into TensorBoard event files.")
    parser.add_argument(
        "--report-log",
        action="append",
        default=[],
        help="Specific report log file(s) to import. Defaults to reports/logs/*.txt under repo root.",
    )
    parser.add_argument("--output-root", default="artifacts/training_runs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    if args.report_log:
        report_logs = [Path(path).resolve() for path in args.report_log]
    else:
        report_logs = sorted((repo_root / "reports" / "logs").glob("*.txt"))

    imported: list[Path] = []
    for report_log in report_logs:
        run_dir = import_log(report_log, output_root=output_root)
        if run_dir is not None:
            imported.append(run_dir)
            print(f"imported={report_log} -> {run_dir}", flush=True)
        else:
            print(f"skipped={report_log}", flush=True)
    print(f"imported_runs={len(imported)}", flush=True)


if __name__ == "__main__":
    main()
