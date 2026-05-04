from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

from lm_experiment_utils import create_run_directory, ensure_directory, slugify_run_name, write_json


PROGRESS_RE = re.compile(
    r"progress \| step=\s*(?P<step>\d+)/(?P<total>\d+)\s+\| stage=(?P<stage>[^|]+)\| span=(?P<span>\d+)\s+\| "
    r"train_loss=(?P<train_loss>[-+0-9.eE]+)\s+\| lr=(?P<lr>[-+0-9.eE]+)\s+\| "
    r"cpu_batch_ms=(?P<cpu_batch_ms>[-+0-9.eE]+)\s+\| prefetch_q=(?P<prefetch_q>[-+0-9.eE]+)\s+\| "
    r"elapsed=(?P<elapsed>[-+0-9.eE]+)s"
)
INTERESTING_MARKERS = (
    "startup | ",
    "prebuild_process_start | ",
    "prebuild_progress | ",
    "flat_preload_skip | ",
    "flat_preload_start | ",
    "flat_preload_done | ",
    "startup | rolling_prefetcher_started | ",
)
PREBUILD_BOOL_FLAGS = ("--prebuild-train-batches", "--prebuild-pin-memory")
PREBUILD_VALUE_FLAGS = ("--prebuild-workers", "--prebuild-worker-threads", "--prebuild-cache-dir")
ROLLING_VALUE_FLAGS = (
    "--rolling-prefetch-workers",
    "--rolling-prefetch-worker-threads",
    "--rolling-prefetch-block-size",
    "--rolling-prefetch-blocks",
    "--rolling-prefetch-cache-dir",
)
@dataclass(frozen=True)
class CommandSpec:
    env_updates: dict[str, str]
    argv: tuple[str, ...]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    mode: str
    max_loaded_shards: int
    preload_flat_shards: bool


def _normalize_command_text(text: str) -> str:
    logical_lines: list[str] = []
    current = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            current += stripped[:-1].rstrip() + " "
            continue
        current += stripped
        logical_lines.append(current)
        current = ""
    if current:
        logical_lines.append(current)
    return " ".join(logical_lines).strip()


def parse_command_text(text: str) -> CommandSpec:
    normalized = _normalize_command_text(text)
    if not normalized:
        raise ValueError("Training command text must not be empty.")
    tokens = shlex.split(normalized, posix=True)
    env_updates: dict[str, str] = {}
    argv_start = 0
    for index, token in enumerate(tokens):
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token) is None:
            argv_start = index
            break
        key, value = token.split("=", maxsplit=1)
        env_updates[key] = value
    else:
        raise ValueError("Training command must include an executable.")
    argv = tuple(tokens[argv_start:])
    if not argv:
        raise ValueError("Training command must include an executable.")
    return CommandSpec(env_updates=env_updates, argv=argv)


def _find_flag(argv: list[str], flag: str) -> int | None:
    for index, token in enumerate(argv):
        if token == flag:
            return index
    return None


def _remove_bool_flag(argv: list[str], flag: str) -> None:
    while True:
        index = _find_flag(argv, flag)
        if index is None:
            return
        del argv[index]


def _remove_value_flag(argv: list[str], flag: str) -> None:
    while True:
        index = _find_flag(argv, flag)
        if index is None:
            return
        del argv[index : min(index + 2, len(argv))]


def _ensure_bool_flag(argv: list[str], flag: str) -> None:
    if _find_flag(argv, flag) is None:
        argv.append(flag)


def _set_value_flag(argv: list[str], flag: str, value: str) -> None:
    _remove_value_flag(argv, flag)
    argv.extend((flag, value))


def _get_value_flag(argv: list[str], flag: str) -> str | None:
    index = _find_flag(argv, flag)
    if index is None or index + 1 >= len(argv):
        return None
    return argv[index + 1]


def rewrite_command(
    command: CommandSpec,
    variant: VariantSpec,
    *,
    rolling_workers: int,
    rolling_worker_threads: int,
    rolling_block_size: int,
    rolling_blocks: int,
) -> CommandSpec:
    argv = list(command.argv)
    base_run_name = _get_value_flag(argv, "--run-name") or "causal_memory"
    _set_value_flag(argv, "--run-name", f"{base_run_name}-{variant.name}")
    _ensure_bool_flag(argv, "--tensorboard")
    _set_value_flag(argv, "--pretokenized-max-loaded-shards", str(int(variant.max_loaded_shards)))
    if variant.preload_flat_shards:
        _ensure_bool_flag(argv, "--preload-flat-shards")
    else:
        _remove_bool_flag(argv, "--preload-flat-shards")
    if variant.mode == "prebuild":
        _ensure_bool_flag(argv, "--prebuild-train-batches")
        for flag in ROLLING_VALUE_FLAGS:
            _remove_value_flag(argv, flag)
    elif variant.mode == "rolling":
        for flag in PREBUILD_BOOL_FLAGS:
            _remove_bool_flag(argv, flag)
        for flag in PREBUILD_VALUE_FLAGS:
            _remove_value_flag(argv, flag)
        _set_value_flag(argv, "--rolling-prefetch-workers", str(max(1, int(rolling_workers))))
        _set_value_flag(argv, "--rolling-prefetch-worker-threads", str(max(1, int(rolling_worker_threads))))
        _set_value_flag(argv, "--rolling-prefetch-block-size", str(max(1, int(rolling_block_size))))
        _set_value_flag(argv, "--rolling-prefetch-blocks", str(max(1, int(rolling_blocks))))
    else:
        raise ValueError(f"Unsupported benchmark mode: {variant.mode!r}")
    return CommandSpec(env_updates=dict(command.env_updates), argv=tuple(argv))


def _resolve_output_root(command: CommandSpec, cwd: Path) -> Path:
    raw_output_root = _get_value_flag(list(command.argv), "--output-root") or "artifacts/training_runs"
    output_root = Path(raw_output_root)
    if not output_root.is_absolute():
        output_root = cwd / output_root
    return output_root


def _locate_run_dir(output_root: Path, run_name: str, started_at: float) -> Path | None:
    if not output_root.exists():
        return None
    slug = slugify_run_name(run_name)
    candidates = [
        path
        for path in output_root.iterdir()
        if path.is_dir()
        and re.fullmatch(r"\d{8}_\d{6}_" + re.escape(slug) + r"(?:_\d{2})?", path.name) is not None
        and path.stat().st_mtime >= started_at - 1.0
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def _terminate_process(process: subprocess.Popen[str], timeout: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=timeout)
            return
        except Exception:
            pass
    process.kill()
    process.wait(timeout=timeout)


def _parse_progress_line(line: str) -> dict[str, Any] | None:
    match = PROGRESS_RE.search(line)
    if match is None:
        return None
    return {
        "step": int(match.group("step")),
        "total_steps": int(match.group("total")),
        "stage": match.group("stage").strip(),
        "span": int(match.group("span")),
        "train_loss": float(match.group("train_loss")),
        "lr": float(match.group("lr")),
        "cpu_batch_ms": float(match.group("cpu_batch_ms")),
        "prefetch_q": float(match.group("prefetch_q")),
        "elapsed_s": float(match.group("elapsed")),
    }


def _command_string(command: CommandSpec) -> str:
    env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in command.env_updates.items())
    argv = " ".join(shlex.quote(token) for token in command.argv)
    return f"{env_prefix} {argv}".strip()


def _default_variants() -> tuple[VariantSpec, ...]:
    return (
        VariantSpec(name="baseline_prebuild_cache8", mode="prebuild", max_loaded_shards=8, preload_flat_shards=False),
        VariantSpec(name="prebuild_cache0", mode="prebuild", max_loaded_shards=0, preload_flat_shards=True),
        VariantSpec(name="rolling_cache8", mode="rolling", max_loaded_shards=8, preload_flat_shards=False),
        VariantSpec(name="rolling_cache0", mode="rolling", max_loaded_shards=0, preload_flat_shards=True),
    )


def run_variant(
    variant: VariantSpec,
    *,
    command: CommandSpec,
    cwd: Path,
    benchmark_dir: Path,
    stop_step: int,
    rolling_workers: int,
    rolling_worker_threads: int,
    rolling_block_size: int,
    rolling_blocks: int,
    dry_run: bool,
) -> dict[str, Any]:
    rewritten = rewrite_command(
        command,
        variant,
        rolling_workers=rolling_workers,
        rolling_worker_threads=rolling_worker_threads,
        rolling_block_size=rolling_block_size,
        rolling_blocks=rolling_blocks,
    )
    variant_dir = ensure_directory(benchmark_dir / variant.name)
    command_text = _command_string(rewritten)
    (variant_dir / "command.sh").write_text(command_text + "\n", encoding="utf-8")
    if dry_run:
        result = {
            "variant": asdict(variant),
            "command": command_text,
            "cwd": str(cwd),
            "dry_run": True,
        }
        write_json(variant_dir / "result.json", result)
        return result

    env = dict(os.environ)
    env.update(rewritten.env_updates)
    log_path = variant_dir / "stdout.log"
    started_at = time.time()
    run_name = _get_value_flag(list(rewritten.argv), "--run-name") or variant.name
    output_root = _resolve_output_root(rewritten, cwd)
    progress_rows: list[dict[str, Any]] = []
    interesting_lines: list[str] = []
    stop_requested = False
    process = subprocess.Popen(
        rewritten.argv,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        start_new_session=(os.name != "nt"),
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log_handle:
        for raw_line in process.stdout:
            log_handle.write(raw_line)
            log_handle.flush()
            sys.stdout.write(f"[{variant.name}] {raw_line}")
            sys.stdout.flush()
            line = raw_line.rstrip("\n")
            if any(marker in line for marker in INTERESTING_MARKERS):
                interesting_lines.append(line)
            progress = _parse_progress_line(line)
            if progress is not None:
                progress["wall_time_s"] = time.time() - started_at
                progress_rows.append(progress)
                if not stop_requested and progress["step"] >= stop_step:
                    stop_requested = True
                    process.terminate()
    if process.poll() is None:
        _terminate_process(process)
    returncode = process.wait()
    finished_at = time.time()
    run_dir = _locate_run_dir(output_root, run_name, started_at)
    result = {
        "variant": asdict(variant),
        "command": command_text,
        "cwd": str(cwd),
        "log_path": str(log_path),
        "run_name": run_name,
        "output_root": str(output_root),
        "run_dir": None if run_dir is None else str(run_dir),
        "tensorboard_dir": None if run_dir is None else str(run_dir / "tensorboard"),
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": finished_at - started_at,
        "returncode": returncode,
        "stopped_at_target_step": stop_requested,
        "target_stop_step": stop_step,
        "progress_rows": progress_rows,
        "interesting_lines": interesting_lines,
    }
    write_json(variant_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run causal-memory prefetch/cache benchmark variants.")
    parser.add_argument("--command")
    parser.add_argument("--command-file")
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--benchmark-root", default="artifacts/benchmarks/prefetch")
    parser.add_argument("--stop-step", type=int, default=125)
    parser.add_argument("--rolling-workers", type=int, default=2)
    parser.add_argument("--rolling-worker-threads", type=int, default=1)
    parser.add_argument("--rolling-block-size", type=int, default=16)
    parser.add_argument("--rolling-blocks", type=int, default=4)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_command(args: argparse.Namespace) -> CommandSpec:
    if bool(args.command) == bool(args.command_file):
        raise ValueError("Provide exactly one of --command or --command-file.")
    command_text = args.command
    if command_text is None:
        command_text = Path(args.command_file).read_text(encoding="utf-8")
    return parse_command_text(command_text)


def main() -> None:
    args = parse_args()
    command = _load_command(args)
    cwd = Path(args.cwd).resolve()
    benchmark_root = ensure_directory(Path(args.benchmark_root))
    benchmark_dir = create_run_directory(benchmark_root, "causal-memory-prefetch-benchmark")
    variants = {variant.name: variant for variant in _default_variants()}
    selected = list(args.variant) if args.variant else list(variants)
    unknown = sorted(set(selected) - set(variants))
    if unknown:
        raise ValueError(f"Unknown variants requested: {', '.join(unknown)}")
    results = []
    for name in selected:
        results.append(
            run_variant(
                variants[name],
                command=command,
                cwd=cwd,
                benchmark_dir=benchmark_dir,
                stop_step=max(1, int(args.stop_step)),
                rolling_workers=max(1, int(args.rolling_workers)),
                rolling_worker_threads=max(1, int(args.rolling_worker_threads)),
                rolling_block_size=max(1, int(args.rolling_block_size)),
                rolling_blocks=max(1, int(args.rolling_blocks)),
                dry_run=bool(args.dry_run),
            )
        )
    write_json(
        benchmark_dir / "manifest.json",
        {
            "cwd": str(cwd),
            "benchmark_dir": str(benchmark_dir),
            "command": _command_string(command),
            "variants": [result["variant"]["name"] for result in results],
            "results": results,
        },
    )
    print(f"benchmark_manifest={benchmark_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
