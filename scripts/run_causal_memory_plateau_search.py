from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PROGRESS_RE = re.compile(
    r"progress \| step=\s*(?P<step>\d+)/(?P<total>\d+).*?"
    r"train_loss=(?P<loss>[-+0-9.eE]+).*?lr=(?P<lr>[-+0-9.eE]+)"
)
EVAL_RE = re.compile(
    r"eval \| step=(?P<step>\d+).*?"
    r"train_loss=(?P<train_loss>[-+0-9.eE]+).*?"
    r"val_loss=(?P<val_loss>[-+0-9.eE]+)"
)


SUMMARY_LOCK = threading.Lock()


@dataclass(frozen=True)
class ScheduleCandidate:
    name: str
    start_lr: float
    max_lr: float
    warmup_steps: int
    decay_kind: str
    min_lr: float
    optimizer: str


@dataclass
class TrialState:
    status: str = "running"
    reason: str = ""
    last_step: int = 0
    last_lr: float | None = None
    band_enter_step: int | None = None
    best_observed_step: int | None = None
    best_observed_loss: float | None = None
    best_train_loss: float | None = None
    best_val_loss: float | None = None


def parse_float_list(values: str) -> list[float]:
    return [float(value.strip()) for value in values.split(",") if value.strip()]


def parse_int_list(values: str) -> list[int]:
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def parse_str_list(values: str) -> list[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def format_lr(value: float) -> str:
    return f"{value:.0e}".replace("-", "m")


def build_grid(args: argparse.Namespace) -> list[ScheduleCandidate]:
    candidates: list[ScheduleCandidate] = []
    for start_lr, max_lr, warmup_steps, decay_kind, min_lr, optimizer in itertools.product(
        parse_float_list(args.start_lrs),
        parse_float_list(args.max_lrs),
        parse_int_list(args.warmup_steps_grid),
        parse_str_list(args.decay_kinds),
        parse_float_list(args.min_lrs),
        parse_str_list(args.optimizers),
    ):
        if min_lr > max_lr:
            continue
        name = (
            f"start{format_lr(start_lr)}_max{format_lr(max_lr)}_"
            f"wu{warmup_steps}_{decay_kind}_min{format_lr(min_lr)}_{optimizer}"
        )
        candidates.append(
            ScheduleCandidate(
                name=name,
                start_lr=start_lr,
                max_lr=max_lr,
                warmup_steps=warmup_steps,
                decay_kind=decay_kind,
                min_lr=min_lr,
                optimizer=optimizer,
            )
        )
    if args.max_trials > 0:
        candidates = candidates[: args.max_trials]
    return candidates


def axis_values(args: argparse.Namespace) -> dict[str, list[float | int | str]]:
    return {
        "start_lr": parse_float_list(args.start_lrs),
        "max_lr": parse_float_list(args.max_lrs),
        "warmup_steps": parse_int_list(args.warmup_steps_grid),
        "decay_kind": parse_str_list(args.decay_kinds),
        "min_lr": parse_float_list(args.min_lrs),
        "optimizer": parse_str_list(args.optimizers),
    }


def candidate_from_indices(
    axes: dict[str, list[float | int | str]],
    indices: tuple[int, int, int, int, int, int],
) -> ScheduleCandidate | None:
    start_lr = float(axes["start_lr"][indices[0]])
    max_lr = float(axes["max_lr"][indices[1]])
    warmup_steps = int(axes["warmup_steps"][indices[2]])
    decay_kind = str(axes["decay_kind"][indices[3]])
    min_lr = float(axes["min_lr"][indices[4]])
    optimizer = str(axes["optimizer"][indices[5]])
    if min_lr > max_lr:
        return None
    name = (
        f"start{format_lr(start_lr)}_max{format_lr(max_lr)}_"
        f"wu{warmup_steps}_{decay_kind}_min{format_lr(min_lr)}_{optimizer}"
    )
    return ScheduleCandidate(
        name=name,
        start_lr=start_lr,
        max_lr=max_lr,
        warmup_steps=warmup_steps,
        decay_kind=decay_kind,
        min_lr=min_lr,
        optimizer=optimizer,
    )


def candidate_key(candidate: ScheduleCandidate) -> tuple[float, float, int, str, float, str]:
    return (
        candidate.start_lr,
        candidate.max_lr,
        candidate.warmup_steps,
        candidate.decay_kind,
        candidate.min_lr,
        candidate.optimizer,
    )


def center_indices(axes: dict[str, list[float | int | str]]) -> tuple[int, int, int, int, int, int]:
    names = ("start_lr", "max_lr", "warmup_steps", "decay_kind", "min_lr", "optimizer")
    return tuple(len(axes[name]) // 2 for name in names)  # type: ignore[return-value]


def probe_indices(
    axes: dict[str, list[float | int | str]],
    center: tuple[int, int, int, int, int, int],
    radius: int,
) -> list[tuple[int, int, int, int, int, int]]:
    names = ("start_lr", "max_lr", "warmup_steps", "decay_kind", "min_lr", "optimizer")
    probes = [center]
    for axis, name in enumerate(names):
        for direction in (-1, 1):
            index = list(center)
            index[axis] = max(0, min(len(axes[name]) - 1, index[axis] + direction * radius))
            probe = tuple(index)  # type: ignore[arg-type]
            if probe not in probes:
                probes.append(probe)
    return probes


def load_candidates(path: Path) -> list[ScheduleCandidate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("candidate JSON must be a list.")
    return [ScheduleCandidate(**item) for item in payload]


def decay_args(candidate: ScheduleCandidate, args: argparse.Namespace) -> tuple[int, int, float]:
    min_ratio = candidate.min_lr / candidate.max_lr
    if candidate.decay_kind == "constant":
        return args.no_decay_start_step, 1, min_ratio
    if candidate.decay_kind == "cosine":
        return max(0, candidate.warmup_steps), args.decay_steps, min_ratio
    if candidate.decay_kind == "cooldown25":
        return 25, args.decay_steps, min_ratio
    if candidate.decay_kind == "cooldown50":
        return 50, args.decay_steps, min_ratio
    if candidate.decay_kind == "hold80":
        return 80, args.decay_steps, min_ratio
    raise ValueError(f"Unknown decay kind: {candidate.decay_kind}")


def terminate(process: subprocess.Popen[str], timeout: float = 20.0) -> None:
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


def observe_loss(state: TrialState, *, step: int, loss: float, args: argparse.Namespace) -> None:
    if state.best_train_loss is None or loss < state.best_train_loss:
        state.best_train_loss = loss
    if loss <= args.target_upper and state.band_enter_step is None:
        state.band_enter_step = step
        state.best_observed_step = step
        state.best_observed_loss = loss
        return
    if state.band_enter_step is None:
        return
    if state.best_observed_loss is None or loss < state.best_observed_loss - args.min_delta:
        state.best_observed_step = step
        state.best_observed_loss = loss


def should_stop(state: TrialState, *, step: int, args: argparse.Namespace) -> bool:
    if state.best_observed_loss is not None and state.best_observed_loss <= args.target_lower:
        state.status = "success"
        state.reason = f"best_observed_loss={state.best_observed_loss:.4f} <= {args.target_lower:.4f}"
        return True
    if args.max_observed_step > 0 and step >= args.max_observed_step:
        state.status = "max_step"
        state.reason = f"reached max_observed_step={args.max_observed_step}"
        return True
    if state.band_enter_step is None or state.best_observed_step is None:
        return False
    if step < state.band_enter_step + args.min_band_steps:
        return False
    stale_steps = step - state.best_observed_step
    if stale_steps >= args.plateau_patience_steps:
        state.status = "plateau"
        state.reason = f"no improvement >= {args.min_delta:g} for {stale_steps} steps"
        return True
    return False


def train_command(args: argparse.Namespace, candidate: ScheduleCandidate, run_name: str) -> list[str]:
    decay_start, decay_steps, min_ratio = decay_args(candidate, args)
    command = [
        sys.executable,
        "-u",
        "scripts/train_causal_memory_lm.py",
        "--jsonl-source",
        args.jsonl_source,
        "--tokenizer",
        "byte_bpe",
        "--subword-vocab-size",
        str(args.subword_vocab_size),
        "--pretokenize-workers",
        str(args.pretokenize_workers),
        "--device",
        args.device,
        "--precision",
        args.precision,
        "--seq-len",
        str(args.seq_len),
        "--dim",
        str(args.dim),
        "--model-kind",
        args.model_kind,
        "--transformer-layers",
        str(args.transformer_layers),
        "--transformer-heads",
        str(args.transformer_heads),
        "--transformer-ff-mult",
        str(args.transformer_ff_mult),
        "--transformer-dropout",
        str(args.transformer_dropout),
        "--s-layers",
        str(args.s_layers),
        "--memory-slots",
        *[str(value) for value in args.memory_slots],
        "--memory-update-intervals",
        *[str(value) for value in args.memory_update_intervals],
        "--prediction-layers",
        str(args.prediction_layers),
        "--s-window",
        str(args.s_window),
        "--memory-topk",
        str(args.memory_topk),
        "--memory-train-mode",
        "dense",
        "--memory-eval-mode",
        "topk",
        "--eval-topk",
        str(args.eval_topk),
        "--pairwise-kind",
        "low_rank_bilinear",
        "--route-kind",
        "low_rank_bilinear",
        "--pairwise-rank",
        str(args.pairwise_rank),
        "--route-rank",
        str(args.route_rank),
        "--pairwise-heads",
        str(args.pairwise_heads),
        "--route-heads",
        str(args.route_heads),
        "--pairwise-anchor-heads",
        str(args.pairwise_anchor_heads),
        "--route-anchor-heads",
        str(args.route_anchor_heads),
        "--implementation",
        "native",
        "--scan-backend",
        "auto",
        "--enable-fused-training",
        "--enable-scan-backward-cuda",
        "--batch-size",
        str(args.batch_size),
        "--stage1-batch-size",
        str(args.batch_size),
        "--stage2-batch-size",
        str(args.batch_size),
        "--stage3-batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(candidate.max_lr),
        "--warmup-start-lr",
        str(candidate.start_lr),
        "--warmup-steps",
        str(candidate.warmup_steps),
        "--lr-decay-start-step",
        str(decay_start),
        "--lr-decay-steps",
        str(decay_steps),
        "--lr-min-ratio",
        str(min_ratio),
        "--optimizer",
        candidate.optimizer,
        "--epochs",
        str(args.epochs),
        "--eval-start-step",
        str(args.eval_interval),
        "--eval-interval",
        str(args.eval_interval),
        "--checkpoint-interval",
        str(args.checkpoint_interval),
        "--eval-sample-interval",
        str(args.eval_sample_interval),
        "--eval-documents",
        str(args.eval_documents),
        "--curriculum-stage1-ratio",
        str(args.curriculum_stage1_ratio),
        "--curriculum-stage2-ratio",
        str(args.curriculum_stage2_ratio),
        "--curriculum-stage1-span",
        "1",
        "--curriculum-stage2-span",
        "1",
        "--curriculum-stage3-span",
        "1",
        "--diagnose-nonfinite-grad",
        "--diagnose-nonfinite-limit",
        str(args.diagnose_nonfinite_limit),
        "--run-name",
        run_name,
    ]
    if args.tensorboard:
        command.append("--tensorboard")
    return command


def run_candidate(args: argparse.Namespace, candidate: ScheduleCandidate, output_dir: Path, summary_path: Path) -> TrialState:
    run_name = f"{args.run_prefix}_{candidate.name}"
    log_path = output_dir / f"{run_name}.log"
    command = train_command(args, candidate, run_name)
    state = TrialState()
    start_time = time.time()
    print(f"trial_start | run={run_name} | candidate={json.dumps(asdict(candidate), sort_keys=True)}", flush=True)
    if args.dry_run:
        print(" ".join(command), flush=True)
        state.status = "dry_run"
        state.reason = "dry_run"
        return state

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=(os.name != "nt"),
    )
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            progress = PROGRESS_RE.search(line)
            if progress is not None:
                step = int(progress.group("step"))
                loss = float(progress.group("loss"))
                state.last_step = step
                state.last_lr = float(progress.group("lr"))
                observe_loss(state, step=step, loss=loss, args=args)
                if should_stop(state, step=step, args=args):
                    terminate(process)
                    break
            evaluation = EVAL_RE.search(line)
            if evaluation is not None:
                step = int(evaluation.group("step"))
                val_loss = float(evaluation.group("val_loss"))
                state.best_val_loss = val_loss if state.best_val_loss is None else min(state.best_val_loss, val_loss)
                state.last_step = max(state.last_step, step)
                if args.plateau_metric == "val":
                    observe_loss(state, step=step, loss=val_loss, args=args)
                    if should_stop(state, step=step, args=args):
                        terminate(process)
                        break
            if args.max_trial_seconds > 0 and time.time() - start_time >= args.max_trial_seconds:
                state.status = "timeout"
                state.reason = f"reached max_trial_seconds={args.max_trial_seconds:g}"
                terminate(process)
                break
    return_code = process.wait()
    if state.status == "running":
        state.status = "completed" if return_code == 0 else "failed"
        state.reason = f"process_exit={return_code}"
    result = {
        "candidate": asdict(candidate),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "log_path": str(log_path),
        "return_code": return_code,
        "run_name": run_name,
        "state": asdict(state),
    }
    with SUMMARY_LOCK:
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True) + "\n")
    print(f"trial_done | run={run_name} | status={state.status} | reason={state.reason}", flush=True)
    return state


def state_score(state: TrialState) -> float:
    if state.status == "failed":
        return float("inf")
    if state.best_observed_loss is not None:
        score = state.best_observed_loss
    elif state.best_val_loss is not None:
        score = state.best_val_loss
    elif state.best_train_loss is not None:
        score = state.best_train_loss
    else:
        score = float("inf")
    if state.status == "success":
        score -= 1.0
    return score


def run_adaptive_search(args: argparse.Namespace, output_dir: Path, summary_path: Path) -> list[TrialState]:
    axes = axis_values(args)
    center = center_indices(axes)
    radius = max(1, int(args.adaptive_initial_radius))
    seen: set[tuple[float, float, int, str, float, str]] = set()
    states: list[TrialState] = []
    best_score = float("inf")
    best_indices = center
    max_trials = args.max_trials if args.max_trials > 0 else 1_000_000

    for round_index in range(1, args.adaptive_rounds + 1):
        print(
            f"adaptive_round | round={round_index} | center={best_indices} | radius={radius}",
            flush=True,
        )
        round_best_score = float("inf")
        round_best_indices = best_indices
        evaluated_this_round = 0
        round_jobs: list[tuple[tuple[int, int, int, int, int, int], ScheduleCandidate]] = []
        for indices in probe_indices(axes, best_indices, radius):
            candidate = candidate_from_indices(axes, indices)
            if candidate is None:
                continue
            key = candidate_key(candidate)
            if key in seen:
                continue
            if len(states) >= max_trials:
                print(f"adaptive_stop | reason=max_trials | max_trials={max_trials}", flush=True)
                return states
            seen.add(key)
            round_jobs.append((indices, candidate))
            if len(states) + len(round_jobs) >= max_trials:
                break
        completed_jobs = run_indexed_jobs_by_optimizer(args, round_jobs, output_dir, summary_path)
        for indices, state in completed_jobs:
            evaluated_this_round += 1
            states.append(state)
            score = state_score(state)
            if score < best_score:
                best_score = score
                best_indices = indices
            if score < round_best_score:
                round_best_score = score
                round_best_indices = indices
            if state.status == "success" and args.stop_on_success:
                print("adaptive_stop | reason=success", flush=True)
                return states
        if evaluated_this_round == 0:
            if radius <= 1:
                print("adaptive_stop | reason=no_new_candidates", flush=True)
                break
            radius = max(1, radius // 2)
            continue
        if round_best_indices == best_indices and radius > 1:
            radius = max(1, radius // 2)
        else:
            best_indices = round_best_indices
    return states


def run_indexed_jobs_by_optimizer(
    args: argparse.Namespace,
    jobs: list[tuple[tuple[int, int, int, int, int, int], ScheduleCandidate]],
    output_dir: Path,
    summary_path: Path,
) -> list[tuple[tuple[int, int, int, int, int, int], TrialState]]:
    if args.parallel_workers <= 1:
        return [
            (indices, run_candidate(args, candidate, output_dir, summary_path))
            for indices, candidate in jobs
        ]
    pending = list(jobs)
    completed: list[tuple[tuple[int, int, int, int, int, int], TrialState]] = []
    while pending:
        optimizer_seen: set[str] = set()
        wave: list[tuple[tuple[int, int, int, int, int, int], ScheduleCandidate]] = []
        remaining: list[tuple[tuple[int, int, int, int, int, int], ScheduleCandidate]] = []
        for indices, candidate in pending:
            if len(wave) < args.parallel_workers and candidate.optimizer not in optimizer_seen:
                optimizer_seen.add(candidate.optimizer)
                wave.append((indices, candidate))
            else:
                remaining.append((indices, candidate))
        print(
            "parallel_wave | "
            + " ".join(candidate.optimizer for _, candidate in wave),
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=len(wave)) as executor:
            futures = {
                executor.submit(run_candidate, args, candidate, output_dir, summary_path): indices
                for indices, candidate in wave
            }
            for future in as_completed(futures):
                completed.append((futures[future], future.result()))
        pending = remaining
    return completed


def run_grid_search(
    args: argparse.Namespace,
    candidates: list[ScheduleCandidate],
    output_dir: Path,
    summary_path: Path,
) -> list[TrialState]:
    indexed_jobs = [
        ((index, 0, 0, 0, 0, 0), candidate)
        for index, candidate in enumerate(candidates)
    ]
    return [
        state
        for _, state in run_indexed_jobs_by_optimizer(args, indexed_jobs, output_dir, summary_path)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-mode", choices=("grid", "adaptive"), default="grid")
    parser.add_argument("--run-prefix", default="plateau7x")
    parser.add_argument("--output-dir", default="artifacts/training_logs/plateau_search")
    parser.add_argument("--candidates-json")
    parser.add_argument("--start-lrs", default="1e-5,5e-5,1e-4")
    parser.add_argument("--max-lrs", default="1e-4,2e-4,3e-4,4e-4,5e-4,6e-4")
    parser.add_argument("--warmup-steps-grid", default="5,10,20")
    parser.add_argument("--decay-kinds", default="cosine,cooldown25,cooldown50,hold80")
    parser.add_argument("--min-lrs", default="5e-5,1e-4,1.5e-4,2e-4,2.5e-4")
    parser.add_argument("--optimizers", default="lion,adafactor,rmsprop,sgd_nesterov,adagrad,adamw_fused")
    parser.add_argument("--max-trials", type=int, default=48)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--adaptive-rounds", type=int, default=6)
    parser.add_argument("--adaptive-initial-radius", type=int, default=2)
    parser.add_argument("--stop-on-success", action="store_true")
    parser.add_argument("--decay-steps", type=int, default=180)
    parser.add_argument("--no-decay-start-step", type=int, default=1_000_000)
    parser.add_argument("--jsonl-source", default="artifacts/data_small/plain_dialogue_20k.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--subword-vocab-size", type=int, default=16384)
    parser.add_argument("--pretokenize-workers", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--model-kind", choices=("causal_memory", "transformer"), default="causal_memory")
    parser.add_argument("--transformer-layers", type=int, default=5)
    parser.add_argument("--transformer-heads", type=int, default=6)
    parser.add_argument("--transformer-ff-mult", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--s-layers", type=int, default=6)
    parser.add_argument("--memory-slots", type=int, nargs="+", default=[384, 96, 24])
    parser.add_argument("--memory-update-intervals", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--prediction-layers", type=int, default=3)
    parser.add_argument("--s-window", type=int, default=256)
    parser.add_argument("--memory-topk", type=int, default=16)
    parser.add_argument("--eval-topk", type=int, default=16)
    parser.add_argument("--pairwise-rank", type=int, default=128)
    parser.add_argument("--route-rank", type=int, default=96)
    parser.add_argument("--pairwise-heads", type=int, default=4)
    parser.add_argument("--route-heads", type=int, default=4)
    parser.add_argument("--pairwise-anchor-heads", type=int, default=0)
    parser.add_argument("--route-anchor-heads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument("--eval-sample-interval", type=int, default=1000)
    parser.add_argument("--eval-documents", type=int, default=8)
    parser.add_argument("--curriculum-stage1-ratio", type=float, default=0.03)
    parser.add_argument("--curriculum-stage2-ratio", type=float, default=0.05)
    parser.add_argument("--diagnose-nonfinite-limit", type=int, default=6)
    parser.add_argument("--target-upper", type=float, default=7.5)
    parser.add_argument("--target-lower", type=float, default=7.0)
    parser.add_argument("--min-delta", type=float, default=0.03)
    parser.add_argument("--min-band-steps", type=int, default=25)
    parser.add_argument("--plateau-patience-steps", type=int, default=50)
    parser.add_argument("--plateau-metric", choices=("train", "val"), default="train")
    parser.add_argument("--max-observed-step", type=int, default=180)
    parser.add_argument("--max-trial-seconds", type=float, default=0.0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(Path(args.candidates_json)) if args.candidates_json else build_grid(args)
    manifest_path = output_dir / f"{args.run_prefix}_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "axes": axis_values(args),
                "candidates": [asdict(item) for item in candidates],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary_path = output_dir / f"{args.run_prefix}_summary.jsonl"
    if summary_path.exists() and not args.dry_run:
        summary_path.unlink()
    if args.search_mode == "adaptive" and args.candidates_json:
        raise ValueError("--search-mode adaptive cannot be combined with --candidates-json.")
    if args.search_mode == "adaptive":
        states = run_adaptive_search(args, output_dir, summary_path)
    else:
        states = run_grid_search(args, candidates, output_dir, summary_path)
    return 0 if all(state.status != "failed" for state in states) else 1


if __name__ == "__main__":
    raise SystemExit(main())
