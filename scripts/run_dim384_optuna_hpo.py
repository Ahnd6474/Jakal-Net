from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from jakal_net import resolve_device
from progressive_b_example import split_train_val
from train_progressive_b_lm import (
    DEFAULT_TEXT,
    TrialPrunedError,
    build_tokenizer,
    load_text_corpus,
    run_single_experiment,
)


def _load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "optuna is required for scripts/run_dim384_optuna_hpo.py. "
            "Install it with `pip install optuna` on the target environment."
        ) from exc
    return optuna


def build_base_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        device=args.device,
        text_file=None,
        text_source=[],
        jsonl_source=[],
        jsonl_text_key=["text"],
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_text_key=args.hf_text_key,
        hf_streaming=args.hf_streaming,
        corpus_max_samples=args.corpus_max_samples,
        corpus_max_chars=args.corpus_max_chars,
        corpus_separator="\n\n",
        training_objective=args.training_objective,
        tokenizer="byte_bpe",
        subword_vocab_size=args.subword_vocab_size,
        subword_character_coverage=args.subword_character_coverage,
        subword_model_type="bpe",
        tokenizer_prefix=args.tokenizer_prefix,
        steps=args.steps,
        epochs=None,
        eval_interval=args.eval_interval,
        eval_steps=args.eval_steps,
        log_interval=args.log_interval,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        data_workers=args.data_workers,
        prefetch_factor=args.prefetch_factor,
        seq_len=args.seq_len,
        model_preset="custom",
        sweep_presets=None,
        dim=384,
        warmup_layers=3,
        final_refine_layers=3,
        lite_layers=5,
        mid_layers=5,
        full_layers=0,
        lite_expand_ratio=1.05,
        lite_compress_ratio=0.90,
        lite_alpha_b=1.0,
        lite_beta_s_to_b=0.2,
        lite_beta_b_to_s=0.1,
        mid_expand_ratio=1.10,
        mid_compress_ratio=0.80,
        mid_alpha_b=1.0,
        mid_beta_s_to_b=0.2,
        mid_beta_b_to_s=0.1,
        full_expand_ratio=1.20,
        full_compress_ratio=0.70,
        full_alpha_b=1.0,
        full_beta_s_to_b=0.2,
        full_beta_b_to_s=0.1,
        learning_rate=3e-4,
        weight_decay=1e-2,
        route_topk=args.route_topk,
        value_norm_kind="rmsnorm",
        norm_position="pre",
        value_residual_scale=0.75,
        state_residual_scale=0.1,
        state_init_mode="neg_half",
        s_window=32,
        route_temperature=1.0,
        route_kind="diagonal_bilinear",
        route_hidden_dim=192,
        pairwise_kind="diagonal_bilinear",
        pairwise_hidden_dim=None,
        precision=args.precision,
        implementation=args.implementation,
        sequence_propagation="window",
        expanded_propagation="topk",
        compressed_propagation="topk",
        route_mode="topk",
        expanded_window=None,
        compressed_window=None,
        disable_layer_norm=False,
        disable_propagation_residual=False,
        alpha_scale=1.0,
        beta_s_to_b_scale=1.0,
        beta_b_to_s_scale=1.0,
        s_delta_scale=0.25,
        b_delta_scale=0.20,
        cross_delta_scale=0.15,
        b_schedule="constant",
        b_schedule_min=1.0,
        b_schedule_max=1.0,
        seed=args.seed,
        sample_tokens=32,
        prompt_text=None,
        temperature=None,
        sample_topk=None,
        teacher_forcing_chunk_size=None,
        run_name=args.study_name,
        output_dir=str(args.output_dir),
        tensorboard=True,
        tensorboard_dir=str(args.tensorboard_dir),
        save_checkpoint=False,
    )


def apply_search_space(base_args: SimpleNamespace, trial: Any) -> str:
    value_norm = trial.suggest_categorical(
        "value_norm",
        ("rmsnorm_pre", "layernorm_pre", "rmsnorm_post"),
    )
    value_norm_kind, norm_position = value_norm.split("_", maxsplit=1)
    base_args.value_norm_kind = value_norm_kind
    base_args.norm_position = norm_position
    base_args.value_residual_scale = trial.suggest_categorical(
        "alpha_v",
        (0.5, 0.75, 1.0),
    )
    base_args.state_residual_scale = trial.suggest_categorical(
        "alpha_s",
        (0.05, 0.1, 0.2),
    )
    base_args.state_init_mode = trial.suggest_categorical(
        "state_init",
        ("zero", "neg_half"),
    )
    base_args.s_window = trial.suggest_categorical("s_window", (16, 32, 64))
    base_args.route_temperature = trial.suggest_categorical(
        "route_temperature",
        (0.7, 1.0, 1.3),
    )
    lite_b_ratio = trial.suggest_categorical(
        "lite_b_ratio",
        ("expand1.05_compress0.90", "expand1.10_compress0.80", "expand1.00_compress1.00"),
    )
    lite_expand, lite_compress = lite_b_ratio.removeprefix("expand").split("_compress")
    base_args.lite_expand_ratio = float(lite_expand)
    base_args.lite_compress_ratio = float(lite_compress)
    mid_b_ratio = trial.suggest_categorical(
        "mid_b_ratio",
        ("expand1.10_compress0.80", "expand1.20_compress0.70", "expand1.05_compress0.90"),
    )
    mid_expand, mid_compress = mid_b_ratio.removeprefix("expand").split("_compress")
    base_args.mid_expand_ratio = float(mid_expand)
    base_args.mid_compress_ratio = float(mid_compress)
    base_args.b_schedule = trial.suggest_categorical(
        "b_schedule",
        ("constant", "up", "down"),
    )
    if base_args.b_schedule == "constant":
        base_args.b_schedule_min = 1.0
        base_args.b_schedule_max = 1.0
    else:
        base_args.b_schedule_min = trial.suggest_categorical(
            "b_schedule_min",
            (0.50, 0.75),
        )
        base_args.b_schedule_max = trial.suggest_categorical(
            "b_schedule_max",
            (1.00, 1.25, 1.50),
        )
    base_args.beta_s_to_b_scale = trial.suggest_categorical(
        "s_to_b_scale",
        (0.50, 0.75, 1.00, 1.25),
    )
    base_args.beta_b_to_s_scale = trial.suggest_categorical(
        "b_to_s_scale",
        (0.50, 0.75, 1.00, 1.25),
    )
    pairwise_kind = trial.suggest_categorical(
        "pairwise_kind",
        ("diagonal_bilinear", "low_rank_bilinear"),
    )
    base_args.pairwise_kind = pairwise_kind
    route_kind = trial.suggest_categorical(
        "route_kind",
        ("diagonal_bilinear", "low_rank_bilinear"),
    )
    base_args.route_kind = route_kind
    route_hidden_dim = trial.suggest_categorical(
        "route_hidden_dim",
        (64, 128, 192, 256),
    )
    base_args.route_hidden_dim = int(route_hidden_dim)
    base_args.pairwise_hidden_dim = (
        int(route_hidden_dim) if pairwise_kind == "low_rank_bilinear" else None
    )
    base_args.warmup_layers = trial.suggest_categorical("warmup_layers", (2, 3, 4))
    base_args.learning_rate = trial.suggest_categorical(
        "learning_rate",
        (2e-4, 3e-4, 4e-4),
    )
    base_args.run_name = f"{base_args.run_name}_trial_{trial.number:03d}"
    return f"{value_norm}_{pairwise_kind}_{route_kind}_{base_args.b_schedule}"


def make_objective(
    *,
    args: argparse.Namespace,
    device: torch.device | str,
    corpus: Any,
    vocab: Any,
    tokenizer_label: str,
    tokenizer_model_path: Path | None,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
):
    def objective(trial: Any) -> float:
        optuna = _load_optuna()
        trial_args = build_base_namespace(args)
        trial_label = apply_search_space(trial_args, trial)
        trial_slug = f"trial_{trial.number:03d}_{trial_label}"
        session_dir = args.output_dir / args.study_name / trial_slug
        try:
            summary = run_single_experiment(
                args=trial_args,
                session_dir=session_dir,
                experiment_name="custom",
                corpus=corpus,
                vocab=vocab,
                tokenizer_label=tokenizer_label,
                tokenizer_model_path=tokenizer_model_path,
                train_tokens=train_tokens,
                val_tokens=val_tokens,
                device=device,
                teacher_forcing=False,
                full_sequence_causal=False,
                trial=trial,
            )
        except TrialPrunedError as exc:
            raise optuna.TrialPruned(str(exc)) from exc
        except FloatingPointError as exc:
            trial.set_user_attr("nonfinite", True)
            raise optuna.TrialPruned(f"non-finite training value: {exc}") from exc
        except torch.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            trial.set_user_attr("oom", True)
            raise optuna.TrialPruned(f"OOM: {exc}") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                trial.set_user_attr("oom", True)
                raise optuna.TrialPruned(f"OOM: {exc}") from exc
            raise
        trial.set_user_attr("run_dir", str(summary["run_dir"]))
        trial.set_user_attr("tensorboard_dir", str(summary["tensorboard_dir"]))
        trial.set_user_attr("best_step", int(summary["best_step"]))
        trial.set_user_attr("parameter_count", int(summary["parameter_count"]))
        return float(summary["best_val_loss"])

    return objective


def query_gpu_headroom() -> tuple[int, int, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 100, 0, 0
    first_line = result.stdout.strip().splitlines()[0]
    util_str, used_str, total_str = [part.strip() for part in first_line.split(",")]
    return int(util_str), int(used_str), int(total_str)


def choose_parallel_workers(args: argparse.Namespace) -> int:
    if args.max_parallel_workers <= 1:
        return 1
    util, used, total = query_gpu_headroom()
    if total <= 0:
        return 1
    if util < 50 and used / total < 0.7:
        return min(2, args.max_parallel_workers)
    return 1


def spawn_parallel_workers(args: argparse.Namespace, worker_count: int) -> int:
    trial_counts = [args.trials // worker_count] * worker_count
    for index in range(args.trials % worker_count):
        trial_counts[index] += 1
    processes: list[subprocess.Popen[str]] = []
    script_path = Path(__file__).resolve()
    for worker_index, worker_trials in enumerate(trial_counts):
        if worker_trials <= 0:
            continue
        command = [
            sys.executable,
            str(script_path),
            "--worker-mode",
            "--worker-index",
            str(worker_index),
            "--trials",
            str(worker_trials),
            "--study-name",
            args.study_name,
            "--storage",
            args.storage,
            "--device",
            args.device,
            "--hf-dataset",
            args.hf_dataset,
            "--hf-split",
            args.hf_split,
            "--hf-text-key",
            args.hf_text_key,
            "--subword-vocab-size",
            str(args.subword_vocab_size),
            "--subword-character-coverage",
            str(args.subword_character_coverage),
            "--seq-len",
            str(args.seq_len),
            "--batch-size",
            str(args.batch_size),
            "--grad-accum-steps",
            str(args.grad_accum_steps),
            "--data-workers",
            str(args.data_workers),
            "--prefetch-factor",
            str(args.prefetch_factor),
            "--route-topk",
            str(args.route_topk),
            "--steps",
            str(args.steps),
            "--eval-interval",
            str(args.eval_interval),
            "--eval-steps",
            str(args.eval_steps),
            "--log-interval",
            str(args.log_interval),
            "--seed",
            str(args.seed + worker_index),
            "--output-dir",
            str(args.output_dir),
            "--tensorboard-dir",
            str(args.tensorboard_dir),
            "--implementation",
            args.implementation,
            "--precision",
            args.precision,
        ]
        if args.hf_config:
            command.extend(["--hf-config", args.hf_config])
        if args.hf_streaming:
            command.append("--hf-streaming")
        if args.corpus_max_samples is not None:
            command.extend(["--corpus-max-samples", str(args.corpus_max_samples)])
        if args.corpus_max_chars is not None:
            command.extend(["--corpus-max-chars", str(args.corpus_max_chars)])
        if args.tokenizer_prefix:
            command.extend(["--tokenizer-prefix", args.tokenizer_prefix])
        processes.append(subprocess.Popen(command))
    exit_code = 0
    for process in processes:
        code = process.wait()
        if code != 0:
            exit_code = code
    return exit_code


def run_worker(args: argparse.Namespace) -> int:
    optuna = _load_optuna()
    if args.storage.startswith("sqlite:///"):
        sqlite_path = Path(args.storage.removeprefix("sqlite:///"))
        if not sqlite_path.is_absolute():
            sqlite_path = (Path.cwd() / sqlite_path).resolve()
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        args.storage = f"sqlite:///{sqlite_path.as_posix()}"
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=max(1, args.eval_interval),
        max_resource=args.steps,
        reduction_factor=3,
    )
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=max(4, min(6, args.trials // 3)),
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    device = resolve_device(args.device)
    if args.tokenizer_prefix is None:
        tokenizer_prefix = (
            Path.cwd()
            / "artifacts"
            / "tokenizers"
            / f"{args.study_name}_byte_bpe_{args.subword_vocab_size}_{args.corpus_max_chars or 'full'}"
        )
        tokenizer_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.tokenizer_prefix = str(tokenizer_prefix)
    corpus = load_text_corpus(
        default_text=DEFAULT_TEXT,
        text_file=None,
        text_sources=[],
        jsonl_sources=[],
        jsonl_text_keys=("text",),
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_text_key=args.hf_text_key,
        hf_streaming=args.hf_streaming,
        max_samples=args.corpus_max_samples,
        max_chars=args.corpus_max_chars,
        separator="\n\n",
    )
    tokenizer_prefix = args.tokenizer_prefix
    vocab, tokenizer_label, tokenizer_model_path = build_tokenizer(
        corpus.text,
        text_path=None if corpus.text_path is None else str(corpus.text_path),
        tokenizer="byte_bpe",
        subword_vocab_size=args.subword_vocab_size,
        subword_character_coverage=args.subword_character_coverage,
        subword_model_type="bpe",
        tokenizer_prefix=tokenizer_prefix,
    )
    tokens = vocab.encode(corpus.text)
    train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.9)

    objective = make_objective(
        args=args,
        device=device,
        corpus=corpus,
        vocab=vocab,
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
        train_tokens=train_tokens,
        val_tokens=val_tokens,
    )
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    summary = {
        "study_name": args.study_name,
        "best_trial_number": None,
        "best_value": None,
        "best_params": None,
        "sampler": "TPESampler",
        "pruner": "HyperbandPruner",
        "trial_count": len(study.trials),
        "completed_trials": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ),
        "pruned_trials": sum(
            1 for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED
        ),
    }
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if complete_trials:
        best_trial = min(complete_trials, key=lambda trial: float(trial.value))
        summary["best_trial_number"] = best_trial.number
        summary["best_value"] = best_trial.value
        summary["best_params"] = best_trial.params
    summary_path = args.output_dir / args.study_name / "optuna_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-dataset", default="wikitext")
    parser.add_argument("--hf-config", default="wikitext-103-raw-v1")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--corpus-max-samples", type=int)
    parser.add_argument("--corpus-max-chars", type=int, default=1000000)
    parser.add_argument("--training-objective", default="last_token")
    parser.add_argument("--subword-vocab-size", type=int, default=1024)
    parser.add_argument("--subword-character-coverage", type=float, default=0.9995)
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--route-topk", type=int, default=128)
    parser.add_argument("--steps", type=int, default=6400)
    parser.add_argument("--eval-interval", type=int, default=800)
    parser.add_argument("--eval-steps", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--study-name", default="dim384_topk128_accum4_eval16_tpe_hyperband_6400")
    parser.add_argument(
        "--storage",
        default="sqlite:///artifacts/optuna/dim384_topk128_accum4_eval16_tpe_hyperband_6400/study.db",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-parallel-workers", type=int, default=1)
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training_runs"))
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=Path("artifacts/tensorboard/dim384_topk128_accum4_eval16_tpe_hyperband_6400"),
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
    )
    parser.add_argument(
        "--implementation",
        choices=("reference", "streaming", "kernel", "native"),
        default="native",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    if not args.worker_mode:
        worker_count = choose_parallel_workers(args)
        print(f"parallel_workers={worker_count}")
        if worker_count > 1:
            return spawn_parallel_workers(args, worker_count)
    exit_code = run_worker(args)
    elapsed = time.perf_counter() - start
    print(f"elapsed_seconds={elapsed:.1f}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
