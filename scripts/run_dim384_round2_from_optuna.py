from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("optuna is required for round2 replay.") from exc
    return optuna


def short_norm_name(value_norm: str) -> str:
    mapping = {
        "layernorm_pre": "lnpre",
        "rmsnorm_pre": "rmspre",
        "rmsnorm_post": "rmspost",
    }
    return mapping.get(value_norm, value_norm.replace("_", ""))


def short_state_init(state_init: str) -> str:
    mapping = {
        "zero": "z",
        "neg_half": "n05",
        "normal": "nrm",
    }
    return mapping.get(state_init, state_init)


def format_decimal(value: float) -> str:
    return str(value).replace(".", "")


def build_run_name(trial_number: int, params: dict[str, object]) -> str:
    return (
        f"round2_t{trial_number:03d}_"
        "dense_"
        f"{short_norm_name(str(params['value_norm']))}_"
        f"av{format_decimal(float(params['alpha_v']))}_"
        f"as{format_decimal(float(params['alpha_s']))}_"
        f"si{short_state_init(str(params['state_init']))}_"
        f"w{int(params['s_window'])}_"
        f"t{format_decimal(float(params['route_temperature']))}_"
        f"wu{int(params['warmup_layers'])}_"
        f"lr{format_decimal(float(params['learning_rate']))}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--study-name",
        default="dim384_round1_random30_seq512",
    )
    parser.add_argument(
        "--storage",
        default="sqlite:///artifacts/optuna/dim384_round1_random30_seq512/study.db",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--steps", type=int, default=640)
    parser.add_argument("--eval-interval", type=int, default=80)
    parser.add_argument("--eval-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="artifacts/training_runs")
    parser.add_argument(
        "--tensorboard-dir",
        default="artifacts/tensorboard/dim384_round2_top5_640",
    )
    parser.add_argument(
        "--tokenizer-prefix",
        default="/workspace/Jakal-Net/artifacts/tokenizers/wiki1m_byte_bpe1024_round1",
    )
    parser.add_argument("--subword-vocab-size", type=int, default=1024)
    parser.add_argument("--subword-character-coverage", type=float, default=0.9995)
    parser.add_argument("--corpus-max-chars", type=int, default=1000000)
    parser.add_argument("--implementation", default="streaming")
    parser.add_argument(
        "--hf-dataset",
        default="wikitext",
    )
    parser.add_argument(
        "--hf-config",
        default="wikitext-103-raw-v1",
    )
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--precision", default="bf16")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    optuna = _load_optuna()
    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    complete_trials.sort(key=lambda trial: float(trial.value))
    selected_trials = complete_trials[: args.top_k]
    if not selected_trials:
        raise RuntimeError("No completed trials found in source study.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "dim384_round2_manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                }
                for trial in selected_trials
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    for rank, trial in enumerate(selected_trials, start=1):
        params = trial.params
        run_name = build_run_name(trial.number, params)
        propagation_topk = int(params.get("route_topk", 4))
        command = [
            sys.executable,
            "scripts/train_progressive_b_lm.py",
            "--device",
            args.device,
            "--hf-dataset",
            args.hf_dataset,
            "--hf-config",
            args.hf_config,
            "--hf-split",
            args.hf_split,
            "--hf-text-key",
            args.hf_text_key,
            "--corpus-max-chars",
            str(args.corpus_max_chars),
            "--training-objective",
            "last_token",
            "--tokenizer",
            "byte_bpe",
            "--subword-vocab-size",
            str(args.subword_vocab_size),
            "--subword-character-coverage",
            str(args.subword_character_coverage),
            "--subword-model-type",
            "bpe",
            "--tokenizer-prefix",
            args.tokenizer_prefix,
            "--steps",
            str(args.steps),
            "--eval-interval",
            str(args.eval_interval),
            "--eval-steps",
            str(args.eval_steps),
            "--log-interval",
            "80",
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--model-preset",
            "custom",
            "--dim",
            "384",
            "--warmup-layers",
            str(int(params["warmup_layers"])),
            "--final-refine-layers",
            "3",
            "--lite-layers",
            "5",
            "--mid-layers",
            "5",
            "--full-layers",
            "0",
            "--lite-expand-ratio",
            "1.05",
            "--lite-compress-ratio",
            "0.90",
            "--lite-alpha-b",
            "1.0",
            "--lite-beta-s-to-b",
            "0.2",
            "--lite-beta-b-to-s",
            "0.1",
            "--mid-expand-ratio",
            "1.10",
            "--mid-compress-ratio",
            "0.80",
            "--mid-alpha-b",
            "1.0",
            "--mid-beta-s-to-b",
            "0.2",
            "--mid-beta-b-to-s",
            "0.1",
            "--learning-rate",
            str(float(params["learning_rate"])),
            "--weight-decay",
            "1e-2",
            "--route-topk",
            str(propagation_topk),
            "--value-norm-kind",
            str(params["value_norm"]).split("_", maxsplit=1)[0],
            "--norm-position",
            str(params["value_norm"]).split("_", maxsplit=1)[1],
            "--value-residual-scale",
            str(float(params["alpha_v"])),
            "--state-residual-scale",
            str(float(params["alpha_s"])),
            "--state-init-mode",
            str(params["state_init"]),
            "--s-window",
            str(int(params["s_window"])),
            "--route-temperature",
            str(float(params["route_temperature"])),
            "--pairwise-kind",
            "diagonal_bilinear",
            "--precision",
            args.precision,
            "--implementation",
            args.implementation,
            "--sequence-propagation",
            "window",
            "--expanded-propagation",
            "topk",
            "--compressed-propagation",
            "topk",
            "--route-mode",
            "dense",
            "--seed",
            str(args.seed + rank),
            "--run-name",
            run_name,
            "--output-dir",
            args.output_dir,
            "--tensorboard",
            "--tensorboard-dir",
            args.tensorboard_dir,
        ]
        print(f"[round2] rank={rank} trial={trial.number} value={trial.value} run_name={run_name}")
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
