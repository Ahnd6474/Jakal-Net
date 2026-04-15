from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from jakal_net import describe_device, resolve_device
from lm_experiment_utils import (
    MODEL_SIZE_PRESETS,
    CorpusLoadResult,
    append_jsonl,
    create_run_directory,
    ensure_directory,
    load_text_corpus,
    parse_preset_names,
    resolve_model_scale_preset,
    slugify_run_name,
    to_jsonable,
    write_csv_rows,
    write_json,
    write_jsonl,
)
from progressive_b_example import (
    ProgressiveBExampleLM,
    TrainingHistory,
    build_char_vocab,
    build_progressive_b_stage_specs,
    perplexity_from_loss,
    sample_prefix_response_batch,
    sample_next_token_batch,
    split_train_val,
    train_prefix_response_model,
    train_next_token_model,
)

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


DEFAULT_TEXT = """
Jakal-Net explores propagation and transition as separate operators.
Progressive B activation starts with stable sequence structure in S.
Then the B path grows from light compression to stronger hierarchical bottlenecks.
The final readout returns to S so token-level prediction remains anchored.
""".strip()
DEFAULT_TOKENIZER_CACHE_DIR = Path(tempfile.gettempdir()) / "jakal_net_tokenizers"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
DIALOGUE_SPECIAL_TOKENS = (USER_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN, PAD_TOKEN)
DEFAULT_DIALOGUE_MESSAGES = (
    (
        {"role": "user", "content": "오늘 학습 상태를 간단히 요약해줘."},
        {"role": "assistant", "content": "현재 손실, 검증 지표, GPU 사용률을 함께 보고 병목을 구분하면 됩니다."},
    ),
    (
        {"role": "user", "content": "라우팅이 한쪽으로 몰리면 어떻게 확인해?"},
        {"role": "assistant", "content": "topk overlap, dead slot ratio, destination load variance를 같이 보면 됩니다."},
    ),
    (
        {"role": "user", "content": "모델이 plateau에 들어간 것 같아."},
        {"role": "assistant", "content": "minibatch loss보다 eval loss의 저점과 반등 시점을 먼저 확인하는 게 맞습니다."},
    ),
)


class TrialPrunedError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DialoguePairText:
    prefix: str
    response: str


@dataclass(frozen=True, slots=True)
class SubwordVocab:
    processor: object
    model_path: Path
    model_type: str

    @property
    def size(self) -> int:
        return int(self.processor.get_piece_size())

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.processor.encode(text, out_type=int)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.processor.decode(list(map(int, token_ids))))

    def token_id(self, piece: str) -> int:
        idx = int(self.processor.piece_to_id(piece))
        if idx < 0 or str(self.processor.id_to_piece(idx)) != piece:
            raise ValueError(
                f"Tokenizer {self.model_path} does not contain required piece {piece!r}."
            )
        return idx


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported precision: {precision!r}.")


def resolve_b_schedule_scale(
    *,
    schedule: str,
    step: int,
    total_steps: int,
    min_scale: float,
    max_scale: float,
) -> float:
    if total_steps <= 1 or schedule == "constant":
        return max_scale
    progress = (step - 1) / (total_steps - 1)
    if schedule == "up":
        return min_scale + (max_scale - min_scale) * progress
    if schedule == "down":
        return max_scale - (max_scale - min_scale) * progress
    raise ValueError(f"Unsupported B schedule: {schedule!r}.")


def estimate_steps_per_epoch(
    *,
    token_count: int,
    seq_len: int,
    batch_size: int,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    contexts_per_epoch = token_count - seq_len
    if contexts_per_epoch <= 0:
        raise ValueError("The corpus must be longer than seq_len.")
    return max(1, math.ceil(contexts_per_epoch / batch_size))


def resolve_tokenizer_prefix(
    *,
    text: str,
    tokenizer: str,
    model_type: str,
    vocab_size: int,
    prefix: str | None,
) -> Path:
    if prefix is not None:
        prefix_path = Path(prefix)
    else:
        corpus_digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
        prefix_path = DEFAULT_TOKENIZER_CACHE_DIR / (
            f"progressive_b_{tokenizer}_{model_type}_{vocab_size}_{corpus_digest}"
        )
    if prefix_path.suffix:
        prefix_path = prefix_path.with_suffix("")
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    return prefix_path


def ensure_tokenizer_training_text(
    *,
    text: str,
    text_path: str | None,
    tokenizer_prefix: Path,
) -> Path:
    if text_path is not None:
        return Path(text_path)
    training_text_path = tokenizer_prefix.parent / f"{tokenizer_prefix.name}_corpus.txt"
    if not training_text_path.exists():
        training_text_path.write_text(text, encoding="utf-8")
    return training_text_path


def build_subword_vocab(
    text: str,
    *,
    text_path: str | None,
    vocab_size: int,
    model_type: str,
    tokenizer_prefix: str | None,
    character_coverage: float,
    user_defined_symbols: Sequence[str] = (),
) -> SubwordVocab:
    if spm is None:
        raise ImportError(
            "sentencepiece is required for --tokenizer subword. "
            "Install dependencies from requirements-base.txt first."
        )
    if vocab_size <= 0:
        raise ValueError("subword-vocab-size must be positive.")

    prefix_path = resolve_tokenizer_prefix(
        text=text,
        tokenizer="subword",
        model_type=model_type,
        vocab_size=vocab_size,
        prefix=tokenizer_prefix,
    )
    model_path = prefix_path.with_suffix(".model")
    vocab_path = prefix_path.with_suffix(".vocab")

    if not model_path.exists() or not vocab_path.exists():
        training_text_path = ensure_tokenizer_training_text(
            text=text,
            text_path=text_path,
            tokenizer_prefix=prefix_path,
        )
        spm.SentencePieceTrainer.train(
            input=str(training_text_path),
            model_prefix=str(prefix_path),
            model_type=model_type,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            user_defined_symbols=list(user_defined_symbols),
            bos_id=-1,
            eos_id=-1,
            pad_id=-1,
            hard_vocab_limit=False,
            split_digits=True,
        )

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return SubwordVocab(
        processor=processor,
        model_path=model_path,
        model_type=model_type,
    )


def build_tokenizer(
    text: str,
    *,
    text_path: str | None,
    tokenizer: str,
    subword_vocab_size: int,
    subword_model_type: str,
    tokenizer_prefix: str | None,
    subword_character_coverage: float,
    user_defined_symbols: Sequence[str] = (),
) -> tuple[object, str, Path | None]:
    if tokenizer == "char":
        return build_char_vocab(text), "char", None
    if tokenizer == "subword":
        subword_vocab = build_subword_vocab(
            text,
            text_path=text_path,
            vocab_size=subword_vocab_size,
            model_type=subword_model_type,
            tokenizer_prefix=tokenizer_prefix,
            character_coverage=subword_character_coverage,
            user_defined_symbols=user_defined_symbols,
        )
        return (
            subword_vocab,
            f"subword/{subword_vocab.model_type}",
            subword_vocab.model_path,
        )
    raise ValueError(f"Unsupported tokenizer: {tokenizer!r}.")


def _message_role(message: dict[str, Any]) -> str:
    role = message.get("role") or message.get("from") or message.get("speaker")
    if not isinstance(role, str):
        return ""
    role = role.lower().strip()
    if role in {"human", "user", "prompt"}:
        return "user"
    if role in {"assistant", "gpt", "bot", "response"}:
        return "assistant"
    return role


def _message_content(message: dict[str, Any]) -> str:
    for key in ("content", "value", "text"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pairs_from_messages(messages: Sequence[dict[str, Any]]) -> list[DialoguePairText]:
    pairs: list[DialoguePairText] = []
    history: list[str] = []
    for message in messages:
        role = _message_role(message)
        content = _message_content(message)
        if not content:
            continue
        if role == "user":
            history.append(f"{USER_TOKEN}\n{content}\n")
        elif role == "assistant" and history:
            prefix = "".join(history) + f"{ASSISTANT_TOKEN}\n"
            pairs.append(DialoguePairText(prefix=prefix, response=content))
            history.append(f"{ASSISTANT_TOKEN}\n{content}\n{EOS_TOKEN}\n")
    return pairs


def _pairs_from_record(record: Any) -> list[DialoguePairText]:
    if not isinstance(record, dict):
        return []
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        normalized = [message for message in messages if isinstance(message, dict)]
        pairs = _pairs_from_messages(normalized)
        if pairs:
            return pairs
    prompt = None
    for key in ("prompt", "instruction", "question", "input"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            prompt = value.strip()
            break
    response = None
    for key in ("response", "output", "answer", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            response = value.strip()
            break
    if prompt and response:
        return [
            DialoguePairText(
                prefix=f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n",
                response=response,
            )
        ]
    return []


def _load_dialogue_pairs_from_jsonl(path: Path) -> list[DialoguePairText]:
    pairs: list[DialoguePairText] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            extracted = _pairs_from_record(record)
            if not extracted:
                raise ValueError(f"No dialogue pair found in {path} line {line_number}.")
            pairs.extend(extracted)
    return pairs


def _load_dialogue_pairs_from_hf(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    streaming: bool,
    max_samples: int | None,
) -> list[DialoguePairText]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required for Hugging Face dialogue loading."
        ) from exc
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    pairs: list[DialoguePairText] = []
    rows_seen = 0
    iterator = dataset if streaming else iter(dataset)
    for row in iterator:
        if max_samples is not None and rows_seen >= max_samples:
            break
        rows_seen += 1
        pairs.extend(_pairs_from_record(row))
    return pairs


def load_dialogue_pairs(
    *,
    jsonl_sources: Sequence[str],
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    hf_streaming: bool,
    max_samples: int | None,
    max_chars: int | None,
) -> tuple[list[DialoguePairText], CorpusLoadResult]:
    from lm_experiment_utils import _expand_sources  # Local helper, kept private to this script.

    paths = _expand_sources(jsonl_sources, directory_suffixes=(".jsonl", ".json"))
    pairs: list[DialoguePairText] = []
    for path in paths:
        pairs.extend(_load_dialogue_pairs_from_jsonl(path))
    if hf_dataset is not None:
        pairs.extend(
            _load_dialogue_pairs_from_hf(
                dataset_name=hf_dataset,
                config_name=hf_config,
                split=hf_split,
                streaming=hf_streaming,
                max_samples=max_samples if not pairs else None,
            )
        )
    if not pairs:
        for messages in DEFAULT_DIALOGUE_MESSAGES:
            pairs.extend(_pairs_from_messages(messages))
    if max_samples is not None:
        pairs = pairs[:max_samples]
    if max_chars is not None:
        selected: list[DialoguePairText] = []
        total_chars = 0
        for pair in pairs:
            pair_chars = len(pair.prefix) + len(pair.response) + len(EOS_TOKEN)
            if total_chars + pair_chars > max_chars:
                break
            selected.append(pair)
            total_chars += pair_chars
        pairs = selected
    if not pairs:
        raise ValueError("No dialogue pairs were loaded.")
    text = "\n\n".join(f"{pair.prefix}{pair.response}\n{EOS_TOKEN}" for pair in pairs)
    metadata = {
        "source_kind": "dialogue_pairs",
        "jsonl_paths": [str(path) for path in paths],
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "max_samples": max_samples,
        "max_chars": max_chars,
    }
    corpus = CorpusLoadResult(
        text=text,
        source_label="dialogue_pairs",
        text_path=None,
        sample_count=len(pairs),
        file_count=len(paths),
        char_count=len(text),
        truncated=False,
        metadata=metadata,
    )
    return pairs, corpus


def split_dialogue_pairs(
    pairs: Sequence[DialoguePairText],
    *,
    train_fraction: float = 0.9,
) -> tuple[list[DialoguePairText], list[DialoguePairText]]:
    split_index = int(len(pairs) * train_fraction)
    split_index = max(1, min(split_index, len(pairs) - 1)) if len(pairs) > 1 else len(pairs)
    return list(pairs[:split_index]), list(pairs[split_index:] or pairs[:split_index])


def encode_dialogue_pairs(
    pairs: Sequence[DialoguePairText],
    *,
    vocab: object,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(vocab.encode(pair.prefix), vocab.encode(pair.response)) for pair in pairs]


def require_subword_special_ids(vocab: object) -> dict[str, int]:
    if not isinstance(vocab, SubwordVocab):
        raise ValueError("prefix_response currently requires --tokenizer subword.")
    return {
        "user": vocab.token_id(USER_TOKEN),
        "assistant": vocab.token_id(ASSISTANT_TOKEN),
        "eos": vocab.token_id(EOS_TOKEN),
        "pad": vocab.token_id(PAD_TOKEN),
    }


@torch.no_grad()
def generate_next_tokens_with_sampling(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    seq_len: int,
    device: torch.device | str,
    temperature: float | None,
    sample_topk: int | None,
    training_objective: str = "last_token",
) -> torch.Tensor:
    if temperature is not None and temperature <= 0.0:
        raise ValueError("temperature must be positive when sampling is enabled.")
    if sample_topk is not None and sample_topk <= 0:
        raise ValueError("sample-topk must be positive.")

    was_training = model.training
    model.eval()
    generated = prompt.to(device).clone()
    for _ in range(max_new_tokens):
        context = generated[-seq_len:].unsqueeze(0)
        if training_objective == "full_sequence_causal":
            logits = model(context, full_sequence_causal=True)[..., -1, :]
        else:
            logits = model(context)
        if temperature is None:
            next_token = torch.argmax(logits, dim=-1)
        else:
            scaled_logits = logits / temperature
            if sample_topk is not None:
                k = min(sample_topk, scaled_logits.shape[-1])
                topk_logits, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
                probs = torch.softmax(topk_logits, dim=-1)
                sampled_offset = torch.multinomial(probs, num_samples=1)
                next_token = topk_indices.gather(-1, sampled_offset).reshape(-1)
            else:
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
        generated = torch.cat((generated, next_token), dim=0)
    if was_training:
        model.train()
    return generated


@torch.no_grad()
def generate_prefix_response_with_sampling(
    model: torch.nn.Module,
    prefix: torch.Tensor,
    *,
    response_len: int,
    bos_token_id: int,
    eos_token_id: int,
    device: torch.device | str,
    temperature: float | None,
    sample_topk: int | None,
) -> torch.Tensor:
    if temperature is not None and temperature <= 0.0:
        raise ValueError("temperature must be positive when sampling is enabled.")
    if sample_topk is not None and sample_topk <= 0:
        raise ValueError("sample-topk must be positive.")
    was_training = model.training
    model.eval()
    prefix = prefix.to(device).unsqueeze(0)
    decoder_input = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
    generated: list[torch.Tensor] = []
    for _ in range(response_len):
        logits = model.forward_response(prefix, decoder_input)[:, -1, :]
        if temperature is None:
            next_token = torch.argmax(logits, dim=-1)
        else:
            scaled_logits = logits / temperature
            if sample_topk is not None:
                k = min(sample_topk, scaled_logits.shape[-1])
                topk_logits, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
                probs = torch.softmax(topk_logits, dim=-1)
                sampled_offset = torch.multinomial(probs, num_samples=1)
                next_token = topk_indices.gather(-1, sampled_offset).reshape(-1)
            else:
                probs = torch.softmax(scaled_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
        generated.append(next_token.detach().cpu())
        decoder_input = torch.cat((decoder_input, next_token.view(1, 1)), dim=-1)
        if int(next_token.item()) == eos_token_id:
            break
    if was_training:
        model.train()
    if not generated:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(generated, dim=0)


def resolve_effective_model_config(
    args: argparse.Namespace,
    preset_name: str,
) -> dict[str, int | str]:
    if preset_name == "custom":
        return {
            "model_preset": "custom",
            "dim": args.dim,
            "warmup_layers": args.warmup_layers,
            "final_refine_layers": args.final_refine_layers,
            "lite_layers": args.lite_layers,
            "mid_layers": args.mid_layers,
            "full_layers": args.full_layers,
            "route_topk": args.route_topk,
        }
    preset = resolve_model_scale_preset(preset_name)
    return {
        "model_preset": preset.name,
        "dim": preset.dim,
        "warmup_layers": preset.warmup_layers,
        "final_refine_layers": preset.final_refine_layers,
        "lite_layers": preset.lite_layers,
        "mid_layers": preset.mid_layers,
        "full_layers": preset.full_layers,
        "route_topk": preset.route_topk,
    }


def build_metrics_rows(history: TrainingHistory, *, steps_per_epoch: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step, (loss, grad_norm) in enumerate(
        zip(history.train_step_losses, history.grad_norms, strict=True),
        start=1,
    ):
        rows.append(
            {
                "record_type": "train_step",
                "step": step,
                "epoch": step / steps_per_epoch,
                "minibatch_loss": loss,
                "grad_norm": grad_norm,
            }
        )
    for step, train_loss, val_loss in zip(
        history.eval_steps,
        history.train_losses,
        history.val_losses,
        strict=True,
    ):
        rows.append(
            {
                "record_type": "eval",
                "step": step,
                "epoch": step / steps_per_epoch,
                "train_loss": train_loss,
                "train_ppl": perplexity_from_loss(train_loss),
                "val_loss": val_loss,
                "val_ppl": perplexity_from_loss(val_loss),
            }
        )
    return rows


def maybe_create_summary_writer(
    *,
    enabled: bool,
    tensorboard_dir: Path,
) -> SummaryWriter | None:
    if not enabled:
        return None
    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard logging requested, but torch.utils.tensorboard is unavailable. "
            "Install dependencies from requirements-base.txt first."
        )
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tensorboard_dir))


def log_history_to_tensorboard(
    writer: SummaryWriter | None,
    *,
    history: TrainingHistory,
    steps_per_epoch: int,
    summary: dict[str, Any],
    prompt_text: str,
    generated_text: str,
    include_scalars: bool = True,
) -> None:
    if writer is None:
        return
    if include_scalars:
        for step, loss in enumerate(history.train_step_losses, start=1):
            writer.add_scalar("train/minibatch_loss", loss, step)
            writer.add_scalar("train/epoch", step / steps_per_epoch, step)
        for step, grad_norm in enumerate(history.grad_norms, start=1):
            writer.add_scalar("train/grad_norm", grad_norm, step)
        for step, train_loss, val_loss in zip(
            history.eval_steps,
            history.train_losses,
            history.val_losses,
            strict=True,
        ):
            writer.add_scalar("eval/train_loss", train_loss, step)
            writer.add_scalar("eval/val_loss", val_loss, step)
            writer.add_scalar("eval/train_ppl", perplexity_from_loss(train_loss), step)
            writer.add_scalar("eval/val_ppl", perplexity_from_loss(val_loss), step)
    writer.add_text("sample/prompt", prompt_text)
    writer.add_text("sample/generated", generated_text)
    writer.add_hparams(
        {
            "seq_len": int(summary["seq_len"]),
            "batch_size": int(summary["batch_size"]),
            "dim": int(summary["dim"]),
            "warmup_layers": int(summary["warmup_layers"]),
            "final_refine_layers": int(summary["final_refine_layers"]),
            "lite_layers": int(summary["lite_layers"]),
            "mid_layers": int(summary["mid_layers"]),
            "full_layers": int(summary["full_layers"]),
            "route_topk": int(summary["route_topk"]),
            "learning_rate": float(summary["learning_rate"]),
            "weight_decay": float(summary["weight_decay"]),
            "value_norm_layernorm": bool(summary["value_norm_kind"] == "layernorm"),
            "value_norm_rmsnorm": bool(summary["value_norm_kind"] == "rmsnorm"),
            "norm_position_pre": bool(summary["norm_position"] == "pre"),
            "propagation_residual": bool(summary["propagation_residual"]),
            "route_mode_dense": bool(summary["route_mode"] == "dense"),
            "sequence_dense": bool(summary["sequence_propagation"] == "dense"),
            "expanded_dense": bool(summary["expanded_propagation"] == "dense"),
            "compressed_dense": bool(summary["compressed_propagation"] == "dense"),
            "subword_vocab_size": int(summary["subword_vocab_size"]),
            "value_residual_scale": float(summary["value_residual_scale"]),
            "state_residual_scale": float(summary["state_residual_scale"]),
            "route_temperature": float(summary["route_temperature"]),
            "alpha_scale": float(summary["alpha_scale"]),
            "beta_s_to_b_scale": float(summary["beta_s_to_b_scale"]),
            "beta_b_to_s_scale": float(summary["beta_b_to_s_scale"]),
            "s_delta_scale": float(summary["s_delta_scale"]),
            "b_delta_scale": float(summary["b_delta_scale"]),
            "cross_delta_scale": float(summary["cross_delta_scale"]),
            "b_schedule_min": float(summary["b_schedule_min"]),
            "b_schedule_max": float(summary["b_schedule_max"]),
        },
        {
            "hparam/final_train_loss": float(summary["final_train_loss"]),
            "hparam/final_val_loss": float(summary["final_val_loss"]),
            "hparam/best_val_loss": float(summary["best_val_loss"]),
            "hparam/runtime_seconds": float(summary["runtime_seconds"]),
        },
        run_name="hparams",
    )
    writer.flush()


def print_eval_table(history: TrainingHistory, *, steps_per_epoch: int) -> None:
    for index, (step, train_loss, val_loss) in enumerate(
        zip(
            history.eval_steps,
            history.train_losses,
            history.val_losses,
            strict=True,
        ),
        start=1,
    ):
        print(
            f"eval {index:02d} | step={step:>5d} | epoch={step / steps_per_epoch:.3f} | "
            f"train_loss={train_loss:.4f} | train_ppl={perplexity_from_loss(train_loss):.2f} | "
            f"val_loss={val_loss:.4f} | val_ppl={perplexity_from_loss(val_loss):.2f}"
        )


def select_prompt_tokens(
    *,
    args: argparse.Namespace,
    vocab: object,
    train_tokens: torch.Tensor,
) -> tuple[torch.Tensor, str]:
    if args.prompt_text:
        prompt = args.prompt_text
        prompt_ids = vocab.encode(prompt)
        if prompt_ids.numel() == 0:
            raise ValueError("--prompt-text must encode to at least one token.")
        return prompt_ids, prompt
    prompt_ids = train_tokens[: args.seq_len]
    return prompt_ids, vocab.decode(prompt_ids.tolist())


def select_prefix_response_prompt_tokens(
    *,
    args: argparse.Namespace,
    vocab: object,
    train_response_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, str]:
    if args.prompt_text:
        prompt = args.prompt_text
        if USER_TOKEN not in prompt and ASSISTANT_TOKEN not in prompt:
            prompt = f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n"
        prompt_ids = vocab.encode(prompt)
        if prompt_ids.numel() == 0:
            raise ValueError("--prompt-text must encode to at least one token.")
        return prompt_ids, prompt
    if not train_response_pairs:
        raise ValueError("prefix_response prompt selection requires at least one train pair.")
    prompt_ids = train_response_pairs[0][0]
    return prompt_ids, vocab.decode(prompt_ids.tolist())


def format_tensorboard_sample(
    *,
    prompt_text: str,
    generated_text: str,
) -> str:
    return (
        "### Prompt\n\n"
        f"```\n{prompt_text}\n```\n\n"
        "### Generated\n\n"
        f"```\n{generated_text}\n```"
    )


def save_run_artifacts(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    corpus: CorpusLoadResult,
    tokenizer_label: str,
    tokenizer_model_path: Path | None,
    effective_config: dict[str, int | str],
    parameter_count: int,
    steps_per_epoch: int,
    history: TrainingHistory,
    metrics_rows: Sequence[dict[str, Any]],
    prompt_text: str,
    generated_text: str,
    summary: dict[str, Any],
    checkpoint_payload: dict[str, Any] | None,
) -> None:
    write_json(run_dir / "corpus.json", corpus)
    write_json(
        run_dir / "config.json",
        {
            "cli_args": vars(args),
            "corpus": corpus,
            "tokenizer_label": tokenizer_label,
            "tokenizer_model_path": tokenizer_model_path,
            "model": effective_config,
            "parameter_count": parameter_count,
            "steps_per_epoch": steps_per_epoch,
        },
    )
    write_json(
        run_dir / "history.json",
        {
            "eval_steps": history.eval_steps,
            "train_step_losses": history.train_step_losses,
            "grad_norms": history.grad_norms,
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
        },
    )
    write_jsonl(run_dir / "metrics.jsonl", metrics_rows)
    write_csv_rows(run_dir / "metrics.csv", metrics_rows)
    write_json(run_dir / "summary.json", summary)
    (run_dir / "sample_prompt.txt").write_text(prompt_text, encoding="utf-8")
    (run_dir / "sample_generated.txt").write_text(generated_text, encoding="utf-8")
    if checkpoint_payload is not None:
        torch.save(checkpoint_payload, run_dir / "checkpoint.pt")


def run_single_experiment(
    *,
    args: argparse.Namespace,
    session_dir: Path,
    experiment_name: str,
    corpus: CorpusLoadResult,
    vocab: object,
    tokenizer_label: str,
    tokenizer_model_path: Path | None,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device | str,
    teacher_forcing: bool,
    full_sequence_causal: bool,
    train_response_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    val_response_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
    special_token_ids: dict[str, int] | None = None,
    trial: Any | None = None,
) -> dict[str, Any]:
    effective_config = resolve_effective_model_config(args, experiment_name)
    autocast_dtype = resolve_autocast_dtype(args.precision)
    autocast_device_type = device.type if isinstance(device, torch.device) else str(device)
    if autocast_dtype is None or autocast_device_type != "cuda":
        autocast_dtype = None
        autocast_device_type = None
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    if grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be positive.")
    effective_batch_size = args.batch_size * grad_accum_steps
    prefix_response = args.training_objective == "prefix_response"
    if prefix_response:
        if train_response_pairs is None or val_response_pairs is None or special_token_ids is None:
            raise ValueError("prefix_response requires encoded dialogue pairs and special ids.")
        if args.response_len <= 0:
            raise ValueError("response-len must be positive.")
        steps_per_epoch = max(1, math.ceil(len(train_response_pairs) / effective_batch_size))
    else:
        steps_per_epoch = estimate_steps_per_epoch(
            token_count=int(train_tokens.numel()),
            seq_len=args.seq_len,
            batch_size=effective_batch_size,
        )
    if args.steps <= 0:
        raise ValueError("steps must be positive.")
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("epochs must be positive.")
        train_steps = max(1, math.ceil(args.epochs * steps_per_epoch))
        requested_epochs = args.epochs
    else:
        train_steps = args.steps
        requested_epochs = train_steps / steps_per_epoch

    stage_specs = build_progressive_b_stage_specs(
        seq_nodes=args.seq_len,
        lite_layers=int(effective_config["lite_layers"]),
        mid_layers=int(effective_config["mid_layers"]),
        full_layers=int(effective_config["full_layers"]),
        lite_expand_ratio=args.lite_expand_ratio,
        lite_compress_ratio=args.lite_compress_ratio,
        lite_alpha_b=args.lite_alpha_b,
        lite_beta_s_to_b=args.lite_beta_s_to_b,
        lite_beta_b_to_s=args.lite_beta_b_to_s,
        mid_expand_ratio=args.mid_expand_ratio,
        mid_compress_ratio=args.mid_compress_ratio,
        mid_alpha_b=args.mid_alpha_b,
        mid_beta_s_to_b=args.mid_beta_s_to_b,
        mid_beta_b_to_s=args.mid_beta_b_to_s,
        full_expand_ratio=args.full_expand_ratio,
        full_compress_ratio=args.full_compress_ratio,
        full_alpha_b=args.full_alpha_b,
        full_beta_s_to_b=args.full_beta_s_to_b,
        full_beta_b_to_s=args.full_beta_b_to_s,
    )
    value_norm_kind = "identity" if args.disable_layer_norm else args.value_norm_kind
    model = ProgressiveBExampleLM(
        vocab_size=vocab.size,
        dim=int(effective_config["dim"]),
        seq_nodes=args.seq_len,
        warmup_layers=int(effective_config["warmup_layers"]),
        stage_specs=stage_specs,
        final_refine_layers=int(effective_config["final_refine_layers"]),
        s_window=args.s_window,
        route_topk=int(effective_config["route_topk"]),
        expanded_topk=int(effective_config["route_topk"]),
        compressed_topk=max(1, min(2, int(effective_config["route_topk"]))),
        sequence_sparse_type=args.sequence_propagation,
        expanded_sparse_type=args.expanded_propagation,
        compressed_sparse_type=args.compressed_propagation,
        route_mode=args.route_mode,
        value_norm_kind=value_norm_kind,
        norm_position=args.norm_position,
        expanded_window=args.expanded_window,
        compressed_window=args.compressed_window,
        implementation=args.implementation,
        propagation_residual=not args.disable_propagation_residual,
        value_residual_scale=args.value_residual_scale,
        state_residual_scale=args.state_residual_scale,
        alpha_scale=args.alpha_scale,
        beta_s_to_b_scale=args.beta_s_to_b_scale,
        beta_b_to_s_scale=args.beta_b_to_s_scale,
        s_delta_scale=args.s_delta_scale,
        b_delta_scale=args.b_delta_scale,
        cross_delta_scale=args.cross_delta_scale,
        route_temperature=args.route_temperature,
        route_kind=args.route_kind,
        route_hidden_dim=args.route_hidden_dim,
        state_init_mode=args.state_init_mode,
        pairwise_kind=args.pairwise_kind,
        pairwise_hidden_dim=args.pairwise_hidden_dim,
        edge_dropout_p=args.edge_dropout_p,
        usage_dropout_base=args.usage_dropout_base,
        usage_dropout_scale=args.usage_dropout_scale,
        usage_dropout_max=args.usage_dropout_max,
        usage_ema_decay=args.usage_ema_decay,
    )
    parameter_count = count_parameters(model)

    run_dir = session_dir / slugify_run_name(experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = (
        Path(args.tensorboard_dir) / session_dir.name / slugify_run_name(experiment_name)
        if args.tensorboard_dir
        else run_dir / "tensorboard"
    )

    print(
        f"experiment={experiment_name} | params={parameter_count:,} | "
        f"dim={effective_config['dim']} | warmup={effective_config['warmup_layers']} | "
        f"stages={effective_config['lite_layers']}/{effective_config['mid_layers']}/{effective_config['full_layers']} | "
        f"refine={effective_config['final_refine_layers']} | route_mode={args.route_mode} | "
        f"route_topk={effective_config['route_topk']} | route_kind={args.route_kind} | "
        f"edge_dropout={args.edge_dropout_p:.3f} | "
        f"usage_dropout={args.usage_dropout_base:.3f}/{args.usage_dropout_scale:.3f}/{args.usage_dropout_max:.3f}"
    )
    print(
        f"schedule_steps={train_steps:,} | approx_epochs={requested_epochs:.3f} | "
        f"steps_per_epoch={steps_per_epoch:,} | batch_size={args.batch_size} | "
        f"grad_accum_steps={grad_accum_steps} | effective_batch_size={effective_batch_size} | "
        f"objective={args.training_objective} | response_len={args.response_len} | "
        f"data_workers={args.data_workers} | "
        f"prefetch_factor={args.prefetch_factor}"
    )

    start_time = time.perf_counter()
    writer = maybe_create_summary_writer(enabled=args.tensorboard, tensorboard_dir=tensorboard_dir)

    def make_eval_sample() -> tuple[str, str]:
        if prefix_response:
            assert train_response_pairs is not None
            assert special_token_ids is not None
            prompt_ids, prompt_text = select_prefix_response_prompt_tokens(
                args=args,
                vocab=vocab,
                train_response_pairs=train_response_pairs,
            )
            response_ids = generate_prefix_response_with_sampling(
                model,
                prompt_ids[-args.seq_len :],
                response_len=max(1, min(args.response_len, args.sample_tokens)),
                bos_token_id=special_token_ids["assistant"],
                eos_token_id=special_token_ids["eos"],
                device=device,
                temperature=args.temperature,
                sample_topk=args.sample_topk,
            )
            return prompt_text, vocab.decode(response_ids.tolist())
        prompt_ids, prompt_text = select_prompt_tokens(args=args, vocab=vocab, train_tokens=train_tokens)
        generated = generate_next_tokens_with_sampling(
            model,
            prompt_ids,
            max_new_tokens=args.sample_tokens,
            seq_len=args.seq_len,
            device=device,
            temperature=args.temperature,
            sample_topk=args.sample_topk,
            training_objective=args.training_objective,
        )
        return prompt_text, vocab.decode(generated.tolist())

    def progress_callback(step: int, total_steps: int, minibatch_loss: float) -> None:
        if args.log_interval <= 0:
            return
        if step != 1 and step != total_steps and step % args.log_interval != 0:
            return
        elapsed = time.perf_counter() - start_time
        print(
            f"progress | experiment={experiment_name} | step={step:>5d}/{total_steps:<5d} | "
            f"epoch={step / steps_per_epoch:.3f} | {100.0 * step / total_steps:5.1f}% | "
            f"minibatch_loss={minibatch_loss:.4f} | elapsed={elapsed:.1f}s"
        )

    def step_tensorboard_callback(step: int, minibatch_loss: float, grad_norm: float) -> None:
        if writer is None:
            return
        writer.add_scalar("train/minibatch_loss", minibatch_loss, step)
        writer.add_scalar("train/epoch", step / steps_per_epoch, step)
        writer.add_scalar("train/grad_norm", grad_norm, step)
        if step == 1 or step == train_steps or (args.log_interval > 0 and step % args.log_interval == 0):
            writer.flush()

    def eval_tensorboard_callback(step: int, train_loss: float, val_loss: float) -> None:
        if writer is None:
            runtime_stats = None
        else:
            writer.add_scalar("eval/train_loss", train_loss, step)
            writer.add_scalar("eval/val_loss", val_loss, step)
            writer.add_scalar("eval/train_ppl", perplexity_from_loss(train_loss), step)
            writer.add_scalar("eval/val_ppl", perplexity_from_loss(val_loss), step)
            if prefix_response:
                assert train_response_pairs is not None
                assert special_token_ids is not None
                stats_batch = sample_prefix_response_batch(
                    train_response_pairs,
                    seq_len=args.seq_len,
                    response_len=args.response_len,
                    batch_size=1,
                    bos_token_id=special_token_ids["assistant"],
                    eos_token_id=special_token_ids["eos"],
                    pad_token_id=special_token_ids["pad"],
                    device=device,
                )
                stats_context = stats_batch.context[:1]
            else:
                stats_batch = sample_next_token_batch(
                    train_tokens,
                    seq_len=args.seq_len,
                    batch_size=1,
                    device=device,
                    teacher_forcing=teacher_forcing,
                    full_sequence_causal=full_sequence_causal,
                )
                stats_context = stats_batch.context[:1]
            runtime_stats = model.collect_runtime_stats(stats_context)
            for key, value in sorted(runtime_stats.items()):
                writer.add_scalar(key, value, step)
            sample_prompt, sample_generated = make_eval_sample()
            writer.add_text(
                "eval/sample",
                format_tensorboard_sample(
                    prompt_text=sample_prompt,
                    generated_text=sample_generated,
                ),
                step,
            )
            print(
                f"eval_sample | step={step}\n"
                f"--- prompt ---\n{sample_prompt}\n\n"
                f"--- sample ---\n{sample_generated}"
            )
            writer.flush()
        if trial is not None:
            trial.report(val_loss, step)
            if trial.should_prune():
                raise TrialPrunedError(f"Trial pruned at step {step} with val_loss={val_loss:.4f}")

    def step_setup_callback(step: int, total_steps: int) -> None:
        model.set_b_schedule_scale(
            resolve_b_schedule_scale(
                schedule=args.b_schedule,
                step=step,
                total_steps=total_steps,
                min_scale=args.b_schedule_min,
                max_scale=args.b_schedule_max,
            )
        )

    try:
        if prefix_response:
            assert train_response_pairs is not None
            assert val_response_pairs is not None
            assert special_token_ids is not None
            history = train_prefix_response_model(
                model,
                train_response_pairs,
                val_response_pairs,
                seq_len=args.seq_len,
                response_len=args.response_len,
                batch_size=args.batch_size,
                bos_token_id=special_token_ids["assistant"],
                eos_token_id=special_token_ids["eos"],
                pad_token_id=special_token_ids["pad"],
                device=device,
                steps=train_steps,
                eval_interval=args.eval_interval,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_accum_steps=grad_accum_steps,
                data_workers=args.data_workers,
                prefetch_factor=args.prefetch_factor,
                step_setup_callback=step_setup_callback,
                progress_callback=progress_callback,
                step_callback=step_tensorboard_callback,
                eval_callback=eval_tensorboard_callback,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
        else:
            history = train_next_token_model(
                model,
                train_tokens,
                val_tokens,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                steps=train_steps,
                eval_interval=args.eval_interval,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_accum_steps=grad_accum_steps,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                teacher_forcing_chunk_size=args.teacher_forcing_chunk_size,
                data_workers=args.data_workers,
                prefetch_factor=args.prefetch_factor,
                step_setup_callback=step_setup_callback,
                progress_callback=progress_callback,
                step_callback=step_tensorboard_callback,
                eval_callback=eval_tensorboard_callback,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
    except Exception:
        if writer is not None:
            writer.flush()
            writer.close()
        raise
    runtime_seconds = time.perf_counter() - start_time
    print_eval_table(history, steps_per_epoch=steps_per_epoch)

    if prefix_response:
        assert special_token_ids is not None
        assert train_response_pairs is not None
        prompt_ids, prompt_text = select_prefix_response_prompt_tokens(
            args=args,
            vocab=vocab,
            train_response_pairs=train_response_pairs,
        )
        response_ids = generate_prefix_response_with_sampling(
            model,
            prompt_ids[-args.seq_len :],
            response_len=args.response_len,
            bos_token_id=special_token_ids["assistant"],
            eos_token_id=special_token_ids["eos"],
            device=device,
            temperature=args.temperature,
            sample_topk=args.sample_topk,
        )
        generated_text = vocab.decode(response_ids.tolist())
        generated = torch.cat((prompt_ids, response_ids), dim=0)
    else:
        prompt_ids, prompt_text = select_prompt_tokens(args=args, vocab=vocab, train_tokens=train_tokens)
        generated = generate_next_tokens_with_sampling(
            model,
            prompt_ids,
            max_new_tokens=args.sample_tokens,
            seq_len=args.seq_len,
            device=device,
            temperature=args.temperature,
            sample_topk=args.sample_topk,
            training_objective=args.training_objective,
        )
        generated_text = vocab.decode(generated.tolist())

    best_index = min(range(len(history.val_losses)), key=history.val_losses.__getitem__)
    best_val_loss = history.val_losses[best_index]
    best_train_loss = history.train_losses[best_index]
    best_step = history.eval_steps[best_index]
    final_train_loss = history.train_losses[-1]
    final_val_loss = history.val_losses[-1]

    summary: dict[str, Any] = {
        "experiment": experiment_name,
        "model_preset": effective_config["model_preset"],
        "tokenizer": tokenizer_label,
        "tokenizer_vocab_size": vocab.size,
        "subword_vocab_size": args.subword_vocab_size if args.tokenizer == "subword" else None,
        "tokenizer_model_path": tokenizer_model_path,
        "seq_len": args.seq_len,
        "response_len": args.response_len,
        "batch_size": args.batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "data_workers": args.data_workers,
        "prefetch_factor": args.prefetch_factor,
        "dim": effective_config["dim"],
        "warmup_layers": effective_config["warmup_layers"],
        "final_refine_layers": effective_config["final_refine_layers"],
        "lite_layers": effective_config["lite_layers"],
        "mid_layers": effective_config["mid_layers"],
        "full_layers": effective_config["full_layers"],
        "route_topk": effective_config["route_topk"],
        "value_norm_kind": value_norm_kind,
        "norm_position": args.norm_position,
        "propagation_residual": not args.disable_propagation_residual,
        "route_mode": args.route_mode,
        "sequence_propagation": args.sequence_propagation,
        "expanded_propagation": args.expanded_propagation,
        "compressed_propagation": args.compressed_propagation,
        "value_residual_scale": args.value_residual_scale,
        "state_residual_scale": args.state_residual_scale,
        "route_temperature": args.route_temperature,
        "route_kind": args.route_kind,
        "route_hidden_dim": args.route_hidden_dim,
        "state_init_mode": args.state_init_mode,
        "pairwise_kind": args.pairwise_kind,
        "pairwise_hidden_dim": args.pairwise_hidden_dim,
        "edge_dropout_p": args.edge_dropout_p,
        "usage_dropout_base": args.usage_dropout_base,
        "usage_dropout_scale": args.usage_dropout_scale,
        "usage_dropout_max": args.usage_dropout_max,
        "usage_ema_decay": args.usage_ema_decay,
        "alpha_scale": args.alpha_scale,
        "beta_s_to_b_scale": args.beta_s_to_b_scale,
        "beta_b_to_s_scale": args.beta_b_to_s_scale,
        "s_delta_scale": args.s_delta_scale,
        "b_delta_scale": args.b_delta_scale,
        "cross_delta_scale": args.cross_delta_scale,
        "b_schedule": args.b_schedule,
        "b_schedule_min": args.b_schedule_min,
        "b_schedule_max": args.b_schedule_max,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "precision": args.precision,
        "s_window": args.s_window,
        "steps": train_steps,
        "approx_epochs": requested_epochs,
        "steps_per_epoch": steps_per_epoch,
        "parameter_count": parameter_count,
        "runtime_seconds": runtime_seconds,
        "final_train_loss": final_train_loss,
        "final_train_ppl": perplexity_from_loss(final_train_loss),
        "final_val_loss": final_val_loss,
        "final_val_ppl": perplexity_from_loss(final_val_loss),
        "best_train_loss": best_train_loss,
        "best_train_ppl": perplexity_from_loss(best_train_loss),
        "best_val_loss": best_val_loss,
        "best_val_ppl": perplexity_from_loss(best_val_loss),
        "best_step": best_step,
        "corpus_source_label": corpus.source_label,
        "corpus_char_count": corpus.char_count,
        "corpus_sample_count": corpus.sample_count,
        "corpus_file_count": corpus.file_count,
        "corpus_truncated": corpus.truncated,
        "special_token_ids": special_token_ids or {},
        "run_dir": run_dir,
        "tensorboard_dir": tensorboard_dir if args.tensorboard else None,
        "final_runtime_stats": model.collect_runtime_stats(
            prompt_ids[: args.seq_len].unsqueeze(0).to(device)
        ),
    }
    metrics_rows = build_metrics_rows(history, steps_per_epoch=steps_per_epoch)

    checkpoint_payload = None
    if args.save_checkpoint:
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": vocab.size,
                "tokenizer": tokenizer_label,
                "tokenizer_model_path": None if tokenizer_model_path is None else str(tokenizer_model_path),
                "seq_nodes": args.seq_len,
                "implementation": args.implementation,
                "value_norm_kind": value_norm_kind,
                "norm_position": args.norm_position,
                "propagation_residual": not args.disable_propagation_residual,
                "route_mode": args.route_mode,
                "sequence_propagation": args.sequence_propagation,
                "expanded_propagation": args.expanded_propagation,
                "compressed_propagation": args.compressed_propagation,
                "value_residual_scale": args.value_residual_scale,
                "state_residual_scale": args.state_residual_scale,
                "route_temperature": args.route_temperature,
                "route_kind": args.route_kind,
                "route_hidden_dim": args.route_hidden_dim,
                "state_init_mode": args.state_init_mode,
                "pairwise_kind": args.pairwise_kind,
                "pairwise_hidden_dim": args.pairwise_hidden_dim,
                "edge_dropout_p": args.edge_dropout_p,
                "usage_dropout_base": args.usage_dropout_base,
                "usage_dropout_scale": args.usage_dropout_scale,
                "usage_dropout_max": args.usage_dropout_max,
                "usage_ema_decay": args.usage_ema_decay,
                "alpha_scale": args.alpha_scale,
                "beta_s_to_b_scale": args.beta_s_to_b_scale,
                "beta_b_to_s_scale": args.beta_b_to_s_scale,
                "s_delta_scale": args.s_delta_scale,
                "b_delta_scale": args.b_delta_scale,
                "cross_delta_scale": args.cross_delta_scale,
                "b_schedule": args.b_schedule,
                "b_schedule_min": args.b_schedule_min,
                "b_schedule_max": args.b_schedule_max,
                "stage_specs": [asdict(spec) for spec in stage_specs],
                **{key: int(value) if isinstance(value, int) else value for key, value in effective_config.items()},
            },
            "history": {
                "eval_steps": history.eval_steps,
                "train_step_losses": history.train_step_losses,
                "train_losses": history.train_losses,
                "val_losses": history.val_losses,
            },
            "summary": summary,
            "corpus": to_jsonable(corpus),
        }

    save_run_artifacts(
        run_dir=run_dir,
        args=args,
        corpus=corpus,
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
        effective_config=effective_config,
        parameter_count=parameter_count,
        steps_per_epoch=steps_per_epoch,
        history=history,
        metrics_rows=metrics_rows,
        prompt_text=prompt_text,
        generated_text=generated_text,
        summary=summary,
        checkpoint_payload=checkpoint_payload,
    )

    try:
        log_history_to_tensorboard(
            writer,
            history=history,
            steps_per_epoch=steps_per_epoch,
            summary=summary,
            prompt_text=prompt_text,
            generated_text=generated_text,
            include_scalars=False,
        )
    finally:
        if writer is not None:
            writer.close()

    print("\n--- prompt ---")
    print(prompt_text)
    print("\n--- sample ---")
    print(generated_text)
    print(f"\nartifacts={run_dir}")
    if args.tensorboard:
        print(f"tensorboard={tensorboard_dir}")

    return summary


def main() -> None:
    preset_choices = ("custom", *sorted(MODEL_SIZE_PRESETS))

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--text-file")
    parser.add_argument(
        "--text-source",
        action="append",
        default=[],
        help="Additional UTF-8 text file, directory, or glob. Can be repeated.",
    )
    parser.add_argument(
        "--jsonl-source",
        action="append",
        default=[],
        help="JSONL file, directory, or glob containing text records. Can be repeated.",
    )
    parser.add_argument(
        "--jsonl-text-key",
        action="append",
        default=["text"],
        help="JSONL field to read text from. Can be repeated.",
    )
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--corpus-max-samples", type=int)
    parser.add_argument("--corpus-max-chars", type=int)
    parser.add_argument("--corpus-separator", default="\n\n")
    parser.add_argument(
        "--training-objective",
        choices=("last_token", "teacher_forcing", "full_sequence_causal", "prefix_response"),
        default="last_token",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("char", "subword"),
        default="char",
    )
    parser.add_argument("--subword-vocab-size", type=int, default=256)
    parser.add_argument("--subword-character-coverage", type=float, default=0.9995)
    parser.add_argument(
        "--subword-model-type",
        choices=("bpe", "unigram"),
        default="bpe",
    )
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--response-len", type=int, default=64)
    parser.add_argument(
        "--model-preset",
        choices=preset_choices,
        default="custom",
    )
    parser.add_argument(
        "--sweep-presets",
        help="Comma-separated preset list such as tiny,small,base.",
    )
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--warmup-layers", type=int, default=2)
    parser.add_argument("--final-refine-layers", type=int, default=2)
    parser.add_argument("--lite-layers", type=int, default=2)
    parser.add_argument("--mid-layers", type=int, default=2)
    parser.add_argument("--full-layers", type=int, default=1)
    parser.add_argument("--lite-expand-ratio", type=float, default=1.05)
    parser.add_argument("--lite-compress-ratio", type=float, default=0.90)
    parser.add_argument("--lite-alpha-b", type=float, default=0.3)
    parser.add_argument("--lite-beta-s-to-b", type=float, default=0.25)
    parser.add_argument("--lite-beta-b-to-s", type=float, default=0.15)
    parser.add_argument("--mid-expand-ratio", type=float, default=1.10)
    parser.add_argument("--mid-compress-ratio", type=float, default=0.80)
    parser.add_argument("--mid-alpha-b", type=float, default=0.65)
    parser.add_argument("--mid-beta-s-to-b", type=float, default=0.55)
    parser.add_argument("--mid-beta-b-to-s", type=float, default=0.35)
    parser.add_argument("--full-expand-ratio", type=float, default=1.20)
    parser.add_argument("--full-compress-ratio", type=float, default=0.70)
    parser.add_argument("--full-alpha-b", type=float, default=1.0)
    parser.add_argument("--full-beta-s-to-b", type=float, default=0.9)
    parser.add_argument("--full-beta-b-to-s", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--route-topk", type=int, default=4)
    parser.add_argument(
        "--value-norm-kind",
        choices=("layernorm", "rmsnorm"),
        default="layernorm",
    )
    parser.add_argument(
        "--norm-position",
        choices=("pre", "post"),
        default="post",
    )
    parser.add_argument("--value-residual-scale", type=float, default=1.0)
    parser.add_argument("--state-residual-scale", type=float, default=1.0)
    parser.add_argument(
        "--state-init-mode",
        choices=("zero", "neg_half", "normal"),
        default="zero",
    )
    parser.add_argument("--s-window", type=int, default=8)
    parser.add_argument("--route-temperature", type=float, default=1.0)
    parser.add_argument(
        "--route-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "hadamard_mlp"),
        default="diagonal_bilinear",
    )
    parser.add_argument("--route-hidden-dim", type=int)
    parser.add_argument(
        "--pairwise-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "hadamard_mlp"),
        default="diagonal_bilinear",
    )
    parser.add_argument("--pairwise-hidden-dim", type=int)
    parser.add_argument("--edge-dropout-p", type=float, default=0.0)
    parser.add_argument("--usage-dropout-base", type=float, default=0.0)
    parser.add_argument("--usage-dropout-scale", type=float, default=0.0)
    parser.add_argument("--usage-dropout-max", type=float, default=0.0)
    parser.add_argument("--usage-ema-decay", type=float, default=0.99)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16", "fp16"),
        default="fp32",
    )
    parser.add_argument(
        "--implementation",
        choices=("reference", "streaming", "kernel", "native"),
        default="streaming",
    )
    parser.add_argument(
        "--sequence-propagation",
        choices=("window", "dense"),
        default="window",
    )
    parser.add_argument(
        "--expanded-propagation",
        choices=("dense", "topk", "window"),
        default="topk",
    )
    parser.add_argument(
        "--compressed-propagation",
        choices=("dense", "topk", "window"),
        default="topk",
    )
    parser.add_argument(
        "--route-mode",
        choices=("topk", "dense"),
        default="dense",
    )
    parser.add_argument("--expanded-window", type=int)
    parser.add_argument("--compressed-window", type=int)
    parser.add_argument("--disable-layer-norm", action="store_true")
    parser.add_argument("--disable-propagation-residual", action="store_true")
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-s-to-b-scale", type=float, default=1.0)
    parser.add_argument("--beta-b-to-s-scale", type=float, default=1.0)
    parser.add_argument("--s-delta-scale", type=float, default=0.25)
    parser.add_argument("--b-delta-scale", type=float, default=0.20)
    parser.add_argument("--cross-delta-scale", type=float, default=0.15)
    parser.add_argument(
        "--b-schedule",
        choices=("constant", "up", "down"),
        default="constant",
    )
    parser.add_argument("--b-schedule-min", type=float, default=1.0)
    parser.add_argument("--b-schedule-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--prompt-text")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--sample-topk", type=int)
    parser.add_argument("--teacher-forcing-chunk-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--run-name", default="progressive_b")
    parser.add_argument("--output-dir", default="artifacts/training_runs")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(args.device)}")

    teacher_forcing = args.training_objective == "teacher_forcing"
    full_sequence_causal = args.training_objective == "full_sequence_causal"
    prefix_response = args.training_objective == "prefix_response"
    if args.teacher_forcing_chunk_size is not None and args.teacher_forcing_chunk_size <= 0:
        raise ValueError("teacher-forcing-chunk-size must be positive.")
    if args.teacher_forcing_chunk_size is not None and not teacher_forcing:
        raise ValueError(
            "--teacher-forcing-chunk-size is only supported with "
            "--training-objective teacher_forcing."
        )
    if args.s_window <= 0:
        raise ValueError("s-window must be positive.")
    if args.route_temperature <= 0.0:
        raise ValueError("route-temperature must be positive.")
    if args.data_workers < 0:
        raise ValueError("data-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be positive.")
    if args.response_len <= 0:
        raise ValueError("response-len must be positive.")
    for name in (
        "edge_dropout_p",
        "usage_dropout_base",
        "usage_dropout_scale",
        "usage_dropout_max",
    ):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1).")
    if not 0.0 <= args.usage_ema_decay < 1.0:
        raise ValueError("usage-ema-decay must be in [0, 1).")

    dialogue_pairs: list[DialoguePairText] | None = None
    if prefix_response:
        dialogue_pairs, corpus = load_dialogue_pairs(
            jsonl_sources=args.jsonl_source,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_streaming=args.hf_streaming,
            max_samples=args.corpus_max_samples,
            max_chars=args.corpus_max_chars,
        )
    else:
        corpus = load_text_corpus(
            default_text=DEFAULT_TEXT,
            text_file=args.text_file,
            text_sources=args.text_source,
            jsonl_sources=args.jsonl_source,
            jsonl_text_keys=tuple(dict.fromkeys(args.jsonl_text_key)),
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_text_key=args.hf_text_key,
            hf_streaming=args.hf_streaming,
            max_samples=args.corpus_max_samples,
            max_chars=args.corpus_max_chars,
            separator=args.corpus_separator,
        )
    print(
        f"corpus={corpus.source_label} | chars={corpus.char_count:,} | "
        f"samples={corpus.sample_count:,} | files={corpus.file_count:,} | "
        f"truncated={corpus.truncated}"
    )

    tokenizer_text_path = None if corpus.text_path is None else str(corpus.text_path)
    vocab, tokenizer_label, tokenizer_model_path = build_tokenizer(
        corpus.text,
        text_path=tokenizer_text_path,
        tokenizer=args.tokenizer,
        subword_vocab_size=args.subword_vocab_size,
        subword_model_type=args.subword_model_type,
        tokenizer_prefix=args.tokenizer_prefix,
        subword_character_coverage=args.subword_character_coverage,
        user_defined_symbols=DIALOGUE_SPECIAL_TOKENS if prefix_response else (),
    )
    special_token_ids = require_subword_special_ids(vocab) if prefix_response else None
    if prefix_response:
        tokens = torch.empty(0, dtype=torch.long)
        train_tokens = tokens
        val_tokens = tokens
    else:
        tokens = vocab.encode(corpus.text)
        if tokens.numel() <= args.seq_len + 1:
            raise ValueError("The tokenized corpus must be longer than seq_len + 1.")
        train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.9)
    train_response_pairs = None
    val_response_pairs = None
    if prefix_response:
        assert dialogue_pairs is not None
        train_pair_texts, val_pair_texts = split_dialogue_pairs(dialogue_pairs)
        train_response_pairs = encode_dialogue_pairs(train_pair_texts, vocab=vocab)
        val_response_pairs = encode_dialogue_pairs(val_pair_texts, vocab=vocab)
    print(
        f"tokenizer={tokenizer_label} | tokenizer_vocab={vocab.size:,} | "
        f"train_tokens={train_tokens.numel():,} | val_tokens={val_tokens.numel():,}"
    )
    if prefix_response:
        print(
            f"dialogue_pairs={len(dialogue_pairs or ()):,} | "
            f"train_pairs={len(train_response_pairs or ()):,} | "
            f"val_pairs={len(val_response_pairs or ()):,} | response_len={args.response_len}"
        )
    if tokenizer_model_path is not None:
        print(f"tokenizer_model={tokenizer_model_path}")

    sweep_presets = parse_preset_names(args.sweep_presets)
    if sweep_presets:
        experiment_names = sweep_presets
    else:
        experiment_names = (args.model_preset,)
    for name in experiment_names:
        if name != "custom":
            resolve_model_scale_preset(name)

    session_dir = create_run_directory(args.output_dir, args.run_name)
    ensure_directory(session_dir)
    print(f"session_dir={session_dir}")

    summaries: list[dict[str, Any]] = []
    for experiment_name in experiment_names:
        summary = run_single_experiment(
            args=args,
            session_dir=session_dir,
            experiment_name=experiment_name,
            corpus=corpus,
            vocab=vocab,
            tokenizer_label=tokenizer_label,
            tokenizer_model_path=tokenizer_model_path,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            device=device,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
            train_response_pairs=train_response_pairs,
            val_response_pairs=val_response_pairs,
            special_token_ids=special_token_ids,
        )
        summaries.append(summary)
        append_jsonl(session_dir / "experiments.jsonl", summary)

    write_json(session_dir / "session_summary.json", summaries)
    write_csv_rows(session_dir / "session_summary.csv", summaries)
    if len(summaries) > 1:
        best_summary = min(summaries, key=lambda row: row["best_val_loss"])
        write_json(
            session_dir / "best_run.json",
            {
                "best_experiment": best_summary["experiment"],
                "best_val_loss": best_summary["best_val_loss"],
                "best_val_ppl": best_summary["best_val_ppl"],
                "run_dir": best_summary["run_dir"],
            },
        )
        print(
            "best_experiment="
            f"{best_summary['experiment']} | best_val_loss={best_summary['best_val_loss']:.4f} | "
            f"run_dir={best_summary['run_dir']}"
        )


if __name__ == "__main__":
    main()
