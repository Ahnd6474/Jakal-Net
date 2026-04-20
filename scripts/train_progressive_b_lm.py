from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

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
    sample_next_token_batch,
    sample_prefix_response_batch,
    sample_query_block_pair_batch,
    split_train_val,
    train_prefix_response_model,
    train_next_token_model,
    train_query_block_pair_model,
)


def rng_state_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    torch_state = payload.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_state = payload.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_state, list):
        torch.cuda.set_rng_state_all(cuda_state)


def _route_role_and_suffix(parameter_name: str) -> tuple[str, str] | None:
    if parameter_name.startswith("query_transition.route_fn."):
        return ("query_transition", parameter_name.removeprefix("query_transition.route_fn."))
    markers = (
        (".expand_transition.route_fn.", "expand_transition"),
        (".b_to_s.route_fn.", "b_to_s"),
        (".compress_transition.route_fn.", "compress_transition"),
        (".s_to_b.route_fn.", "s_to_b"),
        (".compressed_adapter.route_fn.", "compressed_adapter"),
    )
    for marker, role in markers:
        if marker in parameter_name:
            _, suffix = parameter_name.split(marker, maxsplit=1)
            return (role, suffix)
    return None


def merge_route_families_for_resume(
    *,
    checkpoint_state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    role_suffix_values: dict[str, dict[str, list[torch.Tensor]]] = {}
    role_module_names: dict[str, set[str]] = {}
    for name, value in checkpoint_state_dict.items():
        match = _route_role_and_suffix(name)
        if match is None or not isinstance(value, torch.Tensor):
            continue
        role, suffix = match
        role_suffix_values.setdefault(role, {}).setdefault(suffix, []).append(value.detach().float())
        role_module_names.setdefault(role, set()).add(name.rsplit(".route_fn.", maxsplit=1)[0])
    if not role_suffix_values:
        return checkpoint_state_dict, {}

    role_suffix_average: dict[str, dict[str, torch.Tensor]] = {}
    for role, suffix_map in role_suffix_values.items():
        averaged_suffixes: dict[str, torch.Tensor] = {}
        for suffix, values in suffix_map.items():
            averaged_suffixes[suffix] = torch.stack(values, dim=0).mean(dim=0)
        role_suffix_average[role] = averaged_suffixes

    merged_state_dict = dict(checkpoint_state_dict)
    for name, reference in model_state_dict.items():
        match = _route_role_and_suffix(name)
        if match is None:
            continue
        role, suffix = match
        averaged_value = role_suffix_average.get(role, {}).get(suffix)
        if averaged_value is None:
            continue
        merged_state_dict[name] = averaged_value.to(dtype=reference.dtype)

    return (
        merged_state_dict,
        {role: len(module_names) for role, module_names in sorted(role_module_names.items())},
    )

try:
    import sentencepiece as spm
except ImportError:
    spm = None

try:
    from tokenizers import ByteLevelBPETokenizer
except ImportError:
    ByteLevelBPETokenizer = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

_SUBWORD_WORKER_PROCESSOR: Any | None = None
_BYTE_BPE_WORKER_TOKENIZER: Any | None = None
_BYTE_BPE_WORKER_SPECIAL_TOKENS: tuple[str, ...] = ()
_BYTE_BPE_WORKER_SPECIAL_PATTERN: re.Pattern[str] | None = None


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
QUERY_BLOCK_START_TOKEN = "<|query_block_start|>"
DIALOGUE_SPECIAL_TOKENS = (
    USER_TOKEN,
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    QUERY_BLOCK_START_TOKEN,
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
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
    source: str = "unknown"


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


@dataclass(frozen=True, slots=True)
class ByteBPEVocab:
    tokenizer: object
    vocab_path: Path
    merges_path: Path
    special_tokens: tuple[str, ...] = ()
    special_pattern: re.Pattern[str] | None = None

    @property
    def size(self) -> int:
        return int(self.tokenizer.get_vocab_size())

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text).ids
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(map(int, token_ids))))

    def token_id(self, piece: str) -> int:
        idx = self.tokenizer.token_to_id(piece)
        if idx is None:
            raise ValueError(
                f"Tokenizer {self.vocab_path} does not contain required token {piece!r}."
            )
        return int(idx)


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
    input_sentence_size: int = 0,
    num_threads: int = 0,
    user_defined_symbols: Sequence[str] = (),
) -> SubwordVocab:
    if spm is None:
        raise ImportError(
            "sentencepiece is required for --tokenizer subword. "
            "Install dependencies from requirements-base.txt first."
        )
    if vocab_size <= 0:
        raise ValueError("subword-vocab-size must be positive.")
    if input_sentence_size < 0:
        raise ValueError("subword-input-sentence-size must be non-negative.")
    if num_threads < 0:
        raise ValueError("subword-num-threads must be non-negative.")

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
        trainer_kwargs: dict[str, Any] = {
            "input": str(training_text_path),
            "model_prefix": str(prefix_path),
            "model_type": model_type,
            "vocab_size": vocab_size,
            "character_coverage": character_coverage,
            "user_defined_symbols": list(user_defined_symbols),
            "bos_id": -1,
            "eos_id": -1,
            "pad_id": -1,
            "hard_vocab_limit": False,
            "split_digits": True,
            "num_threads": num_threads or (os.cpu_count() or 16),
        }
        if input_sentence_size > 0:
            trainer_kwargs["input_sentence_size"] = input_sentence_size
            trainer_kwargs["shuffle_input_sentence"] = True
        spm.SentencePieceTrainer.train(**trainer_kwargs)

    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    return SubwordVocab(
        processor=processor,
        model_path=model_path,
        model_type=model_type,
    )


def build_byte_bpe_vocab(
    text: str,
    *,
    text_path: str | None,
    vocab_size: int,
    tokenizer_prefix: str | None,
    input_sentence_size: int = 0,
    num_threads: int = 0,
    user_defined_symbols: Sequence[str] = (),
) -> ByteBPEVocab:
    if ByteLevelBPETokenizer is None:
        raise ImportError(
            "tokenizers is required for --tokenizer byte_bpe. "
            "Install dependencies from requirements-base.txt first."
        )
    if vocab_size <= 0:
        raise ValueError("subword-vocab-size must be positive.")
    if input_sentence_size < 0:
        raise ValueError("subword-input-sentence-size must be non-negative.")
    if num_threads < 0:
        raise ValueError("subword-num-threads must be non-negative.")

    prefix_path = resolve_tokenizer_prefix(
        text=text,
        tokenizer="byte_bpe",
        model_type="byte_level",
        vocab_size=vocab_size,
        prefix=tokenizer_prefix,
    )
    vocab_path = prefix_path.parent / f"{prefix_path.name}-vocab.json"
    merges_path = prefix_path.parent / f"{prefix_path.name}-merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        training_text_path = ensure_tokenizer_training_text(
            text=text,
            text_path=text_path,
            tokenizer_prefix=prefix_path,
        )
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(
            files=[str(training_text_path)],
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(user_defined_symbols),
        )
        tokenizer.save_model(str(prefix_path.parent), prefix_path.name)

    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    special_tokens = _ensure_byte_bpe_special_tokens(tokenizer, user_defined_symbols)
    return ByteBPEVocab(
        tokenizer=tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=special_tokens,
        special_pattern=_byte_bpe_special_pattern(special_tokens),
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
    subword_input_sentence_size: int = 0,
    subword_num_threads: int = 0,
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
            input_sentence_size=subword_input_sentence_size,
            num_threads=subword_num_threads,
            user_defined_symbols=user_defined_symbols,
        )
        return (
            subword_vocab,
            f"subword/{subword_vocab.model_type}",
            subword_vocab.model_path,
        )
    if tokenizer == "byte_bpe":
        byte_bpe_vocab = build_byte_bpe_vocab(
            text,
            text_path=text_path,
            vocab_size=subword_vocab_size,
            tokenizer_prefix=tokenizer_prefix,
            input_sentence_size=subword_input_sentence_size,
            num_threads=subword_num_threads,
            user_defined_symbols=user_defined_symbols,
        )
        return byte_bpe_vocab, "byte_bpe", byte_bpe_vocab.vocab_path
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
            pairs.append(DialoguePairText(prefix=prefix, response=content, source="messages"))
            history.append(f"{ASSISTANT_TOKEN}\n{content}\n{EOS_TOKEN}\n")
    return pairs


def _pairs_from_record(record: Any) -> list[DialoguePairText]:
    if not isinstance(record, dict):
        return []
    prefix = record.get("prefix")
    response = record.get("response")
    if isinstance(prefix, str) and prefix.strip() and isinstance(response, str) and response.strip():
        source = record.get("source")
        source_label = source.strip() if isinstance(source, str) and source.strip() else "unknown"
        return [DialoguePairText(prefix=prefix.strip(), response=response.strip(), source=source_label)]
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


def _chat_transcript_from_messages(messages: Sequence[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    has_assistant = False
    for message in messages:
        role = _message_role(message)
        content = _message_content(message)
        if not content:
            continue
        if role == "user":
            parts.append(f"{USER_TOKEN}\n{content}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}\n{content}")
            has_assistant = True
    if not parts or not has_assistant:
        return None
    return "\n".join(parts)


def _text_from_record_keys(record: Any, text_keys: Sequence[str]) -> str | None:
    if isinstance(record, str):
        return record.strip() or None
    if not isinstance(record, dict):
        return None
    for key in text_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _chat_stream_from_record(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    prefix = record.get("prefix")
    response = record.get("response")
    if isinstance(prefix, str) and prefix.strip() and isinstance(response, str) and response.strip():
        separator = "" if prefix.endswith(("\n", " ", "\t")) else "\n"
        return f"{prefix}{separator}{response.strip()}"
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        normalized = [message for message in messages if isinstance(message, dict)]
        transcript = _chat_transcript_from_messages(normalized)
        if transcript is not None:
            return transcript
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
        return f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n{response}"
    return None


def load_token_stream_corpus(
    *,
    default_text: str,
    text_file: str | None,
    text_sources: Sequence[str],
    jsonl_sources: Sequence[str],
    jsonl_text_keys: Sequence[str],
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    hf_text_key: str,
    hf_streaming: bool,
    max_samples: int | None,
    max_chars: int | None,
    separator: str,
) -> CorpusLoadResult:
    from lm_experiment_utils import _expand_sources, _truncate_texts

    sources = [text_file] if text_file is not None else []
    sources.extend(text_sources)
    text_paths = _expand_sources(sources, directory_suffixes=(".txt", ".text", ".md"))
    jsonl_paths = _expand_sources(jsonl_sources, directory_suffixes=(".jsonl", ".json"))
    eos_separator = f"\n{EOS_TOKEN}\n"

    pieces: list[str] = []
    chat_record_count = 0
    text_record_count = 0
    for path in text_paths:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"Input text file must not be empty: {path}")
        pieces.append(text.strip())
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip().lstrip("\ufeff")
                if not line:
                    continue
                record = json.loads(line)
                stream = _chat_stream_from_record(record)
                if stream is not None:
                    pieces.append(stream.strip())
                    chat_record_count += 1
                else:
                    text = _text_from_record_keys(record, jsonl_text_keys)
                    if text is None:
                        raise ValueError(
                            f"Could not find chat or text in {path} line {line_number}. "
                            f"Tried text keys: {', '.join(jsonl_text_keys)}."
                        )
                    pieces.append(text)
                    text_record_count += 1
                if max_samples is not None and len(pieces) >= max_samples:
                    break
        if max_samples is not None and len(pieces) >= max_samples:
            break
    if hf_dataset is not None and (max_samples is None or len(pieces) < max_samples):
        hf_corpus = load_text_corpus(
            default_text="",
            text_file=None,
            text_sources=(),
            jsonl_sources=(),
            jsonl_text_keys=jsonl_text_keys,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            hf_split=hf_split,
            hf_text_key=hf_text_key,
            hf_streaming=hf_streaming,
            max_samples=None if pieces else max_samples,
            max_chars=None,
            separator=separator,
        )
        if hf_corpus.text.strip():
            pieces.append(hf_corpus.text.strip())
            text_record_count += hf_corpus.sample_count
    if not pieces:
        pieces.append(default_text.strip())
        text_record_count += 1

    combined_text, sample_count, truncated = _truncate_texts(
        pieces,
        separator=eos_separator,
        max_samples=max_samples,
        max_chars=max_chars,
    )
    combined_text = combined_text.rstrip()
    if not combined_text.endswith(EOS_TOKEN):
        combined_text = f"{combined_text}{eos_separator}"
    else:
        combined_text = f"{combined_text}\n"

    source_parts: list[str] = []
    if text_paths:
        source_parts.append(f"text_files={len(text_paths)}")
    if jsonl_paths:
        source_parts.append(f"jsonl_files={len(jsonl_paths)}")
    if hf_dataset is not None:
        source_parts.append(f"hf_dataset={hf_dataset}:{hf_split}")
    if not source_parts:
        source_parts.append("default_text")

    return CorpusLoadResult(
        text=combined_text,
        source_label=f"token_stream,{', '.join(source_parts)}",
        text_path=None,
        sample_count=sample_count,
        file_count=len(text_paths) + len(jsonl_paths),
        char_count=len(combined_text),
        truncated=truncated,
        metadata={
            "source_kind": "token_stream",
            "text_paths": [str(path) for path in text_paths],
            "jsonl_paths": [str(path) for path in jsonl_paths],
            "hf_dataset": hf_dataset,
            "hf_config": hf_config,
            "hf_split": hf_split,
            "hf_text_key": hf_text_key,
            "hf_streaming": hf_streaming,
            "max_samples": max_samples,
            "max_chars": max_chars,
            "chat_record_count": chat_record_count,
            "text_record_count": text_record_count,
            "eos_at_document_boundaries": True,
            "document_separator": EOS_TOKEN,
        },
    )


def _source_label_from_record(record: Any, *, default: str = "unknown") -> str:
    raw_source = record.get("source") if isinstance(record, dict) else None
    if isinstance(raw_source, str) and raw_source.strip():
        label = raw_source.strip().split(":", 1)[0]
    else:
        label = default
    label = re.sub(r"\s+", "_", label.strip().lower())
    return label or "unknown"


def load_token_stream_source_texts(
    *,
    jsonl_sources: Sequence[str],
    jsonl_text_keys: Sequence[str],
    max_samples: int | None,
    max_chars: int | None,
) -> dict[str, str]:
    from lm_experiment_utils import _expand_sources

    jsonl_paths = _expand_sources(jsonl_sources, directory_suffixes=(".jsonl", ".json"))
    eos_separator = f"\n{EOS_TOKEN}\n"
    source_pieces: dict[str, list[str]] = {}
    sample_count = 0
    char_count = 0
    exhausted = False
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if max_samples is not None and sample_count >= max_samples:
                    exhausted = True
                    break
                line = raw_line.strip().lstrip("\ufeff")
                if not line:
                    continue
                record = json.loads(line)
                stream = _chat_stream_from_record(record)
                if stream is None:
                    text = _text_from_record_keys(record, jsonl_text_keys)
                    if text is None:
                        raise ValueError(
                            f"Could not find chat or text in {path} line {line_number}. "
                            f"Tried text keys: {', '.join(jsonl_text_keys)}."
                        )
                    stream = text
                piece = stream.strip()
                if not piece:
                    continue
                if max_chars is not None:
                    remaining_chars = max_chars - char_count
                    if remaining_chars <= 0:
                        exhausted = True
                        break
                    if len(piece) > remaining_chars:
                        piece = piece[:remaining_chars].rstrip()
                if not piece:
                    exhausted = True
                    break
                label = _source_label_from_record(record, default=path.stem)
                source_pieces.setdefault(label, []).append(piece)
                sample_count += 1
                char_count += len(piece)
                if max_chars is not None and char_count >= max_chars:
                    exhausted = True
                    break
        if exhausted:
            break

    source_texts: dict[str, str] = {}
    for source, pieces in source_pieces.items():
        combined = eos_separator.join(piece.strip() for piece in pieces if piece.strip()).rstrip()
        if not combined:
            continue
        if not combined.endswith(EOS_TOKEN):
            combined = f"{combined}{eos_separator}"
        else:
            combined = f"{combined}\n"
        source_texts[source] = combined
    return source_texts


def _iter_nonempty_text_chunks(text: str, *, chunk_chars: int) -> Iterable[str]:
    for start in range(0, len(text), chunk_chars):
        chunk = text[start : start + chunk_chars]
        if chunk.strip():
            yield chunk


def _byte_bpe_special_tokens(tokenizer: Any) -> tuple[str, ...]:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if get_vocab is None:
        return ()
    vocab = get_vocab()
    if not isinstance(vocab, dict):
        return ()
    specials = [token for token in vocab.keys() if isinstance(token, str) and re.fullmatch(r"<\|[^|\n]+\|>", token)]
    return tuple(sorted(specials, key=len, reverse=True))


def _byte_bpe_special_pattern(special_tokens: Sequence[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    return re.compile("|".join(re.escape(token) for token in special_tokens))


def _ensure_byte_bpe_special_tokens(tokenizer: Any, required_special_tokens: Sequence[str]) -> tuple[str, ...]:
    missing_special_tokens = [
        token for token in required_special_tokens
        if tokenizer.token_to_id(token) is None
    ]
    if missing_special_tokens:
        tokenizer.add_special_tokens(list(missing_special_tokens))
    return _byte_bpe_special_tokens(tokenizer)


def _encode_byte_bpe_preserving_specials(
    tokenizer: Any,
    text: str,
    *,
    special_tokens: Sequence[str],
    special_pattern: re.Pattern[str] | None = None,
) -> list[int]:
    if not special_tokens:
        return list(map(int, tokenizer.encode(text).ids))
    pattern = special_pattern or _byte_bpe_special_pattern(special_tokens)
    if pattern is None:
        return list(map(int, tokenizer.encode(text).ids))
    operations: list[tuple[str, int | str]] = []
    plain_spans: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            operations.append(("plain", len(plain_spans)))
            plain_spans.append(text[cursor : match.start()])
        token_id = tokenizer.token_to_id(match.group(0))
        if token_id is None:
            operations.append(("plain", len(plain_spans)))
            plain_spans.append(match.group(0))
        else:
            operations.append(("special", int(token_id)))
        cursor = match.end()
    if cursor < len(text):
        operations.append(("plain", len(plain_spans)))
        plain_spans.append(text[cursor:])
    token_ids: list[int] = []
    encoded_spans = (
        [list(map(int, encoded.ids)) for encoded in tokenizer.encode_batch(plain_spans)]
        if plain_spans
        else []
    )
    for op_kind, payload in operations:
        if op_kind == "plain":
            token_ids.extend(encoded_spans[int(payload)])
        else:
            token_ids.append(int(payload))
    return token_ids


def _init_byte_bpe_encode_worker(vocab_path: str, merges_path: str, required_special_tokens: Sequence[str] = ()) -> None:
    global _BYTE_BPE_WORKER_TOKENIZER, _BYTE_BPE_WORKER_SPECIAL_TOKENS, _BYTE_BPE_WORKER_SPECIAL_PATTERN
    if ByteLevelBPETokenizer is None:
        raise ImportError("tokenizers is required for parallel byte BPE encoding.")
    _BYTE_BPE_WORKER_TOKENIZER = ByteLevelBPETokenizer(vocab_path, merges_path)
    _BYTE_BPE_WORKER_SPECIAL_TOKENS = _ensure_byte_bpe_special_tokens(
        _BYTE_BPE_WORKER_TOKENIZER,
        required_special_tokens,
    )
    _BYTE_BPE_WORKER_SPECIAL_PATTERN = _byte_bpe_special_pattern(_BYTE_BPE_WORKER_SPECIAL_TOKENS)


def _encode_byte_bpe_text_worker(text: str) -> list[int]:
    if _BYTE_BPE_WORKER_TOKENIZER is None:
        raise RuntimeError("Byte BPE worker was not initialized.")
    return _encode_byte_bpe_preserving_specials(
        _BYTE_BPE_WORKER_TOKENIZER,
        text,
        special_tokens=_BYTE_BPE_WORKER_SPECIAL_TOKENS,
        special_pattern=_BYTE_BPE_WORKER_SPECIAL_PATTERN,
    )


def _encode_byte_bpe_text_batch_worker(texts: Sequence[str]) -> list[list[int]]:
    if _BYTE_BPE_WORKER_TOKENIZER is None:
        raise RuntimeError("Byte BPE worker was not initialized.")
    return [
        _encode_byte_bpe_preserving_specials(
            _BYTE_BPE_WORKER_TOKENIZER,
            text,
            special_tokens=_BYTE_BPE_WORKER_SPECIAL_TOKENS,
            special_pattern=_BYTE_BPE_WORKER_SPECIAL_PATTERN,
        )
        for text in texts
    ]


def _encode_byte_bpe_dialogue_pair_worker(pair: tuple[str, str]) -> tuple[list[int], list[int]]:
    if _BYTE_BPE_WORKER_TOKENIZER is None:
        raise RuntimeError("Byte BPE worker was not initialized.")
    prefix, response = pair
    return (
        _encode_byte_bpe_preserving_specials(
            _BYTE_BPE_WORKER_TOKENIZER,
            prefix,
            special_tokens=_BYTE_BPE_WORKER_SPECIAL_TOKENS,
            special_pattern=_BYTE_BPE_WORKER_SPECIAL_PATTERN,
        ),
        _encode_byte_bpe_preserving_specials(
            _BYTE_BPE_WORKER_TOKENIZER,
            response,
            special_tokens=_BYTE_BPE_WORKER_SPECIAL_TOKENS,
            special_pattern=_BYTE_BPE_WORKER_SPECIAL_PATTERN,
        ),
    )


def encode_text_in_chunks(
    vocab: object,
    text: str,
    *,
    chunk_chars: int = 8_000_000,
    workers: int = 0,
) -> torch.Tensor:
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive.")
    if workers > 1 and isinstance(vocab, ByteBPEVocab):
        token_parts: list[torch.Tensor] = []
        chunks = list(_iter_nonempty_text_chunks(text, chunk_chars=chunk_chars))
        if not chunks:
            return torch.empty(0, dtype=torch.long)
        batch_size = max(1, min(64, len(chunks)))
        chunk_batches = [chunks[index : index + batch_size] for index in range(0, len(chunks), batch_size)]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_byte_bpe_encode_worker,
            initargs=(str(vocab.vocab_path), str(vocab.merges_path), vocab.special_tokens),
        ) as executor:
            for token_id_batch in executor.map(_encode_byte_bpe_text_batch_worker, chunk_batches, chunksize=1):
                for token_ids in token_id_batch:
                    if token_ids:
                        token_parts.append(torch.tensor(token_ids, dtype=torch.long))
        if not token_parts:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(token_parts)

    token_parts: list[torch.Tensor] = []
    for chunk in _iter_nonempty_text_chunks(text, chunk_chars=chunk_chars):
        if isinstance(vocab, ByteBPEVocab):
            encoded = torch.tensor(
                _encode_byte_bpe_preserving_specials(
                    vocab.tokenizer,
                    chunk,
                    special_tokens=vocab.special_tokens or _byte_bpe_special_tokens(vocab.tokenizer),
                    special_pattern=vocab.special_pattern,
                ),
                dtype=torch.long,
            )
        else:
            encoded = vocab.encode(chunk)
        if encoded.numel() > 0:
            token_parts.append(encoded)
    if not token_parts:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(token_parts)


def split_prediction_units(text: str, *, min_chars: int = 8) -> list[str]:
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        line_candidates = [line.strip() for line in paragraph.splitlines() if line.strip()]
        code_like = len(line_candidates) >= 3 and any(
            token in paragraph for token in ("def ", "class ", "{", "}", "=>", ";</", "import ")
        )
        if code_like:
            candidates = line_candidates
        else:
            normalized = re.sub(r"\s+", " ", paragraph)
            candidates = [piece.strip() for piece in SENTENCE_BOUNDARY_RE.split(normalized)]
            if len(candidates) <= 1 and len(line_candidates) > 1:
                candidates = line_candidates
        for candidate in candidates:
            candidate = candidate.strip()
            if len(candidate) >= min_chars:
                units.append(candidate)
    return units


def make_next_sentence_pairs(
    texts: Sequence[str],
    *,
    prefix_sentences: int = 1,
    min_chars: int = 8,
    max_pairs: int | None = None,
) -> list[DialoguePairText]:
    if prefix_sentences <= 0:
        raise ValueError("prefix_sentences must be positive.")
    pairs: list[DialoguePairText] = []
    for text in texts:
        units = split_prediction_units(text, min_chars=min_chars)
        if len(units) <= prefix_sentences:
            continue
        for index in range(prefix_sentences, len(units)):
            prefix = " ".join(units[index - prefix_sentences:index])
            response = units[index]
            pairs.append(DialoguePairText(prefix=prefix, response=response))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def load_next_sentence_pairs(
    *,
    default_text: str,
    text_file: str | None,
    text_sources: Sequence[str],
    jsonl_sources: Sequence[str],
    jsonl_text_keys: Sequence[str],
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    hf_text_key: str,
    hf_streaming: bool,
    max_samples: int | None,
    max_chars: int | None,
    separator: str,
    prefix_sentences: int,
    min_chars: int,
) -> tuple[list[DialoguePairText], CorpusLoadResult]:
    from lm_experiment_utils import _expand_sources

    paths = _expand_sources(jsonl_sources, directory_suffixes=(".jsonl", ".json"))
    explicit_pairs: list[DialoguePairText] = []
    text_jsonl_paths: list[str] = []
    for path in paths:
        path_pairs: list[DialoguePairText] = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                path_pairs.extend(_pairs_from_record(json.loads(line)))
                if max_samples is not None and len(explicit_pairs) + len(path_pairs) >= max_samples:
                    break
        if path_pairs:
            explicit_pairs.extend(path_pairs)
        else:
            text_jsonl_paths.append(str(path))
        if max_samples is not None and len(explicit_pairs) >= max_samples:
            explicit_pairs = explicit_pairs[:max_samples]
            break

    has_text_sources = bool(text_file or text_sources or text_jsonl_paths or hf_dataset or not explicit_pairs)
    if has_text_sources:
        corpus = load_text_corpus(
            default_text=default_text,
            text_file=text_file,
            text_sources=text_sources,
            jsonl_sources=text_jsonl_paths,
            jsonl_text_keys=jsonl_text_keys,
            hf_dataset=hf_dataset,
            hf_config=hf_config,
            hf_split=hf_split,
            hf_text_key=hf_text_key,
            hf_streaming=hf_streaming,
            max_samples=max_samples if not explicit_pairs else None,
            max_chars=max_chars if not explicit_pairs else None,
            separator=separator,
        )
        generated_pairs = make_next_sentence_pairs(
            [corpus.text],
            prefix_sentences=prefix_sentences,
            min_chars=min_chars,
        )
    else:
        pair_text = "\n\n".join(f"{pair.prefix}\n{pair.response}" for pair in explicit_pairs)
        corpus = CorpusLoadResult(
            text=pair_text,
            source_label="next_sentence_pairs",
            text_path=None,
            sample_count=len(explicit_pairs),
            file_count=len(paths),
            char_count=len(pair_text),
            truncated=False,
            metadata={"source_kind": "next_sentence_pairs", "jsonl_paths": [str(path) for path in paths]},
        )
        generated_pairs = []
    pairs = explicit_pairs + generated_pairs
    if max_samples is not None:
        pairs = pairs[:max_samples]
    if max_chars is not None and explicit_pairs:
        selected: list[DialoguePairText] = []
        total_chars = 0
        for pair in pairs:
            pair_chars = len(pair.prefix) + len(pair.response)
            if total_chars + pair_chars > max_chars:
                break
            selected.append(pair)
            total_chars += pair_chars
        pairs = selected
    if not pairs:
        raise ValueError("No next-sentence pairs were loaded.")
    pair_text = "\n\n".join(f"{pair.prefix}\n{pair.response}" for pair in pairs)
    corpus = CorpusLoadResult(
        text=pair_text,
        source_label=f"next_sentence_pairs,{corpus.source_label}",
        text_path=corpus.text_path,
        sample_count=len(pairs),
        file_count=corpus.file_count,
        char_count=len(pair_text),
        truncated=corpus.truncated,
        metadata={
            **corpus.metadata,
            "source_kind": "next_sentence_pairs",
            "explicit_pair_count": len(explicit_pairs),
            "generated_pair_count": len(generated_pairs),
            "prefix_sentences": prefix_sentences,
            "sentence_min_chars": min_chars,
        },
    )
    return pairs, corpus


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


def build_source_balanced_index_groups(
    pairs: Sequence[DialoguePairText],
) -> tuple[tuple[int, ...], ...]:
    buckets: dict[str, list[int]] = {}
    for index, pair in enumerate(pairs):
        source = pair.source.split(":", 1)[0] if pair.source else "unknown"
        buckets.setdefault(source, []).append(index)
    return tuple(tuple(indices) for _, indices in sorted(buckets.items()))


def summarize_source_groups(
    pairs: Sequence[DialoguePairText],
    groups: Sequence[Sequence[int]],
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for group in groups:
        if not group:
            continue
        source = pairs[int(group[0])].source.split(":", 1)[0] if pairs[int(group[0])].source else "unknown"
        summary[source] = len(group)
    return summary


def split_dialogue_pairs_by_source(
    pairs: Sequence[DialoguePairText],
    *,
    train_fraction: float = 0.9,
) -> tuple[list[DialoguePairText], list[DialoguePairText]]:
    buckets: dict[str, list[DialoguePairText]] = {}
    for pair in pairs:
        source = pair.source.split(":", 1)[0] if pair.source else "unknown"
        buckets.setdefault(source, []).append(pair)
    train: list[DialoguePairText] = []
    val: list[DialoguePairText] = []
    for source in sorted(buckets):
        bucket = buckets[source]
        split_index = int(len(bucket) * train_fraction)
        split_index = max(1, min(split_index, len(bucket) - 1)) if len(bucket) > 1 else len(bucket)
        train.extend(bucket[:split_index])
        val.extend(bucket[split_index:] or bucket[:split_index])
    return train, val


def _init_subword_encode_worker(model_path: str) -> None:
    global _SUBWORD_WORKER_PROCESSOR
    if spm is None:
        raise ImportError("sentencepiece is required for parallel subword encoding.")
    _SUBWORD_WORKER_PROCESSOR = spm.SentencePieceProcessor(model_file=model_path)


def _encode_subword_dialogue_pair_worker(pair: tuple[str, str]) -> tuple[list[int], list[int]]:
    if _SUBWORD_WORKER_PROCESSOR is None:
        raise RuntimeError("Subword worker was not initialized.")
    prefix, response = pair
    return (
        list(map(int, _SUBWORD_WORKER_PROCESSOR.encode(prefix, out_type=int))),
        list(map(int, _SUBWORD_WORKER_PROCESSOR.encode(response, out_type=int))),
    )


def encode_dialogue_pairs(
    pairs: Sequence[DialoguePairText],
    *,
    vocab: object,
    workers: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if workers <= 1:
        return [(vocab.encode(pair.prefix), vocab.encode(pair.response)) for pair in pairs]
    pair_count = len(pairs)
    chunksize = max(1, pair_count // (workers * 16)) if pair_count else 1
    text_pairs = ((pair.prefix, pair.response) for pair in pairs)
    encoded: list[tuple[torch.Tensor, torch.Tensor]] = []
    if isinstance(vocab, SubwordVocab):
        if vocab.model_path is None:
            raise ValueError("Parallel subword encoding requires a tokenizer model path.")
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_subword_encode_worker,
            initargs=(str(vocab.model_path),),
        ) as executor:
            for prefix_ids, response_ids in executor.map(
                _encode_subword_dialogue_pair_worker,
                text_pairs,
                chunksize=chunksize,
            ):
                encoded.append(
                    (
                        torch.tensor(prefix_ids, dtype=torch.long),
                        torch.tensor(response_ids, dtype=torch.long),
                    )
                )
        return encoded
    if isinstance(vocab, ByteBPEVocab):
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_byte_bpe_encode_worker,
            initargs=(str(vocab.vocab_path), str(vocab.merges_path), vocab.special_tokens),
        ) as executor:
            for prefix_ids, response_ids in executor.map(
                _encode_byte_bpe_dialogue_pair_worker,
                text_pairs,
                chunksize=chunksize,
            ):
                encoded.append(
                    (
                        torch.tensor(prefix_ids, dtype=torch.long),
                        torch.tensor(response_ids, dtype=torch.long),
                    )
                )
        return encoded
    return [(vocab.encode(pair.prefix), vocab.encode(pair.response)) for pair in pairs]


def require_special_token_ids(vocab: object) -> dict[str, int]:
    if not hasattr(vocab, "token_id"):
        raise ValueError(
            "This objective requires a tokenizer with explicit special-token ids "
            "(use --tokenizer subword or --tokenizer byte_bpe)."
        )
    return {
        "user": vocab.token_id(USER_TOKEN),
        "assistant": vocab.token_id(ASSISTANT_TOKEN),
        "eos": vocab.token_id(EOS_TOKEN),
        "pad": vocab.token_id(PAD_TOKEN),
        "query_block_start": vocab.token_id(QUERY_BLOCK_START_TOKEN),
    }


def append_eos_markers_to_corpus(corpus: CorpusLoadResult, *, separator: str) -> CorpusLoadResult:
    eos_marker = f"\n{EOS_TOKEN}\n"
    text = corpus.text
    if separator and separator in text:
        text = text.replace(separator, eos_marker)
    text = text.rstrip()
    if not text.endswith(EOS_TOKEN):
        text = f"{text}{eos_marker}"
    else:
        text = f"{text}\n"
    return CorpusLoadResult(
        text=text,
        source_label=f"{corpus.source_label}+eos",
        text_path=None,
        sample_count=corpus.sample_count,
        file_count=corpus.file_count,
        char_count=len(text),
        truncated=corpus.truncated,
        metadata={**corpus.metadata, "eos_markers_inserted": True},
    )


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
    training_objective: str = "query_block",
    eos_token_id: int | None = None,
    target_len: int = 1,
    query_block_start_token_id: int | None = None,
    sample_source: str = "query_head",
) -> torch.Tensor:
    if temperature is not None and temperature <= 0.0:
        raise ValueError("temperature must be positive when sampling is enabled.")
    if sample_topk is not None and sample_topk <= 0:
        raise ValueError("sample-topk must be positive.")
    if sample_source != "query_head":
        raise ValueError(f"Unsupported sample_source: {sample_source!r}.")

    def sample_from_logits(step_logits: torch.Tensor) -> torch.Tensor:
        if temperature is None:
            return torch.argmax(step_logits, dim=-1)
        scaled_logits = step_logits / temperature
        if sample_topk is not None:
            k = min(sample_topk, scaled_logits.shape[-1])
            topk_logits, topk_indices = torch.topk(scaled_logits, k=k, dim=-1)
            probs = torch.softmax(topk_logits, dim=-1)
            sampled_offset = torch.multinomial(probs, num_samples=1)
            return topk_indices.gather(-1, sampled_offset).reshape(-1)
        probs = torch.softmax(scaled_logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).reshape(-1)

    was_training = model.training
    model.eval()
    generated = prompt.to(device).clone()
    if generated.numel() == 0:
        if was_training:
            model.train()
        return generated
    tokens_remaining = max_new_tokens
    while tokens_remaining > 0:
        context = generated[-seq_len:].unsqueeze(0)
        if context.shape[-1] <= 0:
            break
        start_offset = 0
        if training_objective == "query_block":
            content_len = min(max(1, target_len), tokens_remaining)
            query_nodes = content_len
            query_seed_token_ids = None
            if query_block_start_token_id is not None:
                query_seed_token_ids = torch.full(
                    (1, 1),
                    fill_value=query_block_start_token_id,
                    dtype=torch.long,
                    device=context.device,
                )
            logits = model.forward_query_block(
                context,
                target_len=query_nodes,
                query_seed_token_ids=query_seed_token_ids,
            )
            if isinstance(logits, tuple):
                logits = logits[0]
        elif training_objective == "query_next_token":
            logits = model.forward_query_next_token(context)
        else:
            raise ValueError(f"Unsupported query generation objective: {training_objective!r}.")
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        sampled_tokens: list[torch.Tensor] = []
        for offset in range(start_offset, logits.shape[1]):
            step_logits = logits[:, offset, :]
            next_token = sample_from_logits(step_logits)
            sampled_tokens.append(next_token)
            tokens_remaining -= 1
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                tokens_remaining = 0
                break
        if not sampled_tokens:
            break
        generated = torch.cat((generated, torch.cat(sampled_tokens, dim=0)), dim=0)
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
    raise RuntimeError("Prefix-response generation was removed; use query objectives.")


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
    train_step_stats = history.train_step_stats
    for step, (loss, grad_norm) in enumerate(
        zip(history.train_step_losses, history.grad_norms, strict=True),
        start=1,
    ):
        epoch_step = step / steps_per_epoch
        row = {
            "record_type": "train_step",
            "step": step,
            "epoch": epoch_step,
            "epoch_step": epoch_step,
            "minibatch_loss": loss,
            "grad_norm": grad_norm,
        }
        stats = train_step_stats[step - 1] if step - 1 < len(train_step_stats) else {}
        query_head_loss = stats.get("loss/query_head")
        if query_head_loss is not None:
            row["query_head_loss"] = query_head_loss
        avg_query_head_loss = stats.get("loss/avg_query_head")
        if avg_query_head_loss is not None:
            row["avg_ppl"] = perplexity_from_loss(avg_query_head_loss)
        for key, value in sorted(stats.items()):
            if key.startswith("query_block_pos_loss/"):
                bucket = key.removeprefix("query_block_pos_loss/")
                row[f"ppl_distance_{bucket}"] = perplexity_from_loss(value)
        rows.append(row)
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
    purge_step: int | None = None,
) -> SummaryWriter | None:
    if not enabled:
        return None
    if SummaryWriter is None:
        raise ImportError(
            "TensorBoard logging requested, but torch.utils.tensorboard is unavailable. "
            "Install dependencies from requirements-base.txt first."
        )
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tensorboard_dir), purge_step=purge_step)


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
            epoch_step = step / steps_per_epoch
            writer.add_scalar("train/minibatch_loss", loss, step)
            writer.add_scalar("train/epoch", epoch_step, step)
        for step, grad_norm in enumerate(history.grad_norms, start=1):
            writer.add_scalar("train/grad_norm", grad_norm, step)
        for step, stats in enumerate(history.train_step_stats, start=1):
            avg_query_head_loss = stats.get("loss/avg_query_head")
            if avg_query_head_loss is not None:
                writer.add_scalar("train/avg_ppl", perplexity_from_loss(avg_query_head_loss), step)
            bucket_ppl = {
                key.removeprefix("query_block_pos_loss/"): perplexity_from_loss(value)
                for key, value in sorted(stats.items())
                if key.startswith("query_block_pos_loss/")
            }
            if bucket_ppl:
                writer.add_scalars("train/ppl_by_distance", bucket_ppl, step)
        for step, train_loss, val_loss in zip(
            history.eval_steps,
            history.train_losses,
            history.val_losses,
            strict=True,
        ):
            writer.add_scalar("eval/metrics/train_loss", train_loss, step)
            writer.add_scalar("eval/metrics/val_loss", val_loss, step)
            writer.add_scalar("eval/metrics/train_ppl", perplexity_from_loss(train_loss), step)
            writer.add_scalar("eval/metrics/val_ppl", perplexity_from_loss(val_loss), step)
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


def select_pair_prompt_tokens(
    *,
    args: argparse.Namespace,
    vocab: object,
    train_pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, str]:
    if args.prompt_text:
        prompt = args.prompt_text
        prompt_ids = vocab.encode(prompt)
        if prompt_ids.numel() == 0:
            raise ValueError("--prompt-text must encode to at least one token.")
        return prompt_ids, prompt
    if not train_pairs:
        raise ValueError("pair prompt selection requires at least one train pair.")
    prompt_ids = train_pairs[0][0]
    return prompt_ids, vocab.decode(prompt_ids.tolist())


def format_tensorboard_text(text: str) -> str:
    lines = text.splitlines() or [text]
    if not lines:
        lines = [""]
    return "\n".join(f"    {line}" for line in lines)


def format_tensorboard_sample(
    *,
    prompt_text: str,
    generated_text: str,
) -> str:
    return (
        "### Prompt\n\n"
        f"<div>{format_tensorboard_text(prompt_text)}</div>\n\n"
        "### Generated\n\n"
        f"<div>{format_tensorboard_text(generated_text)}</div>"
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
    eval_runtime_rows: Sequence[dict[str, Any]],
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
    if eval_runtime_rows:
        write_jsonl(run_dir / "eval_runtime_metrics.jsonl", eval_runtime_rows)
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
    train_balanced_index_groups: Sequence[Sequence[int]] | None,
    train_balanced_token_groups: Sequence[torch.Tensor] | None,
    val_balanced_index_groups: Sequence[Sequence[int]] | None,
    val_balanced_token_groups: Sequence[torch.Tensor] | None,
    device: torch.device | str,
    teacher_forcing: bool,
    full_sequence_causal: bool,
    query_next_token: bool,
    query_block: bool,
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
    response_objective = False
    pair_query_block = query_block and train_response_pairs is not None
    query_block_start_token_id: int | None = None
    if response_objective:
        if train_response_pairs is None or val_response_pairs is None or special_token_ids is None:
            raise ValueError("response objectives require encoded pairs and special ids.")
        if args.response_len <= 0:
            raise ValueError("response-len must be positive.")
        steps_per_epoch = max(1, math.ceil(len(train_response_pairs) / effective_batch_size))
    elif pair_query_block:
        if train_response_pairs is None or val_response_pairs is None or special_token_ids is None:
            raise ValueError("pair query_block requires encoded pairs and special ids.")
        steps_per_epoch = max(1, math.ceil(len(train_response_pairs) / effective_batch_size))
        query_block_start_token_id = special_token_ids["query_block_start"]
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
    effective_eval_interval = steps_per_epoch if args.eval_every_epoch else args.eval_interval
    if effective_eval_interval <= 0:
        raise ValueError("eval interval must be positive.")
    eval_on_first_step = not args.eval_every_epoch

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
        query_refine_layers=args.query_refine_layers,
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
        block_residual=args.block_residual,
        query_residual=args.query_residual,
        value_residual_scale=args.value_residual_scale,
        state_residual_scale=args.state_residual_scale,
        alpha_scale=args.alpha_scale,
        beta_s_to_b_scale=args.beta_s_to_b_scale,
        beta_b_to_s_scale=args.beta_b_to_s_scale,
        warmup_delta_scale=args.warmup_delta_scale,
        s_delta_scale=args.s_delta_scale,
        b_delta_scale=args.b_delta_scale,
        cross_delta_scale=args.cross_delta_scale,
        route_temperature=args.route_temperature,
        route_kind=args.route_kind,
        route_hidden_dim=args.route_hidden_dim,
        state_init_mode=args.state_init_mode,
        pairwise_kind=args.pairwise_kind,
        pairwise_hidden_dim=args.pairwise_hidden_dim,
        query_topk=args.query_topk or int(effective_config["route_topk"]),
        edge_dropout_p=args.edge_dropout_p,
        usage_dropout_base=args.usage_dropout_base,
        usage_dropout_scale=args.usage_dropout_scale,
        usage_dropout_max=args.usage_dropout_max,
        usage_ema_decay=args.usage_ema_decay,
        b_slot_dropout_p=args.b_slot_dropout_p,
        s_to_b_slot_dropout_p=args.s_to_b_slot_dropout_p,
        b_to_s_slot_dropout_p=args.b_to_s_slot_dropout_p,
        include_query_head=query_next_token or query_block,
        share_route_families=args.share_route_families,
    )
    if args.freeze_position_encoding:
        for parameter in model.position_encoding.parameters():
            parameter.requires_grad_(False)
    parameter_count = count_parameters(model)

    run_dir = session_dir / slugify_run_name(experiment_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = (
        Path(args.tensorboard_dir) / session_dir.name / slugify_run_name(experiment_name)
        if args.tensorboard_dir
        else run_dir / "tensorboard"
    )

    print(
        f"experiment={experiment_name} | params={parameter_count:,} | "
        f"dim={effective_config['dim']} | warmup={effective_config['warmup_layers']} | "
        f"stages={effective_config['lite_layers']}/{effective_config['mid_layers']}/{effective_config['full_layers']} | "
        f"refine={effective_config['final_refine_layers']} | qrefine={args.query_refine_layers} | route_mode={args.route_mode} | "
        f"route_topk={effective_config['route_topk']} | route_kind={args.route_kind} | "
        f"query_topk={args.query_topk or int(effective_config['route_topk'])} | "
        f"block_residual={args.block_residual} | query_residual={args.query_residual} | "
        f"share_route_families={args.share_route_families} | "
        f"edge_dropout={args.edge_dropout_p:.3f} | "
        f"usage_dropout={args.usage_dropout_base:.3f}/{args.usage_dropout_scale:.3f}/{args.usage_dropout_max:.3f} | "
        f"b_slot_dropout={args.b_slot_dropout_p:.3f} | "
        f"slot_dropout(s_to_b/b_to_s)={args.s_to_b_slot_dropout_p:.3f}/{args.b_to_s_slot_dropout_p:.3f}"
    )
    print(
        f"schedule_steps={train_steps:,} | approx_epochs={requested_epochs:.3f} | "
        f"steps_per_epoch={steps_per_epoch:,} | batch_size={args.batch_size} | "
        f"grad_accum_steps={grad_accum_steps} | effective_batch_size={effective_batch_size} | "
        f"objective={args.training_objective} | target_len={args.target_len} | "
        f"response_len={args.response_len} | "
        f"eval_interval={effective_eval_interval:,} | eval_every_epoch={args.eval_every_epoch} | "
        f"data_workers={args.data_workers} | "
        f"prefetch_factor={args.prefetch_factor}"
    )
    if (
        args.b_diversity_loss_weight > 0.0
        or args.b_diversity_unweighted_per_layer
        or args.route_concentration_loss_weight > 0.0
    ):
        print(
            "aux_losses | "
            f"b_diversity_weight={args.b_diversity_loss_weight:.6g} | "
            f"b_diversity_per_layer_weight={args.b_diversity_per_layer_weight:.6g} | "
            f"b_cosine_margin={args.b_cosine_margin:.3f} | "
            f"b_cosine_early_margin={args.b_cosine_early_margin:.3f} | "
            f"b_diversity_unweighted_per_layer={args.b_diversity_unweighted_per_layer} | "
            f"route_concentration_weight={args.route_concentration_loss_weight:.6g} | "
            f"route_load_cap={args.route_load_cap:.3f} | "
            f"edge_prob_cap={args.edge_prob_cap:.3f}"
        )
    print(
        f"learning_rate={args.learning_rate:.6g} | "
        f"lr_schedule={args.learning_rate_schedule} | "
        f"lr_warmup_steps={args.learning_rate_warmup_steps} | "
        f"lr_warmup_start={args.learning_rate_warmup_start} | "
        f"lr_min_ratio={args.learning_rate_min_ratio:.3f} | "
        f"query_block_front_weight={args.query_block_front_weight:.3f} | "
        f"freeze_position_encoding={args.freeze_position_encoding} | "
        f"grad_breakdown={args.grad_breakdown} | "
        f"grad_breakdown_start_step={args.grad_breakdown_start_step}"
    )

    start_time = time.perf_counter()
    writer: SummaryWriter | None = None
    eval_runtime_rows: list[dict[str, Any]] = []
    resume_checkpoint: dict[str, Any] | None = None
    resume_step = 0
    optimizer_state_dict: dict[str, object] | None = None
    if args.resume_checkpoint:
        resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        resume_step = int(resume_checkpoint.get("step") or 0)
        resume_model_state_dict = resume_checkpoint["model_state_dict"]
        if args.share_route_families:
            resume_model_state_dict, route_merge_counts = merge_route_families_for_resume(
                checkpoint_state_dict=resume_model_state_dict,
                model_state_dict=model.state_dict(),
            )
            if route_merge_counts:
                route_merge_counts_text = ", ".join(
                    f"{role}={count}" for role, count in sorted(route_merge_counts.items())
                )
                print(f"resume_route_merge_by_role | {route_merge_counts_text}")
        load_result = model.load_state_dict(resume_model_state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "resume_state_dict_load | "
                f"missing={list(load_result.missing_keys)} | "
                f"unexpected={list(load_result.unexpected_keys)}"
            )
        raw_optimizer_state = resume_checkpoint.get("optimizer_state_dict")
        optimizer_state_dict = (
            None
            if args.reset_optimizer_on_resume
            else (raw_optimizer_state if isinstance(raw_optimizer_state, dict) else None)
        )
        restore_rng_state(resume_checkpoint.get("rng_state"))
        if resume_step >= train_steps:
            raise ValueError(
                f"Resume checkpoint step {resume_step} is already >= requested train_steps {train_steps}."
            )
        print(
            f"resume_checkpoint={args.resume_checkpoint} | resume_step={resume_step:,} | "
            f"optimizer_reset={args.reset_optimizer_on_resume}"
        )
    writer = maybe_create_summary_writer(
        enabled=args.tensorboard,
        tensorboard_dir=tensorboard_dir,
        purge_step=resume_step + 1 if resume_step > 0 else None,
    )

    best_checkpoint_val_loss = math.inf
    if resume_checkpoint is not None and resume_checkpoint.get("val_loss") is not None:
        best_checkpoint_val_loss = float(resume_checkpoint["val_loss"])
    progress_bar: Any | None = None
    last_progress_step = resume_step
    if args.tqdm:
        if tqdm is None:
            raise ImportError("tqdm is required when --tqdm is set.")
        progress_bar = tqdm(
            total=requested_epochs,
            initial=resume_step / steps_per_epoch,
            unit="epoch",
            dynamic_ncols=True,
            desc=experiment_name,
        )

    def make_checkpoint_payload(
        *,
        step: int,
        train_loss: float | None,
        val_loss: float | None,
        checkpoint_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        return {
            "checkpoint_kind": checkpoint_kind,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "model_state_dict": checkpoint_model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "rng_state": rng_state_payload(),
            "config": {
                "vocab_size": vocab.size,
                "tokenizer": tokenizer_label,
                "tokenizer_model_path": None if tokenizer_model_path is None else str(tokenizer_model_path),
                "training_objective": args.training_objective,
                "seq_nodes": args.seq_len,
                "target_len": args.target_len,
                "response_len": args.response_len,
                "implementation": args.implementation,
                "precision": args.precision,
                "value_norm_kind": value_norm_kind,
                "freeze_position_encoding": args.freeze_position_encoding,
                "norm_position": args.norm_position,
                "propagation_residual": not args.disable_propagation_residual,
                "route_mode": args.route_mode,
                "query_topk": args.query_topk or int(effective_config["route_topk"]),
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
                "b_slot_dropout_p": args.b_slot_dropout_p,
                "s_to_b_slot_dropout_p": args.s_to_b_slot_dropout_p,
                "b_to_s_slot_dropout_p": args.b_to_s_slot_dropout_p,
                "b_diversity_unweighted_per_layer": args.b_diversity_unweighted_per_layer,
                "b_diversity_per_layer_weight": args.b_diversity_per_layer_weight,
                "b_cosine_early_margin": args.b_cosine_early_margin,
                "learning_rate_schedule": args.learning_rate_schedule,
                "learning_rate_warmup_steps": args.learning_rate_warmup_steps,
                "learning_rate_warmup_start": args.learning_rate_warmup_start,
                "learning_rate_min_ratio": args.learning_rate_min_ratio,
                "query_block_front_weight": args.query_block_front_weight,
                "alpha_scale": args.alpha_scale,
                "beta_s_to_b_scale": args.beta_s_to_b_scale,
                "beta_b_to_s_scale": args.beta_b_to_s_scale,
                "warmup_delta_scale": args.warmup_delta_scale,
                "s_delta_scale": args.s_delta_scale,
                "b_delta_scale": args.b_delta_scale,
                "cross_delta_scale": args.cross_delta_scale,
                "b_schedule": args.b_schedule,
                "b_schedule_min": args.b_schedule_min,
                "b_schedule_max": args.b_schedule_max,
                "stage_specs": [asdict(spec) for spec in stage_specs],
                **{key: int(value) if isinstance(value, int) else value for key, value in effective_config.items()},
            },
            "metadata": {
                "experiment": experiment_name,
                "parameter_count": parameter_count,
                "corpus_source_label": corpus.source_label,
                "corpus_char_count": corpus.char_count,
                "corpus_sample_count": corpus.sample_count,
                "steps_per_epoch": steps_per_epoch,
                "train_steps": train_steps,
            },
        }

    def save_checkpoint(
        *,
        step: int,
        train_loss: float | None,
        val_loss: float | None,
        checkpoint_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        checkpoint_kind: str,
        filename: str,
    ) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / filename
        torch.save(
            make_checkpoint_payload(
                step=step,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_model=checkpoint_model,
                optimizer=optimizer,
                checkpoint_kind=checkpoint_kind,
            ),
            path,
        )
        write_json(
            checkpoint_dir / f"{path.stem}.json",
            {
                "path": str(path),
                "checkpoint_kind": checkpoint_kind,
                "step": step,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        )
        return path

    def training_checkpoint_callback(
        step: int,
        train_loss: float,
        val_loss: float,
        checkpoint_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        nonlocal best_checkpoint_val_loss
        if not args.save_checkpoint:
            return
        last_path = save_checkpoint(
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            checkpoint_model=checkpoint_model,
            optimizer=optimizer,
            checkpoint_kind="last",
            filename="last.pt",
        )
        if val_loss is None:
            interval_path = save_checkpoint(
                step=step,
                train_loss=train_loss,
                val_loss=None,
                checkpoint_model=checkpoint_model,
                optimizer=optimizer,
                checkpoint_kind="interval",
                filename=f"step_{step:06d}.pt",
            )
            save_checkpoint(
                step=step,
                train_loss=train_loss,
                val_loss=None,
                checkpoint_model=checkpoint_model,
                optimizer=optimizer,
                checkpoint_kind="last",
                filename="last.pt",
            )
            checkpoint_message = f"checkpoint | step={step} | interval | path={interval_path}"
        elif val_loss < best_checkpoint_val_loss:
            best_checkpoint_val_loss = val_loss
            best_path = save_checkpoint(
                step=step,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_model=checkpoint_model,
                optimizer=optimizer,
                checkpoint_kind="best",
                filename="best.pt",
            )
            checkpoint_message = (
                f"checkpoint | step={step} | best_val_loss={val_loss:.4f} | path={best_path}"
            )
        else:
            checkpoint_message = (
                f"checkpoint | step={step} | val_loss={val_loss:.4f} | path={last_path}"
            )
        if progress_bar is not None:
            progress_bar.write(checkpoint_message)
        else:
            print(checkpoint_message)

    def select_eval_prompt() -> tuple[torch.Tensor, str]:
        if response_objective:
            assert train_response_pairs is not None
            assert special_token_ids is not None
            prompt_ids, prompt_text = select_prefix_response_prompt_tokens(
                args=args,
                vocab=vocab,
                train_response_pairs=train_response_pairs,
            )
            return prompt_ids, prompt_text
        if pair_query_block:
            assert train_response_pairs is not None
            prompt_ids, prompt_text = select_pair_prompt_tokens(
                args=args,
                vocab=vocab,
                train_pairs=train_response_pairs,
            )
        else:
            prompt_ids, prompt_text = select_prompt_tokens(args=args, vocab=vocab, train_tokens=train_tokens)
        return prompt_ids, prompt_text

    def make_eval_sample(
        prompt_ids: torch.Tensor,
        prompt_text: str,
        *,
        sample_source: str = "query_head",
    ) -> tuple[str, str]:
        if response_objective:
            assert special_token_ids is not None
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
        generated = generate_next_tokens_with_sampling(
            model,
            prompt_ids,
            max_new_tokens=args.sample_tokens,
            seq_len=args.seq_len,
            device=device,
            temperature=args.temperature,
            sample_topk=args.sample_topk,
            training_objective=args.training_objective,
            target_len=args.target_len,
            eos_token_id=(
                vocab.token_id(EOS_TOKEN)
                if args.training_objective in {"query_next_token", "query_block"}
                and hasattr(vocab, "token_id")
                else None
            ),
            query_block_start_token_id=query_block_start_token_id if query_block else None,
            sample_source=sample_source,
        )
        return prompt_text, vocab.decode(generated.tolist())

    def progress_callback(step: int, total_steps: int, minibatch_loss: float) -> None:
        nonlocal last_progress_step
        if progress_bar is not None:
            progress_bar.update((step - last_progress_step) / steps_per_epoch)
            last_progress_step = step
            progress_bar.set_postfix(
                step=f"{step}/{total_steps}",
                loss=f"{minibatch_loss:.4f}",
            )
        if args.log_interval <= 0:
            return
        if step != 1 and step != total_steps and step % args.log_interval != 0:
            return
        elapsed = time.perf_counter() - start_time
        progress_message = (
            f"progress | experiment={experiment_name} | step={step:>5d}/{total_steps:<5d} | "
            f"epoch={step / steps_per_epoch:.3f} | {100.0 * step / total_steps:5.1f}% | "
            f"minibatch_loss={minibatch_loss:.4f}"
        )
        latest_loss_stats = getattr(model, "last_loss_stats", None) or {}
        if "loss/query_head" in latest_loss_stats:
            progress_message += f" | query_loss={latest_loss_stats['loss/query_head']:.4f}"
        progress_message += f" | elapsed={elapsed:.1f}s"
        if progress_bar is not None:
            progress_bar.write(progress_message)
        else:
            print(progress_message)

    def step_tensorboard_callback(step: int, minibatch_loss: float, grad_norm: float) -> None:
        if writer is None:
            return
        epoch_step = step / steps_per_epoch
        writer.add_scalar("train/minibatch_loss", minibatch_loss, step)
        writer.add_scalar("train/epoch", epoch_step, step)
        writer.add_scalar("train/grad_norm", grad_norm, step)
        if step == 1 or step == train_steps or (args.log_interval > 0 and step % args.log_interval == 0):
            writer.flush()

    def loss_stats_tensorboard_callback(step: int, stats: dict[str, float]) -> None:
        if writer is None:
            return
        average_query_head_loss = stats.get("loss/avg_query_head")
        if average_query_head_loss is not None and args.training_objective == "query_block":
            writer.add_scalar(
                "train/avg_ppl",
                perplexity_from_loss(average_query_head_loss),
                step,
            )
        bucket_ppl = {
            key.removeprefix("query_block_pos_loss/"): perplexity_from_loss(value)
            for key, value in sorted(stats.items())
            if key.startswith("query_block_pos_loss/")
        }
        if bucket_ppl:
            writer.add_scalars("train/ppl_by_distance", bucket_ppl, step)
        if step == 1 or step == train_steps or (args.log_interval > 0 and step % args.log_interval == 0):
            writer.flush()

    def grad_stats_tensorboard_callback(step: int, stats: dict[str, float]) -> None:
        if writer is None:
            return
        for key, value in sorted(stats.items()):
            writer.add_scalar(f"debug/{key}", value, step)
        if step == 1 or step == train_steps or (args.log_interval > 0 and step % args.log_interval == 0):
            writer.flush()

    def eval_tensorboard_callback(step: int, train_loss: float, val_loss: float) -> None:
        if writer is not None:
            writer.add_scalar("eval/metrics/train_loss", train_loss, step)
            writer.add_scalar("eval/metrics/val_loss", val_loss, step)
            writer.add_scalar("eval/metrics/train_ppl", perplexity_from_loss(train_loss), step)
            writer.add_scalar("eval/metrics/val_ppl", perplexity_from_loss(val_loss), step)
            prompt_ids, prompt_text = select_eval_prompt()
            sample_prompt, sample_generated = make_eval_sample(prompt_ids, prompt_text, sample_source="query_head")
            writer.add_text(
                "eval/sample_query_head" if pair_query_block else "eval/sample",
                format_tensorboard_sample(
                    prompt_text=sample_prompt,
                    generated_text=sample_generated,
                ),
                step,
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
        if response_objective:
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
                steps=train_steps - resume_step,
                eval_interval=effective_eval_interval,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_accum_steps=grad_accum_steps,
                data_workers=args.data_workers,
                prefetch_factor=args.prefetch_factor,
                balanced_index_groups=train_balanced_index_groups,
                start_step=resume_step,
                total_steps=train_steps,
                optimizer_state_dict=optimizer_state_dict,
                checkpoint_interval=args.checkpoint_interval,
                learning_rate_schedule=args.learning_rate_schedule,
                learning_rate_warmup_steps=args.learning_rate_warmup_steps,
                learning_rate_warmup_start=args.learning_rate_warmup_start,
                learning_rate_min_ratio=args.learning_rate_min_ratio,
                step_setup_callback=step_setup_callback,
                progress_callback=progress_callback,
                step_callback=step_tensorboard_callback,
                eval_callback=eval_tensorboard_callback,
                checkpoint_callback=training_checkpoint_callback,
                eval_on_first_step=eval_on_first_step,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
            )
        elif pair_query_block:
            assert train_response_pairs is not None
            assert val_response_pairs is not None
            assert special_token_ids is not None
            history = train_query_block_pair_model(
                model,
                train_response_pairs,
                val_response_pairs,
                seq_len=args.seq_len,
                target_len=args.target_len,
                batch_size=args.batch_size,
                query_block_start_token_id=special_token_ids["query_block_start"],
                eos_token_id=special_token_ids["eos"],
                pad_token_id=special_token_ids["pad"],
                device=device,
                steps=train_steps - resume_step,
                eval_interval=effective_eval_interval,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_accum_steps=grad_accum_steps,
                data_workers=args.data_workers,
                prefetch_factor=args.prefetch_factor,
                step_setup_callback=step_setup_callback,
                progress_callback=progress_callback,
                step_callback=step_tensorboard_callback,
                loss_stats_callback=loss_stats_tensorboard_callback,
                grad_stats_callback=grad_stats_tensorboard_callback if args.grad_breakdown else None,
                eval_callback=eval_tensorboard_callback,
                checkpoint_callback=training_checkpoint_callback,
                eval_on_first_step=eval_on_first_step,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
                balanced_index_groups=train_balanced_index_groups,
                val_balanced_index_groups=val_balanced_index_groups,
                start_step=resume_step,
                total_steps=train_steps,
                optimizer_state_dict=optimizer_state_dict,
                checkpoint_interval=args.checkpoint_interval,
                b_diversity_loss_weight=args.b_diversity_loss_weight,
                b_diversity_unweighted_per_layer=args.b_diversity_unweighted_per_layer,
                b_diversity_per_layer_weight=args.b_diversity_per_layer_weight,
                b_cosine_margin=args.b_cosine_margin,
                b_cosine_early_margin=args.b_cosine_early_margin,
                route_concentration_loss_weight=args.route_concentration_loss_weight,
                route_load_cap=args.route_load_cap,
                edge_prob_cap=args.edge_prob_cap,
                learning_rate_schedule=args.learning_rate_schedule,
                learning_rate_warmup_steps=args.learning_rate_warmup_steps,
                learning_rate_warmup_start=args.learning_rate_warmup_start,
                learning_rate_min_ratio=args.learning_rate_min_ratio,
                query_block_front_weight=args.query_block_front_weight,
                grad_breakdown_start_step=args.grad_breakdown_start_step if args.grad_breakdown else None,
            )
        else:
            history = train_next_token_model(
                model,
                train_tokens,
                val_tokens,
                seq_len=args.seq_len,
                batch_size=args.batch_size,
                device=device,
                steps=train_steps - resume_step,
                eval_interval=effective_eval_interval,
                eval_steps=args.eval_steps,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                grad_accum_steps=grad_accum_steps,
                teacher_forcing=teacher_forcing,
                full_sequence_causal=full_sequence_causal,
                query_next_token=query_next_token,
                query_block=query_block,
                target_len=args.target_len if query_block else 1,
                teacher_forcing_chunk_size=args.teacher_forcing_chunk_size,
                data_workers=args.data_workers,
                prefetch_factor=args.prefetch_factor,
                step_setup_callback=step_setup_callback,
                progress_callback=progress_callback,
                step_callback=step_tensorboard_callback,
                loss_stats_callback=loss_stats_tensorboard_callback,
                grad_stats_callback=grad_stats_tensorboard_callback if args.grad_breakdown else None,
                eval_callback=eval_tensorboard_callback,
                checkpoint_callback=training_checkpoint_callback,
                eval_on_first_step=eval_on_first_step,
                autocast_device_type=autocast_device_type,
                autocast_dtype=autocast_dtype,
                balanced_token_groups=train_balanced_token_groups,
                val_balanced_token_groups=val_balanced_token_groups,
                query_block_start_token_id=query_block_start_token_id if query_block else None,
                start_step=resume_step,
                total_steps=train_steps,
                optimizer_state_dict=optimizer_state_dict,
                checkpoint_interval=args.checkpoint_interval,
                b_diversity_loss_weight=args.b_diversity_loss_weight,
                b_diversity_unweighted_per_layer=args.b_diversity_unweighted_per_layer,
                b_diversity_per_layer_weight=args.b_diversity_per_layer_weight,
                b_cosine_margin=args.b_cosine_margin,
                b_cosine_early_margin=args.b_cosine_early_margin,
                route_concentration_loss_weight=args.route_concentration_loss_weight,
                route_load_cap=args.route_load_cap,
                edge_prob_cap=args.edge_prob_cap,
                learning_rate_schedule=args.learning_rate_schedule,
                learning_rate_warmup_steps=args.learning_rate_warmup_steps,
                learning_rate_warmup_start=args.learning_rate_warmup_start,
                learning_rate_min_ratio=args.learning_rate_min_ratio,
                query_block_front_weight=args.query_block_front_weight,
                grad_breakdown_start_step=args.grad_breakdown_start_step if args.grad_breakdown else None,
            )
    except Exception:
        if progress_bar is not None:
            progress_bar.close()
        if writer is not None:
            writer.flush()
            writer.close()
        raise
    if progress_bar is not None:
        progress_bar.close()
    runtime_seconds = time.perf_counter() - start_time
    print_eval_table(history, steps_per_epoch=steps_per_epoch)

    if response_objective:
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
            target_len=args.target_len,
            eos_token_id=(
                vocab.token_id(EOS_TOKEN)
                if args.training_objective in {"query_next_token", "query_block"}
                and hasattr(vocab, "token_id")
                else None
            ),
            query_block_start_token_id=query_block_start_token_id if query_block else None,
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
        "subword_vocab_size": args.subword_vocab_size if args.tokenizer in {"subword", "byte_bpe"} else None,
        "tokenizer_model_path": tokenizer_model_path,
        "seq_len": args.seq_len,
        "target_len": args.target_len,
        "response_len": args.response_len,
        "sentence_prefix_count": args.sentence_prefix_count,
        "sentence_min_chars": args.sentence_min_chars,
        "batch_size": args.batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "data_workers": args.data_workers,
        "prefetch_factor": args.prefetch_factor,
        "pretokenize_workers": args.pretokenize_workers,
        "dim": effective_config["dim"],
        "warmup_layers": effective_config["warmup_layers"],
        "final_refine_layers": effective_config["final_refine_layers"],
        "lite_layers": effective_config["lite_layers"],
        "mid_layers": effective_config["mid_layers"],
        "full_layers": effective_config["full_layers"],
        "route_topk": effective_config["route_topk"],
        "query_topk": args.query_topk or int(effective_config["route_topk"]),
        "value_norm_kind": value_norm_kind,
        "freeze_position_encoding": args.freeze_position_encoding,
        "norm_position": args.norm_position,
        "propagation_residual": not args.disable_propagation_residual,
        "block_residual": args.block_residual,
        "query_residual": args.query_residual,
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
        "b_slot_dropout_p": args.b_slot_dropout_p,
        "s_to_b_slot_dropout_p": args.s_to_b_slot_dropout_p,
        "b_to_s_slot_dropout_p": args.b_to_s_slot_dropout_p,
        "b_diversity_unweighted_per_layer": args.b_diversity_unweighted_per_layer,
        "b_diversity_per_layer_weight": args.b_diversity_per_layer_weight,
        "b_cosine_early_margin": args.b_cosine_early_margin,
        "alpha_scale": args.alpha_scale,
        "beta_s_to_b_scale": args.beta_s_to_b_scale,
        "beta_b_to_s_scale": args.beta_b_to_s_scale,
        "warmup_delta_scale": args.warmup_delta_scale,
        "s_delta_scale": args.s_delta_scale,
        "b_delta_scale": args.b_delta_scale,
        "cross_delta_scale": args.cross_delta_scale,
        "b_schedule": args.b_schedule,
        "b_schedule_min": args.b_schedule_min,
        "b_schedule_max": args.b_schedule_max,
        "learning_rate": args.learning_rate,
        "learning_rate_schedule": args.learning_rate_schedule,
        "learning_rate_warmup_steps": args.learning_rate_warmup_steps,
        "learning_rate_warmup_start": args.learning_rate_warmup_start,
        "learning_rate_min_ratio": args.learning_rate_min_ratio,
        "query_block_front_weight": args.query_block_front_weight,
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

    final_checkpoint_payload = None
    if args.save_checkpoint:
        final_checkpoint_payload = make_checkpoint_payload(
            step=train_steps,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            checkpoint_model=model,
            optimizer=None,
            checkpoint_kind="final",
        )
        final_checkpoint_payload["history"] = {
            "eval_steps": history.eval_steps,
            "train_step_losses": history.train_step_losses,
            "train_losses": history.train_losses,
            "val_losses": history.val_losses,
        }
        final_checkpoint_payload["summary"] = summary
        final_checkpoint_payload["corpus"] = to_jsonable(corpus)

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
        eval_runtime_rows=eval_runtime_rows,
        prompt_text=prompt_text,
        generated_text=generated_text,
        summary=summary,
        checkpoint_payload=final_checkpoint_payload,
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
        choices=(
            "query_next_token",
            "query_block",
        ),
        default="query_block",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("char", "subword", "byte_bpe"),
        default="byte_bpe",
    )
    parser.add_argument("--subword-vocab-size", type=int, default=256)
    parser.add_argument("--subword-character-coverage", type=float, default=0.9995)
    parser.add_argument(
        "--subword-input-sentence-size",
        type=int,
        default=0,
        help="Randomly sample this many sentences for tokenizer training. 0 uses all sentences.",
    )
    parser.add_argument(
        "--subword-num-threads",
        type=int,
        default=0,
        help="Tokenizer trainer threads. 0 uses os.cpu_count().",
    )
    parser.add_argument(
        "--subword-model-type",
        choices=("bpe", "unigram"),
        default="bpe",
    )
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument(
        "--eval-every-epoch",
        action="store_true",
        help="Run eval, TensorBoard sample logging, and eval-tied checkpoints once per epoch.",
    )
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--balance-batch-by-source",
        action="store_true",
        help="For response objectives, sample each batch evenly across source buckets.",
    )
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Show tqdm progress with epoch progress and ETA.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pretokenize-workers", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--target-len",
        type=int,
        default=1,
        help="Number of future tokens predicted by query_block in one causal query-slot block.",
    )
    parser.add_argument(
        "--b-diversity-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for reducing high cosine similarity between compressed B nodes.",
    )
    parser.add_argument("--b-diversity-unweighted-per-layer", action="store_true")
    parser.add_argument(
        "--b-diversity-per-layer-weight",
        type=float,
        default=1.0,
        help="Weight applied to the averaged per-layer B diversity penalty when enabled.",
    )
    parser.add_argument(
        "--b-cosine-margin",
        type=float,
        default=0.20,
        help="Compressed B node cosine similarity above this margin is penalized.",
    )
    parser.add_argument(
        "--b-cosine-early-margin",
        type=float,
        default=0.35,
        help="Higher cosine margin used for earlier B layers when per-layer B diversity loss is enabled.",
    )
    parser.add_argument(
        "--route-concentration-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary weight for penalizing route/edge concentration onto a single node.",
    )
    parser.add_argument(
        "--route-load-cap",
        type=float,
        default=0.25,
        help="Maximum allowed normalized destination route load before concentration loss applies.",
    )
    parser.add_argument(
        "--edge-prob-cap",
        type=float,
        default=0.55,
        help="Maximum allowed single-edge probability before concentration loss applies.",
    )
    parser.add_argument("--response-len", type=int, default=64)
    parser.add_argument(
        "--sentence-prefix-count",
        type=int,
        default=1,
        help="For next_sentence_response, condition on this many previous sentence-like units.",
    )
    parser.add_argument(
        "--sentence-min-chars",
        type=int,
        default=8,
        help="For next_sentence_response, ignore sentence-like units shorter than this.",
    )
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
    parser.add_argument("--query-refine-layers", type=int)
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
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("none", "cosine"),
        default="none",
    )
    parser.add_argument("--learning-rate-warmup-steps", type=int, default=0)
    parser.add_argument("--learning-rate-warmup-start", type=float)
    parser.add_argument("--learning-rate-min-ratio", type=float, default=0.0)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--query-block-front-weight", type=float, default=1.0)
    parser.add_argument("--grad-breakdown", action="store_true")
    parser.add_argument("--grad-breakdown-start-step", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--route-topk", type=int, default=4)
    parser.add_argument(
        "--query-topk",
        type=int,
        help="Top-k source slots used by query_next_token query-conditioned propagation. Defaults to route-topk.",
    )
    parser.add_argument(
        "--value-norm-kind",
        choices=("layernorm", "rmsnorm"),
        default="layernorm",
    )
    parser.add_argument("--freeze-position-encoding", action="store_true")
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
        choices=("diagonal_bilinear", "low_rank_bilinear", "hadamard_mlp", "additive_low_rank"),
        default="diagonal_bilinear",
    )
    parser.add_argument("--route-hidden-dim", type=int)
    parser.add_argument(
        "--share-route-families",
        action="store_true",
        help="Share route parameters across the 6 transition roles and average per-role route weights when resuming older checkpoints.",
    )
    parser.add_argument(
        "--pairwise-kind",
        choices=("diagonal_bilinear", "low_rank_bilinear", "hadamard_mlp", "additive_low_rank"),
        default="diagonal_bilinear",
    )
    parser.add_argument("--pairwise-hidden-dim", type=int)
    parser.add_argument("--edge-dropout-p", type=float, default=0.0)
    parser.add_argument("--usage-dropout-base", type=float, default=0.0)
    parser.add_argument("--usage-dropout-scale", type=float, default=0.0)
    parser.add_argument("--usage-dropout-max", type=float, default=0.0)
    parser.add_argument("--usage-ema-decay", type=float, default=0.99)
    parser.add_argument("--b-slot-dropout-p", type=float, default=0.0)
    parser.add_argument("--s-to-b-slot-dropout-p", type=float, default=0.0)
    parser.add_argument("--b-to-s-slot-dropout-p", type=float, default=0.0)
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
    parser.add_argument("--block-residual", action="store_true")
    parser.add_argument("--query-residual", action="store_true")
    parser.add_argument("--alpha-scale", type=float, default=1.0)
    parser.add_argument("--beta-s-to-b-scale", type=float, default=1.0)
    parser.add_argument("--beta-b-to-s-scale", type=float, default=1.0)
    parser.add_argument("--warmup-delta-scale", type=float, default=0.0)
    parser.add_argument("--s-delta-scale", type=float, default=0.10)
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
    parser.add_argument("--session-dir")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(args.device)}")

    teacher_forcing = False
    full_sequence_causal = False
    query_next_token = args.training_objective == "query_next_token"
    query_block = args.training_objective == "query_block"
    pair_query_block_objective = query_block
    prefix_response = False
    next_sentence_response = False
    response_objective = False
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
    if args.query_topk is not None and args.query_topk <= 0:
        raise ValueError("query-topk must be positive.")
    if args.query_refine_layers is not None and args.query_refine_layers <= 0:
        raise ValueError("query-refine-layers must be positive.")
    if args.data_workers < 0:
        raise ValueError("data-workers must be non-negative.")
    if args.pretokenize_workers < 0:
        raise ValueError("pretokenize-workers must be non-negative.")
    if args.prefetch_factor <= 0:
        raise ValueError("prefetch-factor must be positive.")
    if args.checkpoint_interval < 0:
        raise ValueError("checkpoint-interval must be non-negative.")
    if args.response_len <= 0:
        raise ValueError("response-len must be positive.")
    if args.target_len <= 0:
        raise ValueError("target-len must be positive.")
    if args.b_diversity_loss_weight < 0.0:
        raise ValueError("b-diversity-loss-weight must be non-negative.")
    if args.b_diversity_per_layer_weight < 0.0:
        raise ValueError("b-diversity-per-layer-weight must be non-negative.")
    if args.route_concentration_loss_weight < 0.0:
        raise ValueError("route-concentration-loss-weight must be non-negative.")
    if args.learning_rate_warmup_steps < 0:
        raise ValueError("learning-rate-warmup-steps must be non-negative.")
    if args.learning_rate_warmup_start is not None and args.learning_rate_warmup_start < 0.0:
        raise ValueError("learning-rate-warmup-start must be non-negative.")
    if not 0.0 <= args.learning_rate_min_ratio <= 1.0:
        raise ValueError("learning-rate-min-ratio must be in [0, 1].")
    if args.query_block_front_weight < 1.0:
        raise ValueError("query-block-front-weight must be at least 1.0.")
    if args.grad_breakdown_start_step < 0:
        raise ValueError("grad-breakdown-start-step must be non-negative.")
    if not 0.0 <= args.b_cosine_margin < 1.0:
        raise ValueError("b-cosine-margin must be in [0, 1).")
    if not 0.0 <= args.b_cosine_early_margin < 1.0:
        raise ValueError("b-cosine-early-margin must be in [0, 1).")
    if not 0.0 < args.route_load_cap <= 1.0:
        raise ValueError("route-load-cap must be in (0, 1].")
    if not 0.0 < args.edge_prob_cap <= 1.0:
        raise ValueError("edge-prob-cap must be in (0, 1].")
    if args.sentence_prefix_count <= 0:
        raise ValueError("sentence-prefix-count must be positive.")
    if args.sentence_min_chars <= 0:
        raise ValueError("sentence-min-chars must be positive.")
    if args.subword_input_sentence_size < 0:
        raise ValueError("subword-input-sentence-size must be non-negative.")
    if args.subword_num_threads < 0:
        raise ValueError("subword-num-threads must be non-negative.")
    for name in (
        "edge_dropout_p",
        "usage_dropout_base",
        "usage_dropout_scale",
        "usage_dropout_max",
        "b_slot_dropout_p",
        "s_to_b_slot_dropout_p",
        "b_to_s_slot_dropout_p",
    ):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1).")
    if not 0.0 <= args.usage_ema_decay < 1.0:
        raise ValueError("usage-ema-decay must be in [0, 1).")

    dialogue_pairs: list[DialoguePairText] | None = None
    if prefix_response or pair_query_block_objective:
        dialogue_pairs, corpus = load_dialogue_pairs(
            jsonl_sources=args.jsonl_source,
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_streaming=args.hf_streaming,
            max_samples=args.corpus_max_samples,
            max_chars=args.corpus_max_chars,
        )
    elif next_sentence_response:
        dialogue_pairs, corpus = load_next_sentence_pairs(
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
            prefix_sentences=args.sentence_prefix_count,
            min_chars=args.sentence_min_chars,
        )
    elif query_next_token:
        corpus = load_token_stream_corpus(
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
    if query_next_token and not corpus.metadata.get("eos_at_document_boundaries"):
        corpus = append_eos_markers_to_corpus(corpus, separator=args.corpus_separator)
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
        subword_input_sentence_size=args.subword_input_sentence_size,
        subword_num_threads=args.subword_num_threads,
        user_defined_symbols=DIALOGUE_SPECIAL_TOKENS if response_objective or query_next_token or query_block else (),
    )
    if response_objective:
        special_token_ids = require_special_token_ids(vocab)
    elif query_block and hasattr(vocab, "token_id"):
        special_token_ids = require_special_token_ids(vocab)
        query_block_start_token_id = special_token_ids["query_block_start"]
    else:
        special_token_ids = None
    effective_pretokenize_workers = min(args.pretokenize_workers, os.cpu_count() or 1)
    if not response_objective and effective_pretokenize_workers > 1 and isinstance(vocab, ByteBPEVocab):
        print(f"parallel_byte_bpe_encoding | workers={effective_pretokenize_workers}", flush=True)
    if response_objective or pair_query_block_objective:
        tokens = torch.empty(0, dtype=torch.long)
        train_tokens = tokens
        val_tokens = tokens
    else:
        required_future_tokens = args.target_len if query_block else 1
        if (query_next_token or query_block) and args.balance_batch_by_source and args.jsonl_source:
            source_texts = load_token_stream_source_texts(
                jsonl_sources=args.jsonl_source,
                jsonl_text_keys=tuple(dict.fromkeys(args.jsonl_text_key)),
                max_samples=args.corpus_max_samples,
                max_chars=args.corpus_max_chars,
            )
            source_train_parts: list[torch.Tensor] = []
            source_val_parts: list[torch.Tensor] = []
            source_val_counts: dict[str, int] = {}
            source_counts: dict[str, int] = {}
            skipped_sources: dict[str, int] = {}
            min_tokens = args.seq_len + required_future_tokens
            for source, source_text in sorted(source_texts.items()):
                source_tokens = encode_text_in_chunks(
                    vocab,
                    source_text,
                    workers=effective_pretokenize_workers,
                )
                if source_tokens.numel() <= min_tokens:
                    skipped_sources[source] = int(source_tokens.numel())
                    continue
                source_train, source_val = split_train_val(source_tokens, train_fraction=0.9)
                if source_train.numel() <= min_tokens:
                    skipped_sources[source] = int(source_train.numel())
                    continue
                source_counts[source] = int(source_train.numel())
                source_train_parts.append(source_train)
                if source_val.numel() > min_tokens:
                    source_val_counts[source] = int(source_val.numel())
                    source_val_parts.append(source_val)
            if not source_train_parts:
                raise ValueError("No source group has enough tokens for balanced query/block batches.")
            train_tokens = torch.cat(source_train_parts)
            source_views: list[torch.Tensor] = []
            offset = 0
            for source_train in source_train_parts:
                next_offset = offset + source_train.numel()
                source_views.append(train_tokens[offset:next_offset])
                offset = next_offset
            train_balanced_token_groups = tuple(source_views)
            val_tokens = torch.cat(source_val_parts) if source_val_parts else train_tokens
            val_balanced_token_groups = None
            if source_val_parts:
                val_views: list[torch.Tensor] = []
                offset = 0
                for source_val in source_val_parts:
                    next_offset = offset + source_val.numel()
                    val_views.append(val_tokens[offset:next_offset])
                    offset = next_offset
                val_balanced_token_groups = tuple(val_views)
            print(
                "balanced_token_batches | "
                f"train_groups={len(train_balanced_token_groups)} | "
                f"val_groups={len(val_balanced_token_groups or ())} | "
                f"train_token_counts={source_counts} | "
                f"val_token_counts={source_val_counts}"
                + (f" | skipped={skipped_sources}" if skipped_sources else "")
            )
        else:
            tokens = encode_text_in_chunks(
                vocab,
                corpus.text,
                workers=effective_pretokenize_workers,
            )
            if tokens.numel() <= args.seq_len + required_future_tokens:
                raise ValueError("The tokenized corpus must be longer than seq_len + target_len.")
            train_tokens, val_tokens = split_train_val(tokens, train_fraction=0.9)
    train_response_pairs = None
    val_response_pairs = None
    train_balanced_index_groups = None
    val_balanced_index_groups = None
    if 'val_balanced_token_groups' not in locals():
        val_balanced_token_groups = None
    if response_objective or pair_query_block_objective or not (
        (query_next_token or query_block) and args.balance_batch_by_source and args.jsonl_source
    ):
        train_balanced_token_groups = None
    if response_objective or pair_query_block_objective:
        assert dialogue_pairs is not None
        if args.balance_batch_by_source:
            train_pair_texts, val_pair_texts = split_dialogue_pairs_by_source(dialogue_pairs)
        else:
            train_pair_texts, val_pair_texts = split_dialogue_pairs(dialogue_pairs)
        if args.balance_batch_by_source:
            train_balanced_index_groups = build_source_balanced_index_groups(train_pair_texts)
            val_balanced_index_groups = build_source_balanced_index_groups(val_pair_texts)
            print(
                "balanced_source_batches | "
                f"groups={len(train_balanced_index_groups)} | "
                f"counts={summarize_source_groups(train_pair_texts, train_balanced_index_groups)}"
            )
        print(
            "encoding_response_pairs | "
            f"train={len(train_pair_texts):,} | val={len(val_pair_texts):,} | "
            f"pretokenize_workers={effective_pretokenize_workers}"
        )
        encode_start = time.perf_counter()
        train_response_pairs = encode_dialogue_pairs(
            train_pair_texts,
            vocab=vocab,
            workers=effective_pretokenize_workers,
        )
        val_response_pairs = encode_dialogue_pairs(
            val_pair_texts,
            vocab=vocab,
            workers=effective_pretokenize_workers,
        )
        print(
            "encoded_response_pairs | "
            f"elapsed={time.perf_counter() - encode_start:.1f}s"
        )
    print(
        f"tokenizer={tokenizer_label} | tokenizer_vocab={vocab.size:,} | "
        f"train_tokens={train_tokens.numel():,} | val_tokens={val_tokens.numel():,}"
    )
    if response_objective:
        print(
            f"response_pairs={len(dialogue_pairs or ()):,} | "
            f"train_pairs={len(train_response_pairs or ()):,} | "
            f"val_pairs={len(val_response_pairs or ()):,} | response_len={args.response_len}"
        )
    elif query_block:
        print(
            f"query_block_pairs={len(dialogue_pairs or ()):,} | "
            f"train_pairs={len(train_response_pairs or ()):,} | "
            f"val_pairs={len(val_response_pairs or ()):,} | target_len={args.target_len}"
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

    if args.session_dir:
        session_dir = ensure_directory(args.session_dir)
    else:
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
            train_balanced_index_groups=train_balanced_index_groups,
            train_balanced_token_groups=train_balanced_token_groups,
            val_balanced_index_groups=val_balanced_index_groups,
            val_balanced_token_groups=val_balanced_token_groups,
            device=device,
            teacher_forcing=teacher_forcing,
            full_sequence_causal=full_sequence_causal,
            query_next_token=query_next_token,
            query_block=query_block,
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
