from __future__ import annotations

import hashlib
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

try:
    from tokenizers import ByteLevelBPETokenizer
except ImportError:
    ByteLevelBPETokenizer = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


DEFAULT_TOKENIZER_CACHE_DIR = Path(tempfile.gettempdir()) / "jakal_net_tokenizers"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|eos|>"
PAD_TOKEN = "<|pad|>"
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")

_BYTE_BPE_WORKER_TOKENIZER: Any | None = None
_BYTE_BPE_WORKER_SPECIAL_TOKENS: tuple[str, ...] = ()
_BYTE_BPE_WORKER_SPECIAL_PATTERN: re.Pattern[str] | None = None
_HF_WORKER_TOKENIZER: Any | None = None


@dataclass(frozen=True, slots=True)
class DialoguePairText:
    prefix: str
    response: str
    source: str = "unknown"


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
        return torch.tensor(self.tokenizer.encode(text).ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(map(int, token_ids))))

    def token_id(self, piece: str) -> int:
        idx = self.tokenizer.token_to_id(piece)
        if idx is None:
            raise ValueError(
                f"Tokenizer {self.vocab_path} does not contain required token {piece!r}."
            )
        return int(idx)


@dataclass(frozen=True, slots=True)
class HFTokenizerVocab:
    tokenizer: object
    model_path: Path

    @property
    def size(self) -> int:
        return int(len(self.tokenizer))

    def encode(self, text: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(
            self.tokenizer.decode(
                list(map(int, token_ids)),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    def token_id(self, piece: str) -> int:
        idx = self.tokenizer.convert_tokens_to_ids(piece)
        if idx is None or int(idx) < 0:
            raise ValueError(
                f"Tokenizer {self.model_path} does not contain required token {piece!r}."
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
            f"jakal_net_{tokenizer}_{model_type}_{vocab_size}_{corpus_digest}"
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


def build_byte_bpe_vocab(
    text: str,
    *,
    text_path: str | None,
    vocab_size: int,
    tokenizer_prefix: str | None,
    user_defined_symbols: Sequence[str] = (),
) -> ByteBPEVocab:
    if ByteLevelBPETokenizer is None:
        raise ImportError("tokenizers is required for --tokenizer byte_bpe.")
    if vocab_size <= 0:
        raise ValueError("subword-vocab-size must be positive.")

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


def build_hf_auto_vocab(
    text: str,
    *,
    model_name_or_path: str,
    tokenizer_prefix: str | None,
    vocab_size: int,
    trust_remote_code: bool = False,
) -> HFTokenizerVocab:
    if AutoTokenizer is None:
        raise ImportError("transformers is required for --tokenizer hf_auto.")
    if not model_name_or_path:
        raise ValueError("hf_tokenizer_model must be provided for --tokenizer hf_auto.")

    prefix_path = resolve_tokenizer_prefix(
        text=text or model_name_or_path,
        tokenizer="hf_auto",
        model_type="hf",
        vocab_size=vocab_size,
        prefix=tokenizer_prefix,
    )
    model_dir = prefix_path.parent / prefix_path.name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(model_dir))
    return HFTokenizerVocab(tokenizer=tokenizer, model_path=model_dir)


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
    hf_tokenizer_model: str | None = None,
    hf_trust_remote_code: bool = False,
) -> tuple[object, str, Path | None]:
    del subword_model_type
    del subword_character_coverage
    del subword_input_sentence_size
    del subword_num_threads
    if tokenizer == "byte_bpe":
        byte_bpe_vocab = build_byte_bpe_vocab(
            text,
            text_path=text_path,
            vocab_size=subword_vocab_size,
            tokenizer_prefix=tokenizer_prefix,
            user_defined_symbols=user_defined_symbols,
        )
        return byte_bpe_vocab, "byte_bpe", byte_bpe_vocab.vocab_path
    if tokenizer == "hf_auto":
        hf_vocab = build_hf_auto_vocab(
            text,
            model_name_or_path=str(hf_tokenizer_model or ""),
            tokenizer_prefix=tokenizer_prefix,
            vocab_size=subword_vocab_size,
            trust_remote_code=hf_trust_remote_code,
        )
        return hf_vocab, "hf_auto", hf_vocab.model_path
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
    response_text = None
    for key in ("response", "output", "answer", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            response_text = value.strip()
            break
    if prompt and response_text:
        return [
            DialoguePairText(
                prefix=f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n",
                response=response_text,
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
    response_text = None
    for key in ("response", "output", "answer", "completion"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            response_text = value.strip()
            break
    if prompt and response_text:
        return f"{USER_TOKEN}\n{prompt}\n{ASSISTANT_TOKEN}\n{response_text}"
    return None


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
    specials = [
        token
        for token in vocab.keys()
        if isinstance(token, str) and re.fullmatch(r"<\|[^|\n]+\|>", token)
    ]
    return tuple(sorted(specials, key=len, reverse=True))


def _byte_bpe_special_pattern(special_tokens: Sequence[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    return re.compile("|".join(re.escape(token) for token in special_tokens))


def _ensure_byte_bpe_special_tokens(tokenizer: Any, required_special_tokens: Sequence[str]) -> tuple[str, ...]:
    missing_special_tokens = [
        token for token in required_special_tokens if tokenizer.token_to_id(token) is None
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


def _init_byte_bpe_encode_worker(
    vocab_path: str,
    merges_path: str,
    required_special_tokens: Sequence[str] = (),
) -> None:
    global _BYTE_BPE_WORKER_TOKENIZER, _BYTE_BPE_WORKER_SPECIAL_TOKENS, _BYTE_BPE_WORKER_SPECIAL_PATTERN
    if ByteLevelBPETokenizer is None:
        raise ImportError("tokenizers is required for parallel byte BPE encoding.")
    _BYTE_BPE_WORKER_TOKENIZER = ByteLevelBPETokenizer(vocab_path, merges_path)
    _BYTE_BPE_WORKER_SPECIAL_TOKENS = _ensure_byte_bpe_special_tokens(
        _BYTE_BPE_WORKER_TOKENIZER,
        required_special_tokens,
    )
    _BYTE_BPE_WORKER_SPECIAL_PATTERN = _byte_bpe_special_pattern(
        _BYTE_BPE_WORKER_SPECIAL_TOKENS
    )


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


def _init_hf_encode_worker(model_path: str) -> None:
    global _HF_WORKER_TOKENIZER
    if AutoTokenizer is None:
        raise ImportError("transformers is required for parallel hf_auto encoding.")
    _HF_WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
    )


def _encode_hf_text_batch_worker(texts: Sequence[str]) -> list[list[int]]:
    if _HF_WORKER_TOKENIZER is None:
        raise RuntimeError("HF tokenizer worker was not initialized.")
    if not texts:
        return []
    encoded = _HF_WORKER_TOKENIZER(
        list(texts),
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [list(map(int, token_ids)) for token_ids in encoded["input_ids"]]


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
        chunk_batches = [
            chunks[index : index + batch_size] for index in range(0, len(chunks), batch_size)
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_byte_bpe_encode_worker,
            initargs=(str(vocab.vocab_path), str(vocab.merges_path), vocab.special_tokens),
        ) as executor:
            for token_id_batch in executor.map(
                _encode_byte_bpe_text_batch_worker,
                chunk_batches,
                chunksize=1,
            ):
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
                    special_tokens=vocab.special_tokens
                    or _byte_bpe_special_tokens(vocab.tokenizer),
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
            prefix = " ".join(units[index - prefix_sentences : index])
            response = units[index]
            pairs.append(DialoguePairText(prefix=prefix, response=response))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs
