from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import math
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from lm_experiment_utils import (
    _expand_sources,
    _load_hf_dataset_texts,
    append_jsonl,
    create_run_directory,
    ensure_directory,
    write_csv_rows,
    write_json,
)
from train_progressive_b_lm import (
    ASSISTANT_TOKEN,
    ByteBPEVocab,
    ByteLevelBPETokenizer,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    _encode_byte_bpe_text_worker,
    _chat_stream_from_record,
    _init_byte_bpe_encode_worker,
    _iter_nonempty_text_chunks,
    _message_content,
    _message_role,
    _text_from_record_keys,
    build_tokenizer,
    count_parameters,
    encode_text_in_chunks,
    resolve_autocast_dtype,
)

from jakal_net import describe_device, resolve_device
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


BOS_TOKEN = "<|bos|>"
TEXT_TOKEN = "<|text|>"
CODE_TOKEN = "<|code|>"
MATH_TOKEN = "<|math|>"
CONT_TOKEN = "<|cont|>"
DIALOGUE_TOKEN = "<|dialogue|>"
INSTRUCTION_TOKEN = "<|instruction|>"
RESPONSE_TOKEN = "<|response|>"
EOT_TOKEN = "<|eot|>"

CAUSAL_DOC_SPECIAL_TOKENS = (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    TEXT_TOKEN,
    CODE_TOKEN,
    MATH_TOKEN,
    CONT_TOKEN,
    DIALOGUE_TOKEN,
    INSTRUCTION_TOKEN,
    RESPONSE_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    EOT_TOKEN,
)

MODE_TOKEN_BY_KIND = {
    "text": TEXT_TOKEN,
    "code": CODE_TOKEN,
    "math": MATH_TOKEN,
    "dialogue": DIALOGUE_TOKEN,
    "instruction": INSTRUCTION_TOKEN,
}

CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cu",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".json",
    ".kt",
    ".m",
    ".mdx",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".scala",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True, slots=True)
class SerializedDocument:
    kind: str
    source: str
    body: str


@dataclass(frozen=True, slots=True)
class TrainingCurriculumStage:
    name: str
    document_span: int
    freeze_memory: bool
    freeze_propagation: bool
    freeze_skip: bool


@dataclass(frozen=True, slots=True)
class TokenizedDocument:
    kind: str
    source: str
    chunks: tuple["DocumentChunk", ...]
    token_count: int


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    context: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    is_continuation: bool


@dataclass(frozen=True, slots=True)
class DocumentBatch:
    context: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    reset_mask: torch.Tensor


_DISPLAY_MATH_PATTERN = re.compile(r"(?s)\$\$.*?\$\$")
_INLINE_DOLLAR_MATH_PATTERN = re.compile(r"(?s)(?<!\$)\$(?!\$)(?:\\.|[^$\\\n])+\$(?!\$)")
_BRACKET_MATH_PATTERN = re.compile(r"(?s)\\\[.*?\\\]")
_INLINE_PAREN_MATH_PATTERN = re.compile(r"(?s)\\\(.*?\\\)")
_MATH_ENV_PATTERN = re.compile(
    r"(?s)\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{(?P=env)\}"
)
_EXECUTE_BLOCK_PATTERN = re.compile(r"(?s)<execute>.*?</execute>")
_CODE_FENCE_PATTERN = re.compile(
    r"(?ms)(?:(?<=\A)|(?<=\n))(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:\n(?P=fence)[ \t]*(?=\n|$)|\Z)"
)


def _coerce_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _segment_token(kind: str) -> str:
    if kind not in MODE_TOKEN_BY_KIND:
        raise ValueError(f"Unsupported segment kind: {kind!r}")
    return MODE_TOKEN_BY_KIND[kind]


def _merge_segment_records(segments: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for segment in segments:
        kind = segment["kind"]
        text = segment["text"]
        if not text:
            continue
        if merged and merged[-1]["kind"] == kind:
            merged[-1]["text"] += text
        else:
            merged.append({"kind": kind, "text": text})
    return merged


def _segment_message_content_exact(content: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    cursor = 0
    patterns = (
        ("code", _CODE_FENCE_PATTERN),
        ("code", _EXECUTE_BLOCK_PATTERN),
        ("math", _DISPLAY_MATH_PATTERN),
        ("math", _INLINE_DOLLAR_MATH_PATTERN),
        ("math", _BRACKET_MATH_PATTERN),
        ("math", _INLINE_PAREN_MATH_PATTERN),
        ("math", _MATH_ENV_PATTERN),
    )
    while cursor < len(content):
        best_kind: str | None = None
        best_match: re.Match[str] | None = None
        for kind, pattern in patterns:
            match = pattern.search(content, cursor)
            if match is None:
                continue
            if best_match is None or match.start() < best_match.start():
                best_kind = kind
                best_match = match
        if best_match is None:
            tail = content[cursor:]
            if tail:
                segments.append({"kind": "text", "text": tail})
            break
        if best_match.start() > cursor:
            segments.append({"kind": "text", "text": content[cursor : best_match.start()]})
        assert best_kind is not None
        segments.append({"kind": best_kind, "text": best_match.group(0)})
        cursor = best_match.end()
    return _merge_segment_records(segments)


def _normalize_segment_records(segments: Sequence[object]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        kind_value = segment.get("kind")
        text_value = _coerce_text(segment.get("text"))
        if text_value is None:
            continue
        kind = str(kind_value).strip().lower() if isinstance(kind_value, str) else "text"
        if kind not in MODE_TOKEN_BY_KIND:
            kind = "text"
        normalized.append({"kind": kind, "text": text_value})
    return _merge_segment_records(normalized)


def _serialize_segments(segments: Sequence[dict[str, str]]) -> str | None:
    parts: list[str] = []
    for segment in segments:
        text = segment["text"]
        if not text.strip():
            continue
        parts.append(f"{_segment_token(segment['kind'])}\n{text}")
    if not parts:
        return None
    return "\n".join(parts)


def _message_segments(message: dict[str, Any]) -> list[dict[str, str]]:
    segments = message.get("segments")
    if isinstance(segments, list):
        normalized = _normalize_segment_records(segments)
        if normalized:
            return normalized
    content = _message_content(message)
    if not content:
        return []
    return _segment_message_content_exact(content)


def _tokenize_document_payload_worker(payload: tuple[str, str, str]) -> tuple[str, str, list[int]]:
    kind, source, body = payload
    token_ids: list[int] = []
    for chunk in _iter_nonempty_text_chunks(body, chunk_chars=8_000_000):
        token_ids.extend(_encode_byte_bpe_text_worker(chunk))
    return kind, source, token_ids


class DocumentChunkBatcher:
    def __init__(
        self,
        documents: Sequence[TokenizedDocument],
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")
        self.documents = tuple(documents)
        self.batch_size = batch_size
        self.device = device
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size

    def _sample_document_index(self) -> int:
        return random.randrange(len(self.documents))

    def next_batch(self) -> DocumentBatch:
        contexts: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        reset_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        for item_index in range(self.batch_size):
            if self.needs_reset[item_index]:
                self.current_doc[item_index] = self._sample_document_index()
                self.current_chunk[item_index] = 0
                reset_mask[item_index] = True
            document = self.documents[self.current_doc[item_index]]
            chunk = document.chunks[self.current_chunk[item_index]]
            contexts.append(chunk.context)
            targets.append(chunk.target)
            masks.append(chunk.loss_mask)
            next_chunk = self.current_chunk[item_index] + 1
            if next_chunk >= len(document.chunks):
                self.needs_reset[item_index] = True
                self.current_chunk[item_index] = 0
            else:
                self.needs_reset[item_index] = False
                self.current_chunk[item_index] = next_chunk
        return DocumentBatch(
            context=torch.stack(contexts, dim=0).to(self.device),
            target=torch.stack(targets, dim=0).to(self.device),
            loss_mask=torch.stack(masks, dim=0).to(self.device),
            reset_mask=reset_mask.to(self.device),
        )


def _is_probably_code(text: str, *, source: str = "") -> bool:
    lower_source = source.lower()
    if any(lower_source.endswith(suffix) for suffix in CODE_SUFFIXES):
        return True
    hints = (
        "def ",
        "class ",
        "import ",
        "#include",
        "public static",
        "fn ",
        "let ",
        "const ",
        "return ",
        "SELECT ",
        "{",
        "};",
    )
    hit_count = sum(1 for hint in hints if hint in text)
    return hit_count >= 2


def _normalize_dialogue_body(messages: Sequence[dict[str, Any]]) -> str | None:
    parts: list[str] = []
    for message in messages:
        role = _message_role(message)
        segments = _message_segments(message)
        serialized_segments = _serialize_segments(segments)
        if serialized_segments is None:
            continue
        if role == "user":
            parts.append(f"{USER_TOKEN}\n{serialized_segments}\n{EOT_TOKEN}")
        elif role == "assistant":
            parts.append(f"{ASSISTANT_TOKEN}\n{serialized_segments}\n{EOT_TOKEN}")
        elif role in {"system", "developer"}:
            parts.append(f"{INSTRUCTION_TOKEN}\n{serialized_segments}\n{EOT_TOKEN}")
    if not parts:
        return None
    return "\n".join(parts)


def _record_source(record: dict[str, Any], *, fallback: str) -> str:
    value = record.get("source")
    if isinstance(value, str) and value.strip():
        source = value.strip()
        return source if source == fallback else f"{source} ({fallback})"
    return fallback


def _record_kind(record: dict[str, Any], *, text: str, source: str) -> str:
    value = record.get("kind")
    if isinstance(value, str):
        kind = value.strip().lower()
        if kind in MODE_TOKEN_BY_KIND:
            return kind
        if kind in {"mixed", "wiki"}:
            return "text"
    return "code" if _is_probably_code(text, source=source) else "text"


def _record_to_document(record: Any, *, source: str) -> SerializedDocument | None:
    if isinstance(record, str):
        text = record.strip()
        if not text:
            return None
        return SerializedDocument(
            kind="code" if _is_probably_code(text, source=source) else "text",
            source=source,
            body=text,
        )
    if not isinstance(record, dict):
        return None

    record_source = _record_source(record, fallback=source)
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        normalized = [message for message in messages if isinstance(message, dict)]
        dialogue_body = _normalize_dialogue_body(normalized)
        if dialogue_body:
            return SerializedDocument(kind="dialogue", source=record_source, body=dialogue_body)

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
        return SerializedDocument(
            kind="instruction",
            source=record_source,
            body=f"{TEXT_TOKEN}\n{prompt}\n{RESPONSE_TOKEN}\n{TEXT_TOKEN}\n{response}",
        )

    segments = record.get("segments")
    if isinstance(segments, list):
        serialized_segments = _serialize_segments(_normalize_segment_records(segments))
        if serialized_segments is not None:
            top_kind = str(record.get("kind") or "text").strip().lower()
            return SerializedDocument(
                kind=top_kind if top_kind in MODE_TOKEN_BY_KIND else "text",
                source=record_source,
                body=serialized_segments,
            )

    text = _text_from_record_keys(record, ("text", "content", "body", "code", "document"))
    if text is None:
        transcript = _chat_stream_from_record(record)
        if transcript is not None:
            text = transcript.strip()
    if text is None or not text.strip():
        return None
    body = text.strip()
    kind = _record_kind(record, text=body, source=record_source)
    return SerializedDocument(kind=kind, source=record_source, body=body)


def load_serialized_documents(
    *,
    text_file: str | None,
    text_sources: Sequence[str],
    jsonl_sources: Sequence[str],
    hf_dataset: str | None,
    hf_config: str | None,
    hf_split: str,
    hf_text_key: str,
    hf_streaming: bool,
    max_samples: int | None,
) -> list[SerializedDocument]:
    documents: list[SerializedDocument] = []
    seen = 0
    for raw_source in ([text_file] if text_file else []) + list(text_sources):
        if raw_source is None:
            continue
        for path in _expand_sources((raw_source,), directory_suffixes=(".txt", ".md", ".py", ".json", ".cpp")):
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            documents.append(
                SerializedDocument(
                    kind="code" if _is_probably_code(text, source=str(path)) else "text",
                    source=str(path),
                    body=text.strip(),
                )
            )
            seen += 1
            if max_samples is not None and seen >= max_samples:
                return documents
    for raw_source in jsonl_sources:
        for path in _expand_sources((raw_source,), directory_suffixes=(".jsonl", ".json")):
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    line = raw_line.strip().lstrip("\ufeff")
                    if not line:
                        continue
                    document = _record_to_document(
                        json.loads(line),
                        source=f"{path}:{line_number}",
                    )
                    if document is None:
                        continue
                    documents.append(document)
                    seen += 1
                    if max_samples is not None and seen >= max_samples:
                        return documents
    if hf_dataset is not None and (max_samples is None or seen < max_samples):
        remaining = None if max_samples is None else max_samples - seen
        hf_texts = _load_hf_dataset_texts(
            dataset_name=hf_dataset,
            config_name=hf_config,
            split=hf_split,
            text_key=hf_text_key,
            streaming=hf_streaming,
            max_samples=remaining,
        )
        for index, text in enumerate(hf_texts):
            if not text.strip():
                continue
            documents.append(SerializedDocument(kind="text", source=f"{hf_dataset}:{index}", body=text.strip()))
    if not documents:
        documents.append(
            SerializedDocument(
                kind="text",
                source="default_text",
                body="Jakal-Net causal memory training sample document.",
            )
        )
    return documents


def render_document_for_tokenizer(document: SerializedDocument) -> str:
    mode_token = MODE_TOKEN_BY_KIND[document.kind]
    return f"{BOS_TOKEN}\n{mode_token}\n{document.body}\n{EOS_TOKEN}"


def build_training_text(documents: Sequence[SerializedDocument]) -> str:
    return "\n\n".join(render_document_for_tokenizer(document) for document in documents)


def build_special_token_id_map(vocab: object) -> dict[str, int]:
    if not hasattr(vocab, "token_id"):
        raise ValueError("The causal-memory path requires a tokenizer with explicit special-token ids.")
    return {
        "bos": vocab.token_id(BOS_TOKEN),
        "eos": vocab.token_id(EOS_TOKEN),
        "pad": vocab.token_id(PAD_TOKEN),
        "text": vocab.token_id(TEXT_TOKEN),
        "code": vocab.token_id(CODE_TOKEN),
        "math": vocab.token_id(MATH_TOKEN),
        "cont": vocab.token_id(CONT_TOKEN),
        "dialogue": vocab.token_id(DIALOGUE_TOKEN),
        "instruction": vocab.token_id(INSTRUCTION_TOKEN),
        "response": vocab.token_id(RESPONSE_TOKEN),
        "user": vocab.token_id(USER_TOKEN),
        "assistant": vocab.token_id(ASSISTANT_TOKEN),
        "eot": vocab.token_id(EOT_TOKEN),
    }


def make_document_chunks(
    *,
    content_ids: torch.Tensor,
    mode_token_id: int,
    seq_len: int,
    bos_token_id: int,
    cont_token_id: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[DocumentChunk, ...]:
    if seq_len <= 2:
        raise ValueError("seq_len must be larger than 2 to fit boundary tokens.")
    payload_capacity = seq_len - 2
    content_ids = content_ids.detach().cpu().to(dtype=torch.long)
    chunks: list[DocumentChunk] = []
    cursor = 0
    first = True
    if content_ids.numel() == 0:
        context = torch.tensor([bos_token_id, mode_token_id], dtype=torch.long)
        target = torch.tensor([mode_token_id, eos_token_id], dtype=torch.long)
        loss_mask = torch.ones(2, dtype=torch.float32)
        return (
            DocumentChunk(
                context=F.pad(context, (0, seq_len - 2), value=pad_token_id),
                target=F.pad(target, (0, seq_len - 2), value=pad_token_id),
                loss_mask=F.pad(loss_mask, (0, seq_len - 2), value=0.0),
                is_continuation=False,
            ),
        )
    while cursor < content_ids.numel():
        prefix = [bos_token_id, mode_token_id] if first else [cont_token_id, mode_token_id]
        take = min(payload_capacity, content_ids.numel() - cursor)
        content_slice = content_ids[cursor : cursor + take]
        context = torch.tensor(prefix, dtype=torch.long)
        if content_slice.numel() > 0:
            context = torch.cat((context, content_slice), dim=0)
        target = torch.empty_like(context)
        target[:-1] = context[1:]
        target[-1] = eos_token_id if cursor + take >= content_ids.numel() else cont_token_id
        loss_mask = torch.ones(context.shape[0], dtype=torch.float32)
        pad = seq_len - context.shape[0]
        chunks.append(
            DocumentChunk(
                context=F.pad(context, (0, pad), value=pad_token_id),
                target=F.pad(target, (0, pad), value=pad_token_id),
                loss_mask=F.pad(loss_mask, (0, pad), value=0.0),
                is_continuation=not first,
            )
        )
        cursor += take
        first = False
    return tuple(chunks)


def tokenize_documents(
    documents: Sequence[SerializedDocument],
    *,
    vocab: object,
    seq_len: int,
    special_token_ids: dict[str, int],
    workers: int,
) -> list[TokenizedDocument]:
    tokenized_documents: list[TokenizedDocument] = []
    mode_token_id_map = {
        "text": special_token_ids["text"],
        "code": special_token_ids["code"],
        "math": special_token_ids["math"],
        "dialogue": special_token_ids["dialogue"],
        "instruction": special_token_ids["instruction"],
    }
    token_total = 0
    chunk_total = 0
    progress_interval = 2_000

    def append_document(kind: str, source: str, content_ids: torch.Tensor, index: int) -> None:
        nonlocal token_total, chunk_total
        chunks = make_document_chunks(
            content_ids=content_ids,
            mode_token_id=mode_token_id_map[kind],
            seq_len=seq_len,
            bos_token_id=special_token_ids["bos"],
            cont_token_id=special_token_ids["cont"],
            eos_token_id=special_token_ids["eos"],
            pad_token_id=special_token_ids["pad"],
        )
        token_count = int(content_ids.numel())
        token_total += token_count
        chunk_total += len(chunks)
        tokenized_documents.append(
            TokenizedDocument(
                kind=kind,
                source=source,
                chunks=chunks,
                token_count=token_count,
            )
        )
        if index % progress_interval == 0 or index == len(documents):
            avg_chunks = chunk_total / max(1, index)
            avg_tokens = token_total / max(1, index)
            print(
                f"pretokenize_progress | documents={index:,}/{len(documents):,} | chunks={chunk_total:,} | avg_chunks_per_doc={avg_chunks:.2f} | avg_tokens_per_doc={avg_tokens:.1f}",
                flush=True,
            )

    if workers > 1 and isinstance(vocab, ByteBPEVocab):
        payload_count = len(documents)
        payloads = ((document.kind, document.source, document.body) for document in documents)
        chunksize = max(8, min(512, payload_count // max(1, workers * 32) or 8))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_byte_bpe_encode_worker,
            initargs=(str(vocab.vocab_path), str(vocab.merges_path)),
        ) as executor:
            for index, (kind, source, token_ids) in enumerate(
                executor.map(_tokenize_document_payload_worker, payloads, chunksize=chunksize),
                start=1,
            ):
                if token_ids:
                    content_ids = torch.tensor(token_ids, dtype=torch.long)
                else:
                    content_ids = torch.empty(0, dtype=torch.long)
                append_document(kind, source, content_ids, index)
        return tokenized_documents

    for index, document in enumerate(documents, start=1):
        content_ids = encode_text_in_chunks(vocab, document.body, workers=workers)
        append_document(document.kind, document.source, content_ids, index)
    return tokenized_documents


def save_pretokenized_bundle(
    path: Path,
    *,
    documents: Sequence[TokenizedDocument],
    vocab_size: int,
    tokenizer_label: str,
    tokenizer_model_path: str | None,
    corpus_info: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [
            {
                "kind": document.kind,
                "source": document.source,
                "token_count": document.token_count,
                "chunks": [
                    {
                        "context": chunk.context,
                        "target": chunk.target,
                        "loss_mask": chunk.loss_mask,
                        "is_continuation": chunk.is_continuation,
                    }
                    for chunk in document.chunks
                ],
            }
            for document in documents
        ],
        "vocab_size": vocab_size,
        "tokenizer_label": tokenizer_label,
        "tokenizer_model_path": tokenizer_model_path,
        "corpus_info": corpus_info,
    }
    torch.save(payload, path)


def load_pretokenized_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "documents" not in bundle:
        raise ValueError(f"Invalid pretokenized bundle: {path}")
    documents = [
        TokenizedDocument(
            kind=item["kind"],
            source=item["source"],
            token_count=int(item["token_count"]),
            chunks=tuple(
                DocumentChunk(
                    context=chunk["context"].detach().cpu().to(dtype=torch.long),
                    target=chunk["target"].detach().cpu().to(dtype=torch.long),
                    loss_mask=chunk["loss_mask"].detach().cpu().to(dtype=torch.float32),
                    is_continuation=bool(chunk["is_continuation"]),
                )
                for chunk in item["chunks"]
            ),
        )
        for item in bundle["documents"]
    ]
    return {
        "documents": documents,
        "vocab_size": int(bundle["vocab_size"]),
        "tokenizer_label": bundle.get("tokenizer_label"),
        "tokenizer_model_path": bundle.get("tokenizer_model_path"),
        "corpus_info": bundle.get("corpus_info") or {},
    }


def split_train_val_documents(
    documents: Sequence[TokenizedDocument],
    *,
    train_fraction: float,
) -> tuple[list[TokenizedDocument], list[TokenizedDocument]]:
    if len(documents) < 2:
        return list(documents), list(documents)
    split_index = int(len(documents) * train_fraction)
    split_index = max(1, min(split_index, len(documents) - 1))
    return list(documents[:split_index]), list(documents[split_index:])


def compute_masked_loss(logits: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        target.reshape(-1),
        reduction="none",
    )
    mask = loss_mask.reshape(-1).float()
    denom = mask.sum().clamp_min(1.0)
    return (flat_loss * mask).sum() / denom


def _set_module_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _set_parameter_collection_requires_grad(parameters: Sequence[torch.nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def resolve_curriculum_stage(
    *,
    step: int,
    total_steps: int,
    stage1_ratio: float,
    stage2_ratio: float,
    stage2_span: int,
    stage3_span: int,
) -> TrainingCurriculumStage:
    stage1_end = int(total_steps * stage1_ratio)
    stage2_end = int(total_steps * stage2_ratio)
    if stage1_end > 0 and step <= stage1_end:
        return TrainingCurriculumStage(
            name="stage1",
            document_span=1,
            freeze_memory=True,
            freeze_propagation=True,
            freeze_skip=True,
        )
    if stage2_end > 0 and step <= stage2_end:
        return TrainingCurriculumStage(
            name="stage2",
            document_span=max(1, stage2_span),
            freeze_memory=False,
            freeze_propagation=True,
            freeze_skip=True,
        )
    return TrainingCurriculumStage(
        name="stage3",
        document_span=max(1, stage3_span),
        freeze_memory=False,
        freeze_propagation=False,
        freeze_skip=False,
    )


def apply_training_curriculum(model: CausalHierarchicalMemoryLM, stage: TrainingCurriculumStage) -> None:
    _set_module_requires_grad(model.memory_levels, not stage.freeze_memory)
    _set_module_requires_grad(model.level_transitions, not stage.freeze_memory)
    _set_module_requires_grad(model.level_norms, not stage.freeze_memory)
    _set_module_requires_grad(model.read_projections, not stage.freeze_memory)
    _set_parameter_collection_requires_grad(model.read_gates, not stage.freeze_memory)
    model.read_template_val.requires_grad_(not stage.freeze_memory)

    for level in model.memory_levels:
        _set_module_requires_grad(level.propagation, not stage.freeze_propagation)

    _set_module_requires_grad(model.skip_transitions, not stage.freeze_skip)
    _set_parameter_collection_requires_grad(tuple(model.skip_gates.values()), not stage.freeze_skip)


def override_batch_reset(batch: DocumentBatch, *, reset_all: bool) -> DocumentBatch:
    if not reset_all:
        return batch
    return DocumentBatch(
        context=batch.context,
        target=batch.target,
        loss_mask=batch.loss_mask,
        reset_mask=torch.ones_like(batch.reset_mask, dtype=torch.bool),
    )


def build_run_name(args: argparse.Namespace) -> str:
    memory_slug = "-".join(str(slot) for slot in args.memory_slots)
    return (
        f"causal-memory-doc-s{args.s_layers}-b{memory_slug}-p{args.prediction_layers}"
        f"-dim{args.dim}-seq{args.seq_len}"
    )


def estimate_steps_per_epoch(*, documents: Sequence[TokenizedDocument], batch_size: int) -> int:
    total_chunks = sum(len(document.chunks) for document in documents)
    return max(1, math.ceil(total_chunks / max(1, batch_size)))


def perplexity_from_loss(loss_value: float) -> float:
    clamped = min(loss_value, 20.0)
    return float(math.exp(clamped))


def compute_learning_rate(
    *,
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int,
    min_ratio: float,
) -> float:
    if total_steps <= 1:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * (step / warmup_steps)
    denom = max(1, total_steps - warmup_steps)
    progress = max(0.0, min(1.0, (step - warmup_steps) / denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def detach_memory_state(memory_state: Sequence[Any] | None) -> tuple[Any, ...] | None:
    if memory_state is None:
        return None
    return tuple(
        layer.with_tensors(state=layer.state.detach(), val=layer.val.detach())
        for layer in memory_state
    )


def memory_state_is_finite(memory_state: Sequence[Any] | None) -> bool:
    if memory_state is None:
        return True
    return all(
        bool(torch.isfinite(layer.state).all().item()) and bool(torch.isfinite(layer.val).all().item())
        for layer in memory_state
    )


def load_decode_vocab(*, tokenizer_label: str, tokenizer_model_path: str | None) -> object | None:
    if tokenizer_label != "byte_bpe" or tokenizer_model_path is None or ByteLevelBPETokenizer is None:
        return None
    vocab_path = Path(tokenizer_model_path)
    if not vocab_path.exists():
        return None
    if vocab_path.name.endswith("-vocab.json"):
        merges_path = vocab_path.with_name(vocab_path.name.replace("-vocab.json", "-merges.txt"))
    else:
        merges_path = vocab_path.with_suffix(".txt")
    if not merges_path.exists():
        return None
    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    return ByteBPEVocab(tokenizer=tokenizer, vocab_path=vocab_path, merges_path=merges_path)


def _decode_visible_tokens(vocab: object, token_ids: torch.Tensor, loss_mask: torch.Tensor | None = None) -> str:
    if not hasattr(vocab, "decode"):
        return ""
    ids = token_ids.detach().cpu().to(dtype=torch.long)
    if loss_mask is not None:
        visible = int(loss_mask.detach().cpu().sum().item())
        ids = ids[:visible]
    return str(vocab.decode(ids.tolist()))


def _eval_sample_priority(document: TokenizedDocument) -> tuple[int, str]:
    if document.source.startswith("mixed_dialogue:"):
        return (0, document.source)
    if document.kind == "dialogue":
        return (1, document.source)
    return (2, document.source)


@torch.no_grad()
def log_eval_samples_to_tensorboard(
    writer: SummaryWriter | None,
    model: CausalHierarchicalMemoryLM,
    documents: Sequence[TokenizedDocument],
    *,
    vocab: object | None,
    device: torch.device,
    precision: str,
    step: int,
    max_samples: int = 6,
) -> None:
    if writer is None or vocab is None or not documents:
        return
    seen_sources: set[str] = set()
    selected: list[TokenizedDocument] = []
    for document in sorted(documents, key=_eval_sample_priority):
        if document.source in seen_sources:
            continue
        seen_sources.add(document.source)
        selected.append(document)
        if len(selected) >= max_samples:
            break
    if not selected:
        return

    model_was_training = model.training
    model.eval()
    autocast_dtype = resolve_autocast_dtype(precision)
    autocast_supported = (
        autocast_dtype is not None
        and (device.type == "cuda" or (device.type == "cpu" and autocast_dtype == torch.bfloat16))
    )
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=autocast_dtype)
        if autocast_supported
        else nullcontext()
    )

    with autocast_context:
        sample_sections: list[str] = []
        for index, document in enumerate(selected, start=1):
            chunk = document.chunks[0]
            output = model(
                chunk.context.unsqueeze(0).to(device),
                reset_mask=torch.tensor([True], device=device, dtype=torch.bool),
                return_memory_state=False,
            )
            assert isinstance(output, torch.Tensor)
            predicted = output.argmax(dim=-1).squeeze(0).detach().cpu().to(dtype=torch.long)
            sample_sections.append(
                f"## sample {index:02d}\n\n"
                f"kind: {document.kind}\n\n"
                f"source: {document.source}\n\n"
                f"### context\n{_decode_visible_tokens(vocab, chunk.context, chunk.loss_mask)}\n\n"
                f"### target\n{_decode_visible_tokens(vocab, chunk.target, chunk.loss_mask)}\n\n"
                f"### prediction\n{_decode_visible_tokens(vocab, predicted, chunk.loss_mask)}"
            )
        writer.add_text("eval_samples/mixed_corpus", "\n\n---\n\n".join(sample_sections), step)
    if model_was_training:
        model.train()


def run_model(
    model: CausalHierarchicalMemoryLM,
    batch: DocumentBatch,
    *,
    memory_state: Sequence[Any] | None,
    precision: str,
    grad_enabled: bool,
) -> tuple[torch.Tensor, tuple[Any, ...] | None]:
    autocast_dtype = resolve_autocast_dtype(precision)
    autocast_supported = (
        autocast_dtype is not None
        and (
            batch.context.device.type == "cuda"
            or (batch.context.device.type == "cpu" and autocast_dtype == torch.bfloat16)
        )
    )
    autocast_context = (
        torch.autocast(device_type=batch.context.device.type, dtype=autocast_dtype)
        if autocast_supported
        else nullcontext()
    )
    grad_context = torch.enable_grad() if grad_enabled else torch.no_grad()
    with grad_context:
        with autocast_context:
            output = model(
                batch.context,
                memory_state=memory_state,
                reset_mask=batch.reset_mask,
                return_memory_state=True,
            )
            assert isinstance(output, MemoryScanOutput)
            loss = compute_masked_loss(output.logits, batch.target, batch.loss_mask)
    return loss, output.memory_state


@torch.no_grad()
def estimate_eval_loss(
    model: CausalHierarchicalMemoryLM,
    documents: Sequence[TokenizedDocument],
    *,
    eval_documents: int,
    device: torch.device,
    precision: str,
) -> float:
    if not documents:
        raise ValueError("documents must not be empty.")
    model_was_training = model.training
    model.eval()
    sampled_documents = random.sample(list(documents), k=min(eval_documents, len(documents)))
    total_loss = 0.0
    total_weight = 0.0
    for document in sampled_documents:
        memory_state: tuple[Any, ...] | None = None
        for chunk_index, chunk in enumerate(document.chunks):
            batch = DocumentBatch(
                context=chunk.context.unsqueeze(0).to(device),
                target=chunk.target.unsqueeze(0).to(device),
                loss_mask=chunk.loss_mask.unsqueeze(0).to(device),
                reset_mask=torch.tensor([chunk_index == 0], device=device, dtype=torch.bool),
            )
            loss, memory_state = run_model(
                model,
                batch,
                memory_state=memory_state,
                precision=precision,
                grad_enabled=False,
            )
            memory_state = detach_memory_state(memory_state)
            weight = float(chunk.loss_mask.sum().item())
            total_loss += float(loss.item()) * weight
            total_weight += weight
    if model_was_training:
        model.train()
    return total_loss / max(1.0, total_weight)


def save_checkpoint(
    path: Path,
    *,
    model: CausalHierarchicalMemoryLM,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    train_loss: float,
    val_loss: float | None,
    tokenizer_label: str,
    tokenizer_model_path: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "tokenizer_label": tokenizer_label,
            "tokenizer_model_path": tokenizer_model_path,
        },
        path,
    )


def load_checkpoint(path: Path, *, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint: {path}")
    return checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the document-chunked causal hierarchical memory LM.")
    parser.add_argument("--text-file")
    parser.add_argument("--text-source", action="append", default=[])
    parser.add_argument("--jsonl-source", action="append", default=[])
    parser.add_argument("--hf-dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-text-key", default="text")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--max-samples", type=int)

    parser.add_argument("--tokenizer", choices=("byte_bpe",), default="byte_bpe")
    parser.add_argument("--subword-vocab-size", type=int, default=16384)
    parser.add_argument("--subword-model-type", default="bpe")
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--subword-character-coverage", type=float, default=1.0)
    parser.add_argument("--subword-input-sentence-size", type=int, default=0)
    parser.add_argument("--subword-num-threads", type=int, default=0)
    parser.add_argument("--pretokenize-workers", type=int, default=8)
    parser.add_argument("--pretokenized-path")
    parser.add_argument("--save-pretokenized", action="store_true")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--s-layers", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, nargs="+", default=[256, 64, 16])
    parser.add_argument("--prediction-layers", type=int, default=2)
    parser.add_argument("--s-window", type=int, default=256)
    parser.add_argument("--s-microbatch-size", type=int, default=0)
    parser.add_argument("--prediction-window", type=int, default=64)
    parser.add_argument("--memory-topk", type=int, default=16)
    parser.add_argument("--scan-checkpoint-chunk-size", type=int, default=0)
    parser.add_argument("--scan-backend", choices=("auto", "python", "native"), default="auto")
    parser.add_argument(
        "--pairwise-kind",
        choices=("low_rank_bilinear", "diagonal_bilinear", "bilinear", "additive_low_rank"),
        default="low_rank_bilinear",
    )
    parser.add_argument(
        "--route-kind",
        choices=("low_rank_bilinear", "diagonal_bilinear", "bilinear", "additive_low_rank"),
        default="low_rank_bilinear",
    )
    parser.add_argument("--pairwise-rank", type=int, default=64)
    parser.add_argument("--route-rank", type=int, default=64)
    parser.add_argument("--implementation", choices=("reference", "streaming", "kernel", "native"), default="streaming")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-documents", type=int, default=8)
    parser.add_argument("--curriculum-stage1-ratio", type=float, default=0.1)
    parser.add_argument("--curriculum-stage2-ratio", type=float, default=0.4)
    parser.add_argument("--curriculum-stage2-span", type=int, default=4)
    parser.add_argument("--curriculum-stage3-span", type=int, default=8)

    parser.add_argument("--output-root", default="artifacts/training_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--tensorboard", action="store_true")
    return parser.parse_args()


def summarize_documents(documents: Sequence[SerializedDocument]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for document in documents:
        summary[document.kind] = summary.get(document.kind, 0) + 1
    return summary


def summarize_tokenized_documents(documents: Sequence[TokenizedDocument]) -> dict[str, int]:
    return {
        "documents": len(documents),
        "chunks": sum(len(document.chunks) for document in documents),
        "tokens": sum(document.token_count for document in documents),
    }


def main() -> None:
    args = parse_args()
    if args.pretokenize_workers < 0:
        raise ValueError("pretokenize-workers must be non-negative.")
    if args.grad_accum_steps <= 0:
        raise ValueError("grad-accum-steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if args.seq_len <= 2:
        raise ValueError("seq-len must be larger than 2.")
    if args.epochs <= 0.0:
        raise ValueError("epochs must be positive.")
    if not 0.0 <= args.curriculum_stage1_ratio <= 1.0:
        raise ValueError("curriculum-stage1-ratio must be between 0 and 1.")
    if not 0.0 <= args.curriculum_stage2_ratio <= 1.0:
        raise ValueError("curriculum-stage2-ratio must be between 0 and 1.")
    if args.curriculum_stage2_ratio < args.curriculum_stage1_ratio:
        raise ValueError("curriculum-stage2-ratio must be greater than or equal to curriculum-stage1-ratio.")
    if args.curriculum_stage2_span <= 0 or args.curriculum_stage3_span <= 0:
        raise ValueError("curriculum-stage spans must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(device)}", flush=True)

    tokenizer_label: str
    tokenizer_model_path: str | None
    vocab_size: int
    corpus_metadata: dict[str, Any]
    documents: list[TokenizedDocument]

    if args.pretokenized_path and Path(args.pretokenized_path).exists():
        bundle = load_pretokenized_bundle(Path(args.pretokenized_path))
        documents = list(bundle["documents"])
        tokenizer_label = str(bundle.get("tokenizer_label") or "unknown")
        tokenizer_model_path = bundle.get("tokenizer_model_path")
        vocab_size = int(bundle["vocab_size"])
        corpus_metadata = dict(bundle.get("corpus_info") or {})
        print(
            f"loaded_pretokenized | path={args.pretokenized_path} | documents={len(documents):,} | tokenizer={tokenizer_label}",
            flush=True,
        )
    else:
        serialized_documents = load_serialized_documents(
            text_file=args.text_file,
            text_sources=tuple(args.text_source),
            jsonl_sources=tuple(args.jsonl_source),
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_config,
            hf_split=args.hf_split,
            hf_text_key=args.hf_text_key,
            hf_streaming=args.hf_streaming,
            max_samples=args.max_samples,
        )
        document_summary = summarize_documents(serialized_documents)
        print(
            f"documents={len(serialized_documents):,} | kinds={document_summary}",
            flush=True,
        )
        training_text = build_training_text(serialized_documents)
        vocab, tokenizer_label, tokenizer_path = build_tokenizer(
            training_text,
            text_path=None,
            tokenizer=args.tokenizer,
            subword_vocab_size=args.subword_vocab_size,
            subword_model_type=args.subword_model_type,
            tokenizer_prefix=args.tokenizer_prefix,
            subword_character_coverage=args.subword_character_coverage,
            subword_input_sentence_size=args.subword_input_sentence_size,
            subword_num_threads=args.subword_num_threads,
            user_defined_symbols=CAUSAL_DOC_SPECIAL_TOKENS,
        )
        tokenizer_model_path = None if tokenizer_path is None else str(tokenizer_path)
        special_token_ids = build_special_token_id_map(vocab)
        workers = min(args.pretokenize_workers, max(1, torch.get_num_threads()))
        documents = tokenize_documents(
            serialized_documents,
            vocab=vocab,
            seq_len=args.seq_len,
            special_token_ids=special_token_ids,
            workers=workers,
        )
        vocab_size = int(getattr(vocab, "size"))
        corpus_metadata = {
            "document_summary": document_summary,
            "tokenized_summary": summarize_tokenized_documents(documents),
            "special_tokens": special_token_ids,
        }
        print(
            f"tokenizer={tokenizer_label} | tokenizer_model={tokenizer_model_path} | tokenized={corpus_metadata['tokenized_summary']}",
            flush=True,
        )
        if args.save_pretokenized and args.pretokenized_path:
            save_pretokenized_bundle(
                Path(args.pretokenized_path),
                documents=documents,
                vocab_size=vocab_size,
                tokenizer_label=tokenizer_label,
                tokenizer_model_path=tokenizer_model_path,
                corpus_info=corpus_metadata,
            )
            print(f"saved_pretokenized | path={args.pretokenized_path}", flush=True)

    train_documents, val_documents = split_train_val_documents(documents, train_fraction=args.train_fraction)
    decode_vocab = load_decode_vocab(
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
    )
    steps_per_epoch = estimate_steps_per_epoch(documents=train_documents, batch_size=args.batch_size)
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))

    model = CausalHierarchicalMemoryLM(
        vocab_size=vocab_size,
        dim=args.dim,
        max_seq_len=args.seq_len,
        s_layers=args.s_layers,
        memory_slots=tuple(args.memory_slots),
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        s_microbatch_size=None if args.s_microbatch_size <= 0 else args.s_microbatch_size,
        prediction_window=args.prediction_window,
        memory_topk=args.memory_topk,
        scan_checkpoint_chunk_size=None if args.scan_checkpoint_chunk_size <= 0 else args.scan_checkpoint_chunk_size,
        scan_backend=args.scan_backend,
        pairwise_kind=args.pairwise_kind,
        route_kind=args.route_kind,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        implementation=args.implementation,
    ).to(device)
    parameter_count = count_parameters(model)
    print(
        f"model=causal_memory_doc | params={parameter_count:,} | dim={args.dim} | seq_len={args.seq_len} | "
        f"s_window={args.s_window} | s_microbatch_size={args.s_microbatch_size} | "
        f"scan_backend={args.scan_backend} | scan_checkpoint_chunk_size={args.scan_checkpoint_chunk_size} | memory_slots={args.memory_slots}",
        flush=True,
    )

    run_name = args.run_name or build_run_name(args)
    run_dir = create_run_directory(args.output_root, run_name)
    checkpoints_dir = ensure_directory(run_dir / "checkpoints")
    write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "tokenizer_label": tokenizer_label,
            "tokenizer_model_path": tokenizer_model_path,
            "parameter_count": parameter_count,
            "corpus_metadata": corpus_metadata,
        },
    )

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("tensorboard is not installed.")
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if args.precision == "fp16" and device.type == "cuda" else None
    batcher = DocumentChunkBatcher(train_documents, batch_size=args.batch_size, device=device)

    history_rows: list[dict[str, Any]] = []
    train_memory_state: tuple[Any, ...] | None = None
    active_stage_name: str | None = None
    start_step = 0
    best_val_loss = float("inf")
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))
        checkpoint_val_loss = checkpoint.get("val_loss")
        if checkpoint_val_loss is not None:
            best_val_loss = float(checkpoint_val_loss)
        print(
            f"resumed_checkpoint | path={checkpoint_path} | step={start_step} | "
            f"train_loss={checkpoint.get('train_loss')} | val_loss={checkpoint.get('val_loss')}",
            flush=True,
        )
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for step in range(start_step + 1, total_steps + 1):
        stage = resolve_curriculum_stage(
            step=step,
            total_steps=total_steps,
            stage1_ratio=args.curriculum_stage1_ratio,
            stage2_ratio=args.curriculum_stage2_ratio,
            stage2_span=args.curriculum_stage2_span,
            stage3_span=args.curriculum_stage3_span,
        )
        if stage.name != active_stage_name:
            apply_training_curriculum(model, stage)
            train_memory_state = None
            active_stage_name = stage.name
            print(
                f"curriculum | step={step} | stage={stage.name} | span={stage.document_span} | "
                f"freeze_memory={stage.freeze_memory} | freeze_propagation={stage.freeze_propagation} | freeze_skip={stage.freeze_skip}",
                flush=True,
            )

        lr = compute_learning_rate(
            step=step,
            total_steps=total_steps,
            base_lr=args.learning_rate,
            warmup_steps=args.warmup_steps,
            min_ratio=args.lr_min_ratio,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        current_memory_state = None if stage.freeze_memory else train_memory_state
        span_losses: list[float] = []
        loss_is_finite = True
        for span_index in range(stage.document_span):
            batch = override_batch_reset(
                batcher.next_batch(),
                reset_all=stage.freeze_memory,
            )
            loss, next_memory_state = run_model(
                model,
                batch,
                memory_state=current_memory_state,
                precision=args.precision,
                grad_enabled=True,
            )
            loss_is_finite = bool(torch.isfinite(loss.detach()).item())
            if not loss_is_finite:
                print(
                    f"warning | step={step} | span_index={span_index} | non-finite loss; skipping optimizer step",
                    flush=True,
                )
                break
            span_losses.append(float(loss.item()))
            scaled_loss = loss / (args.grad_accum_steps * stage.document_span)
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            current_memory_state = None if stage.freeze_memory else next_memory_state

        if loss_is_finite:
            train_memory_state = None if stage.freeze_memory else detach_memory_state(current_memory_state)
            if not memory_state_is_finite(train_memory_state):
                print(f"warning | step={step} | non-finite memory state; resetting memory", flush=True)
                train_memory_state = None
        else:
            optimizer.zero_grad(set_to_none=True)
            train_memory_state = None

        should_step_optimizer = step % args.grad_accum_steps == 0 or step == total_steps
        if loss_is_finite and should_step_optimizer:
            if scaler is not None:
                scaler.unscale_(optimizer)
            try:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        args.grad_clip,
                        error_if_nonfinite=True,
                    ).item()
                )
            except RuntimeError as exc:
                grad_norm = float("nan")
                print(f"warning | step={step} | non-finite grad norm; skipping optimizer step | {exc}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                train_memory_state = None
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = float("nan")

        train_loss = float(sum(span_losses) / max(1, len(span_losses)))
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/document_span", stage.document_span, step)
            if not math.isnan(grad_norm):
                writer.add_scalar("train/grad_norm", grad_norm, step)

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"progress | step={step:5d}/{total_steps} | stage={stage.name} | span={stage.document_span} | "
                f"train_loss={train_loss:.4f} | lr={lr:.6g} | elapsed={elapsed:.1f}s",
                flush=True,
            )

        val_loss = None
        if step == 1 or step % args.eval_interval == 0 or step == total_steps:
            val_loss = estimate_eval_loss(
                model,
                val_documents,
                eval_documents=args.eval_documents,
                device=device,
                precision=args.precision,
            )
            val_ppl = perplexity_from_loss(val_loss)
            print(
                f"eval | step={step} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("eval/val_loss", val_loss, step)
                writer.add_scalar("eval/val_ppl", val_ppl, step)
                log_eval_samples_to_tensorboard(
                    writer,
                    model,
                    val_documents,
                    vocab=decode_vocab,
                    device=device,
                    precision=args.precision,
                    step=step,
                )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    tokenizer_label=tokenizer_label,
                    tokenizer_model_path=tokenizer_model_path,
                )
                write_json(
                    checkpoints_dir / "best.json",
                    {"step": step, "train_loss": train_loss, "val_loss": val_loss},
                )

        if step == total_steps or step % args.eval_interval == 0:
            save_checkpoint(
                checkpoints_dir / "last.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                args=args,
                train_loss=train_loss,
                val_loss=val_loss,
                tokenizer_label=tokenizer_label,
                tokenizer_model_path=tokenizer_model_path,
            )
            write_json(
                checkpoints_dir / "last.json",
                {"step": step, "train_loss": train_loss, "val_loss": val_loss},
            )

        row = {
            "step": step,
            "stage": stage.name,
            "document_span": stage.document_span,
            "train_loss": train_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "val_loss": val_loss,
            "val_ppl": None if val_loss is None else perplexity_from_loss(val_loss),
        }
        history_rows.append(row)
        append_jsonl(run_dir / "history.jsonl", row)

    write_csv_rows(run_dir / "history.csv", history_rows)
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
