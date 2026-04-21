from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import nullcontext
import json
import math
import os
import queue
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

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
    _encode_byte_bpe_preserving_specials,
    _encode_byte_bpe_text_batch_worker,
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
    resolve_tokenizer_prefix,
)

from jakal_net import describe_device, resolve_device
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput, ModelRecurrentState

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
    loss_mode: str = "all"


@dataclass(frozen=True, slots=True)
class TrainingCurriculumStage:
    name: str
    document_span: int
    batch_size: int
    grad_accum_steps: int
    freeze_memory: bool
    freeze_propagation: bool
    freeze_skip: bool
    bucket_weights: dict[str, float]


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


@dataclass(frozen=True, slots=True)
class FlatDocumentRef:
    shard_index: int
    document_index: int


@dataclass(frozen=True, slots=True)
class FlatPretokenizedShard:
    path: Path
    kind: tuple[str, ...]
    source: tuple[str, ...]
    token_count: torch.Tensor
    document_chunk_offsets: torch.Tensor
    chunk_token_offsets: torch.Tensor
    context_flat: torch.Tensor
    loss_mask_flat: torch.Tensor
    is_continuation: torch.Tensor
    seq_len: int
    pad_token_id: int
    eos_token_id: int
    cont_token_id: int
    special_token_ids: dict[str, int]
    vocab_size: int
    tokenizer_label: str | None
    tokenizer_model_path: str | None
    corpus_info: dict[str, Any]

    @property
    def num_documents(self) -> int:
        return len(self.kind)

    def document_chunk_range(self, document_index: int) -> tuple[int, int]:
        start = int(self.document_chunk_offsets[document_index].item())
        end = int(self.document_chunk_offsets[document_index + 1].item())
        return start, end

    def chunk_count(self, document_index: int) -> int:
        start, end = self.document_chunk_range(document_index)
        return end - start

    def build_chunk_tensors(
        self,
        chunk_index: int,
        *,
        document_index: int,
        is_last_chunk: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        token_start = int(self.chunk_token_offsets[chunk_index].item())
        token_end = int(self.chunk_token_offsets[chunk_index + 1].item())
        active_context = self.context_flat[token_start:token_end]
        active_length = int(active_context.shape[0])
        context = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
        if active_length > 0:
            context[:active_length] = active_context.to(dtype=torch.long)
        target = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
        if active_length > 1:
            target[: active_length - 1] = active_context[1:].to(dtype=torch.long)
        if active_length > 0:
            target[active_length - 1] = self.eos_token_id if is_last_chunk else self.cont_token_id
        loss_mask = torch.zeros(self.seq_len, dtype=torch.float32)
        document_kind = self.kind[document_index].lower()
        if active_length > 0:
            if document_kind in {"dialogue", "instruction"} and active_length > 2:
                content_ids = active_context[2:].to(dtype=torch.long)
                content_visibility = _content_target_visibility(
                    content_ids,
                    loss_mode="assistant_only",
                    special_token_ids=self.special_token_ids,
                )
                loss_mask[1 : 1 + content_visibility.shape[0]] = content_visibility.to(dtype=torch.float32)
            else:
                loss_mask[:active_length] = self.loss_mask_flat[token_start:token_end].to(dtype=torch.float32)
        return context, target, loss_mask, bool(self.is_continuation[chunk_index].item())


@dataclass(frozen=True, slots=True)
class FlatPretokenizedDirectory:
    shards: tuple[FlatPretokenizedShard, ...]
    document_refs: tuple[FlatDocumentRef, ...]
    vocab_size: int
    tokenizer_label: str | None
    tokenizer_model_path: str | None
    corpus_info: dict[str, Any]


SCHEDULED_BUCKET_ORDER = (
    "mixed_dialogue",
    "arxiv",
    "dialogue",
    "wiki",
    "code",
    "math",
    "pubmed",
    "docs",
    "math_qa",
    "reasoning",
)


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


def _content_target_visibility(
    content_ids: torch.Tensor,
    *,
    loss_mode: str,
    special_token_ids: dict[str, int],
) -> torch.Tensor:
    content_ids = content_ids.detach().to(dtype=torch.long)
    if loss_mode == "all":
        return torch.ones(content_ids.shape[0], dtype=torch.float32, device=content_ids.device)
    if loss_mode != "assistant_only":
        raise ValueError(f"Unsupported loss_mode: {loss_mode!r}")
    assistant_start_mask = (content_ids == int(special_token_ids["assistant"])) | (
        content_ids == int(special_token_ids["response"])
    )
    assistant_stop_mask = content_ids == int(special_token_ids["eot"])
    visibility = torch.zeros(content_ids.shape[0], dtype=torch.float32, device=content_ids.device)
    start_positions = torch.nonzero(assistant_start_mask, as_tuple=False).flatten().tolist()
    stop_positions = torch.nonzero(assistant_stop_mask, as_tuple=False).flatten().tolist()
    for start_index in start_positions:
        stop_index = content_ids.shape[0] - 1
        for candidate in stop_positions:
            if candidate >= start_index:
                stop_index = candidate
                break
        visibility[start_index : stop_index + 1] = 1.0
    return visibility


def _tokenize_document_payload_worker(payload: tuple[str, str, str]) -> tuple[str, str, list[int]]:
    kind, source, body = payload
    token_ids: list[int] = []
    for chunk in _iter_nonempty_text_chunks(body, chunk_chars=8_000_000):
        token_ids.extend(_encode_byte_bpe_text_worker(chunk))
    return kind, source, token_ids


def _tokenize_document_payload_batch_worker(
    payloads: Sequence[tuple[str, str, str]],
) -> list[tuple[str, str, list[int]]]:
    encoded_payloads: list[tuple[str, str, list[int]]] = []
    for kind, source, body in payloads:
        token_ids: list[int] = []
        chunks = list(_iter_nonempty_text_chunks(body, chunk_chars=8_000_000))
        if chunks:
            for encoded_chunk in _encode_byte_bpe_text_batch_worker(chunks):
                token_ids.extend(encoded_chunk)
        encoded_payloads.append((kind, source, token_ids))
    return encoded_payloads


def document_sampling_bucket(document: SerializedDocument | TokenizedDocument) -> str:
    source = document.source.lower()
    kind = document.kind.lower()
    return document_sampling_bucket_from_kind_source(kind=kind, source=source)


def document_sampling_bucket_from_kind_source(*, kind: str, source: str) -> str:
    if source.startswith("mixed_dialogue:"):
        return "mixed_dialogue"
    if source.startswith("math_qa:"):
        return "math_qa"
    if source.startswith("reasoning_qa:"):
        return "reasoning"
    if "ccdv/arxiv-classification" in source or source.startswith("arxiv:"):
        return "arxiv"
    if "medrag/pubmed" in source or source.startswith("pubmed:"):
        return "pubmed"
    if kind == "dialogue":
        return "dialogue"
    if kind == "code":
        return "code"
    if kind == "math":
        return "math"
    if "wikipedia" in source or source.startswith("wiki:"):
        return "wiki"
    return "docs"


def summarize_document_buckets(documents: Sequence[SerializedDocument | TokenizedDocument]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for document in documents:
        bucket = document_sampling_bucket(document)
        summary[bucket] = summary.get(bucket, 0) + 1
    return {bucket: summary[bucket] for bucket in SCHEDULED_BUCKET_ORDER if bucket in summary}


def resolve_bucket_weights(*, step: int, total_steps: int) -> dict[str, float]:
    if total_steps <= 1:
        progress = 1.0
    else:
        progress = max(0.0, min(1.0, (step - 1) / (total_steps - 1)))
    conditioned_progress = 0.0 if progress <= 0.20 else (progress - 0.20) / 0.80
    pure_front = 0.20 + 0.80 * ((1.0 - progress) ** 1.5)
    conditioned_late = conditioned_progress ** 1.5
    return {
        "mixed_dialogue": 1.0,
        "arxiv": 1.0,
        "pubmed": 1.0,
        "dialogue": pure_front,
        "wiki": pure_front,
        "code": pure_front,
        "math": pure_front,
        "docs": pure_front,
        "math_qa": conditioned_late,
        "reasoning": conditioned_late,
    }


def resolve_stage_bucket_weights(*, stage_name: str) -> dict[str, float]:
    if stage_name == "stage1":
        return {
            "mixed_dialogue": 1.0,
            "arxiv": 1.0,
            "pubmed": 1.0,
            "dialogue": 1.0,
            "wiki": 1.0,
            "code": 1.0,
            "math": 1.0,
            "docs": 1.0,
            "math_qa": 0.0,
            "reasoning": 0.0,
        }
    if stage_name == "stage2":
        return {
            "mixed_dialogue": 1.0,
            "arxiv": 1.0,
            "pubmed": 1.0,
            "dialogue": 0.6,
            "wiki": 0.6,
            "code": 0.6,
            "math": 0.6,
            "docs": 0.6,
            "math_qa": 0.35,
            "reasoning": 0.35,
        }
    return {
        "mixed_dialogue": 1.0,
        "arxiv": 1.0,
        "pubmed": 1.0,
        "dialogue": 0.2,
        "wiki": 0.2,
        "code": 0.2,
        "math": 0.2,
        "docs": 0.2,
        "math_qa": 1.0,
        "reasoning": 1.0,
    }


def sample_documents_uniform_by_bucket(
    documents: Sequence[TokenizedDocument],
    *,
    sample_count: int,
) -> list[TokenizedDocument]:
    if sample_count <= 0 or not documents:
        return []
    bucket_to_documents: dict[str, list[TokenizedDocument]] = {}
    for document in documents:
        bucket_to_documents.setdefault(document_sampling_bucket(document), []).append(document)
    if not bucket_to_documents:
        return random.sample(list(documents), k=min(sample_count, len(documents)))
    ordered_buckets = [bucket for bucket in SCHEDULED_BUCKET_ORDER if bucket_to_documents.get(bucket)]
    bucket_queues = {
        bucket: random.sample(bucket_to_documents[bucket], k=len(bucket_to_documents[bucket]))
        for bucket in ordered_buckets
    }
    selected: list[TokenizedDocument] = []
    while ordered_buckets and len(selected) < sample_count:
        next_buckets: list[str] = []
        for bucket in ordered_buckets:
            queue = bucket_queues[bucket]
            if not queue:
                continue
            selected.append(queue.pop())
            if queue:
                next_buckets.append(bucket)
            if len(selected) >= sample_count:
                break
        ordered_buckets = next_buckets
    return selected[:sample_count]


def flat_document_ref_bucket(collection: FlatPretokenizedDirectory, reference: FlatDocumentRef) -> str:
    shard = collection.shards[reference.shard_index]
    return document_sampling_bucket_from_kind_source(
        kind=shard.kind[reference.document_index].lower(),
        source=shard.source[reference.document_index].lower(),
    )


def summarize_flat_document_buckets(
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
) -> dict[str, int]:
    summary: dict[str, int] = {}
    for reference in documents:
        bucket = flat_document_ref_bucket(collection, reference)
        summary[bucket] = summary.get(bucket, 0) + 1
    return {bucket: summary[bucket] for bucket in SCHEDULED_BUCKET_ORDER if bucket in summary}


def sample_flat_documents_uniform_by_bucket(
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    *,
    sample_count: int,
) -> list[FlatDocumentRef]:
    if sample_count <= 0 or not documents:
        return []
    bucket_to_documents: dict[str, list[FlatDocumentRef]] = {}
    for reference in documents:
        bucket_to_documents.setdefault(flat_document_ref_bucket(collection, reference), []).append(reference)
    if not bucket_to_documents:
        return random.sample(list(documents), k=min(sample_count, len(documents)))
    ordered_buckets = [bucket for bucket in SCHEDULED_BUCKET_ORDER if bucket_to_documents.get(bucket)]
    bucket_queues = {
        bucket: random.sample(bucket_to_documents[bucket], k=len(bucket_to_documents[bucket]))
        for bucket in ordered_buckets
    }
    selected: list[FlatDocumentRef] = []
    while ordered_buckets and len(selected) < sample_count:
        next_buckets: list[str] = []
        for bucket in ordered_buckets:
            queue = bucket_queues[bucket]
            if not queue:
                continue
            selected.append(queue.pop())
            if queue:
                next_buckets.append(bucket)
            if len(selected) >= sample_count:
                break
        ordered_buckets = next_buckets
    return selected[:sample_count]


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
        self.pin_memory = device.type == "cuda"
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size
        self.bucket_to_indices: dict[str, tuple[int, ...]] = {}
        for index, document in enumerate(self.documents):
            bucket = document_sampling_bucket(document)
            self.bucket_to_indices.setdefault(bucket, []).append(index)
        self.bucket_to_indices = {
            bucket: tuple(indices)
            for bucket, indices in self.bucket_to_indices.items()
            if indices
        }
        self.active_buckets = tuple(self.bucket_to_indices)
        self.active_bucket_weights = tuple(1.0 for _ in self.active_buckets)

    def set_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size == self.batch_size:
            return
        self.batch_size = batch_size
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size

    def _sample_document_index(self) -> int:
        if not self.active_buckets:
            return random.randrange(len(self.documents))
        bucket = random.choices(self.active_buckets, weights=self.active_bucket_weights, k=1)[0]
        return random.choice(self.bucket_to_indices[bucket])

    def set_bucket_weights(self, weights: dict[str, float]) -> None:
        active = [
            (bucket, max(0.0, float(weights.get(bucket, 0.0))))
            for bucket in SCHEDULED_BUCKET_ORDER
            if bucket in self.bucket_to_indices
        ]
        active = [(bucket, weight) for bucket, weight in active if weight > 0.0]
        if not active:
            active = [(bucket, 1.0) for bucket in self.bucket_to_indices]
        self.active_buckets = tuple(bucket for bucket, _ in active)
        self.active_bucket_weights = tuple(weight for _, weight in active)

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
        context = torch.stack(contexts, dim=0)
        target = torch.stack(targets, dim=0)
        loss_mask = torch.stack(masks, dim=0)
        if self.pin_memory:
            context = context.pin_memory()
            target = target.pin_memory()
            loss_mask = loss_mask.pin_memory()
            reset_mask = reset_mask.pin_memory()
        return DocumentBatch(
            context=context,
            target=target,
            loss_mask=loss_mask,
            reset_mask=reset_mask,
        )


def move_batch_to_device(
    batch: DocumentBatch,
    *,
    device: torch.device,
    non_blocking: bool = True,
) -> DocumentBatch:
    return DocumentBatch(
        context=batch.context.to(device, non_blocking=non_blocking),
        target=batch.target.to(device, non_blocking=non_blocking),
        loss_mask=batch.loss_mask.to(device, non_blocking=non_blocking),
        reset_mask=batch.reset_mask.to(device, non_blocking=non_blocking),
    )


@dataclass(frozen=True, slots=True)
class _PrefetchFailure:
    error: BaseException


class AsyncDocumentBatchPrefetcher:
    def __init__(
        self,
        batcher: DocumentChunkBatcher,
        *,
        device: torch.device,
        prefetch_batches: int,
    ) -> None:
        self.batcher = batcher
        self.device = device
        self.prefetch_batches = max(1, prefetch_batches)
        self._queue: queue.Queue[DocumentBatch | _PrefetchFailure] = queue.Queue(maxsize=self.prefetch_batches)
        self._stop = threading.Event()
        self._stats_lock = threading.Lock()
        self._produced_batches = 0
        self._total_batch_build_seconds = 0.0
        self._thread = threading.Thread(target=self._worker, name="cmem_batch_prefetch", daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                started_at = time.perf_counter()
                batch = self.batcher.next_batch()
                elapsed = time.perf_counter() - started_at
                with self._stats_lock:
                    self._produced_batches += 1
                    self._total_batch_build_seconds += elapsed
                placed = False
                while not placed and not self._stop.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        placed = True
                    except queue.Full:
                        continue
            except BaseException as exc:  # pragma: no cover - defensive background thread path
                while not self._stop.is_set():
                    try:
                        self._queue.put(_PrefetchFailure(exc), timeout=0.1)
                        return
                    except queue.Full:
                        continue

    def next_batch(self) -> DocumentBatch:
        item = self._queue.get()
        if isinstance(item, _PrefetchFailure):
            raise RuntimeError("Async batch prefetcher failed.") from item.error
        return move_batch_to_device(item, device=self.device, non_blocking=True)

    def stats(self) -> tuple[float, int]:
        with self._stats_lock:
            average = (
                self._total_batch_build_seconds / self._produced_batches
                if self._produced_batches > 0
                else 0.0
            )
        return average, self._queue.qsize()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


class FlatDocumentChunkBatcher:
    def __init__(
        self,
        collection: FlatPretokenizedDirectory,
        documents: Sequence[FlatDocumentRef],
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")
        self.collection = collection
        self.documents = tuple(documents)
        self.batch_size = batch_size
        self.device = device
        self.pin_memory = device.type == "cuda"
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size
        self.bucket_to_indices: dict[str, tuple[int, ...]] = {}
        for index, reference in enumerate(self.documents):
            bucket = flat_document_ref_bucket(self.collection, reference)
            self.bucket_to_indices.setdefault(bucket, []).append(index)
        self.bucket_to_indices = {
            bucket: tuple(indices)
            for bucket, indices in self.bucket_to_indices.items()
            if indices
        }
        self.active_buckets = tuple(self.bucket_to_indices)
        self.active_bucket_weights = tuple(1.0 for _ in self.active_buckets)

    def set_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size == self.batch_size:
            return
        self.batch_size = batch_size
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size

    def _sample_document_index(self) -> int:
        if not self.active_buckets:
            return random.randrange(len(self.documents))
        bucket = random.choices(self.active_buckets, weights=self.active_bucket_weights, k=1)[0]
        return random.choice(self.bucket_to_indices[bucket])

    def set_bucket_weights(self, weights: dict[str, float]) -> None:
        active = [
            (bucket, max(0.0, float(weights.get(bucket, 0.0))))
            for bucket in SCHEDULED_BUCKET_ORDER
            if bucket in self.bucket_to_indices
        ]
        active = [(bucket, weight) for bucket, weight in active if weight > 0.0]
        if not active:
            active = [(bucket, 1.0) for bucket in self.bucket_to_indices]
        self.active_buckets = tuple(bucket for bucket, _ in active)
        self.active_bucket_weights = tuple(weight for _, weight in active)

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
            reference = self.documents[self.current_doc[item_index]]
            shard = self.collection.shards[reference.shard_index]
            chunk_start, chunk_end = shard.document_chunk_range(reference.document_index)
            chunk_index = chunk_start + self.current_chunk[item_index]
            context, target, loss_mask, _ = shard.build_chunk_tensors(
                chunk_index,
                document_index=reference.document_index,
                is_last_chunk=chunk_index == chunk_end - 1,
            )
            contexts.append(context)
            targets.append(target)
            masks.append(loss_mask)
            next_chunk = self.current_chunk[item_index] + 1
            if chunk_start + next_chunk >= chunk_end:
                self.needs_reset[item_index] = True
                self.current_chunk[item_index] = 0
            else:
                self.needs_reset[item_index] = False
                self.current_chunk[item_index] = next_chunk
        context = torch.stack(contexts, dim=0)
        target = torch.stack(targets, dim=0)
        loss_mask = torch.stack(masks, dim=0)
        if self.pin_memory:
            context = context.pin_memory()
            target = target.pin_memory()
            loss_mask = loss_mask.pin_memory()
            reset_mask = reset_mask.pin_memory()
        return DocumentBatch(
            context=context,
            target=target,
            loss_mask=loss_mask,
            reset_mask=reset_mask,
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
            loss_mode="all",
        )
    if not isinstance(record, dict):
        return None

    record_source = _record_source(record, fallback=source)
    messages = record.get("messages") or record.get("conversations")
    if isinstance(messages, list):
        normalized = [message for message in messages if isinstance(message, dict)]
        dialogue_body = _normalize_dialogue_body(normalized)
        if dialogue_body:
            return SerializedDocument(kind="dialogue", source=record_source, body=dialogue_body, loss_mode="assistant_only")

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
            loss_mode="assistant_only",
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
                loss_mode="assistant_only" if top_kind in {"dialogue", "instruction"} else "all",
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
    return SerializedDocument(kind=kind, source=record_source, body=body, loss_mode="all")


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
                    loss_mode="all",
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
            documents.append(SerializedDocument(kind="text", source=f"{hf_dataset}:{index}", body=text.strip(), loss_mode="all"))
    if not documents:
        documents.append(
            SerializedDocument(
                kind="text",
                source="default_text",
                body="Jakal-Net causal memory training sample document.",
                loss_mode="all",
            )
        )
    return documents


def render_document_for_tokenizer(document: SerializedDocument) -> str:
    mode_token = MODE_TOKEN_BY_KIND[document.kind]
    return f"{BOS_TOKEN}\n{mode_token}\n{document.body}\n{EOS_TOKEN}"


def build_training_text(documents: Sequence[SerializedDocument]) -> str:
    return "\n\n".join(render_document_for_tokenizer(document) for document in documents)


def byte_bpe_tokenizer_cache_exists(
    *,
    tokenizer_prefix: str | None,
    vocab_size: int,
) -> bool:
    if not tokenizer_prefix:
        return False
    prefix_path = resolve_tokenizer_prefix(
        text="",
        tokenizer="byte_bpe",
        model_type="byte_level",
        vocab_size=vocab_size,
        prefix=tokenizer_prefix,
    )
    vocab_path = prefix_path.parent / f"{prefix_path.name}-vocab.json"
    merges_path = prefix_path.parent / f"{prefix_path.name}-merges.txt"
    return vocab_path.exists() and merges_path.exists()


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
    content_target_visibility: torch.Tensor | None = None,
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
    if content_target_visibility is None:
        content_target_visibility = torch.ones(content_ids.shape[0], dtype=torch.float32)
    else:
        content_target_visibility = content_target_visibility.detach().cpu().to(dtype=torch.float32)
        if content_target_visibility.shape != content_ids.shape:
            raise ValueError("content_target_visibility must have the same shape as content_ids.")
    chunks: list[DocumentChunk] = []
    cursor = 0
    first = True
    if content_ids.numel() == 0:
        context = torch.tensor([bos_token_id, mode_token_id], dtype=torch.long)
        target = torch.tensor([mode_token_id, eos_token_id], dtype=torch.long)
        loss_mask = torch.zeros(2, dtype=torch.float32)
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
        content_visibility_slice = content_target_visibility[cursor : cursor + take]
        context = torch.tensor(prefix, dtype=torch.long)
        if content_slice.numel() > 0:
            context = torch.cat((context, content_slice), dim=0)
        target = torch.empty_like(context)
        target[:-1] = context[1:]
        target[-1] = eos_token_id if cursor + take >= content_ids.numel() else cont_token_id
        loss_mask = torch.zeros(target.shape[0], dtype=torch.float32)
        if content_visibility_slice.numel() > 0:
            loss_mask[1 : 1 + content_visibility_slice.shape[0]] = content_visibility_slice
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
    processing_device: torch.device | str | None = None,
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

    def append_document(kind: str, source: str, content_ids: torch.Tensor, *, loss_mode: str, index: int) -> None:
        nonlocal token_total, chunk_total
        visibility_input = (
            content_ids.to(device=processing_device, dtype=torch.long)
            if processing_device is not None and content_ids.numel() > 0
            else content_ids
        )
        target_visibility = _content_target_visibility(
            visibility_input,
            loss_mode=loss_mode,
            special_token_ids=special_token_ids,
        )
        chunks = make_document_chunks(
            content_ids=content_ids,
            content_target_visibility=target_visibility,
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
        payload_batch_size = max(32, min(512, payload_count // max(1, workers * 8) or 128))
        payload_batches = [
            [
                (document.kind, document.source, document.body)
                for document in documents[batch_start : batch_start + payload_batch_size]
            ]
            for batch_start in range(0, payload_count, payload_batch_size)
        ]
        total_payload_batches = len(payload_batches)
        chunksize = max(1, min(32, total_payload_batches // max(1, workers * 2) or 1))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_byte_bpe_encode_worker,
            initargs=(str(vocab.vocab_path), str(vocab.merges_path), vocab.special_tokens),
        ) as executor:
            index = 0
            for encoded_batch in executor.map(
                _tokenize_document_payload_batch_worker,
                payload_batches,
                chunksize=chunksize,
            ):
                for kind, source, token_ids in encoded_batch:
                    index += 1
                    if token_ids:
                        content_ids = torch.tensor(token_ids, dtype=torch.long)
                    else:
                        content_ids = torch.empty(0, dtype=torch.long)
                    document = documents[index - 1]
                    append_document(document.kind, document.source, content_ids, loss_mode=document.loss_mode, index=index)
        return tokenized_documents

    for index, document in enumerate(documents, start=1):
        content_ids = encode_text_in_chunks(vocab, document.body, workers=workers)
        append_document(document.kind, document.source, content_ids, loss_mode=document.loss_mode, index=index)
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
    special_tokens = dict(corpus_info.get("special_tokens") or {})
    pad_token_id = special_tokens.get("pad")
    eos_token_id = special_tokens.get("eos")
    cont_token_id = special_tokens.get("cont")
    if pad_token_id is None or eos_token_id is None or cont_token_id is None:
        raise ValueError("Pretokenized bundle save requires pad/eos/cont special token ids in corpus_info.")
    token_dtype = torch.int16 if vocab_size <= int(torch.iinfo(torch.int16).max) else torch.int32
    chunk_offsets: list[int] = [0]
    document_offsets: list[int] = [0]
    flat_context_parts: list[torch.Tensor] = []
    flat_loss_mask_parts: list[torch.Tensor] = []
    chunk_is_continuation: list[bool] = []
    document_kinds: list[str] = []
    document_sources: list[str] = []
    document_token_counts: list[int] = []
    binary_loss_mask = True

    for document in documents:
        document_kinds.append(document.kind)
        document_sources.append(document.source)
        document_token_counts.append(int(document.token_count))
        for chunk in document.chunks:
            context = chunk.context.detach().cpu().to(dtype=torch.long)
            loss_mask = chunk.loss_mask.detach().cpu().to(dtype=torch.float32)
            nonpad = torch.nonzero(context != int(pad_token_id), as_tuple=False)
            active_length = int(nonpad[-1].item() + 1) if nonpad.numel() > 0 else 0
            active_context = context[:active_length].to(dtype=token_dtype)
            active_loss_mask = loss_mask[:active_length]
            if binary_loss_mask and not torch.all((active_loss_mask == 0.0) | (active_loss_mask == 1.0)).item():
                binary_loss_mask = False
            flat_context_parts.append(active_context)
            flat_loss_mask_parts.append(active_loss_mask)
            chunk_is_continuation.append(bool(chunk.is_continuation))
            chunk_offsets.append(chunk_offsets[-1] + active_length)
        document_offsets.append(document_offsets[-1] + len(document.chunks))

    flat_context = (
        torch.cat(flat_context_parts, dim=0)
        if flat_context_parts
        else torch.empty(0, dtype=token_dtype)
    )
    seq_len = int(documents[0].chunks[0].context.shape[0]) if documents and documents[0].chunks else 0
    if binary_loss_mask:
        flat_loss_mask = (
            torch.cat([part.to(dtype=torch.uint8) for part in flat_loss_mask_parts], dim=0)
            if flat_loss_mask_parts
            else torch.empty(0, dtype=torch.uint8)
        )
    else:
        flat_loss_mask = (
            torch.cat([part.to(dtype=torch.float16) for part in flat_loss_mask_parts], dim=0)
            if flat_loss_mask_parts
            else torch.empty(0, dtype=torch.float16)
        )

    payload = {
        "storage_format": "flat_v2",
        "documents": {
            "kind": document_kinds,
            "source": document_sources,
            "token_count": torch.tensor(document_token_counts, dtype=torch.int32),
            "chunk_offsets": torch.tensor(document_offsets, dtype=torch.int64),
        },
        "chunks": {
            "token_offsets": torch.tensor(chunk_offsets, dtype=torch.int64),
            "context_flat": flat_context,
            "loss_mask_flat": flat_loss_mask,
            "is_continuation": torch.tensor(chunk_is_continuation, dtype=torch.bool),
        },
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tokenizer_label": tokenizer_label,
        "tokenizer_model_path": tokenizer_model_path,
        "corpus_info": corpus_info,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid4().hex}")
    try:
        with temp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


def _validate_flat_bundle_payload(bundle: dict[str, Any], *, path: Path) -> None:
    if bundle.get("storage_format") != "flat_v2":
        raise ValueError(f"Expected flat_v2 pretokenized shard: {path}")
    if not isinstance(bundle.get("documents"), dict) or not isinstance(bundle.get("chunks"), dict):
        raise ValueError(f"Invalid flat pretokenized shard structure: {path}")
    corpus_info = bundle.get("corpus_info") or {}
    special_tokens = dict(corpus_info.get("special_tokens") or {})
    for key in ("pad", "eos", "cont"):
        if special_tokens.get(key) is None:
            raise ValueError(f"Flat pretokenized bundle is missing special token '{key}': {path}")
    seq_len = int(bundle.get("seq_len") or 0)
    if seq_len <= 0:
        raise ValueError(f"Flat pretokenized bundle is missing seq_len: {path}")
    document_meta = bundle["documents"]
    chunk_meta = bundle["chunks"]
    for key in ("kind", "source", "token_count", "chunk_offsets"):
        if key not in document_meta:
            raise ValueError(f"Flat pretokenized bundle is missing documents['{key}']: {path}")
    for key in ("token_offsets", "context_flat", "loss_mask_flat", "is_continuation"):
        if key not in chunk_meta:
            raise ValueError(f"Flat pretokenized bundle is missing chunks['{key}']: {path}")


def validate_flat_pretokenized_shard(path: Path) -> None:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid pretokenized shard payload: {path}")
    _validate_flat_bundle_payload(bundle, path=path)


def load_pretokenized_bundle(path: Path) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict) or "documents" not in bundle:
        raise ValueError(f"Invalid pretokenized bundle: {path}")
    storage_format = bundle.get("storage_format")
    if storage_format == "flat_v2":
        corpus_info = bundle.get("corpus_info") or {}
        special_tokens = dict(corpus_info.get("special_tokens") or {})
        pad_token_id = special_tokens.get("pad")
        eos_token_id = special_tokens.get("eos")
        cont_token_id = special_tokens.get("cont")
        if pad_token_id is None or eos_token_id is None or cont_token_id is None:
            raise ValueError(f"Flat pretokenized bundle is missing pad/eos/cont special token ids: {path}")
        document_meta = bundle["documents"]
        chunk_meta = bundle["chunks"]
        document_kinds = list(document_meta["kind"])
        document_sources = list(document_meta["source"])
        document_token_counts = document_meta["token_count"].detach().cpu().to(dtype=torch.int64)
        document_chunk_offsets = document_meta["chunk_offsets"].detach().cpu().to(dtype=torch.int64)
        chunk_token_offsets = chunk_meta["token_offsets"].detach().cpu().to(dtype=torch.int64)
        flat_context = chunk_meta["context_flat"].detach().cpu().to(dtype=torch.long)
        loss_mask_flat = chunk_meta["loss_mask_flat"].detach().cpu()
        chunk_is_continuation = chunk_meta["is_continuation"].detach().cpu().to(dtype=torch.bool)
        seq_len = int(bundle.get("seq_len") or 0)
        if seq_len <= 0:
            raise ValueError(f"Flat pretokenized bundle is missing seq_len: {path}")
        documents: list[TokenizedDocument] = []
        for document_index, kind in enumerate(document_kinds):
            chunk_start = int(document_chunk_offsets[document_index].item())
            chunk_end = int(document_chunk_offsets[document_index + 1].item())
            restored_chunks: list[DocumentChunk] = []
            for chunk_index in range(chunk_start, chunk_end):
                token_start = int(chunk_token_offsets[chunk_index].item())
                token_end = int(chunk_token_offsets[chunk_index + 1].item())
                active_context = flat_context[token_start:token_end]
                active_length = int(active_context.shape[0])
                context = torch.full((seq_len,), int(pad_token_id), dtype=torch.long)
                if active_length > 0:
                    context[:active_length] = active_context
                target = torch.full((seq_len,), int(pad_token_id), dtype=torch.long)
                if active_length > 1:
                    target[: active_length - 1] = active_context[1:]
                if active_length > 0:
                    terminal_token = int(eos_token_id) if chunk_index == chunk_end - 1 else int(cont_token_id)
                    target[active_length - 1] = terminal_token
                loss_mask = torch.zeros(seq_len, dtype=torch.float32)
                if active_length > 0:
                    active_loss_mask = loss_mask_flat[token_start:token_end].to(dtype=torch.float32)
                    loss_mask[:active_length] = active_loss_mask
                restored_chunks.append(
                    DocumentChunk(
                        context=context,
                        target=target,
                        loss_mask=loss_mask,
                        is_continuation=bool(chunk_is_continuation[chunk_index].item()),
                    )
                )
            documents.append(
                TokenizedDocument(
                    kind=kind,
                    source=document_sources[document_index],
                    token_count=int(document_token_counts[document_index].item()),
                    chunks=tuple(restored_chunks),
                )
            )
    else:
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


def load_flat_pretokenized_shard(path: Path) -> FlatPretokenizedShard:
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid pretokenized shard payload: {path}")
    _validate_flat_bundle_payload(bundle, path=path)
    corpus_info = bundle.get("corpus_info") or {}
    special_tokens = dict(corpus_info.get("special_tokens") or {})
    pad_token_id = special_tokens.get("pad")
    eos_token_id = special_tokens.get("eos")
    cont_token_id = special_tokens.get("cont")
    if pad_token_id is None or eos_token_id is None or cont_token_id is None:
        raise ValueError(f"Flat pretokenized bundle is missing pad/eos/cont special token ids: {path}")
    document_meta = bundle["documents"]
    chunk_meta = bundle["chunks"]
    seq_len = int(bundle.get("seq_len") or 0)
    if seq_len <= 0:
        raise ValueError(f"Flat pretokenized bundle is missing seq_len: {path}")
    return FlatPretokenizedShard(
        path=path,
        kind=tuple(str(item) for item in document_meta["kind"]),
        source=tuple(str(item) for item in document_meta["source"]),
        token_count=document_meta["token_count"].detach().cpu().to(dtype=torch.int64),
        document_chunk_offsets=document_meta["chunk_offsets"].detach().cpu().to(dtype=torch.int64),
        chunk_token_offsets=chunk_meta["token_offsets"].detach().cpu().to(dtype=torch.int64),
        context_flat=chunk_meta["context_flat"].detach().cpu().to(dtype=torch.long),
        loss_mask_flat=chunk_meta["loss_mask_flat"].detach().cpu(),
        is_continuation=chunk_meta["is_continuation"].detach().cpu().to(dtype=torch.bool),
        seq_len=seq_len,
        pad_token_id=int(pad_token_id),
        eos_token_id=int(eos_token_id),
        cont_token_id=int(cont_token_id),
        special_token_ids={key: int(value) for key, value in special_tokens.items()},
        vocab_size=int(bundle["vocab_size"]),
        tokenizer_label=bundle.get("tokenizer_label"),
        tokenizer_model_path=bundle.get("tokenizer_model_path"),
        corpus_info=corpus_info,
    )


def load_flat_pretokenized_directory(path: Path, *, load_workers: int = 1) -> FlatPretokenizedDirectory:
    shard_paths = sorted(candidate for candidate in path.glob("*.pt") if candidate.is_file())
    if not shard_paths:
        raise ValueError(f"No pretokenized shard files found in {path}")
    validation_workers = min(max(1, load_workers), len(shard_paths))
    print(
        f"flat_integrity_check | path={path} | shards={len(shard_paths)} | workers={validation_workers}",
        flush=True,
    )
    if validation_workers <= 1:
        for shard_path in shard_paths:
            try:
                validate_flat_pretokenized_shard(shard_path)
            except Exception as exc:
                raise RuntimeError(f"Corrupt pretokenized shard: {shard_path}") from exc
    else:
        with ThreadPoolExecutor(
            max_workers=validation_workers,
            thread_name_prefix="pretok_flat_validate",
        ) as validation_executor:
            validation_futures = {
                validation_executor.submit(validate_flat_pretokenized_shard, shard_path): shard_path
                for shard_path in shard_paths
            }
            for future, shard_path in validation_futures.items():
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Corrupt pretokenized shard: {shard_path}") from exc
    print(f"flat_integrity_ok | path={path} | shards={len(shard_paths)}", flush=True)
    if load_workers <= 1:
        shard_iterator = (load_flat_pretokenized_shard(shard_path) for shard_path in shard_paths)
        executor: ThreadPoolExecutor | None = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=min(max(1, load_workers), len(shard_paths)),
            thread_name_prefix="pretok_flat_load",
        )
        shard_iterator = executor.map(load_flat_pretokenized_shard, shard_paths)
    shards: list[FlatPretokenizedShard] = []
    tokenizer_label: str | None = None
    tokenizer_model_path: str | None = None
    vocab_size: int | None = None
    shard_summaries: list[dict[str, Any]] = []
    try:
        for shard_path, shard in zip(shard_paths, shard_iterator):
            shards.append(shard)
            if tokenizer_label is None:
                tokenizer_label = shard.tokenizer_label
                tokenizer_model_path = shard.tokenizer_model_path
                vocab_size = int(shard.vocab_size)
            shard_summaries.append(
                {
                    "path": str(shard_path),
                    "documents": shard.num_documents,
                    "corpus_info": shard.corpus_info,
                }
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if vocab_size is None:
        raise ValueError(f"Invalid pretokenized shard directory: {path}")
    refs: list[FlatDocumentRef] = []
    for shard_index, shard in enumerate(shards):
        refs.extend(FlatDocumentRef(shard_index=shard_index, document_index=i) for i in range(shard.num_documents))
    return FlatPretokenizedDirectory(
        shards=tuple(shards),
        document_refs=tuple(refs),
        vocab_size=vocab_size,
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
        corpus_info={"shards": shard_summaries, "directory": str(path)},
    )


def load_pretokenized_directory(path: Path, *, load_workers: int = 1) -> dict[str, Any]:
    shard_paths = sorted(candidate for candidate in path.glob("*.pt") if candidate.is_file())
    if not shard_paths:
        raise ValueError(f"No pretokenized shard files found in {path}")
    merged_documents: list[TokenizedDocument] = []
    tokenizer_label: str | None = None
    tokenizer_model_path: str | None = None
    vocab_size: int | None = None
    shard_summaries: list[dict[str, Any]] = []
    if load_workers <= 1:
        bundle_iterator = (load_pretokenized_bundle(shard_path) for shard_path in shard_paths)
        executor: ThreadPoolExecutor | None = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=min(max(1, load_workers), len(shard_paths)),
            thread_name_prefix="pretok_load",
        )
        bundle_iterator = executor.map(load_pretokenized_bundle, shard_paths)
    try:
        for shard_path, bundle in zip(shard_paths, bundle_iterator):
            merged_documents.extend(bundle["documents"])
            if tokenizer_label is None:
                tokenizer_label = bundle.get("tokenizer_label")
                tokenizer_model_path = bundle.get("tokenizer_model_path")
                vocab_size = int(bundle["vocab_size"])
            shard_summaries.append(
                {
                    "path": str(shard_path),
                    "documents": len(bundle["documents"]),
                    "corpus_info": bundle.get("corpus_info") or {},
                }
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    if vocab_size is None:
        raise ValueError(f"Invalid pretokenized shard directory: {path}")
    return {
        "documents": merged_documents,
        "vocab_size": vocab_size,
        "tokenizer_label": tokenizer_label,
        "tokenizer_model_path": tokenizer_model_path,
        "corpus_info": {"shards": shard_summaries, "directory": str(path)},
    }


def split_train_val_documents(
    documents: Sequence[TokenizedDocument],
    *,
    train_fraction: float,
) -> tuple[list[TokenizedDocument], list[TokenizedDocument]]:
    if len(documents) < 2:
        return list(documents), list(documents)
    rng = random.Random(1337)
    bucket_to_documents: dict[str, list[TokenizedDocument]] = {}
    for document in documents:
        bucket_to_documents.setdefault(document_sampling_bucket(document), []).append(document)
    train_documents: list[TokenizedDocument] = []
    val_documents: list[TokenizedDocument] = []
    for bucket in SCHEDULED_BUCKET_ORDER:
        bucket_docs = bucket_to_documents.get(bucket)
        if not bucket_docs:
            continue
        bucket_docs = list(bucket_docs)
        rng.shuffle(bucket_docs)
        if len(bucket_docs) < 2:
            train_documents.extend(bucket_docs)
            continue
        split_index = int(len(bucket_docs) * train_fraction)
        split_index = max(1, min(split_index, len(bucket_docs) - 1))
        train_documents.extend(bucket_docs[:split_index])
        val_documents.extend(bucket_docs[split_index:])
    rng.shuffle(train_documents)
    rng.shuffle(val_documents)
    if not val_documents:
        val_documents = train_documents[-1:]
        train_documents = train_documents[:-1] or val_documents
    return train_documents, val_documents


def split_train_val_flat_documents(
    documents: Sequence[FlatDocumentRef],
    *,
    train_fraction: float,
) -> tuple[list[FlatDocumentRef], list[FlatDocumentRef]]:
    if len(documents) < 2:
        return list(documents), list(documents)
    raise RuntimeError("split_train_val_flat_documents requires flat collection-aware splitting.")


def split_train_val_flat_documents_with_collection(
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    *,
    train_fraction: float,
) -> tuple[list[FlatDocumentRef], list[FlatDocumentRef]]:
    if len(documents) < 2:
        return list(documents), list(documents)
    rng = random.Random(1337)
    bucket_to_documents: dict[str, list[FlatDocumentRef]] = {}
    for reference in documents:
        bucket_to_documents.setdefault(flat_document_ref_bucket(collection, reference), []).append(reference)
    train_documents: list[FlatDocumentRef] = []
    val_documents: list[FlatDocumentRef] = []
    for bucket in SCHEDULED_BUCKET_ORDER:
        bucket_docs = bucket_to_documents.get(bucket)
        if not bucket_docs:
            continue
        bucket_docs = list(bucket_docs)
        rng.shuffle(bucket_docs)
        if len(bucket_docs) < 2:
            train_documents.extend(bucket_docs)
            continue
        split_index = int(len(bucket_docs) * train_fraction)
        split_index = max(1, min(split_index, len(bucket_docs) - 1))
        train_documents.extend(bucket_docs[:split_index])
        val_documents.extend(bucket_docs[split_index:])
    rng.shuffle(train_documents)
    rng.shuffle(val_documents)
    if not val_documents:
        val_documents = train_documents[-1:]
        train_documents = train_documents[:-1] or val_documents
    return train_documents, val_documents


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
    stage1_batch_size: int,
    stage2_batch_size: int,
    stage3_batch_size: int,
    stage1_grad_accum_steps: int,
    stage2_grad_accum_steps: int,
    stage3_grad_accum_steps: int,
) -> TrainingCurriculumStage:
    stage1_end = int(total_steps * stage1_ratio)
    stage2_end = int(total_steps * stage2_ratio)
    if stage1_end > 0 and step <= stage1_end:
        return TrainingCurriculumStage(
            name="stage1",
            document_span=1,
            batch_size=max(1, stage1_batch_size),
            grad_accum_steps=max(1, stage1_grad_accum_steps),
            freeze_memory=True,
            freeze_propagation=True,
            freeze_skip=True,
            bucket_weights=resolve_stage_bucket_weights(stage_name="stage1"),
        )
    if stage2_end > 0 and step <= stage2_end:
        return TrainingCurriculumStage(
            name="stage2",
            document_span=max(1, stage2_span),
            batch_size=max(1, stage2_batch_size),
            grad_accum_steps=max(1, stage2_grad_accum_steps),
            freeze_memory=False,
            freeze_propagation=True,
            freeze_skip=True,
            bucket_weights=resolve_stage_bucket_weights(stage_name="stage2"),
        )
    return TrainingCurriculumStage(
        name="stage3",
        document_span=max(1, stage3_span),
        batch_size=max(1, stage3_batch_size),
        grad_accum_steps=max(1, stage3_grad_accum_steps),
        freeze_memory=False,
        freeze_propagation=False,
        freeze_skip=False,
        bucket_weights=resolve_stage_bucket_weights(stage_name="stage3"),
    )


def apply_training_curriculum(model: CausalHierarchicalMemoryLM, stage: TrainingCurriculumStage) -> None:
    b_module = model.b_module
    _set_module_requires_grad(b_module.memory_levels, not stage.freeze_memory)
    _set_module_requires_grad(b_module.level_transitions, not stage.freeze_memory)
    _set_module_requires_grad(b_module.level_norms, not stage.freeze_memory)
    _set_module_requires_grad(b_module.read_projections, not stage.freeze_memory)
    _set_parameter_collection_requires_grad(b_module.read_gates, not stage.freeze_memory)
    b_module.read_template_val.requires_grad_(not stage.freeze_memory)

    for level in b_module.memory_levels:
        _set_module_requires_grad(level.propagation, not stage.freeze_propagation)

    _set_module_requires_grad(b_module.skip_transitions, not stage.freeze_skip)
    _set_parameter_collection_requires_grad(tuple(b_module.skip_gates.values()), not stage.freeze_skip)


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


def estimate_stage_weighted_steps_per_epoch(
    *,
    documents: Sequence[TokenizedDocument],
    stage1_ratio: float,
    stage2_ratio: float,
    stage1_batch_size: int,
    stage2_batch_size: int,
    stage3_batch_size: int,
) -> int:
    total_chunks = sum(len(document.chunks) for document in documents)
    stage1_weight = max(0.0, stage1_ratio)
    stage2_weight = max(0.0, stage2_ratio - stage1_ratio)
    stage3_weight = max(0.0, 1.0 - stage2_ratio)
    effective_examples_per_step = (
        stage1_weight * max(1, stage1_batch_size)
        + stage2_weight * max(1, stage2_batch_size)
        + stage3_weight * max(1, stage3_batch_size)
    )
    return max(1, math.ceil(total_chunks / max(1.0, effective_examples_per_step)))


def estimate_stage_weighted_steps_per_epoch_flat(
    *,
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    stage1_ratio: float,
    stage2_ratio: float,
    stage1_batch_size: int,
    stage2_batch_size: int,
    stage3_batch_size: int,
) -> int:
    total_chunks = 0
    for reference in documents:
        total_chunks += collection.shards[reference.shard_index].chunk_count(reference.document_index)
    stage1_weight = max(0.0, stage1_ratio)
    stage2_weight = max(0.0, stage2_ratio - stage1_ratio)
    stage3_weight = max(0.0, 1.0 - stage2_ratio)
    effective_examples_per_step = (
        stage1_weight * max(1, stage1_batch_size)
        + stage2_weight * max(1, stage2_batch_size)
        + stage3_weight * max(1, stage3_batch_size)
    )
    return max(1, math.ceil(total_chunks / max(1.0, effective_examples_per_step)))


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


def detach_memory_state(memory_state: Any | None) -> Any | None:
    if memory_state is None:
        return None
    if isinstance(memory_state, ModelRecurrentState):
        detached_knowledge = None
        if memory_state.knowledge_state is not None:
            detached_knowledge = memory_state.knowledge_state.with_tensors(
                state=memory_state.knowledge_state.state.detach(),
                val=memory_state.knowledge_state.val.detach(),
            )
        return ModelRecurrentState(
            memory_state=tuple(
                layer.with_tensors(state=layer.state.detach(), val=layer.val.detach())
                for layer in memory_state.memory_state
            ),
            knowledge_state=detached_knowledge,
        )
    return tuple(
        layer.with_tensors(state=layer.state.detach(), val=layer.val.detach())
        for layer in memory_state
    )


def memory_state_is_finite(memory_state: Any | None) -> bool:
    if memory_state is None:
        return True
    if isinstance(memory_state, ModelRecurrentState):
        memory_layers: list[Any] = list(memory_state.memory_state)
        if memory_state.knowledge_state is not None:
            memory_layers.append(memory_state.knowledge_state)
        return all(
            bool(torch.isfinite(layer.state).all().item()) and bool(torch.isfinite(layer.val).all().item())
            for layer in memory_layers
        )
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
        mask = loss_mask.detach().cpu() > 0
        ids = ids[mask]
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
    selected = sample_documents_uniform_by_bucket(documents, sample_count=max_samples)
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
                f"### context\n{_decode_visible_tokens(vocab, chunk.context, None)}\n\n"
                f"### target\n{_decode_visible_tokens(vocab, chunk.target, chunk.loss_mask)}\n\n"
                f"### prediction\n{_decode_visible_tokens(vocab, predicted, chunk.loss_mask)}"
            )
        writer.add_text("eval_samples/mixed_corpus", "\n\n---\n\n".join(sample_sections), step)
    if model_was_training:
        model.train()


@torch.no_grad()
def log_eval_samples_to_tensorboard_flat(
    writer: SummaryWriter | None,
    model: CausalHierarchicalMemoryLM,
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    *,
    vocab: object | None,
    device: torch.device,
    precision: str,
    step: int,
    max_samples: int = 6,
) -> None:
    if writer is None or vocab is None or not documents:
        return
    selected = sample_flat_documents_uniform_by_bucket(collection, documents, sample_count=max_samples)
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
        for index, reference in enumerate(selected, start=1):
            shard = collection.shards[reference.shard_index]
            chunk_start, _ = shard.document_chunk_range(reference.document_index)
            context, target, loss_mask, _ = shard.build_chunk_tensors(
                chunk_start,
                document_index=reference.document_index,
                is_last_chunk=shard.chunk_count(reference.document_index) == 1,
            )
            output = model(
                context.unsqueeze(0).to(device),
                reset_mask=torch.tensor([True], device=device, dtype=torch.bool),
                return_memory_state=False,
            )
            assert isinstance(output, torch.Tensor)
            predicted = output.argmax(dim=-1).squeeze(0).detach().cpu().to(dtype=torch.long)
            sample_sections.append(
                f"## sample {index:02d}\n\n"
                f"kind: {shard.kind[reference.document_index]}\n\n"
                f"source: {shard.source[reference.document_index]}\n\n"
                f"### context\n{_decode_visible_tokens(vocab, context, None)}\n\n"
                f"### target\n{_decode_visible_tokens(vocab, target, loss_mask)}\n\n"
                f"### prediction\n{_decode_visible_tokens(vocab, predicted, loss_mask)}"
            )
        writer.add_text("eval_samples/mixed_corpus", "\n\n---\n\n".join(sample_sections), step)
    if model_was_training:
        model.train()


def run_model(
    model: CausalHierarchicalMemoryLM,
    batch: DocumentBatch,
    *,
    memory_state: Any | None,
    precision: str,
    grad_enabled: bool,
) -> tuple[torch.Tensor, Any | None]:
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
    return loss, output.recurrent_state


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
    sampled_documents = sample_documents_uniform_by_bucket(documents, sample_count=min(eval_documents, len(documents)))
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


@torch.no_grad()
def estimate_eval_loss_flat(
    model: CausalHierarchicalMemoryLM,
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    *,
    eval_documents: int,
    device: torch.device,
    precision: str,
) -> float:
    if not documents:
        raise ValueError("documents must not be empty.")
    model_was_training = model.training
    model.eval()
    sampled_documents = sample_flat_documents_uniform_by_bucket(
        collection,
        documents,
        sample_count=min(eval_documents, len(documents)),
    )
    total_loss = 0.0
    total_weight = 0.0
    for reference in sampled_documents:
        shard = collection.shards[reference.shard_index]
        chunk_start, chunk_end = shard.document_chunk_range(reference.document_index)
        memory_state: tuple[Any, ...] | None = None
        for chunk_index in range(chunk_start, chunk_end):
            context, target, loss_mask, _ = shard.build_chunk_tensors(
                chunk_index,
                document_index=reference.document_index,
                is_last_chunk=chunk_index == chunk_end - 1,
            )
            batch = DocumentBatch(
                context=context.unsqueeze(0).to(device),
                target=target.unsqueeze(0).to(device),
                loss_mask=loss_mask.unsqueeze(0).to(device),
                reset_mask=torch.tensor([chunk_index == chunk_start], device=device, dtype=torch.bool),
            )
            loss, memory_state = run_model(
                model,
                batch,
                memory_state=memory_state,
                precision=precision,
                grad_enabled=False,
            )
            weight = float(batch.loss_mask.sum().item())
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
    parser.add_argument("--pretokenized-dir")
    parser.add_argument("--pretokenized-load-workers", type=int, default=8)
    parser.add_argument("--prefetch-batches", type=int, default=100)
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
    parser.add_argument("--knowledge-nodes", type=int, default=0)
    parser.add_argument("--knowledge-route-topk", type=int, default=0)
    parser.add_argument("--knowledge-propagation-topk", type=int, default=0)
    parser.add_argument("--knowledge-propagation-layers", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--stage1-batch-size", type=int, default=0)
    parser.add_argument("--stage2-batch-size", type=int, default=0)
    parser.add_argument("--stage3-batch-size", type=int, default=0)
    parser.add_argument("--stage1-grad-accum-steps", type=int, default=0)
    parser.add_argument("--stage2-grad-accum-steps", type=int, default=0)
    parser.add_argument("--stage3-grad-accum-steps", type=int, default=0)
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


def summarize_flat_documents(
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
) -> dict[str, int]:
    chunk_total = 0
    token_total = 0
    for reference in documents:
        shard = collection.shards[reference.shard_index]
        chunk_total += shard.chunk_count(reference.document_index)
        token_total += int(shard.token_count[reference.document_index].item())
    return {
        "documents": len(documents),
        "chunks": chunk_total,
        "tokens": token_total,
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
    if any(
        value < 0
        for value in (
            args.stage1_batch_size,
            args.stage2_batch_size,
            args.stage3_batch_size,
            args.stage1_grad_accum_steps,
            args.stage2_grad_accum_steps,
            args.stage3_grad_accum_steps,
        )
    ):
        raise ValueError("Stage batch sizes and grad accum steps must be non-negative.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"using device: {describe_device(device)}", flush=True)

    stage1_batch_size = args.stage1_batch_size or args.batch_size
    stage2_batch_size = args.stage2_batch_size or args.batch_size
    stage3_batch_size = args.stage3_batch_size or args.batch_size
    stage1_grad_accum_steps = args.stage1_grad_accum_steps or args.grad_accum_steps
    stage2_grad_accum_steps = args.stage2_grad_accum_steps or args.grad_accum_steps
    stage3_grad_accum_steps = args.stage3_grad_accum_steps or args.grad_accum_steps

    tokenizer_label: str
    tokenizer_model_path: str | None
    vocab_size: int
    corpus_metadata: dict[str, Any]
    documents: list[TokenizedDocument] | None = None
    flat_collection: FlatPretokenizedDirectory | None = None

    if args.pretokenized_dir and Path(args.pretokenized_dir).exists():
        flat_collection = load_flat_pretokenized_directory(
            Path(args.pretokenized_dir),
            load_workers=max(1, int(args.pretokenized_load_workers)),
        )
        tokenizer_label = str(flat_collection.tokenizer_label or "unknown")
        tokenizer_model_path = flat_collection.tokenizer_model_path
        vocab_size = int(flat_collection.vocab_size)
        corpus_metadata = dict(flat_collection.corpus_info or {})
        print(
            f"loaded_pretokenized_dir | path={args.pretokenized_dir} | documents={len(flat_collection.document_refs):,} | tokenizer={tokenizer_label}",
            flush=True,
        )
    elif args.pretokenized_path and Path(args.pretokenized_path).exists():
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
        tokenizer_cache_ready = (
            args.tokenizer == "byte_bpe"
            and byte_bpe_tokenizer_cache_exists(
                tokenizer_prefix=args.tokenizer_prefix,
                vocab_size=args.subword_vocab_size,
            )
        )
        if tokenizer_cache_ready:
            training_text = ""
            print(
                f"tokenizer_cache_hit | prefix={args.tokenizer_prefix}",
                flush=True,
            )
        else:
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

    if flat_collection is not None:
        train_documents_flat, val_documents_flat = split_train_val_flat_documents_with_collection(
            flat_collection,
            flat_collection.document_refs,
            train_fraction=args.train_fraction,
        )
        train_bucket_summary = summarize_flat_document_buckets(flat_collection, train_documents_flat)
        val_bucket_summary = summarize_flat_document_buckets(flat_collection, val_documents_flat)
        steps_per_epoch = estimate_stage_weighted_steps_per_epoch_flat(
            collection=flat_collection,
            documents=train_documents_flat,
            stage1_ratio=args.curriculum_stage1_ratio,
            stage2_ratio=args.curriculum_stage2_ratio,
            stage1_batch_size=stage1_batch_size,
            stage2_batch_size=stage2_batch_size,
            stage3_batch_size=stage3_batch_size,
        )
    else:
        assert documents is not None
        train_documents, val_documents = split_train_val_documents(documents, train_fraction=args.train_fraction)
        train_bucket_summary = summarize_document_buckets(train_documents)
        val_bucket_summary = summarize_document_buckets(val_documents)
        steps_per_epoch = estimate_stage_weighted_steps_per_epoch(
            documents=train_documents,
            stage1_ratio=args.curriculum_stage1_ratio,
            stage2_ratio=args.curriculum_stage2_ratio,
            stage1_batch_size=stage1_batch_size,
            stage2_batch_size=stage2_batch_size,
            stage3_batch_size=stage3_batch_size,
        )
    decode_vocab = load_decode_vocab(
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
    )
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
        knowledge_nodes=args.knowledge_nodes,
        knowledge_route_topk=None if args.knowledge_route_topk <= 0 else args.knowledge_route_topk,
        knowledge_propagation_topk=None if args.knowledge_propagation_topk <= 0 else args.knowledge_propagation_topk,
        knowledge_propagation_layers=args.knowledge_propagation_layers,
    ).to(device)
    parameter_count = count_parameters(model)
    print(
        f"model=causal_memory_doc | params={parameter_count:,} | dim={args.dim} | seq_len={args.seq_len} | "
        f"s_window={args.s_window} | s_microbatch_size={args.s_microbatch_size} | "
        f"scan_backend={args.scan_backend} | scan_checkpoint_chunk_size={args.scan_checkpoint_chunk_size} | "
        f"memory_slots={args.memory_slots} | knowledge_nodes={args.knowledge_nodes}",
        flush=True,
    )
    print(
        f"curriculum_batches | stage1=batch{stage1_batch_size}/ga{stage1_grad_accum_steps} | "
        f"stage2=batch{stage2_batch_size}/ga{stage2_grad_accum_steps} | "
        f"stage3=batch{stage3_batch_size}/ga{stage3_grad_accum_steps}",
        flush=True,
    )
    print(
        f"bucket_summary | train={train_bucket_summary} | val={val_bucket_summary}",
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
    if flat_collection is not None:
        batcher: DocumentChunkBatcher | FlatDocumentChunkBatcher = FlatDocumentChunkBatcher(
            flat_collection,
            train_documents_flat,
            batch_size=stage1_batch_size,
            device=device,
        )
    else:
        batcher = DocumentChunkBatcher(train_documents, batch_size=stage1_batch_size, device=device)
    prefetcher = AsyncDocumentBatchPrefetcher(
        batcher,
        device=device,
        prefetch_batches=max(1, int(args.prefetch_batches)),
    )

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
            stage1_batch_size=stage1_batch_size,
            stage2_batch_size=stage2_batch_size,
            stage3_batch_size=stage3_batch_size,
            stage1_grad_accum_steps=stage1_grad_accum_steps,
            stage2_grad_accum_steps=stage2_grad_accum_steps,
            stage3_grad_accum_steps=stage3_grad_accum_steps,
        )
        bucket_weights = stage.bucket_weights
        batcher.set_batch_size(stage.batch_size)
        batcher.set_bucket_weights(bucket_weights)
        if stage.name != active_stage_name:
            prefetcher.close()
            prefetcher = AsyncDocumentBatchPrefetcher(
                batcher,
                device=device,
                prefetch_batches=max(1, int(args.prefetch_batches)),
            )
            apply_training_curriculum(model, stage)
            train_memory_state = None
            active_stage_name = stage.name
            print(
                f"curriculum | step={step} | stage={stage.name} | span={stage.document_span} | "
                f"batch_size={stage.batch_size} | grad_accum_steps={stage.grad_accum_steps} | "
                f"freeze_memory={stage.freeze_memory} | freeze_propagation={stage.freeze_propagation} | freeze_skip={stage.freeze_skip} | "
                f"bucket_weights={{{', '.join(f'{key}:{value:.3f}' for key, value in bucket_weights.items() if value > 0.0)}}}",
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
        span_loss_tensors: list[torch.Tensor] = []
        loss_is_finite = True
        for span_index in range(stage.document_span):
            batch = override_batch_reset(
                prefetcher.next_batch(),
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
            span_loss_tensors.append(loss)
            current_memory_state = None if stage.freeze_memory else next_memory_state

        if loss_is_finite and span_loss_tensors:
            total_span_loss = sum(span_loss_tensors) / (stage.grad_accum_steps * len(span_loss_tensors))
            if scaler is not None:
                scaler.scale(total_span_loss).backward()
            else:
                total_span_loss.backward()
            del total_span_loss
            train_memory_state = None if stage.freeze_memory else detach_memory_state(current_memory_state)
            if not memory_state_is_finite(train_memory_state):
                print(f"warning | step={step} | non-finite memory state; resetting memory", flush=True)
                train_memory_state = None
        else:
            optimizer.zero_grad(set_to_none=True)
            train_memory_state = None

        should_step_optimizer = step % stage.grad_accum_steps == 0 or step == total_steps
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
        avg_cpu_batch_seconds, prefetch_queue_size = prefetcher.stats()
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/document_span", stage.document_span, step)
            writer.add_scalar("train/cpu_batch_ms", avg_cpu_batch_seconds * 1000.0, step)
            writer.add_scalar("train/prefetch_queue_size", prefetch_queue_size, step)
            if not math.isnan(grad_norm):
                writer.add_scalar("train/grad_norm", grad_norm, step)
            for bucket_name, bucket_weight in bucket_weights.items():
                writer.add_scalar(f"train_bucket_weight/{bucket_name}", bucket_weight, step)

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"progress | step={step:5d}/{total_steps} | stage={stage.name} | span={stage.document_span} | "
                f"train_loss={train_loss:.4f} | lr={lr:.6g} | "
                f"cpu_batch_ms={avg_cpu_batch_seconds * 1000.0:.1f} | prefetch_q={prefetch_queue_size} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        val_loss = None
        if step == 1 or step % args.eval_interval == 0 or step == total_steps:
            if flat_collection is not None:
                val_loss = estimate_eval_loss_flat(
                    model,
                    flat_collection,
                    val_documents_flat,
                    eval_documents=args.eval_documents,
                    device=device,
                    precision=args.precision,
                )
            else:
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
                if flat_collection is not None:
                    log_eval_samples_to_tensorboard_flat(
                        writer,
                        model,
                        flat_collection,
                        val_documents_flat,
                        vocab=decode_vocab,
                        device=device,
                        precision=args.precision,
                        step=step,
                    )
                else:
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
    prefetcher.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
