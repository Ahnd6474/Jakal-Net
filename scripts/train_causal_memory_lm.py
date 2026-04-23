from __future__ import annotations

import argparse
from collections import OrderedDict
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
from dataclasses import dataclass, field
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
    HFTokenizerVocab,
    ByteBPEVocab,
    ByteLevelBPETokenizer,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    _encode_byte_bpe_preserving_specials,
    _encode_byte_bpe_text_batch_worker,
    _encode_byte_bpe_text_worker,
    _encode_hf_text_batch_worker,
    _chat_stream_from_record,
    _init_byte_bpe_encode_worker,
    _init_hf_encode_worker,
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

try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoConfig = None
    AutoModel = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from jakal_net import describe_device, resolve_device
from jakal_net.causal_memory_lm import CausalHierarchicalMemoryLM, MemoryScanOutput, ModelRecurrentState
from jakal_net.native_backend import (
    EXPERIMENTAL_FUSED_TRAINING_CHECKPOINT_STRIDE_ENV,
    EXPERIMENTAL_FUSED_TRAINING_ENV,
    EXPERIMENTAL_SCAN_BACKWARD_CUDA_ENV,
)

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
class OptimizerParameterGroupConfig:
    name: str
    lr_scale: float
    weight_decay: float
    parameter_count: int


class SharedEmbeddingRnnAuxHead(torch.nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int,
        vocab_size: int,
        hidden_dim: int | None = None,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        hidden_size = int(hidden_dim or embedding_dim)
        if hidden_size <= 0:
            raise ValueError("hidden_dim must be positive.")
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = hidden_size
        self.num_layers = int(num_layers)
        self.rnn = torch.nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(self.hidden_dim)
        self.output = torch.nn.Linear(self.hidden_dim, int(vocab_size), bias=False)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for name, parameter in self.rnn.named_parameters():
            if "weight" in name:
                torch.nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                torch.nn.init.zeros_(parameter)
        torch.nn.init.normal_(self.output.weight, mean=0.0, std=0.02)

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.rnn(embedded_tokens)
        hidden = self.norm(hidden)
        return self.output(hidden)


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


@dataclass(slots=True)
class FlatPretokenizedShard:
    path: Path
    kind: tuple[str, ...]
    source: tuple[str, ...]
    token_count: torch.Tensor
    document_chunk_offsets: torch.Tensor
    chunk_token_offsets: torch.Tensor
    context_flat: torch.Tensor | None
    loss_mask_flat: torch.Tensor | None
    is_continuation: torch.Tensor | None
    seq_len: int
    pad_token_id: int
    eos_token_id: int
    cont_token_id: int
    special_token_ids: dict[str, int]
    vocab_size: int
    tokenizer_label: str | None
    tokenizer_model_path: str | None
    corpus_info: dict[str, Any]
    _load_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _loaded: bool = False

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
        loss_mode: str = "default",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        with self._load_lock:
            if self.context_flat is None or self.loss_mask_flat is None or self.is_continuation is None:
                self.ensure_loaded()
            if self.context_flat is None or self.loss_mask_flat is None or self.is_continuation is None:
                raise RuntimeError(f"Shard tensors are unavailable: {self.path}")
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
                if loss_mode == "full":
                    loss_mask[:active_length] = 1.0
                elif (
                    document_kind in {"dialogue", "instruction"}
                    and active_length > 2
                    and int(self.special_token_ids.get("assistant", -1)) >= 0
                    and int(self.special_token_ids.get("eot", -1)) >= 0
                ):
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

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            bundle = torch.load(self.path, map_location="cpu")
            if not isinstance(bundle, dict):
                raise ValueError(f"Invalid pretokenized shard payload: {self.path}")
            _validate_flat_bundle_payload(bundle, path=self.path)
            chunk_meta = bundle["chunks"]
            self.context_flat = chunk_meta["context_flat"].detach().cpu().to(dtype=torch.long)
            self.loss_mask_flat = chunk_meta["loss_mask_flat"].detach().cpu()
            self.is_continuation = chunk_meta["is_continuation"].detach().cpu().to(dtype=torch.bool)
            self._loaded = True

    def unload(self) -> None:
        with self._load_lock:
            self.context_flat = None
            self.loss_mask_flat = None
            self.is_continuation = None
            self._loaded = False


@dataclass(slots=True)
class FlatPretokenizedDirectory:
    shards: tuple[FlatPretokenizedShard, ...]
    document_refs: tuple[FlatDocumentRef, ...]
    vocab_size: int
    tokenizer_label: str | None
    tokenizer_model_path: str | None
    corpus_info: dict[str, Any]
    max_loaded_shards: int = 0
    _cache_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _loaded_shard_lru: OrderedDict[int, None] = field(default_factory=OrderedDict, repr=False, compare=False)

    def get_shard(self, shard_index: int) -> FlatPretokenizedShard:
        shard = self.shards[shard_index]
        shard.ensure_loaded()
        max_loaded_shards = max(0, int(self.max_loaded_shards))
        if max_loaded_shards <= 0:
            return shard
        evicted: list[FlatPretokenizedShard] = []
        with self._cache_lock:
            self._loaded_shard_lru.pop(shard_index, None)
            self._loaded_shard_lru[shard_index] = None
            while len(self._loaded_shard_lru) > max_loaded_shards:
                evicted_index, _ = self._loaded_shard_lru.popitem(last=False)
                if evicted_index == shard_index:
                    self._loaded_shard_lru[shard_index] = None
                    break
                evicted.append(self.shards[evicted_index])
        for evicted_shard in evicted:
            evicted_shard.unload()
        return shard


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


def sample_flat_documents_for_logging(
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    *,
    sample_count: int,
) -> list[FlatDocumentRef]:
    if sample_count <= 0 or not documents:
        return []
    mixed_dialogue = [
        reference
        for reference in documents
        if collection.shards[reference.shard_index].source[reference.document_index].startswith("mixed_dialogue:")
    ]
    selected: list[FlatDocumentRef] = []
    if mixed_dialogue:
        selected.extend(random.sample(mixed_dialogue, k=min(sample_count, len(mixed_dialogue))))
    if len(selected) >= sample_count:
        return selected[:sample_count]
    selected_keys = {(reference.shard_index, reference.document_index) for reference in selected}
    for reference in sample_flat_documents_uniform_by_bucket(
        collection,
        documents,
        sample_count=max(sample_count * 2, sample_count),
    ):
        key = (reference.shard_index, reference.document_index)
        if key in selected_keys:
            continue
        selected.append(reference)
        selected_keys.add(key)
        if len(selected) >= sample_count:
            break
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
        active_shards_per_bucket: int = 2,
        shard_rotation_interval: int = 256,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")
        self.collection = collection
        self.documents = tuple(documents)
        self.batch_size = batch_size
        self.device = device
        self.pin_memory = device.type == "cuda"
        self.active_shards_per_bucket = max(1, int(active_shards_per_bucket))
        self.shard_rotation_interval = max(1, int(shard_rotation_interval))
        self._rng = random.Random(1337)
        self._batches_served = 0
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size
        self.bucket_to_indices: dict[str, tuple[int, ...]] = {}
        self.bucket_to_shard_indices: dict[str, dict[int, tuple[int, ...]]] = {}
        for index, reference in enumerate(self.documents):
            bucket = flat_document_ref_bucket(self.collection, reference)
            self.bucket_to_indices.setdefault(bucket, []).append(index)
            self.bucket_to_shard_indices.setdefault(bucket, {}).setdefault(reference.shard_index, []).append(index)
        self.bucket_to_indices = {
            bucket: tuple(indices)
            for bucket, indices in self.bucket_to_indices.items()
            if indices
        }
        self.bucket_to_shard_indices = {
            bucket: {shard_index: tuple(indices) for shard_index, indices in shard_map.items() if indices}
            for bucket, shard_map in self.bucket_to_shard_indices.items()
            if shard_map
        }
        self.active_buckets = tuple(self.bucket_to_indices)
        self.active_bucket_weights = tuple(1.0 for _ in self.active_buckets)
        self.active_bucket_shards = self._sample_active_bucket_shards()

    def set_batch_size(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size == self.batch_size:
            return
        self.batch_size = batch_size
        self.current_doc = [-1] * batch_size
        self.current_chunk = [0] * batch_size
        self.needs_reset = [True] * batch_size

    def _sample_active_bucket_shards(self) -> dict[str, tuple[int, ...]]:
        active_bucket_shards: dict[str, tuple[int, ...]] = {}
        for bucket, shard_map in self.bucket_to_shard_indices.items():
            shard_indices = list(shard_map)
            if len(shard_indices) <= self.active_shards_per_bucket:
                active_bucket_shards[bucket] = tuple(shard_indices)
                continue
            active_bucket_shards[bucket] = tuple(
                self._rng.sample(shard_indices, k=self.active_shards_per_bucket)
            )
        return active_bucket_shards

    def _refresh_active_bucket_shards(self, *, force: bool = False) -> None:
        if force or self._batches_served % self.shard_rotation_interval == 0:
            self.active_bucket_shards = self._sample_active_bucket_shards()

    def _sample_document_index(self) -> int:
        if not self.active_buckets:
            return random.randrange(len(self.documents))
        bucket = random.choices(self.active_buckets, weights=self.active_bucket_weights, k=1)[0]
        active_shards = self.active_bucket_shards.get(bucket) or tuple(self.bucket_to_shard_indices[bucket])
        shard_index = self._rng.choice(active_shards)
        return self._rng.choice(self.bucket_to_shard_indices[bucket][shard_index])

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
        self._refresh_active_bucket_shards(force=True)

    def next_batch(self) -> DocumentBatch:
        self._batches_served += 1
        self._refresh_active_bucket_shards()
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
            shard = self.collection.get_shard(reference.shard_index)
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


class FlatSequentialDocumentBatcher:
    def __init__(
        self,
        collection: FlatPretokenizedDirectory,
        documents: Sequence[FlatDocumentRef],
        *,
        batch_size: int,
        device: torch.device,
        max_chunks_per_document: int = 0,
        full_loss: bool = False,
    ) -> None:
        if not documents:
            raise ValueError("documents must not be empty.")
        self.collection = collection
        self.documents = tuple(sorted(documents, key=lambda ref: (ref.shard_index, ref.document_index)))
        self.batch_size = max(1, int(batch_size))
        self.device = device
        self.pin_memory = device.type == "cuda"
        self.max_chunks_per_document = max(0, int(max_chunks_per_document))
        self.full_loss = bool(full_loss)
        self.current_doc = [-1] * self.batch_size
        self.current_chunk = [0] * self.batch_size
        self.needs_reset = [True] * self.batch_size
        self._next_document_cursor = 0

    def set_batch_size(self, batch_size: int) -> None:
        batch_size = max(1, int(batch_size))
        if batch_size == self.batch_size:
            return
        self.batch_size = batch_size
        self.current_doc = [-1] * self.batch_size
        self.current_chunk = [0] * self.batch_size
        self.needs_reset = [True] * self.batch_size

    def _assign_next_document(self) -> int:
        document_index = self._next_document_cursor
        self._next_document_cursor = (self._next_document_cursor + 1) % len(self.documents)
        return document_index

    def next_batch(self) -> DocumentBatch:
        contexts: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        reset_mask = torch.zeros(self.batch_size, dtype=torch.bool)
        for item_index in range(self.batch_size):
            if self.needs_reset[item_index]:
                self.current_doc[item_index] = self._assign_next_document()
                self.current_chunk[item_index] = 0
                reset_mask[item_index] = True
            reference = self.documents[self.current_doc[item_index]]
            shard = self.collection.get_shard(reference.shard_index)
            chunk_start, chunk_end = shard.document_chunk_range(reference.document_index)
            max_chunk_end = chunk_end
            if self.max_chunks_per_document > 0:
                max_chunk_end = min(chunk_end, chunk_start + self.max_chunks_per_document)
            chunk_index = chunk_start + self.current_chunk[item_index]
            is_last_chunk = chunk_index >= max_chunk_end - 1
            context, target, loss_mask, _ = shard.build_chunk_tensors(
                chunk_index,
                document_index=reference.document_index,
                is_last_chunk=is_last_chunk,
                loss_mode="full" if self.full_loss else "default",
            )
            contexts.append(context)
            targets.append(target)
            masks.append(loss_mask)
            next_chunk = self.current_chunk[item_index] + 1
            if chunk_start + next_chunk >= max_chunk_end:
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


def render_document_for_hf_tokenizer(document: SerializedDocument) -> str:
    body = document.body
    replacements = {
        f"{USER_TOKEN}\n": "User:\n",
        f"{ASSISTANT_TOKEN}\n": "Assistant:\n",
        f"{INSTRUCTION_TOKEN}\n": "System:\n",
        f"{EOT_TOKEN}": "\n",
        f"{TEXT_TOKEN}\n": "",
        f"{CODE_TOKEN}\n": "Code:\n",
        f"{MATH_TOKEN}\n": "Math:\n",
        f"{DIALOGUE_TOKEN}\n": "Dialogue:\n",
        f"{RESPONSE_TOKEN}\n": "Response:\n",
    }
    rendered = body
    for source, target in replacements.items():
        rendered = rendered.replace(source, target)
    rendered = rendered.strip()
    if document.kind == "dialogue":
        return rendered
    prefix = {
        "code": "Code document:\n",
        "math": "Math document:\n",
        "instruction": "Instruction:\n",
    }.get(document.kind, "")
    return f"{prefix}{rendered}".strip()


def build_training_text(documents: Sequence[SerializedDocument], *, tokenizer_label: str = "byte_bpe") -> str:
    if tokenizer_label == "hf_auto":
        return "\n\n".join(render_document_for_hf_tokenizer(document) for document in documents)
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


def _safe_token_id(vocab: object, piece: str, default: int = -1) -> int:
    if not hasattr(vocab, "token_id"):
        return int(default)
    try:
        return int(vocab.token_id(piece))
    except Exception:
        return int(default)


def build_special_token_id_map(vocab: object, *, tokenizer_label: str = "byte_bpe") -> dict[str, int]:
    if tokenizer_label == "hf_auto":
        tokenizer = getattr(vocab, "tokenizer", None)
        bos_piece = getattr(tokenizer, "bos_token", None) if tokenizer is not None else None
        eos_piece = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
        pad_piece = getattr(tokenizer, "pad_token", None) if tokenizer is not None else None
        bos_id = _safe_token_id(vocab, str(bos_piece), -1) if bos_piece is not None else -1
        eos_id = _safe_token_id(vocab, str(eos_piece), bos_id if bos_id >= 0 else 0) if eos_piece is not None else (bos_id if bos_id >= 0 else 0)
        pad_id = _safe_token_id(vocab, str(pad_piece), eos_id if eos_id >= 0 else 0) if pad_piece is not None else (eos_id if eos_id >= 0 else 0)
        return {
            "bos": bos_id if bos_id >= 0 else eos_id,
            "eos": eos_id,
            "pad": pad_id,
            "text": -1,
            "code": -1,
            "math": -1,
            "cont": eos_id,
            "dialogue": -1,
            "instruction": -1,
            "response": -1,
            "user": _safe_token_id(vocab, "<|im_start|>"),
            "assistant": _safe_token_id(vocab, "<|im_start|>"),
            "eot": _safe_token_id(vocab, "<|im_end|>"),
        }
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
    mode_token_id: int | None,
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
    first_prefix = [bos_token_id] if bos_token_id >= 0 else []
    continuation_prefix = [cont_token_id] if cont_token_id >= 0 else []
    if mode_token_id is not None and mode_token_id >= 0:
        first_prefix = [*first_prefix, int(mode_token_id)]
        continuation_prefix = [*continuation_prefix, int(mode_token_id)]
    if content_ids.numel() == 0:
        empty_prefix = first_prefix if first_prefix else [eos_token_id]
        context = torch.tensor(empty_prefix, dtype=torch.long)
        target = torch.empty_like(context)
        if context.numel() > 1:
            target[:-1] = context[1:]
        target[-1] = eos_token_id
        loss_mask = torch.zeros(context.shape[0], dtype=torch.float32)
        return (
            DocumentChunk(
                context=F.pad(context, (0, seq_len - context.shape[0]), value=pad_token_id),
                target=F.pad(target, (0, seq_len - target.shape[0]), value=pad_token_id),
                loss_mask=F.pad(loss_mask, (0, seq_len - loss_mask.shape[0]), value=0.0),
                is_continuation=False,
            ),
        )
    while cursor < content_ids.numel():
        prefix = first_prefix if first else continuation_prefix
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
    tokenizer_label: str = "byte_bpe",
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
        if tokenizer_label == "hf_auto":
            target_visibility = torch.ones(content_ids.shape[0], dtype=torch.float32)
        else:
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
            mode_token_id=None if tokenizer_label == "hf_auto" else mode_token_id_map[kind],
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

    if workers > 1 and isinstance(vocab, HFTokenizerVocab):
        payload_count = len(documents)
        payload_batch_size = max(32, min(512, payload_count // max(1, workers * 8) or 128))
        payload_batches = [
            [
                render_document_for_hf_tokenizer(document)
                for document in documents[batch_start : batch_start + payload_batch_size]
            ]
            for batch_start in range(0, payload_count, payload_batch_size)
        ]
        total_payload_batches = len(payload_batches)
        chunksize = max(1, min(32, total_payload_batches // max(1, workers * 2) or 1))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_hf_encode_worker,
            initargs=(str(vocab.model_path),),
        ) as executor:
            index = 0
            for encoded_batch in executor.map(
                _encode_hf_text_batch_worker,
                payload_batches,
                chunksize=chunksize,
            ):
                for token_ids in encoded_batch:
                    index += 1
                    if token_ids:
                        content_ids = torch.tensor(token_ids, dtype=torch.long)
                    else:
                        content_ids = torch.empty(0, dtype=torch.long)
                    document = documents[index - 1]
                    append_document(document.kind, document.source, content_ids, loss_mode=document.loss_mode, index=index)
        return tokenized_documents

    for index, document in enumerate(documents, start=1):
        source_text = (
            render_document_for_hf_tokenizer(document)
            if tokenizer_label == "hf_auto"
            else document.body
        )
        content_ids = encode_text_in_chunks(vocab, source_text, workers=workers)
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
    meta_path = path.with_suffix(f"{path.suffix}.meta.json")
    meta_payload = {
        "storage_format": "flat_v2",
        "documents": {
            "kind": document_kinds,
            "source": document_sources,
            "token_count": document_token_counts,
            "chunk_offsets": document_offsets,
        },
        "chunks": {
            "token_offsets": chunk_offsets,
        },
        "seq_len": seq_len,
        "vocab_size": vocab_size,
        "tokenizer_label": tokenizer_label,
        "tokenizer_model_path": tokenizer_model_path,
        "corpus_info": corpus_info,
    }
    try:
        with temp_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        write_json(meta_path, meta_payload)
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


def _flat_shard_meta_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.meta.json")


def _flat_shard_meta_payload(shard: FlatPretokenizedShard) -> dict[str, Any]:
    return {
        "storage_format": "flat_v2",
        "documents": {
            "kind": list(shard.kind),
            "source": list(shard.source),
            "token_count": shard.token_count.to(dtype=torch.int64).tolist(),
            "chunk_offsets": shard.document_chunk_offsets.to(dtype=torch.int64).tolist(),
        },
        "chunks": {
            "token_offsets": shard.chunk_token_offsets.to(dtype=torch.int64).tolist(),
        },
        "seq_len": int(shard.seq_len),
        "vocab_size": int(shard.vocab_size),
        "tokenizer_label": shard.tokenizer_label,
        "tokenizer_model_path": shard.tokenizer_model_path,
        "corpus_info": shard.corpus_info,
    }


def load_flat_pretokenized_shard_metadata(path: Path) -> FlatPretokenizedShard:
    meta_path = _flat_shard_meta_path(path)
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("storage_format") != "flat_v2":
            raise ValueError(f"Expected flat_v2 shard metadata: {meta_path}")
        corpus_info = payload.get("corpus_info") or {}
        special_tokens = dict(corpus_info.get("special_tokens") or {})
        pad_token_id = special_tokens.get("pad")
        eos_token_id = special_tokens.get("eos")
        cont_token_id = special_tokens.get("cont")
        if pad_token_id is None or eos_token_id is None or cont_token_id is None:
            raise ValueError(f"Flat pretokenized metadata is missing pad/eos/cont ids: {meta_path}")
        document_meta = payload["documents"]
        chunk_meta = payload["chunks"]
        return FlatPretokenizedShard(
            path=path,
            kind=tuple(str(item) for item in document_meta["kind"]),
            source=tuple(str(item) for item in document_meta["source"]),
            token_count=torch.tensor(document_meta["token_count"], dtype=torch.int64),
            document_chunk_offsets=torch.tensor(document_meta["chunk_offsets"], dtype=torch.int64),
            chunk_token_offsets=torch.tensor(chunk_meta["token_offsets"], dtype=torch.int64),
            context_flat=None,
            loss_mask_flat=None,
            is_continuation=None,
            seq_len=int(payload["seq_len"]),
            pad_token_id=int(pad_token_id),
            eos_token_id=int(eos_token_id),
            cont_token_id=int(cont_token_id),
            special_token_ids={key: int(value) for key, value in special_tokens.items()},
            vocab_size=int(payload["vocab_size"]),
            tokenizer_label=payload.get("tokenizer_label"),
            tokenizer_model_path=payload.get("tokenizer_model_path"),
            corpus_info=corpus_info,
        )
    shard = load_flat_pretokenized_shard(path)
    write_json(meta_path, _flat_shard_meta_payload(shard))
    return shard


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
        _loaded=True,
    )


def load_flat_pretokenized_directory(
    path: Path,
    *,
    load_workers: int = 1,
    max_loaded_shards: int = 0,
    integrity_mode: str = "meta",
    integrity_workers: int = 1,
) -> FlatPretokenizedDirectory:
    shard_paths = sorted(candidate for candidate in path.glob("*.pt") if candidate.is_file())
    if not shard_paths:
        raise ValueError(f"No pretokenized shard files found in {path}")
    integrity_mode_normalized = str(integrity_mode).strip().lower()
    if integrity_mode_normalized not in {"none", "meta", "full"}:
        raise ValueError(f"Unsupported flat integrity mode: {integrity_mode!r}")

    metadata_workers = min(max(1, load_workers), len(shard_paths))
    metadata_executor: ThreadPoolExecutor | None = None
    metadata_futures = None
    preloaded_shards: list[FlatPretokenizedShard] | None = None

    if integrity_mode_normalized == "full":
        validation_workers = min(max(1, integrity_workers), len(shard_paths))
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
    elif integrity_mode_normalized == "meta":
        metadata_workers = min(max(1, integrity_workers), len(shard_paths))
        print(
            f"flat_metadata_index | path={path} | shards={len(shard_paths)} | workers={metadata_workers}",
            flush=True,
        )
        if metadata_workers <= 1:
            preloaded_shards = [load_flat_pretokenized_shard_metadata(shard_path) for shard_path in shard_paths]
        else:
            with ThreadPoolExecutor(
                max_workers=metadata_workers,
                thread_name_prefix="pretok_flat_meta",
            ) as validation_executor:
                metadata_futures = validation_executor.map(load_flat_pretokenized_shard_metadata, shard_paths)
                preloaded_shards = list(metadata_futures)
        print(f"flat_metadata_ok | path={path} | shards={len(shard_paths)}", flush=True)
    else:
        print(f"flat_integrity_skip | path={path} | shards={len(shard_paths)}", flush=True)

    if preloaded_shards is not None:
        shard_iterator = iter(preloaded_shards)
        executor = None
    elif metadata_workers <= 1:
        shard_iterator = (load_flat_pretokenized_shard_metadata(shard_path) for shard_path in shard_paths)
        executor = None
    else:
        executor = ThreadPoolExecutor(
            max_workers=metadata_workers,
            thread_name_prefix="pretok_flat_load",
        )
        shard_iterator = executor.map(load_flat_pretokenized_shard_metadata, shard_paths)
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
    if max_loaded_shards > 0:
        for shard in shards:
            shard.unload()
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
        max_loaded_shards=max_loaded_shards,
    )


def preload_flat_pretokenized_directory(collection: FlatPretokenizedDirectory) -> None:
    total_shards = len(collection.shards)
    print(f"flat_preload_start | shards={total_shards}", flush=True)
    started_at = time.perf_counter()
    for shard_index in range(total_shards):
        collection.get_shard(shard_index)
    elapsed = time.perf_counter() - started_at
    print(f"flat_preload_done | shards={total_shards} | seconds={elapsed:.1f}", flush=True)


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
    stage1_span: int,
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
            document_span=max(1, stage1_span),
            batch_size=max(1, stage1_batch_size),
            grad_accum_steps=max(1, stage1_grad_accum_steps),
            freeze_memory=False,
            freeze_propagation=False,
            freeze_skip=False,
            bucket_weights=resolve_stage_bucket_weights(stage_name="stage1"),
        )
    if stage2_end > 0 and step <= stage2_end:
        return TrainingCurriculumStage(
            name="stage2",
            document_span=max(1, stage2_span),
            batch_size=max(1, stage2_batch_size),
            grad_accum_steps=max(1, stage2_grad_accum_steps),
            freeze_memory=False,
            freeze_propagation=False,
            freeze_skip=False,
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
    _set_module_requires_grad(b_module.memory_levels, True)
    _set_module_requires_grad(b_module.level_transitions, True)
    _set_module_requires_grad(b_module.level_norms, True)
    _set_module_requires_grad(b_module.read_projections, True)
    _set_parameter_collection_requires_grad(b_module.read_gates, True)
    b_module.read_template_val.requires_grad_(True)
    for level in b_module.memory_levels:
        _set_module_requires_grad(level.propagation, True)
    _set_module_requires_grad(b_module.skip_transitions, True)
    _set_parameter_collection_requires_grad(tuple(b_module.skip_gates.values()), True)


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


def compute_decayed_scalar(
    *,
    step: int,
    start_step: int,
    end_step: int,
    initial_value: float,
    final_value: float,
) -> float:
    if step <= start_step:
        return float(initial_value)
    if end_step <= start_step:
        return float(final_value)
    if step >= end_step:
        return float(final_value)
    progress = (step - start_step) / max(1, end_step - start_step)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(final_value + (initial_value - final_value) * cosine)


def compute_warmup_scalar(
    *,
    step: int,
    start_step: int,
    end_step: int,
    initial_value: float = 0.0,
    final_value: float = 1.0,
) -> float:
    if step <= start_step:
        return float(initial_value)
    if end_step <= start_step:
        return float(final_value)
    if step >= end_step:
        return float(final_value)
    progress = (step - start_step) / max(1, end_step - start_step)
    return float(initial_value + (final_value - initial_value) * progress)


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


def clone_memory_state(memory_state: Any | None) -> Any | None:
    if memory_state is None:
        return None
    if isinstance(memory_state, ModelRecurrentState):
        cloned_knowledge = None
        if memory_state.knowledge_state is not None:
            cloned_knowledge = memory_state.knowledge_state.clone()
        return ModelRecurrentState(
            memory_state=tuple(layer.clone() for layer in memory_state.memory_state),
            knowledge_state=cloned_knowledge,
        )
    return tuple(layer.clone() for layer in memory_state)


def copy_memory_state_(destination: Any | None, source: Any | None) -> Any | None:
    if destination is None or source is None:
        return destination
    if isinstance(destination, ModelRecurrentState):
        if not isinstance(source, ModelRecurrentState):
            raise TypeError("source memory state type does not match destination.")
        for dst_layer, src_layer in zip(destination.memory_state, source.memory_state, strict=True):
            dst_layer.state.copy_(src_layer.state)
            dst_layer.val.copy_(src_layer.val)
        if destination.knowledge_state is not None and source.knowledge_state is not None:
            destination.knowledge_state.state.copy_(source.knowledge_state.state)
            destination.knowledge_state.val.copy_(source.knowledge_state.val)
        return destination
    if isinstance(source, ModelRecurrentState):
        raise TypeError("source memory state type does not match destination.")
    for dst_layer, src_layer in zip(destination, source, strict=True):
        dst_layer.state.copy_(src_layer.state)
        dst_layer.val.copy_(src_layer.val)
    return destination


def make_static_document_batch(batch: DocumentBatch) -> DocumentBatch:
    return DocumentBatch(
        context=torch.empty_like(batch.context),
        target=torch.empty_like(batch.target),
        loss_mask=torch.empty_like(batch.loss_mask),
        reset_mask=torch.empty_like(batch.reset_mask),
    )


def copy_document_batch_(destination: DocumentBatch, source: DocumentBatch) -> None:
    destination.context.copy_(source.context)
    destination.target.copy_(source.target)
    destination.loss_mask.copy_(source.loss_mask)
    destination.reset_mask.copy_(source.reset_mask)


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


def summarize_nonfinite_memory_state(memory_state: Any | None) -> list[str]:
    if memory_state is None:
        return []
    layers: list[tuple[str, Any]] = []
    if isinstance(memory_state, ModelRecurrentState):
        layers.extend((f"memory[{index}]", layer) for index, layer in enumerate(memory_state.memory_state))
        if memory_state.knowledge_state is not None:
            layers.append(("knowledge", memory_state.knowledge_state))
    else:
        layers.extend((f"memory[{index}]", layer) for index, layer in enumerate(memory_state))
    diagnostics: list[str] = []
    for layer_name, layer in layers:
        for tensor_name, tensor in (("state", layer.state), ("val", layer.val)):
            finite_mask = torch.isfinite(tensor)
            if bool(finite_mask.all().item()):
                continue
            nonfinite_count = int((~finite_mask).sum().item())
            diagnostics.append(
                f"{layer_name}.{tensor_name}:shape={tuple(tensor.shape)} nonfinite={nonfinite_count}"
            )
    return diagnostics


def summarize_nonfinite_gradients(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    limit: int,
) -> list[str]:
    diagnostics: list[tuple[int, str]] = []
    for parameter_name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite_mask = torch.isfinite(gradient)
        if bool(finite_mask.all().item()):
            continue
        nonfinite_count = int((~finite_mask).sum().item())
        diagnostics.append(
            (
                nonfinite_count,
                f"{parameter_name}:shape={tuple(parameter.shape)} grad_nonfinite={nonfinite_count}",
            )
        )
    diagnostics.sort(key=lambda item: item[0], reverse=True)
    return [message for _, message in diagnostics[: max(1, int(limit))]]


def summarize_gradient_extremes(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    *,
    limit: int,
) -> list[str]:
    diagnostics: list[tuple[float, str]] = []
    for parameter_name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached_gradient = gradient.detach()
        max_abs = float(detached_gradient.abs().max().item())
        grad_norm = float(torch.linalg.vector_norm(detached_gradient.float()).item())
        diagnostics.append(
            (
                grad_norm,
                f"{parameter_name}:shape={tuple(parameter.shape)} grad_norm={grad_norm:.6g} grad_max_abs={max_abs:.6g}",
            )
        )
    diagnostics.sort(key=lambda item: item[0], reverse=True)
    return [message for _, message in diagnostics[: max(1, int(limit))]]


def load_decode_vocab(*, tokenizer_label: str, tokenizer_model_path: str | None) -> object | None:
    if tokenizer_model_path is None:
        return None
    if tokenizer_label == "hf_auto":
        if AutoTokenizer is None:
            return None
        tokenizer_path = Path(tokenizer_model_path)
        if not tokenizer_path.exists():
            return None
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
        return HFTokenizerVocab(tokenizer=tokenizer, model_path=tokenizer_path)
    if tokenizer_label != "byte_bpe" or ByteLevelBPETokenizer is None:
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


def infer_hf_hidden_size(*, model_name_or_path: str, trust_remote_code: bool = False) -> int:
    if AutoConfig is None:
        raise ImportError("transformers is required to infer HF embedding dimensions.")
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not infer hidden_size from HF config: {model_name_or_path}")
    return int(hidden_size)


@torch.no_grad()
def initialize_model_embedding_from_hf(
    *,
    model: CausalHierarchicalMemoryLM,
    vocab: HFTokenizerVocab,
    model_name_or_path: str,
    trust_remote_code: bool = False,
) -> dict[str, int]:
    if AutoModelForCausalLM is None or AutoModel is None:
        raise ImportError("transformers is required to load HF pretrained embeddings.")
    hf_model: object
    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
        )
    except Exception:
        hf_model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,
        )
    try:
        if not hasattr(hf_model, "get_input_embeddings"):
            raise RuntimeError(f"Model {model_name_or_path} does not expose input embeddings.")
        hf_embedding = hf_model.get_input_embeddings().weight.detach().cpu()
        target_weight = model.s_module.token_embedding.weight.detach()
        if target_weight.shape[1] != hf_embedding.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: target={target_weight.shape[1]}, hf={hf_embedding.shape[1]}"
            )
        copied = 0
        skipped = 0
        for token, token_id in vocab.tokenizer.get_vocab().items():
            source_id = vocab.tokenizer.convert_tokens_to_ids(token)
            if source_id is None or int(source_id) < 0:
                skipped += 1
                continue
            source_index = int(source_id)
            if source_index >= hf_embedding.shape[0] or int(token_id) >= target_weight.shape[0]:
                skipped += 1
                continue
            target_weight[int(token_id)].copy_(hf_embedding[source_index])
            copied += 1
        if model.lm_head.weight.data_ptr() != model.s_module.token_embedding.weight.data_ptr():
            model.lm_head.weight.copy_(model.s_module.token_embedding.weight)
        return {"copied": copied, "skipped": skipped}
    finally:
        del hf_model


def _decode_visible_tokens(
    vocab: object,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    ignore_token_ids: set[int] | None = None,
) -> str:
    if not hasattr(vocab, "decode"):
        return ""
    ids = token_ids.detach().cpu().to(dtype=torch.long)
    if loss_mask is not None:
        mask = loss_mask.detach().cpu() > 0
        ids = ids[mask]
    if ignore_token_ids:
        ids = torch.tensor(
            [int(token_id) for token_id in ids.tolist() if int(token_id) not in ignore_token_ids],
            dtype=torch.long,
        )
    return str(vocab.decode(ids.tolist()))


def _format_visible_text(
    vocab: object,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    *,
    ignore_token_ids: set[int] | None = None,
    empty_label: str,
) -> str:
    decoded = _decode_visible_tokens(
        vocab,
        token_ids,
        loss_mask,
        ignore_token_ids=ignore_token_ids,
    )
    if decoded.strip():
        return decoded
    ids = token_ids.detach().cpu().to(dtype=torch.long)
    if loss_mask is not None:
        ids = ids[loss_mask.detach().cpu() > 0]
    if ignore_token_ids:
        ids = torch.tensor(
            [int(token_id) for token_id in ids.tolist() if int(token_id) not in ignore_token_ids],
            dtype=torch.long,
        )
    return f"[{empty_label}] ids={ids[:32].tolist()}"


def _format_prediction_text(
    vocab: object,
    token_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    ignore_token_ids: set[int],
) -> str:
    decoded = _decode_visible_tokens(
        vocab,
        token_ids,
        loss_mask,
        ignore_token_ids=ignore_token_ids,
    )
    if decoded.strip():
        return decoded
    masked_ids = token_ids.detach().cpu().to(dtype=torch.long)[loss_mask.detach().cpu() > 0]
    return f"[no visible decoded text] ids={masked_ids[:32].tolist()}"


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
    max_samples: int = 1,
) -> None:
    if writer is None or vocab is None or not documents:
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
        for index, document in enumerate(documents[:max_samples], start=1):
            chunk = document.chunks[0]
            ignored_token_ids = set()
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
                f"### context\n{_format_visible_text(vocab, chunk.context, None, ignore_token_ids=ignored_token_ids, empty_label='no visible context')}\n\n"
                f"### target\n{_format_visible_text(vocab, chunk.target, chunk.loss_mask, ignore_token_ids=ignored_token_ids, empty_label='no visible target')}\n\n"
                f"### prediction\n{_format_prediction_text(vocab, predicted, chunk.loss_mask, ignore_token_ids=ignored_token_ids)}"
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
    max_samples: int = 1,
) -> None:
    if writer is None or vocab is None or not documents:
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
        for index, reference in enumerate(documents[:max_samples], start=1):
            shard = collection.get_shard(reference.shard_index)
            chunk_start, _ = shard.document_chunk_range(reference.document_index)
            context, target, loss_mask, _ = shard.build_chunk_tensors(
                chunk_start,
                document_index=reference.document_index,
                is_last_chunk=shard.chunk_count(reference.document_index) == 1,
            )
            ignored_token_ids = {
                int(shard.pad_token_id),
                int(shard.eos_token_id),
                int(shard.cont_token_id),
                *[int(token_id) for token_id in shard.special_token_ids.values()],
            }
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
                f"### context\n{_format_visible_text(vocab, context, None, ignore_token_ids=ignored_token_ids, empty_label='no visible context')}\n\n"
                f"### target\n{_format_visible_text(vocab, target, loss_mask, ignore_token_ids=ignored_token_ids, empty_label='no visible target')}\n\n"
                f"### prediction\n{_format_prediction_text(vocab, predicted, loss_mask, ignore_token_ids=ignored_token_ids)}"
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
) -> tuple[torch.Tensor, float, Any | None]:
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
            main_loss = compute_masked_loss(output.logits, batch.target, batch.loss_mask)
    return main_loss, float(main_loss.detach().item()), output.recurrent_state


def run_model_loss_tensor(
    model: CausalHierarchicalMemoryLM,
    batch: DocumentBatch,
    *,
    memory_state: Any | None,
    precision: str,
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
    with torch.enable_grad():
        with autocast_context:
            output = model(
                batch.context,
                memory_state=memory_state,
                reset_mask=batch.reset_mask,
                return_memory_state=True,
            )
            assert isinstance(output, MemoryScanOutput)
            main_loss = compute_masked_loss(output.logits, batch.target, batch.loss_mask)
    return main_loss, output.recurrent_state


class Stage1CudaGraphRunner:
    def __init__(
        self,
        *,
        model: CausalHierarchicalMemoryLM,
        batch: DocumentBatch,
        precision: str,
    ) -> None:
        if batch.context.device.type != "cuda":
            raise ValueError("CUDA graph runner requires CUDA tensors.")
        self.model = model
        self.precision = precision
        self.static_batch = make_static_document_batch(batch)
        copy_document_batch_(self.static_batch, batch)
        zero_memory = model.initialize_memory_state(
            batch.context.shape[0],
            device=batch.context.device,
            dtype=model.s_module.token_embedding.weight.dtype,
        )
        self.zero_memory_state = clone_memory_state(zero_memory)
        self.static_memory_state = clone_memory_state(zero_memory)
        self.graph = torch.cuda.CUDAGraph()
        self.loss_tensor: torch.Tensor | None = None
        self.output_memory_state: Any | None = None
        self._capture()

    def _capture(self) -> None:
        warmup_stream = torch.cuda.Stream(device=self.static_batch.context.device)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            self.model.zero_grad(set_to_none=True)
            loss, _ = run_model_loss_tensor(
                self.model,
                self.static_batch,
                memory_state=self.static_memory_state,
                precision=self.precision,
            )
            loss.backward()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        copy_memory_state_(self.static_memory_state, self.zero_memory_state)
        self.model.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.loss_tensor, self.output_memory_state = run_model_loss_tensor(
                self.model,
                self.static_batch,
                memory_state=self.static_memory_state,
                precision=self.precision,
            )
            self.loss_tensor.backward()

    def replay(
        self,
        batch: DocumentBatch,
        *,
        memory_state: Any | None,
    ) -> tuple[torch.Tensor, float, Any | None]:
        copy_document_batch_(self.static_batch, batch)
        if memory_state is None:
            copy_memory_state_(self.static_memory_state, self.zero_memory_state)
        else:
            copy_memory_state_(self.static_memory_state, memory_state)
        self.graph.replay()
        assert self.loss_tensor is not None
        loss_tensor = self.loss_tensor
        return (
            loss_tensor,
            float(loss_tensor.detach().item()),
            detach_memory_state(self.output_memory_state),
        )


class Stage1ForwardCudaGraphRunner:
    def __init__(
        self,
        *,
        model: CausalHierarchicalMemoryLM,
        batch: DocumentBatch,
        precision: str,
    ) -> None:
        if batch.context.device.type != "cuda":
            raise ValueError("CUDA graph runner requires CUDA tensors.")
        self.model = model
        self.precision = precision
        self.static_batch = make_static_document_batch(batch)
        copy_document_batch_(self.static_batch, batch)
        zero_memory = model.initialize_memory_state(
            batch.context.shape[0],
            device=batch.context.device,
            dtype=model.s_module.token_embedding.weight.dtype,
        )
        self.zero_memory_state = clone_memory_state(zero_memory)
        self.static_memory_state = clone_memory_state(zero_memory)
        self.graph = torch.cuda.CUDAGraph()
        self.loss_tensor: torch.Tensor | None = None
        self.output_memory_state: Any | None = None
        self._capture()

    def _capture(self) -> None:
        warmup_stream = torch.cuda.Stream(device=self.static_batch.context.device)
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            _ = run_model_loss_tensor(
                self.model,
                self.static_batch,
                memory_state=self.static_memory_state,
                precision=self.precision,
            )
        torch.cuda.current_stream().wait_stream(warmup_stream)
        copy_memory_state_(self.static_memory_state, self.zero_memory_state)
        with torch.cuda.graph(self.graph):
            self.loss_tensor, self.output_memory_state = run_model_loss_tensor(
                self.model,
                self.static_batch,
                memory_state=self.static_memory_state,
                precision=self.precision,
            )

    def replay(
        self,
        batch: DocumentBatch,
        *,
        memory_state: Any | None,
    ) -> tuple[torch.Tensor, float, Any | None]:
        copy_document_batch_(self.static_batch, batch)
        if memory_state is None:
            copy_memory_state_(self.static_memory_state, self.zero_memory_state)
        else:
            copy_memory_state_(self.static_memory_state, memory_state)
        self.graph.replay()
        assert self.loss_tensor is not None
        return (
            self.loss_tensor,
            float(self.loss_tensor.detach().item()),
            detach_memory_state(self.output_memory_state),
        )


def run_rnn_aux_model(
    model: CausalHierarchicalMemoryLM,
    rnn_aux_head: SharedEmbeddingRnnAuxHead,
    batch: DocumentBatch,
    *,
    precision: str,
    grad_enabled: bool,
) -> tuple[torch.Tensor, float]:
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
            embedded_tokens = model.s_module.token_embedding(batch.context)
            aux_logits = rnn_aux_head(embedded_tokens)
            aux_loss = compute_masked_loss(aux_logits, batch.target, batch.loss_mask)
    return aux_loss, float(aux_loss.detach().item())


@torch.no_grad()
def estimate_eval_loss(
    model: CausalHierarchicalMemoryLM,
    documents: Sequence[TokenizedDocument],
    *,
    eval_documents: int,
    device: torch.device,
    precision: str,
    rnn_aux_head: SharedEmbeddingRnnAuxHead | None = None,
) -> tuple[float, float | None]:
    if not documents:
        raise ValueError("documents must not be empty.")
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    total_weight = 0.0
    for document in documents[: min(eval_documents, len(documents))]:
        memory_state: tuple[Any, ...] | None = None
        for chunk_index, chunk in enumerate(document.chunks):
            batch = DocumentBatch(
                context=chunk.context.unsqueeze(0).to(device),
                target=chunk.target.unsqueeze(0).to(device),
                loss_mask=chunk.loss_mask.unsqueeze(0).to(device),
                reset_mask=torch.tensor([chunk_index == 0], device=device, dtype=torch.bool),
            )
            loss, _, memory_state = run_model(
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
            if rnn_aux_head is not None:
                aux_loss, _ = run_rnn_aux_model(
                    model,
                    rnn_aux_head,
                    batch,
                    precision=precision,
                    grad_enabled=False,
                )
                total_aux_loss += float(aux_loss.item()) * weight
    if model_was_training:
        model.train()
    denom = max(1.0, total_weight)
    aux_value = None if rnn_aux_head is None else (total_aux_loss / denom)
    return total_loss / denom, aux_value


@torch.no_grad()
def estimate_eval_loss_flat(
    model: CausalHierarchicalMemoryLM,
    collection: FlatPretokenizedDirectory,
    documents: Sequence[FlatDocumentRef],
    *,
    eval_documents: int,
    device: torch.device,
    precision: str,
    rnn_aux_head: SharedEmbeddingRnnAuxHead | None = None,
) -> tuple[float, float | None]:
    if not documents:
        raise ValueError("documents must not be empty.")
    model_was_training = model.training
    model.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    total_weight = 0.0
    for reference in documents[: min(eval_documents, len(documents))]:
        shard = collection.get_shard(reference.shard_index)
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
            loss, _, memory_state = run_model(
                model,
                batch,
                memory_state=memory_state,
                precision=precision,
                grad_enabled=False,
            )
            weight = float(batch.loss_mask.sum().item())
            total_loss += float(loss.item()) * weight
            total_weight += weight
            if rnn_aux_head is not None:
                aux_loss, _ = run_rnn_aux_model(
                    model,
                    rnn_aux_head,
                    batch,
                    precision=precision,
                    grad_enabled=False,
                )
                total_aux_loss += float(aux_loss.item()) * weight
    if model_was_training:
        model.train()
    denom = max(1.0, total_weight)
    aux_value = None if rnn_aux_head is None else (total_aux_loss / denom)
    return total_loss / denom, aux_value


def save_checkpoint(
    path: Path,
    *,
    model: CausalHierarchicalMemoryLM,
    rnn_aux_head: SharedEmbeddingRnnAuxHead | None,
    optimizer: torch.optim.Optimizer,
    rnn_aux_optimizer: torch.optim.Optimizer | None,
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
            "rnn_aux_optimizer_state_dict": None if rnn_aux_optimizer is None else rnn_aux_optimizer.state_dict(),
            "rnn_aux_state_dict": None if rnn_aux_head is None else rnn_aux_head.state_dict(),
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

    parser.add_argument("--tokenizer", choices=("byte_bpe", "hf_auto"), default="byte_bpe")
    parser.add_argument("--subword-vocab-size", type=int, default=16384)
    parser.add_argument("--subword-model-type", default="bpe")
    parser.add_argument("--tokenizer-prefix")
    parser.add_argument("--hf-tokenizer-model")
    parser.add_argument("--hf-embedding-model")
    parser.add_argument("--hf-trust-remote-code", action="store_true")
    parser.add_argument("--match-dim-to-hf-embedding", action="store_true")
    parser.add_argument("--subword-character-coverage", type=float, default=1.0)
    parser.add_argument("--subword-input-sentence-size", type=int, default=0)
    parser.add_argument("--subword-num-threads", type=int, default=0)
    parser.add_argument("--pretokenize-workers", type=int, default=8)
    parser.add_argument("--pretokenized-path")
    parser.add_argument("--pretokenized-dir")
    parser.add_argument("--pretokenized-load-workers", type=int, default=8)
    parser.add_argument("--flat-integrity-mode", choices=("none", "meta", "full"), default="meta")
    parser.add_argument("--flat-integrity-workers", type=int, default=8)
    parser.add_argument("--pretokenized-max-loaded-shards", type=int, default=8)
    parser.add_argument("--preload-flat-shards", action="store_true")
    parser.add_argument("--pretokenized-active-shards-per-bucket", type=int, default=2)
    parser.add_argument("--pretokenized-shard-rotation-interval", type=int, default=256)
    parser.add_argument("--prefetch-batches", type=int, default=100)
    parser.add_argument("--save-pretokenized", action="store_true")

    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--s-layers", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, nargs="+", default=[256, 64, 16])
    parser.add_argument("--memory-update-intervals", type=int, nargs="+")
    parser.add_argument("--prediction-layers", type=int, default=2)
    parser.add_argument("--s-window", type=int, default=256)
    parser.add_argument("--s-microbatch-size", type=int, default=0)
    parser.add_argument("--prediction-window", type=int, default=64)
    parser.add_argument("--checkpoint-sequence-layers", action="store_true")
    parser.add_argument("--checkpoint-prediction-layers", action="store_true")
    parser.add_argument("--memory-topk", type=int, default=16)
    parser.add_argument("--scan-checkpoint-chunk-size", type=int, default=0)
    parser.add_argument("--scan-backend", choices=("auto", "python", "native"), default="auto")
    parser.add_argument("--enable-fused-training", action="store_true")
    parser.add_argument("--fused-training-checkpoint-stride", type=int, default=0)
    parser.add_argument("--enable-scan-backward-cuda", action="store_true")
    parser.add_argument(
        "--pairwise-kind",
        choices=("low_rank_bilinear", "diagonal_bilinear", "bilinear", "additive_low_rank", "scaled_cosine"),
        default="low_rank_bilinear",
    )
    parser.add_argument(
        "--route-kind",
        choices=("low_rank_bilinear", "diagonal_bilinear", "bilinear", "additive_low_rank", "query_norm_dot"),
        default="low_rank_bilinear",
    )
    parser.add_argument("--pairwise-rank", type=int, default=64)
    parser.add_argument("--route-rank", type=int, default=64)
    parser.add_argument("--pairwise-heads", type=int, default=1)
    parser.add_argument("--pairwise-frozen-heads", type=int, default=0)
    parser.add_argument("--pairwise-anchor-heads", type=int, default=0)
    parser.add_argument("--pairwise-anchor-kind", choices=("scaled_cosine", "diagonal_bilinear"), default="scaled_cosine")
    parser.add_argument("--route-heads", type=int, default=1)
    parser.add_argument("--route-frozen-heads", type=int, default=0)
    parser.add_argument("--route-anchor-heads", type=int, default=0)
    parser.add_argument("--route-anchor-kind", choices=("fixed_projection", "query_norm_dot", "diagonal_bilinear"), default="fixed_projection")
    parser.add_argument("--unit-norm-values", action="store_true")
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
    parser.add_argument("--optimizer", choices=("adamw", "adamw_fused", "adamw8bit"), default="adamw_fused")
    parser.add_argument("--embedding-lr-mult", type=float, default=1.0)
    parser.add_argument("--embedding-warmup-start", type=int, default=1500)
    parser.add_argument("--embedding-warmup-end", type=int, default=2000)
    parser.add_argument("--b-lr-mult", type=float, default=0.4)
    parser.add_argument("--b-propagation-lr-mult", type=float, default=0.15)
    parser.add_argument("--b-propagation-warmup-start", type=int, default=0)
    parser.add_argument("--b-propagation-warmup-end", type=int, default=1500)
    parser.add_argument("--b-route-lr-mult", type=float, default=0.05)
    parser.add_argument("--b-route-warmup-start", type=int, default=0)
    parser.add_argument("--b-route-warmup-end", type=int, default=1500)
    parser.add_argument("--rnn-aux-head", action="store_true")
    parser.add_argument("--rnn-aux-hidden-dim", type=int, default=1024)
    parser.add_argument("--rnn-aux-layers", type=int, default=2)
    parser.add_argument("--rnn-aux-loss-weight", type=float, default=1.0)
    parser.add_argument("--rnn-aux-loss-weight-final", type=float, default=0.05)
    parser.add_argument("--rnn-aux-decay-start", type=int, default=2000)
    parser.add_argument("--rnn-aux-decay-end", type=int, default=5000)
    parser.add_argument("--rnn-aux-lr-mult", type=float, default=1.0)
    parser.add_argument("--rnn-aux-lr-final-ratio", type=float, default=0.1)
    parser.add_argument("--rnn-aux-batch-size", type=int, default=128)
    parser.add_argument("--rnn-aux-prefetch-batches", type=int, default=32)
    parser.add_argument("--rnn-aux-updates-per-step", type=int, default=1)
    parser.add_argument("--rnn-aux-max-chunks-per-doc", type=int, default=0)
    parser.add_argument("--rnn-pretrain-steps", type=int, default=0)
    parser.add_argument("--rnn-pretrain-log-interval", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--lr-min-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--diagnose-nonfinite-grad", action="store_true")
    parser.add_argument("--diagnose-nonfinite-limit", type=int, default=12)
    parser.add_argument("--stop-on-nonfinite-grad", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument(
        "--eval-start-step",
        type=int,
        default=200,
        help="First training step to run evaluation. Use >1 to avoid expensive startup eval.",
    )
    parser.add_argument(
        "--eval-sample-interval",
        type=int,
        default=1000,
        help="Interval for TensorBoard eval sample text logging. Set larger than eval-interval to reduce overhead.",
    )
    parser.add_argument("--eval-documents", type=int, default=8)
    parser.add_argument("--curriculum-stage1-ratio", type=float, default=0.1)
    parser.add_argument("--curriculum-stage2-ratio", type=float, default=0.4)
    parser.add_argument("--curriculum-stage1-span", type=int, default=2)
    parser.add_argument("--curriculum-stage2-span", type=int, default=4)
    parser.add_argument("--curriculum-stage3-span", type=int, default=8)

    parser.add_argument("--output-root", default="artifacts/training_runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument(
        "--cuda-graph-stage1",
        action="store_true",
        help="Capture stage1 forward+backward into a CUDA graph when shapes are static.",
    )
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


def serialize_flat_document_refs(documents: Sequence[FlatDocumentRef]) -> list[dict[str, int]]:
    return [
        {
            "shard_index": int(reference.shard_index),
            "document_index": int(reference.document_index),
        }
        for reference in documents
    ]


def serialize_tokenized_document_refs(documents: Sequence[TokenizedDocument]) -> list[dict[str, str]]:
    return [
        {
            "kind": document.kind,
            "source": document.source,
        }
        for document in documents
    ]


def configure_native_runtime_flags(args: argparse.Namespace) -> None:
    if args.enable_fused_training:
        os.environ[EXPERIMENTAL_FUSED_TRAINING_ENV] = "1"
    else:
        os.environ.pop(EXPERIMENTAL_FUSED_TRAINING_ENV, None)
    if args.fused_training_checkpoint_stride > 0:
        os.environ[EXPERIMENTAL_FUSED_TRAINING_CHECKPOINT_STRIDE_ENV] = str(
            int(args.fused_training_checkpoint_stride)
        )
    else:
        os.environ.pop(EXPERIMENTAL_FUSED_TRAINING_CHECKPOINT_STRIDE_ENV, None)
    if args.enable_scan_backward_cuda:
        os.environ[EXPERIMENTAL_SCAN_BACKWARD_CUDA_ENV] = "1"
    else:
        os.environ.pop(EXPERIMENTAL_SCAN_BACKWARD_CUDA_ENV, None)


def _parameter_group_name(name: str) -> str:
    if name.startswith("rnn_aux_head."):
        return "rnn_aux"
    if name.startswith("s_module.token_embedding."):
        return "embedding"
    if name.startswith("b_module."):
        if ".route_fn." in name:
            return "b_route"
        if ".pairwise_fn." in name:
            return "b_propagation"
        return "b_module"
    return "default"


def build_parameter_groups(
    model: CausalHierarchicalMemoryLM,
    *,
    learning_rate: float,
    weight_decay: float,
    embedding_lr_mult: float,
    b_lr_mult: float,
    b_propagation_lr_mult: float,
    b_route_lr_mult: float,
) -> tuple[list[dict[str, Any]], tuple[OptimizerParameterGroupConfig, ...]]:
    lr_scales = {
        "default": 1.0,
        "embedding": float(embedding_lr_mult),
        "b_module": float(b_lr_mult),
        "b_propagation": float(b_propagation_lr_mult),
        "b_route": float(b_route_lr_mult),
    }
    grouped: dict[str, dict[str, Any]] = {
        group_name: {
            "params": [],
            "lr": learning_rate * scale,
            "weight_decay": weight_decay,
            "group_name": group_name,
            "lr_scale": scale,
        }
        for group_name, scale in lr_scales.items()
    }
    counts = {group_name: 0 for group_name in lr_scales}
    seen: set[int] = set()
    named_parameter_sources: list[tuple[str, torch.nn.Parameter]] = list(model.named_parameters())
    for parameter_name, parameter in named_parameter_sources:
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        group_name = _parameter_group_name(parameter_name)
        grouped[group_name]["params"].append(parameter)
        counts[group_name] += int(parameter.numel())
    param_groups = [group for group in grouped.values() if group["params"]]
    group_configs = tuple(
        OptimizerParameterGroupConfig(
            name=group_name,
            lr_scale=lr_scales[group_name],
            weight_decay=weight_decay,
            parameter_count=counts[group_name],
        )
        for group_name in ("default", "embedding", "b_module", "b_propagation", "b_route", "rnn_aux")
        if group_name in counts and counts[group_name] > 0
    )
    return param_groups, group_configs


def build_optimizer(
    param_groups: Sequence[dict[str, Any]],
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> torch.optim.Optimizer:
    optimizer_name = name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw_fused":
        if device.type == "cuda":
            try:
                return torch.optim.AdamW(
                    param_groups,
                    lr=learning_rate,
                    weight_decay=weight_decay,
                    fused=True,
                )
            except TypeError:
                pass
        return torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("bitsandbytes is required for --optimizer adamw8bit.") from exc
        return AdamW8bit(param_groups, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name!r}")


def build_rnn_aux_optimizer(
    model: CausalHierarchicalMemoryLM,
    rnn_aux_head: SharedEmbeddingRnnAuxHead | None,
    *,
    name: str,
    learning_rate: float,
    weight_decay: float,
    rnn_aux_lr_mult: float,
    device: torch.device,
) -> tuple[torch.optim.Optimizer | None, tuple[OptimizerParameterGroupConfig, ...]]:
    if rnn_aux_head is None:
        return None, ()
    head_parameters = [parameter for parameter in rnn_aux_head.parameters() if parameter.requires_grad]
    embedding_parameters = [
        parameter for parameter in model.s_module.token_embedding.parameters() if parameter.requires_grad
    ]
    parameters: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for parameter in head_parameters + embedding_parameters:
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        parameters.append(parameter)
    if not parameters:
        return None, ()
    lr_scale = float(rnn_aux_lr_mult)
    param_groups = [
        {
            "params": parameters,
            "lr": learning_rate * lr_scale,
            "weight_decay": weight_decay,
            "group_name": "rnn_aux",
            "lr_scale": lr_scale,
        }
    ]
    optimizer = build_optimizer(
        param_groups,
        name=name,
        learning_rate=learning_rate * lr_scale,
        weight_decay=weight_decay,
        device=device,
    )
    return optimizer, (
        OptimizerParameterGroupConfig(
            name="rnn_aux",
            lr_scale=lr_scale,
            weight_decay=weight_decay,
            parameter_count=sum(int(parameter.numel()) for parameter in parameters),
        ),
    )


def main() -> None:
    args = parse_args()
    if args.pretokenize_workers < 0:
        raise ValueError("pretokenize-workers must be non-negative.")
    if args.pretokenized_load_workers <= 0:
        raise ValueError("pretokenized-load-workers must be positive.")
    if args.flat_integrity_workers <= 0:
        raise ValueError("flat-integrity-workers must be positive.")
    if args.pretokenized_max_loaded_shards < 0:
        raise ValueError("pretokenized-max-loaded-shards must be non-negative.")
    if args.pretokenized_active_shards_per_bucket <= 0:
        raise ValueError("pretokenized-active-shards-per-bucket must be positive.")
    if args.pretokenized_shard_rotation_interval <= 0:
        raise ValueError("pretokenized-shard-rotation-interval must be positive.")
    configure_native_runtime_flags(args)
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
    if args.curriculum_stage1_span <= 0 or args.curriculum_stage2_span <= 0 or args.curriculum_stage3_span <= 0:
        raise ValueError("curriculum-stage spans must be positive.")
    if min(
        args.embedding_lr_mult,
        args.b_lr_mult,
        args.b_propagation_lr_mult,
        args.b_route_lr_mult,
        args.rnn_aux_lr_mult,
        args.rnn_aux_lr_final_ratio,
        args.rnn_aux_loss_weight,
        args.rnn_aux_loss_weight_final,
    ) < 0.0:
        raise ValueError("Learning-rate multipliers must be non-negative.")
    if args.rnn_aux_layers <= 0:
        raise ValueError("rnn-aux-layers must be positive.")
    if args.rnn_aux_batch_size <= 0:
        raise ValueError("rnn-aux-batch-size must be positive.")
    if args.rnn_aux_prefetch_batches <= 0:
        raise ValueError("rnn-aux-prefetch-batches must be positive.")
    if args.rnn_aux_updates_per_step <= 0:
        raise ValueError("rnn-aux-updates-per-step must be positive.")
    if args.rnn_aux_max_chunks_per_doc < 0:
        raise ValueError("rnn-aux-max-chunks-per-doc must be non-negative.")
    if args.rnn_pretrain_steps < 0:
        raise ValueError("rnn-pretrain-steps must be non-negative.")
    if args.rnn_pretrain_log_interval <= 0:
        raise ValueError("rnn-pretrain-log-interval must be positive.")
    if args.tokenizer == "hf_auto" and not args.hf_tokenizer_model and not (args.pretokenized_dir or args.pretokenized_path):
        raise ValueError("--hf-tokenizer-model is required when --tokenizer hf_auto is used without pretokenized input.")
    if args.hf_embedding_model and args.tokenizer != "hf_auto" and not (args.pretokenized_dir or args.pretokenized_path):
        raise ValueError("--hf-embedding-model currently requires --tokenizer hf_auto.")
    if args.embedding_warmup_start < 0 or args.embedding_warmup_end < 0:
        raise ValueError("embedding warmup steps must be non-negative.")
    if args.embedding_warmup_end < args.embedding_warmup_start:
        raise ValueError("embedding-warmup-end must be greater than or equal to embedding-warmup-start.")
    if min(
        args.b_propagation_warmup_start,
        args.b_propagation_warmup_end,
        args.b_route_warmup_start,
        args.b_route_warmup_end,
    ) < 0:
        raise ValueError("B warmup steps must be non-negative.")
    if args.b_propagation_warmup_end < args.b_propagation_warmup_start:
        raise ValueError("b-propagation-warmup-end must be greater than or equal to b-propagation-warmup-start.")
    if args.b_route_warmup_end < args.b_route_warmup_start:
        raise ValueError("b-route-warmup-end must be greater than or equal to b-route-warmup-start.")
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
            max_loaded_shards=max(0, int(args.pretokenized_max_loaded_shards)),
            integrity_mode=str(args.flat_integrity_mode),
            integrity_workers=max(1, int(args.flat_integrity_workers)),
        )
        if args.preload_flat_shards:
            preload_flat_pretokenized_directory(flat_collection)
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
            training_text = build_training_text(serialized_documents, tokenizer_label=args.tokenizer)
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
            user_defined_symbols=CAUSAL_DOC_SPECIAL_TOKENS if args.tokenizer == "byte_bpe" else (),
            hf_tokenizer_model=args.hf_tokenizer_model,
            hf_trust_remote_code=args.hf_trust_remote_code,
        )
        tokenizer_model_path = None if tokenizer_path is None else str(tokenizer_path)
        special_token_ids = build_special_token_id_map(vocab, tokenizer_label=tokenizer_label)
        workers = min(args.pretokenize_workers, max(1, torch.get_num_threads()))
        documents = tokenize_documents(
            serialized_documents,
            vocab=vocab,
            seq_len=args.seq_len,
            special_token_ids=special_token_ids,
            workers=workers,
            tokenizer_label=tokenizer_label,
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
        print("startup | split_train_val_flat_done", flush=True)
        train_bucket_summary = summarize_flat_document_buckets(flat_collection, train_documents_flat)
        val_bucket_summary = summarize_flat_document_buckets(flat_collection, val_documents_flat)
        print("startup | summarize_flat_buckets_done", flush=True)
        steps_per_epoch = estimate_stage_weighted_steps_per_epoch_flat(
            collection=flat_collection,
            documents=train_documents_flat,
            stage1_ratio=args.curriculum_stage1_ratio,
            stage2_ratio=args.curriculum_stage2_ratio,
            stage1_batch_size=stage1_batch_size,
            stage2_batch_size=stage2_batch_size,
            stage3_batch_size=stage3_batch_size,
        )
        print("startup | estimate_steps_flat_done", flush=True)
        fixed_eval_documents_flat = sample_flat_documents_uniform_by_bucket(
            flat_collection,
            val_documents_flat,
            sample_count=min(args.eval_documents, len(val_documents_flat)),
        )
        print("startup | fixed_eval_documents_flat_done", flush=True)
        fixed_eval_sample_documents_flat = sample_flat_documents_for_logging(
            flat_collection,
            val_documents_flat,
            sample_count=1,
        )
        print("startup | fixed_eval_sample_documents_flat_done", flush=True)
    else:
        assert documents is not None
        train_documents, val_documents = split_train_val_documents(documents, train_fraction=args.train_fraction)
        print("startup | split_train_val_done", flush=True)
        train_bucket_summary = summarize_document_buckets(train_documents)
        val_bucket_summary = summarize_document_buckets(val_documents)
        print("startup | summarize_buckets_done", flush=True)
        steps_per_epoch = estimate_stage_weighted_steps_per_epoch(
            documents=train_documents,
            stage1_ratio=args.curriculum_stage1_ratio,
            stage2_ratio=args.curriculum_stage2_ratio,
            stage1_batch_size=stage1_batch_size,
            stage2_batch_size=stage2_batch_size,
            stage3_batch_size=stage3_batch_size,
        )
        print("startup | estimate_steps_done", flush=True)
        fixed_eval_documents = sample_documents_uniform_by_bucket(
            val_documents,
            sample_count=min(args.eval_documents, len(val_documents)),
        )
        print("startup | fixed_eval_documents_done", flush=True)
        fixed_eval_sample_documents = sample_documents_uniform_by_bucket(
            val_documents,
            sample_count=1,
        )
        print("startup | fixed_eval_sample_documents_done", flush=True)
    decode_vocab = load_decode_vocab(
        tokenizer_label=tokenizer_label,
        tokenizer_model_path=tokenizer_model_path,
    )
    if args.match_dim_to_hf_embedding:
        hf_dim_source = args.hf_embedding_model or args.hf_tokenizer_model
        if not hf_dim_source:
            raise ValueError("--match-dim-to-hf-embedding requires --hf-embedding-model or --hf-tokenizer-model.")
        args.dim = infer_hf_hidden_size(
            model_name_or_path=hf_dim_source,
            trust_remote_code=args.hf_trust_remote_code,
        )
        print(f"hf_embedding_dim | source={hf_dim_source} | dim={args.dim}", flush=True)
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))

    model = CausalHierarchicalMemoryLM(
        vocab_size=vocab_size,
        dim=args.dim,
        max_seq_len=args.seq_len,
        s_layers=args.s_layers,
        memory_slots=tuple(args.memory_slots),
        memory_update_intervals=None if args.memory_update_intervals is None else tuple(args.memory_update_intervals),
        prediction_layers=args.prediction_layers,
        s_window=args.s_window,
        s_microbatch_size=None if args.s_microbatch_size <= 0 else args.s_microbatch_size,
        prediction_window=args.prediction_window,
        checkpoint_sequence_layers=args.checkpoint_sequence_layers,
        checkpoint_prediction_layers=args.checkpoint_prediction_layers,
        memory_topk=args.memory_topk,
        scan_checkpoint_chunk_size=None if args.scan_checkpoint_chunk_size <= 0 else args.scan_checkpoint_chunk_size,
        scan_backend=args.scan_backend,
        pairwise_kind=args.pairwise_kind,
        route_kind=args.route_kind,
        pairwise_rank=args.pairwise_rank,
        route_rank=args.route_rank,
        pairwise_heads=args.pairwise_heads,
        pairwise_frozen_heads=args.pairwise_frozen_heads,
        pairwise_anchor_heads=args.pairwise_anchor_heads,
        pairwise_anchor_kind=args.pairwise_anchor_kind,
        route_heads=args.route_heads,
        route_frozen_heads=args.route_frozen_heads,
        route_anchor_heads=args.route_anchor_heads,
        route_anchor_kind=args.route_anchor_kind,
        implementation=args.implementation,
        unit_norm_values=args.unit_norm_values,
        knowledge_nodes=args.knowledge_nodes,
        knowledge_route_topk=None if args.knowledge_route_topk <= 0 else args.knowledge_route_topk,
        knowledge_propagation_topk=None if args.knowledge_propagation_topk <= 0 else args.knowledge_propagation_topk,
        knowledge_propagation_layers=args.knowledge_propagation_layers,
    ).to(device)
    if args.hf_embedding_model:
        if not isinstance(decode_vocab, HFTokenizerVocab):
            raise ValueError("--hf-embedding-model requires --tokenizer hf_auto.")
        embedding_init_stats = initialize_model_embedding_from_hf(
            model=model,
            vocab=decode_vocab,
            model_name_or_path=args.hf_embedding_model,
            trust_remote_code=args.hf_trust_remote_code,
        )
        print(
            f"hf_embedding_init | model={args.hf_embedding_model} | copied={embedding_init_stats['copied']:,} | skipped={embedding_init_stats['skipped']:,}",
            flush=True,
        )
    print("startup | model_init_done", flush=True)
    rnn_aux_head = None
    if args.rnn_aux_head:
        rnn_aux_head = SharedEmbeddingRnnAuxHead(
            embedding_dim=args.dim,
            vocab_size=vocab_size,
            hidden_dim=None if args.rnn_aux_hidden_dim <= 0 else args.rnn_aux_hidden_dim,
            num_layers=args.rnn_aux_layers,
        ).to(device)
    print("startup | rnn_aux_init_done", flush=True)
    parameter_count = count_parameters(model)
    rnn_aux_parameter_count = 0 if rnn_aux_head is None else count_parameters(rnn_aux_head)
    print(
        f"model=causal_memory_doc | params={parameter_count:,} | dim={args.dim} | seq_len={args.seq_len} | "
        f"s_window={args.s_window} | s_microbatch_size={args.s_microbatch_size} | "
        f"scan_backend={args.scan_backend} | scan_checkpoint_chunk_size={args.scan_checkpoint_chunk_size} | "
        f"memory_slots={args.memory_slots} | memory_update_intervals={args.memory_update_intervals} | knowledge_nodes={args.knowledge_nodes} | "
        f"unit_norm_values={args.unit_norm_values} | "
        f"optimizer={args.optimizer} | checkpoint_sequence={args.checkpoint_sequence_layers} | "
        f"checkpoint_prediction={args.checkpoint_prediction_layers}",
        flush=True,
    )
    if rnn_aux_head is not None:
        print(
            f"rnn_aux | params={rnn_aux_parameter_count:,} | hidden_dim={rnn_aux_head.hidden_dim} | "
            f"layers={rnn_aux_head.num_layers} | batch_size={args.rnn_aux_batch_size} | "
            f"updates_per_step={args.rnn_aux_updates_per_step} | loss_weight={args.rnn_aux_loss_weight} | "
            f"loss_weight_final={args.rnn_aux_loss_weight_final} | decay_start={args.rnn_aux_decay_start} | "
            f"decay_end={args.rnn_aux_decay_end}",
            flush=True,
        )
    print(
        f"native_runtime | fused_training={args.enable_fused_training} | "
        f"fused_training_checkpoint_stride={args.fused_training_checkpoint_stride} | "
        f"scan_backward_cuda={args.enable_scan_backward_cuda}",
        flush=True,
    )
    print(
        f"curriculum_batches | stage1=batch{stage1_batch_size}/ga{stage1_grad_accum_steps} | "
        f"stage2=batch{stage2_batch_size}/ga{stage2_grad_accum_steps} | "
        f"stage3=batch{stage3_batch_size}/ga{stage3_grad_accum_steps}",
        flush=True,
    )
    print(
        f"curriculum_spans | stage1={args.curriculum_stage1_span} | "
        f"stage2={args.curriculum_stage2_span} | stage3={args.curriculum_stage3_span}",
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
            "rnn_aux_parameter_count": rnn_aux_parameter_count,
            "corpus_metadata": corpus_metadata,
            "fixed_eval_documents": (
                serialize_flat_document_refs(fixed_eval_documents_flat)
                if flat_collection is not None
                else serialize_tokenized_document_refs(fixed_eval_documents)
            ),
            "fixed_eval_sample_documents": (
                serialize_flat_document_refs(fixed_eval_sample_documents_flat)
                if flat_collection is not None
                else serialize_tokenized_document_refs(fixed_eval_sample_documents)
            ),
        },
    )

    writer = None
    if args.tensorboard:
        if SummaryWriter is None:
            raise ImportError("tensorboard is not installed.")
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))

    param_groups, param_group_configs = build_parameter_groups(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        embedding_lr_mult=args.embedding_lr_mult,
        b_lr_mult=args.b_lr_mult,
        b_propagation_lr_mult=args.b_propagation_lr_mult,
        b_route_lr_mult=args.b_route_lr_mult,
    )
    rnn_aux_optimizer, rnn_aux_group_configs = build_rnn_aux_optimizer(
        model,
        rnn_aux_head,
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rnn_aux_lr_mult=args.rnn_aux_lr_mult,
        device=device,
    )
    print(
        "optimizer_groups | "
        + " | ".join(
            f"{group.name}:count={group.parameter_count:,},lr_scale={group.lr_scale:.4f}"
            for group in (param_group_configs + rnn_aux_group_configs)
        ),
        flush=True,
    )
    optimizer = build_optimizer(
        param_groups,
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
    )
    print("startup | main_optimizer_done", flush=True)
    scaler = torch.cuda.amp.GradScaler() if args.precision == "fp16" and device.type == "cuda" else None
    if flat_collection is not None:
        batcher: DocumentChunkBatcher | FlatDocumentChunkBatcher = FlatDocumentChunkBatcher(
            flat_collection,
            train_documents_flat,
            batch_size=stage1_batch_size,
            device=device,
            active_shards_per_bucket=args.pretokenized_active_shards_per_bucket,
            shard_rotation_interval=args.pretokenized_shard_rotation_interval,
        )
    else:
        batcher = DocumentChunkBatcher(train_documents, batch_size=stage1_batch_size, device=device)
    print("startup | batcher_done", flush=True)
    prefetcher = AsyncDocumentBatchPrefetcher(
        batcher,
        device=device,
        prefetch_batches=max(1, int(args.prefetch_batches)),
    )
    print("startup | prefetcher_done", flush=True)
    rnn_prefetcher: AsyncDocumentBatchPrefetcher | None = None
    if rnn_aux_head is not None:
        if flat_collection is not None:
            rnn_batcher = FlatSequentialDocumentBatcher(
                flat_collection,
                train_documents_flat,
                batch_size=args.rnn_aux_batch_size,
                device=device,
                max_chunks_per_document=args.rnn_aux_max_chunks_per_doc,
                full_loss=True,
            )
        else:
            rnn_batcher = DocumentChunkBatcher(
                train_documents,
                batch_size=args.rnn_aux_batch_size,
                device=device,
            )
        rnn_prefetcher = AsyncDocumentBatchPrefetcher(
            rnn_batcher,
            device=device,
            prefetch_batches=max(1, int(args.rnn_aux_prefetch_batches)),
        )
        print("startup | rnn_prefetcher_done", flush=True)

    run_rnn_pretrain = rnn_aux_head is not None and args.rnn_pretrain_steps > 0
    run_concurrent_rnn = rnn_aux_head is not None and not run_rnn_pretrain

    history_rows: list[dict[str, Any]] = []
    train_memory_state: tuple[Any, ...] | None = None
    active_stage_name: str | None = None
    start_step = 0
    best_val_loss = float("inf")
    if args.resume_checkpoint:
        checkpoint_path = Path(args.resume_checkpoint)
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if rnn_aux_head is not None and checkpoint.get("rnn_aux_state_dict") is not None:
            rnn_aux_head.load_state_dict(checkpoint["rnn_aux_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if rnn_aux_optimizer is not None and checkpoint.get("rnn_aux_optimizer_state_dict") is not None:
            rnn_aux_optimizer.load_state_dict(checkpoint["rnn_aux_optimizer_state_dict"])
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
    if rnn_aux_optimizer is not None:
        rnn_aux_optimizer.zero_grad(set_to_none=True)
    start_time = time.time()
    grad_clip_parameters = list(model.parameters())
    grad_clip_named_parameters = [
        (parameter_name, parameter)
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    rnn_grad_clip_parameters: list[torch.nn.Parameter] = []
    if rnn_aux_head is not None:
        rnn_grad_clip_parameters.extend(parameter for parameter in rnn_aux_head.parameters() if parameter.requires_grad)
        rnn_grad_clip_parameters.extend(
            parameter for parameter in model.s_module.token_embedding.parameters() if parameter.requires_grad
        )

    if run_rnn_pretrain and rnn_aux_optimizer is not None and rnn_prefetcher is not None:
        print(
            f"rnn_pretrain | steps={args.rnn_pretrain_steps} | batch_size={args.rnn_aux_batch_size} | "
            f"log_interval={args.rnn_pretrain_log_interval}",
            flush=True,
        )
        for rnn_pretrain_step in range(1, args.rnn_pretrain_steps + 1):
            rnn_batch = rnn_prefetcher.next_batch()
            rnn_aux_optimizer.zero_grad(set_to_none=True)
            rnn_loss, rnn_loss_value = run_rnn_aux_model(
                model,
                rnn_aux_head,
                rnn_batch,
                precision=args.precision,
                grad_enabled=True,
            )
            if not bool(torch.isfinite(rnn_loss.detach()).item()):
                raise RuntimeError(f"Non-finite RNN pretrain loss at step {rnn_pretrain_step}.")
            if scaler is not None:
                scaler.scale(rnn_loss).backward()
                scaler.unscale_(rnn_aux_optimizer)
            else:
                rnn_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                rnn_grad_clip_parameters,
                args.grad_clip,
                error_if_nonfinite=True,
            )
            if scaler is not None:
                scaler.step(rnn_aux_optimizer)
                scaler.update()
            else:
                rnn_aux_optimizer.step()
            if writer is not None:
                writer.add_scalar("pretrain/rnn_loss", float(rnn_loss_value), rnn_pretrain_step)
            if (
                rnn_pretrain_step == 1
                or rnn_pretrain_step == args.rnn_pretrain_steps
                or rnn_pretrain_step % args.rnn_pretrain_log_interval == 0
            ):
                print(
                    f"rnn_pretrain_progress | step={rnn_pretrain_step}/{args.rnn_pretrain_steps} | "
                    f"loss={rnn_loss_value:.4f}",
                    flush=True,
                )
        rnn_aux_optimizer.zero_grad(set_to_none=True)
        rnn_prefetcher.close()
        rnn_prefetcher = None
        print("rnn_pretrain | complete", flush=True)

    stage1_cuda_graph_runner: Stage1CudaGraphRunner | Stage1ForwardCudaGraphRunner | None = None

    for step in range(start_step + 1, total_steps + 1):
        stage = resolve_curriculum_stage(
            step=step,
            total_steps=total_steps,
            stage1_ratio=args.curriculum_stage1_ratio,
            stage2_ratio=args.curriculum_stage2_ratio,
            stage1_span=args.curriculum_stage1_span,
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
            stage1_cuda_graph_runner = None
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
        rnn_aux_lr_scale = 0.0
        rnn_aux_weight = 0.0
        if run_concurrent_rnn:
            rnn_aux_lr_scale = compute_decayed_scalar(
                step=step,
                start_step=max(0, int(args.rnn_aux_decay_start)),
                end_step=max(int(args.rnn_aux_decay_start), int(args.rnn_aux_decay_end)),
                initial_value=1.0,
                final_value=float(args.rnn_aux_lr_final_ratio),
            )
            rnn_aux_weight = compute_decayed_scalar(
                step=step,
                start_step=max(0, int(args.rnn_aux_decay_start)),
                end_step=max(int(args.rnn_aux_decay_start), int(args.rnn_aux_decay_end)),
                initial_value=float(args.rnn_aux_loss_weight),
                final_value=float(args.rnn_aux_loss_weight_final),
            )
        embedding_lr_scale = compute_warmup_scalar(
            step=step,
            start_step=max(0, int(args.embedding_warmup_start)),
            end_step=max(int(args.embedding_warmup_start), int(args.embedding_warmup_end)),
            initial_value=0.0,
            final_value=1.0,
        )
        b_propagation_lr_scale = compute_warmup_scalar(
            step=step,
            start_step=max(0, int(args.b_propagation_warmup_start)),
            end_step=max(int(args.b_propagation_warmup_start), int(args.b_propagation_warmup_end)),
            initial_value=0.0,
            final_value=1.0,
        )
        b_route_lr_scale = compute_warmup_scalar(
            step=step,
            start_step=max(0, int(args.b_route_warmup_start)),
            end_step=max(int(args.b_route_warmup_start), int(args.b_route_warmup_end)),
            initial_value=0.0,
            final_value=1.0,
        )
        for param_group in optimizer.param_groups:
            group_lr = lr * float(param_group.get("lr_scale", 1.0))
            if param_group.get("group_name") == "embedding":
                group_lr *= embedding_lr_scale
            elif param_group.get("group_name") == "b_propagation":
                group_lr *= b_propagation_lr_scale
            elif param_group.get("group_name") == "b_route":
                group_lr *= b_route_lr_scale
            param_group["lr"] = group_lr
        if rnn_aux_optimizer is not None:
            for param_group in rnn_aux_optimizer.param_groups:
                group_lr = args.learning_rate * float(param_group.get("lr_scale", 1.0))
                group_lr *= rnn_aux_lr_scale
                param_group["lr"] = group_lr

        current_memory_state = None if stage.freeze_memory else train_memory_state
        span_losses: list[float] = []
        last_span_loss_tensor: torch.Tensor | None = None
        loss_is_finite = True
        used_cuda_graph = False
        for span_index in range(stage.document_span):
            batch = override_batch_reset(
                prefetcher.next_batch(),
                reset_all=stage.freeze_memory,
            )
            use_stage1_cuda_graph = (
                args.cuda_graph_stage1
                and device.type == "cuda"
                and scaler is None
                and not run_concurrent_rnn
                and stage.name == "stage1"
                and stage.document_span == 1
                and stage.grad_accum_steps == 1
                and not stage.freeze_memory
                and span_index == 0
            )
            if use_stage1_cuda_graph:
                if (
                    stage1_cuda_graph_runner is None
                    or tuple(stage1_cuda_graph_runner.static_batch.context.shape) != tuple(batch.context.shape)
                ):
                    stage1_cuda_graph_runner = Stage1ForwardCudaGraphRunner(
                        model=model,
                        batch=batch,
                        precision=args.precision,
                    )
                    print("cuda_graph | mode=forward_only", flush=True)
                loss, main_loss_value, next_memory_state = stage1_cuda_graph_runner.replay(
                    batch,
                    memory_state=current_memory_state,
                )
                used_cuda_graph = isinstance(stage1_cuda_graph_runner, Stage1CudaGraphRunner)
            else:
                loss, main_loss_value, next_memory_state = run_model(
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
            span_losses.append(main_loss_value)
            last_span_loss_tensor = loss
            current_memory_state = None if stage.freeze_memory else next_memory_state

        if loss_is_finite and last_span_loss_tensor is not None and not used_cuda_graph:
            total_span_loss = last_span_loss_tensor / stage.grad_accum_steps
            if scaler is not None:
                scaler.scale(total_span_loss).backward()
            else:
                total_span_loss.backward()
            del total_span_loss
            train_memory_state = None if stage.freeze_memory else detach_memory_state(current_memory_state)
            if not memory_state_is_finite(train_memory_state):
                print(f"warning | step={step} | non-finite memory state; resetting memory", flush=True)
                train_memory_state = None
        elif loss_is_finite and used_cuda_graph:
            train_memory_state = detach_memory_state(current_memory_state)
            if not memory_state_is_finite(train_memory_state):
                print(f"warning | step={step} | non-finite memory state; resetting memory", flush=True)
                train_memory_state = None
        else:
            optimizer.zero_grad(set_to_none=True)
            if rnn_aux_optimizer is not None:
                rnn_aux_optimizer.zero_grad(set_to_none=True)
            train_memory_state = None

        should_step_optimizer = step % stage.grad_accum_steps == 0 or step == total_steps
        if loss_is_finite and should_step_optimizer:
            if scaler is not None:
                scaler.unscale_(optimizer)
            try:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        grad_clip_parameters,
                        args.grad_clip,
                        error_if_nonfinite=True,
                    ).item()
                )
            except RuntimeError as exc:
                grad_norm = float("nan")
                print(f"warning | step={step} | non-finite grad norm; skipping optimizer step | {exc}", flush=True)
                if args.diagnose_nonfinite_grad:
                    gradient_diagnostics = summarize_nonfinite_gradients(
                        grad_clip_named_parameters,
                        limit=args.diagnose_nonfinite_limit,
                    )
                    if gradient_diagnostics:
                        print(
                            "diagnose_nonfinite_grad | "
                            + " | ".join(gradient_diagnostics),
                            flush=True,
                        )
                    else:
                        gradient_extremes = summarize_gradient_extremes(
                            grad_clip_named_parameters,
                            limit=args.diagnose_nonfinite_limit,
                        )
                        if gradient_extremes:
                            print(
                                "diagnose_grad_extremes | "
                                + " | ".join(gradient_extremes),
                                flush=True,
                            )
                    memory_diagnostics = summarize_nonfinite_memory_state(current_memory_state)
                    if memory_diagnostics:
                        print(
                            "diagnose_nonfinite_memory | "
                            + " | ".join(memory_diagnostics),
                            flush=True,
                        )
                    else:
                        print("diagnose_nonfinite_memory | all_finite", flush=True)
                optimizer.zero_grad(set_to_none=True)
                train_memory_state = None
                if args.stop_on_nonfinite_grad:
                    raise
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = float("nan")

        train_loss = float(span_losses[-1] if span_losses else float("nan"))
        train_aux_loss = 0.0
        avg_rnn_batch_seconds = 0.0
        rnn_prefetch_queue_size = 0
        if (
            run_concurrent_rnn
            and rnn_aux_head is not None
            and rnn_aux_optimizer is not None
            and rnn_prefetcher is not None
            and rnn_aux_weight > 0.0
        ):
            rnn_aux_optimizer.zero_grad(set_to_none=True)
            rnn_losses: list[float] = []
            rnn_loss_tensors: list[torch.Tensor] = []
            rnn_loss_is_finite = True
            for _ in range(max(1, int(args.rnn_aux_updates_per_step))):
                rnn_batch = rnn_prefetcher.next_batch()
                rnn_loss, rnn_loss_value = run_rnn_aux_model(
                    model,
                    rnn_aux_head,
                    rnn_batch,
                    precision=args.precision,
                    grad_enabled=True,
                )
                if not bool(torch.isfinite(rnn_loss.detach()).item()):
                    rnn_loss_is_finite = False
                    print(
                        f"warning | step={step} | non-finite rnn aux loss; skipping rnn optimizer step",
                        flush=True,
                    )
                    break
                rnn_losses.append(rnn_loss_value)
                rnn_loss_tensors.append(rnn_loss * float(rnn_aux_weight))
            if rnn_loss_is_finite and rnn_loss_tensors:
                total_rnn_aux_loss = sum(rnn_loss_tensors) / max(1, len(rnn_loss_tensors))
                if scaler is not None:
                    scaler.scale(total_rnn_aux_loss).backward()
                    scaler.unscale_(rnn_aux_optimizer)
                else:
                    total_rnn_aux_loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(
                        rnn_grad_clip_parameters,
                        args.grad_clip,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    print(
                        f"warning | step={step} | non-finite rnn grad norm; skipping rnn optimizer step | {exc}",
                        flush=True,
                    )
                    rnn_aux_optimizer.zero_grad(set_to_none=True)
                else:
                    if scaler is not None:
                        scaler.step(rnn_aux_optimizer)
                    else:
                        rnn_aux_optimizer.step()
                    rnn_aux_optimizer.zero_grad(set_to_none=True)
                    train_aux_loss = float(sum(rnn_losses) / max(1, len(rnn_losses)))
            else:
                rnn_aux_optimizer.zero_grad(set_to_none=True)
            avg_rnn_batch_seconds, rnn_prefetch_queue_size = rnn_prefetcher.stats()
        avg_cpu_batch_seconds, prefetch_queue_size = prefetcher.stats()
        if writer is not None:
            writer.add_scalar("train/loss", train_loss, step)
            writer.add_scalar("train/rnn_aux_loss", train_aux_loss, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/embedding_lr_scale", embedding_lr_scale, step)
            writer.add_scalar("train/b_propagation_lr_scale", b_propagation_lr_scale, step)
            writer.add_scalar("train/b_route_lr_scale", b_route_lr_scale, step)
            writer.add_scalar("train/rnn_aux_lr_scale", rnn_aux_lr_scale, step)
            writer.add_scalar("train/rnn_aux_weight", rnn_aux_weight, step)
            writer.add_scalar("train/document_span", stage.document_span, step)
            writer.add_scalar("train/cpu_batch_ms", avg_cpu_batch_seconds * 1000.0, step)
            writer.add_scalar("train/prefetch_queue_size", prefetch_queue_size, step)
            writer.add_scalar("train/rnn_cpu_batch_ms", avg_rnn_batch_seconds * 1000.0, step)
            writer.add_scalar("train/rnn_prefetch_queue_size", rnn_prefetch_queue_size, step)
            if not math.isnan(grad_norm):
                writer.add_scalar("train/grad_norm", grad_norm, step)
            for bucket_name, bucket_weight in bucket_weights.items():
                writer.add_scalar(f"train_bucket_weight/{bucket_name}", bucket_weight, step)

        if step == 1 or step % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"progress | step={step:5d}/{total_steps} | stage={stage.name} | span={stage.document_span} | "
                f"train_loss={train_loss:.4f} | rnn_aux_loss={train_aux_loss:.4f} | lr={lr:.6g} | "
                f"embed_lr_scale={embedding_lr_scale:.3f} | "
                f"edge_lr_scale={b_propagation_lr_scale:.3f} | route_lr_scale={b_route_lr_scale:.3f} | "
                f"cpu_batch_ms={avg_cpu_batch_seconds * 1000.0:.1f} | prefetch_q={prefetch_queue_size} | "
                f"rnn_cpu_batch_ms={avg_rnn_batch_seconds * 1000.0:.1f} | rnn_prefetch_q={rnn_prefetch_queue_size} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

        val_loss = None
        val_rnn_aux_loss = None
        should_eval = (
            step == total_steps
            or (
                step >= max(1, int(args.eval_start_step))
                and step % args.eval_interval == 0
            )
        )
        if should_eval:
            if flat_collection is not None:
                val_loss, val_rnn_aux_loss = estimate_eval_loss_flat(
                    model,
                    flat_collection,
                    fixed_eval_documents_flat,
                    eval_documents=len(fixed_eval_documents_flat),
                    device=device,
                    precision=args.precision,
                    rnn_aux_head=rnn_aux_head if run_concurrent_rnn else None,
                )
            else:
                val_loss, val_rnn_aux_loss = estimate_eval_loss(
                    model,
                    fixed_eval_documents,
                    eval_documents=len(fixed_eval_documents),
                    device=device,
                    precision=args.precision,
                    rnn_aux_head=rnn_aux_head if run_concurrent_rnn else None,
                )
            val_ppl = perplexity_from_loss(val_loss)
            val_rnn_aux_text = "n/a" if val_rnn_aux_loss is None else f"{val_rnn_aux_loss:.4f}"
            print(
                f"eval | step={step} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_rnn_aux_loss={val_rnn_aux_text} | val_ppl={val_ppl:.2f}",
                flush=True,
            )
            if writer is not None:
                writer.add_scalar("eval/val_loss", val_loss, step)
                if val_rnn_aux_loss is not None:
                    writer.add_scalar("eval/rnn_aux_loss", float(val_rnn_aux_loss), step)
                writer.add_scalar("eval/val_ppl", val_ppl, step)
                should_log_eval_samples = (
                    step == total_steps
                    or (
                        args.eval_sample_interval > 0
                        and step >= max(1, int(args.eval_start_step))
                        and step % args.eval_sample_interval == 0
                    )
                )
                if should_log_eval_samples:
                    if flat_collection is not None:
                        log_eval_samples_to_tensorboard_flat(
                            writer,
                            model,
                            flat_collection,
                            fixed_eval_sample_documents_flat,
                            vocab=decode_vocab,
                            device=device,
                            precision=args.precision,
                            step=step,
                            max_samples=1,
                        )
                    else:
                        log_eval_samples_to_tensorboard(
                            writer,
                            model,
                            fixed_eval_sample_documents,
                            vocab=decode_vocab,
                            device=device,
                            precision=args.precision,
                            step=step,
                            max_samples=1,
                        )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    checkpoints_dir / "best.pt",
                    model=model,
                    rnn_aux_head=rnn_aux_head,
                    optimizer=optimizer,
                    rnn_aux_optimizer=rnn_aux_optimizer,
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
                rnn_aux_head=rnn_aux_head,
                optimizer=optimizer,
                rnn_aux_optimizer=rnn_aux_optimizer,
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
            "train_rnn_aux_loss": train_aux_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "embedding_lr_scale": embedding_lr_scale,
            "b_propagation_lr_scale": b_propagation_lr_scale,
            "b_route_lr_scale": b_route_lr_scale,
            "rnn_aux_lr_scale": rnn_aux_lr_scale,
            "rnn_aux_weight": rnn_aux_weight,
            "val_loss": val_loss,
            "val_rnn_aux_loss": val_rnn_aux_loss,
            "val_ppl": None if val_loss is None else perplexity_from_loss(val_loss),
        }
        history_rows.append(row)
        append_jsonl(run_dir / "history.jsonl", row)

    write_csv_rows(run_dir / "history.csv", history_rows)
    prefetcher.close()
    if rnn_prefetcher is not None:
        rnn_prefetcher.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
