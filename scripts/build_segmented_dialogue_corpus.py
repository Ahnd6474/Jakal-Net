from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from lm_experiment_utils import write_json


_DISPLAY_MATH_PATTERN = re.compile(r"(?s)\$\$.*?\$\$")
_BRACKET_MATH_PATTERN = re.compile(r"(?s)\\\[.*?\\\]")
_INLINE_PAREN_MATH_PATTERN = re.compile(r"(?s)\\\(.*?\\\)")
_MATH_ENV_PATTERN = re.compile(
    r"(?s)\\begin\{(?P<env>equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{(?P=env)\}"
)
_CODE_FENCE_PATTERN = re.compile(
    r"(?ms)(?:(?<=\A)|(?<=\n))(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:\n(?P=fence)[ \t]*(?=\n|$)|\Z)"
)


@dataclass(frozen=True, slots=True)
class DialogueSource:
    label: str
    dataset: str
    config: str | None
    split: str
    messages_key: str
    role_key: str
    content_key: str


@dataclass(frozen=True, slots=True)
class DocumentSource:
    label: str
    dataset: str
    config: str | None
    split: str
    text_key: str
    kind: str


def parse_dialogue_source(spec: str) -> DialogueSource:
    if "=" not in spec:
        raise ValueError(
            "HF dialogue source spec must be label=dataset|config|split|messages_key|role_key|content_key."
        )
    label, payload = spec.split("=", 1)
    parts = payload.split("|")
    if len(parts) != 6:
        raise ValueError(
            "HF dialogue source spec must be label=dataset|config|split|messages_key|role_key|content_key."
        )
    dataset, config, split, messages_key, role_key, content_key = parts
    return DialogueSource(
        label=label.strip(),
        dataset=dataset.strip(),
        config=config.strip() or None,
        split=split.strip(),
        messages_key=messages_key.strip(),
        role_key=role_key.strip(),
        content_key=content_key.strip(),
    )


def parse_document_source(spec: str) -> DocumentSource:
    if "=" not in spec:
        raise ValueError(
            "HF document source spec must be label=dataset|config|split|text_key|kind."
        )
    label, payload = spec.split("=", 1)
    parts = payload.split("|")
    if len(parts) != 5:
        raise ValueError(
            "HF document source spec must be label=dataset|config|split|text_key|kind."
        )
    dataset, config, split, text_key, kind = parts
    normalized_kind = kind.strip().lower()
    if normalized_kind not in {"text", "wiki", "code", "math"}:
        raise ValueError(f"Unsupported document kind: {normalized_kind!r}")
    return DocumentSource(
        label=label.strip(),
        dataset=dataset.strip(),
        config=config.strip() or None,
        split=split.strip(),
        text_key=text_key.strip(),
        kind=normalized_kind,
    )


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def _nested_get(value: Any, dotted_key: str) -> Any:
    current = value
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _normalize_role(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    role = value.strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant", "bot", "model"}:
        return "assistant"
    if role in {"system", "developer"}:
        return role
    return None


def _merge_segments(segments: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    for segment in segments:
        if not segment["text"]:
            continue
        if merged and merged[-1]["kind"] == segment["kind"]:
            merged[-1]["text"] += segment["text"]
        else:
            merged.append(segment)
    return merged


def segment_message_content_exact(content: str) -> list[dict[str, str]]:
    segments: list[dict[str, str]] = []
    cursor = 0
    patterns = (
        ("code", _CODE_FENCE_PATTERN),
        ("math", _DISPLAY_MATH_PATTERN),
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
    return _merge_segments(segments)


def normalize_dialogue_messages(
    messages: list[dict[str, Any]],
    *,
    role_key: str,
    content_key: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = _normalize_role(_nested_get(message, role_key) if "." in role_key else message.get(role_key))
        content = _nested_get(message, content_key) if "." in content_key else message.get(content_key)
        if role is None or not isinstance(content, str) or not content:
            continue
        segments = segment_message_content_exact(content)
        if not segments:
            continue
        normalized.append({"role": role, "segments": segments})
    return normalized


def write_document(handle: Any, *, kind: str, source: str, messages: list[dict[str, Any]]) -> None:
    handle.write(
        json.dumps(
            {"kind": kind, "source": source, "messages": messages},
            ensure_ascii=False,
        )
        + "\n"
    )


def write_segment_document(handle: Any, *, kind: str, source: str, text: str) -> None:
    normalized_kind = "text" if kind == "wiki" else kind
    handle.write(
        json.dumps(
            {
                "kind": normalized_kind,
                "source": source,
                "segments": [{"kind": normalized_kind, "text": text}],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def add_hf_dialogue_source(
    *,
    output_handle: Any,
    source: DialogueSource,
    streaming: bool,
    max_records: int | None,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to build the segmented dialogue corpus.") from exc

    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=streaming,
    )
    written = 0
    for record_index, row in enumerate(dataset, start=1):
        if max_records is not None and record_index > max_records:
            break
        messages = _nested_get(row, source.messages_key)
        if not isinstance(messages, list):
            continue
        normalized = normalize_dialogue_messages(
            [item for item in messages if isinstance(item, dict)],
            role_key=source.role_key,
            content_key=source.content_key,
        )
        if not normalized:
            continue
        write_document(
            output_handle,
            kind="dialogue",
            source=f"{source.label}:{record_index}",
            messages=normalized,
        )
        written += 1
    return written


def add_hf_document_source(
    *,
    output_handle: Any,
    source: DocumentSource,
    streaming: bool,
    max_records: int | None,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to build the segmented dialogue corpus.") from exc

    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=streaming,
    )
    written = 0
    for record_index, row in enumerate(dataset, start=1):
        if max_records is not None and record_index > max_records:
            break
        text = _nested_get(row, source.text_key)
        if not isinstance(text, str) or not text.strip():
            continue
        write_segment_document(
            output_handle,
            kind=source.kind,
            source=f"{source.label}:{record_index}",
            text=text,
        )
        written += 1
    return written


def add_dialogue_jsonl(
    *,
    output_handle: Any,
    path: Path,
    source_label: str,
    max_records: int | None,
) -> int:
    written = 0
    for record_index, record in enumerate(iter_jsonl_records(path), start=1):
        if max_records is not None and record_index > max_records:
            break
        messages = record.get("messages") or record.get("conversations")
        if not isinstance(messages, list):
            continue
        normalized = normalize_dialogue_messages(
            [item for item in messages if isinstance(item, dict)],
            role_key="role",
            content_key="content",
        )
        if not normalized:
            continue
        write_document(
            output_handle,
            kind="dialogue",
            source=f"{source_label}:{record_index}",
            messages=normalized,
        )
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a segmented dialogue corpus with exact code/math labels.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dialogue-jsonl", action="append", default=[])
    parser.add_argument("--hf-dialogue-source", action="append", default=[])
    parser.add_argument("--hf-document-source", action="append", default=[])
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--no-hf-streaming", dest="hf_streaming", action="store_false")
    parser.add_argument("--max-records-per-source", type=int, default=50000)
    parser.add_argument("--skip-failed-sources", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hf_sources = [parse_dialogue_source(spec) for spec in args.hf_dialogue_source]
    document_sources = [parse_document_source(spec) for spec in args.hf_document_source]
    source_counts: dict[str, int] = {}
    started = time.perf_counter()

    with output.open("w", encoding="utf-8") as handle:
        for dialogue_path in args.dialogue_jsonl:
            path = Path(dialogue_path)
            label = f"jsonl:{path.stem}"
            count = add_dialogue_jsonl(
                output_handle=handle,
                path=path,
                source_label=label,
                max_records=args.max_records_per_source,
            )
            source_counts[label] = count
            print(f"source={label} documents={count:,}", flush=True)

        for source in hf_sources:
            try:
                count = add_hf_dialogue_source(
                    output_handle=handle,
                    source=source,
                    streaming=args.hf_streaming,
                    max_records=args.max_records_per_source,
                )
            except Exception as exc:
                if not args.skip_failed_sources:
                    raise
                print(f"source={source.label} failed={type(exc).__name__}: {exc}", flush=True)
                source_counts[source.label] = 0
                continue
            source_counts[source.label] = count
            print(f"source={source.label} documents={count:,}", flush=True)

        for source in document_sources:
            try:
                count = add_hf_document_source(
                    output_handle=handle,
                    source=source,
                    streaming=args.hf_streaming,
                    max_records=args.max_records_per_source,
                )
            except Exception as exc:
                if not args.skip_failed_sources:
                    raise
                print(f"source={source.label} failed={type(exc).__name__}: {exc}", flush=True)
                source_counts[source.label] = 0
                continue
            source_counts[source.label] = count
            print(f"source={source.label} documents={count:,}", flush=True)

    total_documents = sum(source_counts.values())
    meta = {
        "output": str(output),
        "total_documents": total_documents,
        "source_counts": source_counts,
        "hf_dialogue_sources": [asdict(source) for source in hf_sources],
        "hf_document_sources": [asdict(source) for source in document_sources],
        "dialogue_jsonl": args.dialogue_jsonl,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(output.suffix + ".meta.json"), meta)
    print(f"output={output} total_documents={total_documents:,}", flush=True)


if __name__ == "__main__":
    main()
