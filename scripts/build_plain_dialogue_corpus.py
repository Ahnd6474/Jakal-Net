from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

from train_causal_memory_lm import _segment_message_content_exact


DEFAULT_DIALOGUE_SOURCES = (
    "ultrachat=HuggingFaceH4/ultrachat_200k||train_sft|messages|role|content",
)


def _apply_shard(dataset: Any, *, num_shards: int, shard_index: int) -> Any:
    if num_shards <= 1:
        return dataset
    if not hasattr(dataset, "shard"):
        raise ValueError("This dataset object does not support sharding.")
    try:
        return dataset.shard(num_shards=num_shards, index=shard_index, contiguous=True)
    except TypeError:
        return dataset.shard(num_shards, shard_index)


def _shard_suffix(*, num_shards: int, shard_index: int) -> str:
    if num_shards <= 1:
        return ""
    return f":shard{shard_index + 1:02d}of{num_shards:02d}"


def parse_hf_source(spec: str) -> tuple[str, str, str, str, str, str]:
    try:
        label, remainder = spec.split("=", maxsplit=1)
        dataset_name, config, split, messages_key, role_key, content_key = remainder.split("|", maxsplit=5)
    except ValueError as exc:
        raise ValueError(
            "HF dialogue source must look like "
            "'label=dataset_name|config|split|messages_key|role_key|content_key'."
        ) from exc
    return (
        label.strip(),
        dataset_name.strip(),
        config.strip(),
        split.strip(),
        messages_key.strip(),
        role_key.strip(),
        content_key.strip(),
    )


def normalize_role(value: object) -> str:
    if not isinstance(value, str):
        return ""
    role = value.strip().lower()
    if role in {"human", "prompt"}:
        return "user"
    if role in {"gpt", "bot", "response"}:
        return "assistant"
    return role


def build_plain_dialogue_record(*, source: str, messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized: list[dict[str, Any]] = []
    has_user = False
    has_assistant = False
    for message in messages:
        role = normalize_role(message.get("role"))
        if role not in {"user", "assistant", "system"}:
            return None
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        segments = _segment_message_content_exact(content)
        if not segments:
            continue
        if any(segment.get("kind") != "text" for segment in segments):
            return None
        normalized.append(
            {
                "role": role,
                "segments": [{"kind": "text", "text": str(segment["text"])} for segment in segments],
            }
        )
        has_user = has_user or role == "user"
        has_assistant = has_assistant or role == "assistant"
    if len(normalized) < 2 or not has_user or not has_assistant:
        return None
    return {
        "kind": "dialogue",
        "source": source,
        "messages": normalized,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a plain text-only dialogue corpus from HF chat datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-dialogue-source", action="append", default=[])
    parser.add_argument("--source-label", action="append", default=[])
    parser.add_argument("--no-default-sources", action="store_true")
    parser.add_argument("--max-records-per-source", type=int, default=50000)
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--skip-failed-sources", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards).")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    source_specs = list(args.hf_dialogue_source)
    if not args.no_default_sources:
        source_specs.extend(spec for spec in DEFAULT_DIALOGUE_SOURCES if spec not in source_specs)
    selected_labels = {label.strip() for label in args.source_label if label.strip()}

    source_counts: dict[str, int] = {}
    source_scanned: dict[str, int] = {}
    written = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for spec in source_specs:
            label, dataset_name, config, split, messages_key, role_key, content_key = parse_hf_source(spec)
            if selected_labels and label not in selected_labels:
                continue
            try:
                dataset = load_dataset(
                    dataset_name,
                    None if config in {"", "None", "null"} else config,
                    split=split,
                    streaming=args.hf_streaming,
                )
                dataset = _apply_shard(dataset, num_shards=args.num_shards, shard_index=args.shard_index)
            except Exception:
                if args.skip_failed_sources:
                    continue
                raise
            count = 0
            scanned = 0
            for row in dataset:
                scanned += 1
                raw_messages = row.get(messages_key)
                if not isinstance(raw_messages, list):
                    continue
                messages = []
                for raw_message in raw_messages:
                    if not isinstance(raw_message, dict):
                        messages = []
                        break
                    messages.append(
                        {
                            "role": raw_message.get(role_key),
                            "content": raw_message.get(content_key),
                        }
                    )
                if not messages:
                    continue
                record = build_plain_dialogue_record(
                    source=f"plain_dialogue:{label}{_shard_suffix(num_shards=args.num_shards, shard_index=args.shard_index)}:{count + 1}",
                    messages=messages,
                )
                if record is None:
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                written += 1
                if count >= args.max_records_per_source:
                    break
            source_counts[label] = count
            source_scanned[label] = scanned

    meta = {
        "version": 1,
        "output": str(output_path),
        "records": written,
        "source_counts": source_counts,
        "source_scanned": source_scanned,
        "sources": source_specs,
        "filter": "exact_text_only_dialogue",
        "selected_labels": sorted(selected_labels),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
