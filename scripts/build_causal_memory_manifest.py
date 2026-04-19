from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a document-level manifest for causal-memory LM training.")
    parser.add_argument(
        "--output-jsonl",
        default="artifacts/data/causal_memory_manifest_1m.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--output-meta",
        default="artifacts/data/causal_memory_manifest_1m.meta.json",
        help="Output metadata JSON path.",
    )
    parser.add_argument("--target-documents", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wiki-target", type=int, default=300_000)
    parser.add_argument("--math-target", type=int, default=200_000)
    parser.add_argument("--dialogue-target", type=int, default=200_000)
    parser.add_argument("--code-target", type=int, default=100_000)
    parser.add_argument("--mixed-target", type=int, default=200_000)
    parser.add_argument("--cache-size", type=int, default=10_000)
    return parser.parse_args()


def normalize_text(text: str, *, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0].rstrip() or text[:max_chars].rstrip()
    return text


def normalize_code(text: str, *, max_chars: int) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def normalize_messages(messages: list[dict[str, Any]], *, max_messages: int, max_chars: int) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        cleaned = normalize_text(content, max_chars=max_chars)
        if not cleaned:
            continue
        if normalized and normalized[-1]["role"] == role:
            continue
        normalized.append({"role": role, "content": cleaned})
        if len(normalized) >= max_messages:
            break
    if len(normalized) < 4:
        return []
    if len(normalized) % 2 != 0:
        normalized = normalized[:-1]
    return normalized


class Reservoir:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.items: list[str] = []
        self.seen = 0

    def add(self, text: str) -> None:
        if not text:
            return
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(text)
            return
        index = random.randrange(self.seen)
        if index < self.limit:
            self.items[index] = text

    def sample(self) -> str:
        return random.choice(self.items)

    def __len__(self) -> int:
        return len(self.items)


def canonical_digest(record: dict[str, Any]) -> bytes:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).digest()


def print_progress(total_written: int, counts: Counter[str]) -> None:
    print(
        "progress",
        json.dumps(
            {
                "documents": total_written,
                "wiki": counts["wiki"],
                "math": counts["math"],
                "dialogue": counts["dialogue"],
                "code": counts["code"],
                "mixed": counts["mixed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


def maybe_write_record(
    handle: Any,
    *,
    record: dict[str, Any],
    category: str,
    source_label: str,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
) -> bool:
    if total_written_ref[0] >= total_limit:
        return False
    digest = canonical_digest(record)
    if digest in seen_hashes:
        return False
    seen_hashes.add(digest)
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")
    total_written_ref[0] += 1
    counts[category] += 1
    source_counts[source_label] += 1
    if total_written_ref[0] % 10_000 == 0:
        print_progress(total_written_ref[0], counts)
    return True


def collect_wikipedia(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    wiki_cache: Reservoir,
) -> None:
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for record in dataset:
        if counts["wiki"] >= target or total_written_ref[0] >= total_limit:
            break
        text = normalize_text(str(record.get("text", "")), max_chars=4_500)
        title = normalize_text(str(record.get("title", "")), max_chars=160)
        if len(text) < 500:
            continue
        body = f"{title}\n\n{text}" if title else text
        manifest_record = {
            "text": body,
            "source": f"wikimedia/wikipedia:{record.get('id', '')}",
            "kind": "text",
        }
        if maybe_write_record(
            handle,
            record=manifest_record,
            category="wiki",
            source_label="wikimedia/wikipedia",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        ):
            wiki_cache.add(body)


def collect_math(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    math_cache: Reservoir,
) -> None:
    dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for index, record in enumerate(dataset):
        if counts["math"] >= target or total_written_ref[0] >= total_limit:
            break
        text = normalize_text(str(record.get("text", "")), max_chars=4_000)
        if len(text) < 300:
            continue
        manifest_record = {
            "text": text,
            "source": f"open-web-math/open-web-math:{index}",
            "kind": "text",
        }
        if maybe_write_record(
            handle,
            record=manifest_record,
            category="math",
            source_label="open-web-math/open-web-math",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        ):
            math_cache.add(text)


def collect_dialogue(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
) -> None:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    for record in dataset:
        if counts["dialogue"] >= target or total_written_ref[0] >= total_limit:
            break
        normalized = normalize_messages(list(record.get("messages") or []), max_messages=8, max_chars=900)
        if not normalized:
            continue
        manifest_record = {
            "messages": normalized,
            "source": f"HuggingFaceH4/ultrachat_200k:{record.get('prompt_id', counts['dialogue'])}",
            "kind": "dialogue",
        }
        maybe_write_record(
            handle,
            record=manifest_record,
            category="dialogue",
            source_label="HuggingFaceH4/ultrachat_200k",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        )


def collect_code_instruction_dataset(
    handle: Any,
    *,
    dataset_name: str,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    code_cache: Reservoir,
) -> None:
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    for index, record in enumerate(dataset):
        if counts["code"] >= target or total_written_ref[0] >= total_limit:
            break
        instruction = normalize_text(str(record.get("instruction", "")), max_chars=600)
        extra_input = normalize_text(str(record.get("input", "")), max_chars=600)
        output = normalize_code(str(record.get("output", "")), max_chars=5_000)
        if len(output) < 40:
            continue
        parts = []
        if instruction:
            parts.append(f"# Instruction\n{instruction}")
        if extra_input:
            parts.append(f"# Context\n{extra_input}")
        parts.append(f"# Solution\n{output}")
        body = "\n\n".join(parts)
        manifest_record = {
            "code": body,
            "source": f"{dataset_name}:{index}",
            "kind": "code",
        }
        if maybe_write_record(
            handle,
            record=manifest_record,
            category="code",
            source_label=dataset_name,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        ):
            code_cache.add(output)


def collect_leetcode(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    code_cache: Reservoir,
) -> None:
    dataset = load_dataset("greengerong/leetcode", split="train", streaming=True)
    for index, record in enumerate(dataset):
        if counts["code"] >= target or total_written_ref[0] >= total_limit:
            break
        solution = normalize_code(str(record.get("python", "")), max_chars=5_000)
        if len(solution) < 40:
            continue
        title = normalize_text(str(record.get("title", "")), max_chars=200)
        content = normalize_text(str(record.get("content", "")), max_chars=1_200)
        body = "\n\n".join(part for part in (title, content, solution) if part)
        manifest_record = {
            "code": body,
            "source": f"greengerong/leetcode:{index}",
            "kind": "code",
        }
        if maybe_write_record(
            handle,
            record=manifest_record,
            category="code",
            source_label="greengerong/leetcode",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        ):
            code_cache.add(solution)


def collect_code_contests(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    code_cache: Reservoir,
) -> None:
    dataset = load_dataset("deepmind/code_contests", split="train", streaming=True)
    for record in dataset:
        if counts["code"] >= target or total_written_ref[0] >= total_limit:
            break
        title = normalize_text(str(record.get("name", "")), max_chars=160)
        description = normalize_text(str(record.get("description", "")), max_chars=1_200)
        solutions = record.get("solutions") or {}
        for solution_index, solution in enumerate(list(solutions.get("solution") or [])):
            if counts["code"] >= target or total_written_ref[0] >= total_limit:
                break
            normalized = normalize_code(str(solution), max_chars=5_000)
            if len(normalized) < 40:
                continue
            body = "\n\n".join(part for part in (title, description, normalized) if part)
            manifest_record = {
                "code": body,
                "source": f"deepmind/code_contests:{title}:{solution_index}",
                "kind": "code",
            }
            if maybe_write_record(
                handle,
                record=manifest_record,
                category="code",
                source_label="deepmind/code_contests",
                seen_hashes=seen_hashes,
                counts=counts,
                source_counts=source_counts,
                total_limit=total_limit,
                total_written_ref=total_written_ref,
            ):
                code_cache.add(normalized)


def collect_mbpp(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    code_cache: Reservoir,
) -> None:
    dataset = load_dataset("mbpp", split="train", streaming=True)
    for index, record in enumerate(dataset):
        if counts["code"] >= target or total_written_ref[0] >= total_limit:
            break
        prompt = normalize_text(str(record.get("text", "")), max_chars=800)
        code = normalize_code(str(record.get("code", "")), max_chars=4_000)
        if len(code) < 40:
            continue
        body = "\n\n".join(part for part in (prompt, code) if part)
        manifest_record = {
            "code": body,
            "source": f"mbpp:{index}",
            "kind": "code",
        }
        if maybe_write_record(
            handle,
            record=manifest_record,
            category="code",
            source_label="mbpp",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        ):
            code_cache.add(code)


def collect_sql_mixed(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    math_cache: Reservoir,
    wiki_cache: Reservoir,
) -> None:
    dataset = load_dataset("b-mc2/sql-create-context", split="train", streaming=True)
    for index, record in enumerate(dataset):
        if counts["mixed"] >= target or total_written_ref[0] >= total_limit:
            break
        question = normalize_text(str(record.get("question", "")), max_chars=600)
        context = normalize_text(str(record.get("context", "")), max_chars=1_600)
        answer = normalize_code(str(record.get("answer", "")), max_chars=1_200)
        if len(context) < 120 or len(answer) < 10:
            continue
        sections = []
        if question:
            sections.append(f"Question\n{question}")
        sections.append(f"Context\n{context}")
        sections.append(f"SQL\n{answer}")
        if len(wiki_cache):
            sections.append(f"Reference\n{wiki_cache.sample()[:800]}")
        if len(math_cache):
            sections.append(f"Math Note\n{math_cache.sample()[:800]}")
        manifest_record = {
            "text": "\n\n".join(sections),
            "source": f"b-mc2/sql-create-context:{index}",
            "kind": "text",
        }
        maybe_write_record(
            handle,
            record=manifest_record,
            category="mixed",
            source_label="b-mc2/sql-create-context",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        )


def synthesize_mixed(
    handle: Any,
    *,
    target: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_limit: int,
    total_written_ref: list[int],
    wiki_cache: Reservoir,
    math_cache: Reservoir,
    code_cache: Reservoir,
) -> None:
    while counts["mixed"] < target and total_written_ref[0] < total_limit:
        if not len(wiki_cache) or not len(math_cache) or not len(code_cache):
            break
        body = (
            "Reference\n"
            f"{wiki_cache.sample()[:1_400]}\n\n"
            "Math\n"
            f"{math_cache.sample()[:1_200]}\n\n"
            "Code\n"
            f"{code_cache.sample()[:1_400]}"
        )
        manifest_record = {
            "text": body,
            "source": f"synthetic_mix:{counts['mixed']}",
            "kind": "text",
        }
        maybe_write_record(
            handle,
            record=manifest_record,
            category="mixed",
            source_label="synthetic_mix",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=total_limit,
            total_written_ref=total_written_ref,
        )


def backfill_with_wiki_or_math(
    handle: Any,
    *,
    target_total: int,
    seen_hashes: set[bytes],
    counts: Counter[str],
    source_counts: Counter[str],
    total_written_ref: list[int],
    wiki_cache: Reservoir,
    math_cache: Reservoir,
) -> None:
    while total_written_ref[0] < target_total:
        category = "wiki" if total_written_ref[0] % 2 == 0 else "math"
        cache = wiki_cache if category == "wiki" else math_cache
        if not len(cache):
            cache = math_cache if category == "wiki" else wiki_cache
            category = "math" if category == "wiki" else "wiki"
        if not len(cache):
            break
        manifest_record = {
            "text": cache.sample(),
            "source": f"cache_backfill:{category}:{total_written_ref[0]}",
            "kind": "text",
        }
        if not maybe_write_record(
            handle,
            record=manifest_record,
            category=category,
            source_label="cache_backfill",
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=target_total,
            total_written_ref=total_written_ref,
        ):
            continue


def main() -> None:
    args = parse_args()
    quota_sum = args.wiki_target + args.math_target + args.dialogue_target + args.code_target + args.mixed_target
    if quota_sum != args.target_documents:
        raise ValueError(f"Quota sum {quota_sum:,} must equal target-documents {args.target_documents:,}.")

    random.seed(args.seed)

    output_path = Path(args.output_jsonl)
    meta_path = Path(args.output_meta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[bytes] = set()
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_written_ref = [0]

    wiki_cache = Reservoir(args.cache_size)
    math_cache = Reservoir(args.cache_size)
    code_cache = Reservoir(args.cache_size)

    with output_path.open("w", encoding="utf-8") as handle:
        collect_wikipedia(
            handle,
            target=args.wiki_target,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=args.target_documents,
            total_written_ref=total_written_ref,
            wiki_cache=wiki_cache,
        )
        collect_math(
            handle,
            target=args.math_target,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=args.target_documents,
            total_written_ref=total_written_ref,
            math_cache=math_cache,
        )
        collect_dialogue(
            handle,
            target=args.dialogue_target,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=args.target_documents,
            total_written_ref=total_written_ref,
        )
        for dataset_name in (
            "flytech/python-codes-25k",
            "sahil2801/CodeAlpaca-20k",
            "iamtarun/python_code_instructions_18k_alpaca",
        ):
            collect_code_instruction_dataset(
                handle,
                dataset_name=dataset_name,
                target=args.code_target,
                seen_hashes=seen_hashes,
                counts=counts,
                source_counts=source_counts,
                total_limit=args.target_documents,
                total_written_ref=total_written_ref,
                code_cache=code_cache,
            )
            if counts["code"] >= args.code_target or total_written_ref[0] >= args.target_documents:
                break
        if counts["code"] < args.code_target and total_written_ref[0] < args.target_documents:
            collect_leetcode(
                handle,
                target=args.code_target,
                seen_hashes=seen_hashes,
                counts=counts,
                source_counts=source_counts,
                total_limit=args.target_documents,
                total_written_ref=total_written_ref,
                code_cache=code_cache,
            )
        if counts["code"] < args.code_target and total_written_ref[0] < args.target_documents:
            collect_code_contests(
                handle,
                target=args.code_target,
                seen_hashes=seen_hashes,
                counts=counts,
                source_counts=source_counts,
                total_limit=args.target_documents,
                total_written_ref=total_written_ref,
                code_cache=code_cache,
            )
        if counts["code"] < args.code_target and total_written_ref[0] < args.target_documents:
            collect_mbpp(
                handle,
                target=args.code_target,
                seen_hashes=seen_hashes,
                counts=counts,
                source_counts=source_counts,
                total_limit=args.target_documents,
                total_written_ref=total_written_ref,
                code_cache=code_cache,
            )

        collect_sql_mixed(
            handle,
            target=args.mixed_target,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=args.target_documents,
            total_written_ref=total_written_ref,
            math_cache=math_cache,
            wiki_cache=wiki_cache,
        )
        synthesize_mixed(
            handle,
            target=args.mixed_target,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_limit=args.target_documents,
            total_written_ref=total_written_ref,
            wiki_cache=wiki_cache,
            math_cache=math_cache,
            code_cache=code_cache,
        )
        backfill_with_wiki_or_math(
            handle,
            target_total=args.target_documents,
            seen_hashes=seen_hashes,
            counts=counts,
            source_counts=source_counts,
            total_written_ref=total_written_ref,
            wiki_cache=wiki_cache,
            math_cache=math_cache,
        )

    meta = {
        "path": str(output_path),
        "record_count": total_written_ref[0],
        "counts": dict(sorted(counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "seed": args.seed,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except Exception:
        sys.stdout.flush()
        sys.stderr.flush()
        raise
