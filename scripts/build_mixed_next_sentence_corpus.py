from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from lm_experiment_utils import _extract_text_from_json_record, write_json
from train_progressive_b_lm import DialoguePairText, _pairs_from_record, make_next_sentence_pairs


DEFAULT_HF_SOURCES = (
    "arxiv=ccdv/arxiv-summarization||train|article",
    "pubmed=ccdv/pubmed-summarization||train|article",
    "wiki=wikimedia/wikipedia|20231101.en|train|text",
    "code=codeparrot/codeparrot-clean||train|content",
)


@dataclass(frozen=True, slots=True)
class HfSource:
    label: str
    dataset: str
    config: str | None
    split: str
    text_key: str


def parse_hf_source(spec: str) -> HfSource:
    if "=" not in spec:
        raise ValueError(
            "HF source spec must be label=dataset|config|split|text_key, "
            f"got {spec!r}."
        )
    label, payload = spec.split("=", 1)
    parts = payload.split("|")
    if len(parts) != 4:
        raise ValueError(
            "HF source spec must be label=dataset|config|split|text_key, "
            f"got {spec!r}."
        )
    dataset, config, split, text_key = parts
    return HfSource(
        label=label.strip(),
        dataset=dataset.strip(),
        config=config.strip() or None,
        split=split.strip(),
        text_key=text_key.strip(),
    )


def iter_jsonl_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                yield record


def write_pair(handle: Any, pair: DialoguePairText, *, source: str) -> None:
    handle.write(
        json.dumps(
            {"prefix": pair.prefix, "response": pair.response, "source": source},
            ensure_ascii=False,
        )
        + "\n"
    )


def add_dialogue_jsonl(
    *,
    output_handle: Any,
    path: Path,
    source_label: str,
    max_pairs: int | None,
) -> int:
    written = 0
    for record in iter_jsonl_records(path):
        for pair in _pairs_from_record(record):
            write_pair(output_handle, pair, source=source_label)
            written += 1
            if max_pairs is not None and written >= max_pairs:
                return written
    return written


def add_hf_source(
    *,
    output_handle: Any,
    source: HfSource,
    streaming: bool,
    max_records: int | None,
    max_pairs: int | None,
    prefix_sentences: int,
    min_chars: int,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to build the mixed corpus.") from exc

    dataset = load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=streaming,
    )
    written = 0
    records_seen = 0
    for row in dataset:
        if max_records is not None and records_seen >= max_records:
            break
        records_seen += 1
        text = _extract_text_from_json_record(row, (source.text_key,))
        if not text:
            continue
        remaining = None if max_pairs is None else max_pairs - written
        if remaining is not None and remaining <= 0:
            break
        pairs = make_next_sentence_pairs(
            [text],
            prefix_sentences=prefix_sentences,
            min_chars=min_chars,
            max_pairs=remaining,
        )
        for pair in pairs:
            write_pair(output_handle, pair, source=source.label)
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--dialogue-jsonl", action="append", default=[])
    parser.add_argument("--hf-source", action="append", default=[])
    parser.add_argument("--no-default-hf-sources", action="store_true")
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--no-hf-streaming", dest="hf_streaming", action="store_false")
    parser.add_argument("--max-records-per-source", type=int, default=20000)
    parser.add_argument("--max-pairs-per-source", type=int, default=100000)
    parser.add_argument("--max-dialogue-pairs", type=int)
    parser.add_argument("--sentence-prefix-count", type=int, default=1)
    parser.add_argument("--sentence-min-chars", type=int, default=8)
    parser.add_argument("--skip-failed-sources", action="store_true")
    args = parser.parse_args()

    if args.sentence_prefix_count <= 0:
        raise ValueError("sentence-prefix-count must be positive.")
    if args.sentence_min_chars <= 0:
        raise ValueError("sentence-min-chars must be positive.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hf_specs = ([] if args.no_default_hf_sources else list(DEFAULT_HF_SOURCES)) + args.hf_source
    hf_sources = [parse_hf_source(spec) for spec in hf_specs]
    source_counts: dict[str, int] = {}
    started = time.perf_counter()

    with output.open("w", encoding="utf-8") as handle:
        for dialogue_path in args.dialogue_jsonl:
            path = Path(dialogue_path)
            label = f"dialogue:{path.stem}"
            count = add_dialogue_jsonl(
                output_handle=handle,
                path=path,
                source_label=label,
                max_pairs=args.max_dialogue_pairs,
            )
            source_counts[label] = count
            print(f"source={label} pairs={count:,}", flush=True)

        for source in hf_sources:
            try:
                count = add_hf_source(
                    output_handle=handle,
                    source=source,
                    streaming=args.hf_streaming,
                    max_records=args.max_records_per_source,
                    max_pairs=args.max_pairs_per_source,
                    prefix_sentences=args.sentence_prefix_count,
                    min_chars=args.sentence_min_chars,
                )
            except Exception as exc:
                if not args.skip_failed_sources:
                    raise
                print(f"source={source.label} failed={type(exc).__name__}: {exc}", flush=True)
                source_counts[source.label] = 0
                continue
            source_counts[source.label] = count
            print(f"source={source.label} pairs={count:,}", flush=True)

    total_pairs = sum(source_counts.values())
    meta = {
        "output": str(output),
        "total_pairs": total_pairs,
        "source_counts": source_counts,
        "hf_sources": [asdict(source) for source in hf_sources],
        "dialogue_jsonl": args.dialogue_jsonl,
        "sentence_prefix_count": args.sentence_prefix_count,
        "sentence_min_chars": args.sentence_min_chars,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(output.suffix + ".meta.json"), meta)
    print(f"output={output} total_pairs={total_pairs:,}")


if __name__ == "__main__":
    main()
