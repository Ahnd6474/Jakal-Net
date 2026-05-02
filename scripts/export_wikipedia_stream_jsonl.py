from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream Wikipedia from Hugging Face and save it as JSONL.")
    parser.add_argument("--dataset", default="wikimedia/wikipedia")
    parser.add_argument("--config", default="20231101.en")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=2_000_000)
    parser.add_argument("--progress-interval", type=int, default=10_000)
    parser.add_argument("--hard-exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    count = 0

    dataset = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
    )
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in dataset:
            text = str(record.get(args.text_key) or "").strip()
            if not text:
                continue
            payload = {
                "text": text,
                "source": f"{args.dataset}:{record.get('id', count)}",
                "title": record.get("title"),
                "url": record.get("url"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            if count % max(1, args.progress_interval) == 0:
                print(f"exported={count:,}", flush=True)
            if count >= args.limit:
                break
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, output_path)
    print(f"saved={output_path}", flush=True)
    print(f"documents={count:,}", flush=True)
    if args.hard_exit:
        os._exit(0)


if __name__ == "__main__":
    main()
