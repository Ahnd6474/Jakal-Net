from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def _resolve_event_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidates = sorted(path.glob("events.out.tfevents.*"))
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event files found under {path}")
    return candidates[-1]


def _parse_steps(spec: str) -> list[int]:
    if ":" in spec:
        start_str, end_str = spec.split(":", maxsplit=1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"Invalid step range: {spec}")
        return list(range(start, end + 1))
    return [int(part) for part in spec.split(",") if part.strip()]


def _scalar_map(acc: event_accumulator.EventAccumulator) -> dict[str, dict[int, float]]:
    return {
        tag: {event.step: event.value for event in acc.Scalars(tag)}
        for tag in acc.Tags().get("scalars", [])
    }


def _top_tags(
    scalars: dict[str, dict[int, float]],
    *,
    step: int,
    prefix: str,
    limit: int,
) -> list[tuple[float, str]]:
    rows: list[tuple[float, str]] = []
    for tag, values in scalars.items():
        if not tag.startswith(prefix):
            continue
        value = values.get(step)
        if value is None:
            continue
        rows.append((value, tag))
    rows.sort(reverse=True)
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("event_path", help="TensorBoard event file or directory containing one.")
    parser.add_argument(
        "--steps",
        default="1025:1032",
        help="Single step, comma list, or inclusive range like 1025:1032.",
    )
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument(
        "--tag",
        action="append",
        default=[
            "train/minibatch_loss",
            "train/grad_norm",
            "train/query_block_avg_ppl",
        ],
        help="Scalar tag to print for every step. Can be repeated.",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[
            "debug/grad_preclip/",
            "debug/grad_postclip/",
            "debug/activation_grad/",
        ],
        help="Scalar tag prefix whose top values should be printed. Can be repeated.",
    )
    args = parser.parse_args()

    event_path = _resolve_event_path(args.event_path)
    acc = event_accumulator.EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    acc.Reload()
    scalars = _scalar_map(acc)

    steps = _parse_steps(args.steps)
    print(f"event_file={event_path}")
    for step in steps:
        print(f"=== step {step} ===")
        for tag in args.tag:
            value = scalars.get(tag, {}).get(step)
            if value is not None:
                print(f"{tag}={value:.6g}")
        for prefix in args.prefix:
            top_rows = _top_tags(scalars, step=step, prefix=prefix, limit=args.topk)
            if not top_rows:
                continue
            print(f"{prefix}top{min(args.topk, len(top_rows))}:")
            for value, tag in top_rows:
                print(f"  {tag}={value:.6g}")


if __name__ == "__main__":
    main()
