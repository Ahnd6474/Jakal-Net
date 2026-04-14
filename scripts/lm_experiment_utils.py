from __future__ import annotations

import csv
import glob
import json
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class CorpusLoadResult:
    text: str
    source_label: str
    text_path: Path | None
    sample_count: int
    file_count: int
    char_count: int
    truncated: bool
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class ModelScalePreset:
    name: str
    dim: int
    warmup_layers: int
    final_refine_layers: int
    lite_layers: int
    mid_layers: int
    full_layers: int
    route_topk: int


MODEL_SIZE_PRESETS: dict[str, ModelScalePreset] = {
    "tiny": ModelScalePreset(
        name="tiny",
        dim=32,
        warmup_layers=1,
        final_refine_layers=1,
        lite_layers=1,
        mid_layers=1,
        full_layers=1,
        route_topk=2,
    ),
    "small": ModelScalePreset(
        name="small",
        dim=48,
        warmup_layers=2,
        final_refine_layers=1,
        lite_layers=1,
        mid_layers=1,
        full_layers=1,
        route_topk=4,
    ),
    "base": ModelScalePreset(
        name="base",
        dim=64,
        warmup_layers=2,
        final_refine_layers=2,
        lite_layers=2,
        mid_layers=2,
        full_layers=1,
        route_topk=4,
    ),
    "medium": ModelScalePreset(
        name="medium",
        dim=96,
        warmup_layers=2,
        final_refine_layers=2,
        lite_layers=2,
        mid_layers=2,
        full_layers=2,
        route_topk=6,
    ),
    "large": ModelScalePreset(
        name="large",
        dim=128,
        warmup_layers=3,
        final_refine_layers=2,
        lite_layers=3,
        mid_layers=3,
        full_layers=2,
        route_topk=8,
    ),
}


def resolve_model_scale_preset(name: str) -> ModelScalePreset:
    try:
        return MODEL_SIZE_PRESETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_SIZE_PRESETS))
        raise ValueError(f"Unknown model preset {name!r}. Expected one of: {choices}.") from exc


def parse_preset_names(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def slugify_run_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    collapsed = "-".join(part for part in cleaned.split("-") if part)
    return collapsed or "run"


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def create_run_directory(output_root: str | Path, run_name: str) -> Path:
    root = ensure_directory(output_root)
    run_dir = root / f"{timestamp_slug()}_{slugify_run_name(run_name)}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{timestamp_slug()}_{slugify_run_name(run_name)}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False))
            handle.write("\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False))
        handle.write("\n")


def write_csv_rows(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_jsonable(value) for key, value in row.items()})


def _read_utf8_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Input text file must not be empty: {path}")
    return text


def _expand_sources(
    sources: Sequence[str],
    *,
    directory_suffixes: tuple[str, ...],
) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        candidate = Path(source)
        matches: list[Path] = []
        if candidate.exists():
            if candidate.is_dir():
                for suffix in directory_suffixes:
                    matches.extend(path for path in candidate.rglob(f"*{suffix}") if path.is_file())
            else:
                matches.append(candidate)
        else:
            matches.extend(Path(match) for match in glob.glob(source, recursive=True))
        for match in matches:
            resolved_match = match.resolve()
            if resolved_match in seen:
                continue
            seen.add(resolved_match)
            resolved.append(resolved_match)
    return sorted(resolved)


def _extract_text_from_json_record(record: Any, text_keys: Sequence[str]) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for key in text_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    messages = record.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
        if parts:
            return "\n".join(parts)
    return None


def _load_jsonl_texts(path: Path, text_keys: Sequence[str]) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            extracted = _extract_text_from_json_record(record, text_keys)
            if extracted is None:
                raise ValueError(
                    f"Could not find text in {path} line {line_number}. "
                    f"Tried keys: {', '.join(text_keys)}."
                )
            texts.append(extracted)
    if not texts:
        raise ValueError(f"Input JSONL file must not be empty: {path}")
    return texts


def _truncate_texts(
    pieces: Sequence[str],
    *,
    separator: str,
    max_samples: int | None,
    max_chars: int | None,
) -> tuple[str, int, bool]:
    selected: list[str] = []
    truncated = False
    current_chars = 0
    for index, piece in enumerate(pieces):
        if max_samples is not None and index >= max_samples:
            truncated = True
            break
        next_piece = piece
        if max_chars is not None:
            budget = max_chars - current_chars
            separator_budget = len(separator) if selected else 0
            budget -= separator_budget
            if budget <= 0:
                truncated = True
                break
            if len(next_piece) > budget:
                next_piece = next_piece[:budget]
                truncated = True
        if not next_piece:
            continue
        if selected:
            current_chars += len(separator)
        selected.append(next_piece)
        current_chars += len(next_piece)
    text = separator.join(selected)
    return text, len(selected), truncated


def _load_hf_dataset_texts(
    *,
    dataset_name: str,
    config_name: str | None,
    split: str,
    text_key: str,
    streaming: bool,
    max_samples: int | None,
) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required for --hf-dataset corpus loading. "
            "Install it with `python -m pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    texts: list[str] = []
    iterator = dataset if streaming else iter(dataset)
    for index, row in enumerate(iterator):
        if max_samples is not None and index >= max_samples:
            break
        extracted = _extract_text_from_json_record(row, (text_key,))
        if extracted is None:
            raise ValueError(
                f"Could not find text key {text_key!r} in Hugging Face dataset row."
            )
        texts.append(extracted)
    if not texts:
        raise ValueError(
            f"Hugging Face dataset {dataset_name!r} split {split!r} did not yield any text."
        )
    return texts


def load_text_corpus(
    *,
    default_text: str,
    text_file: str | None = None,
    text_sources: Sequence[str] = (),
    jsonl_sources: Sequence[str] = (),
    jsonl_text_keys: Sequence[str] = ("text",),
    hf_dataset: str | None = None,
    hf_config: str | None = None,
    hf_split: str = "train",
    hf_text_key: str = "text",
    hf_streaming: bool = False,
    max_samples: int | None = None,
    max_chars: int | None = None,
    separator: str = "\n\n",
) -> CorpusLoadResult:
    sources = [text_file] if text_file is not None else []
    sources.extend(text_sources)

    text_paths = _expand_sources(sources, directory_suffixes=(".txt", ".text", ".md"))
    jsonl_paths = _expand_sources(jsonl_sources, directory_suffixes=(".jsonl", ".json"))

    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when provided.")
    if max_chars is not None and max_chars <= 0:
        raise ValueError("max_chars must be positive when provided.")

    if not text_paths and not jsonl_paths and hf_dataset is None:
        text = default_text
        return CorpusLoadResult(
            text=text,
            source_label="default_text",
            text_path=None,
            sample_count=1,
            file_count=0,
            char_count=len(text),
            truncated=False,
            metadata={"source_kind": "default_text"},
        )

    text_pieces: list[str] = []
    for path in text_paths:
        text_pieces.append(_read_utf8_text(path))
    for path in jsonl_paths:
        text_pieces.extend(_load_jsonl_texts(path, jsonl_text_keys))
    if hf_dataset is not None:
        hf_max_samples = max_samples if not text_pieces else None
        text_pieces.extend(
            _load_hf_dataset_texts(
                dataset_name=hf_dataset,
                config_name=hf_config,
                split=hf_split,
                text_key=hf_text_key,
                streaming=hf_streaming,
                max_samples=hf_max_samples,
            )
        )

    combined_text, sample_count, truncated = _truncate_texts(
        text_pieces,
        separator=separator,
        max_samples=max_samples,
        max_chars=max_chars,
    )
    if not combined_text.strip():
        raise ValueError("The resolved corpus is empty after loading.")

    direct_text_path = text_paths[0] if len(text_paths) == 1 and not jsonl_paths and hf_dataset is None else None
    source_parts: list[str] = []
    if text_paths:
        source_parts.append(f"text_files={len(text_paths)}")
    if jsonl_paths:
        source_parts.append(f"jsonl_files={len(jsonl_paths)}")
    if hf_dataset is not None:
        source_parts.append(f"hf_dataset={hf_dataset}:{hf_split}")

    metadata: dict[str, object] = {
        "text_paths": [str(path) for path in text_paths],
        "jsonl_paths": [str(path) for path in jsonl_paths],
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "hf_text_key": hf_text_key,
        "hf_streaming": hf_streaming,
        "max_samples": max_samples,
        "max_chars": max_chars,
        "separator": separator,
    }

    return CorpusLoadResult(
        text=combined_text,
        source_label=", ".join(source_parts) if source_parts else "external_corpus",
        text_path=direct_text_path,
        sample_count=sample_count,
        file_count=len(text_paths) + len(jsonl_paths),
        char_count=len(combined_text),
        truncated=truncated,
        metadata=metadata,
    )
