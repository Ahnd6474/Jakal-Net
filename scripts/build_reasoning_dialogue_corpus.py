from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lm_experiment_utils import write_json


DEFAULT_REASONING_SOURCES = (
    "anli_r1=anli||train_r1|anli",
    "anli_r2=anli||train_r2|anli",
    "anli_r3=anli||train_r3|anli",
    "proofwriter=tasksource/proofwriter||train|proofwriter",
    "logicnli=tasksource/logicnli||train|logicnli",
    "boardgameqa=tasksource/Boardgame-QA||train|boardgameqa",
    "defeasible_atomic=tasksource/defeasible-nli|atomic|train|defeasible_nli",
    "defeasible_snli=tasksource/defeasible-nli|snli|train|defeasible_nli",
    "defeasible_social=tasksource/defeasible-nli|social|train|defeasible_nli",
    # Optional but often unavailable in this environment; keep as explicit opt-in later.
    # "boardgameqa=<dataset>|<config>|<split>|boardgameqa",
    # "alpha_nli=<dataset>|<config>|<split>|alpha_nli",
    # "evidence_inference=<dataset>|<config>|<split>|evidence_inference",
)


@dataclass(frozen=True, slots=True)
class ReasoningSource:
    label: str
    dataset: str
    config: str | None
    split: str
    template: str


def parse_reasoning_source(spec: str) -> ReasoningSource:
    if "=" not in spec:
        raise ValueError("Reasoning source spec must be label=dataset|config|split|template.")
    label, payload = spec.split("=", 1)
    parts = payload.split("|")
    if len(parts) != 4:
        raise ValueError("Reasoning source spec must be label=dataset|config|split|template.")
    dataset, config, split, template = parts
    return ReasoningSource(
        label=label.strip(),
        dataset=dataset.strip(),
        config=config.strip() or None,
        split=split.strip(),
        template=template.strip(),
    )


def _class_names(dataset: Any, key: str) -> dict[int, str]:
    features = getattr(dataset, "features", None)
    if features is None or key not in features:
        return {}
    feature = features[key]
    names = getattr(feature, "names", None)
    if not isinstance(names, list):
        return {}
    return {index: str(name) for index, name in enumerate(names)}


def _normalize_label_text(value: object, *, names: dict[int, str] | None = None) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        if names and value in names:
            return names[value]
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _write_dialogue(handle: Any, *, source: str, prompt: str, answer: str) -> None:
    handle.write(
        json.dumps(
            {
                "kind": "dialogue",
                "source": f"reasoning_qa:{source}",
                "messages": [
                    {"role": "user", "segments": [{"kind": "text", "text": prompt}]},
                    {"role": "assistant", "segments": [{"kind": "text", "text": answer}]},
                ],
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def _build_anli_prompt(row: dict[str, Any], *, label_names: dict[int, str]) -> tuple[str, str] | None:
    premise = row.get("premise")
    hypothesis = row.get("hypothesis")
    label = _normalize_label_text(row.get("label"), names=label_names)
    if not isinstance(premise, str) or not premise.strip():
        return None
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    if label is None:
        return None
    reason = row.get("reason")
    answer = f"label: {label}"
    if isinstance(reason, str) and reason.strip():
        answer = f"{answer}\nreason: {reason.strip()}"
    prompt = (
        "Natural language inference task.\n"
        f"Premise: {premise.strip()}\n"
        f"Hypothesis: {hypothesis.strip()}\n"
        "Answer with one label: entailment, neutral, or contradiction."
    )
    return prompt, answer


def _build_proofwriter_prompt(row: dict[str, Any]) -> tuple[str, str] | None:
    theory = row.get("theory")
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(theory, str) or not theory.strip():
        return None
    if not isinstance(question, str) or not question.strip():
        return None
    answer_text = _normalize_label_text(answer)
    if answer_text is None:
        return None
    prompt = (
        "Reasoning task.\n"
        f"Theory: {theory.strip()}\n"
        f"Question: {question.strip()}\n"
        "Answer with one of: True, False, Unknown."
    )
    proofs = row.get("allProofs")
    response = answer_text
    if isinstance(proofs, str) and proofs.strip():
        response = f"{response}\nproof: {proofs.strip()}"
    return prompt, response


def _build_logicnli_prompt(row: dict[str, Any], *, label_names: dict[int, str]) -> tuple[str, str] | None:
    premise = row.get("premise")
    hypothesis = row.get("hypothesis")
    label = _normalize_label_text(row.get("label"), names=label_names)
    if not isinstance(premise, str) or not premise.strip():
        return None
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    if label is None:
        return None
    prompt = (
        "Logical natural language inference task.\n"
        f"Premise: {premise.strip()}\n"
        f"Hypothesis: {hypothesis.strip()}\n"
        "Answer with the correct entailment label."
    )
    return prompt, label


def _build_boardgameqa_prompt(row: dict[str, Any]) -> tuple[str, str] | None:
    example = row.get("example")
    goal = row.get("goal")
    label = _normalize_label_text(row.get("label"))
    if not isinstance(example, str) or not example.strip():
        return None
    if not isinstance(goal, str) or not goal.strip():
        return None
    if label is None:
        return None
    parts = [
        "Boardgame reasoning task.",
        f"Scenario: {example.strip()}",
        f"Goal: {goal.strip()}",
        "Answer with one of: proved, disproved, unknown.",
    ]
    rules = row.get("rules")
    preferences = row.get("preferences")
    if isinstance(rules, str) and rules.strip():
        parts.insert(2, f"Rules: {rules.strip()}")
    if isinstance(preferences, str) and preferences.strip():
        parts.insert(3, f"Preferences: {preferences.strip()}")
    prompt = "\n".join(parts)
    proof = row.get("proof")
    answer = label
    if isinstance(proof, str) and proof.strip():
        answer = f"{answer}\nproof: {proof.strip()}"
    return prompt, answer


def _build_defeasible_nli_prompt(row: dict[str, Any]) -> tuple[str, str] | None:
    hypothesis = row.get("Hypothesis")
    update = row.get("Update")
    update_type = _normalize_label_text(row.get("UpdateType"))
    premise = row.get("Premise")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None
    if not isinstance(update, str) or not update.strip():
        return None
    if update_type is None:
        return None
    parts = ["Defeasible reasoning task."]
    if isinstance(premise, str) and premise.strip():
        parts.append(f"Premise: {premise.strip()}")
    parts.append(f"Hypothesis: {hypothesis.strip()}")
    parts.append(f"Update: {update.strip()}")
    parts.append("Does the update strengthen or weaken the hypothesis?")
    return "\n".join(parts), update_type


def _row_to_dialogue(row: dict[str, Any], *, template: str, label_names: dict[int, str]) -> tuple[str, str] | None:
    if template == "anli":
        return _build_anli_prompt(row, label_names=label_names)
    if template == "proofwriter":
        return _build_proofwriter_prompt(row)
    if template == "logicnli":
        return _build_logicnli_prompt(row, label_names=label_names)
    if template == "boardgameqa":
        return _build_boardgameqa_prompt(row)
    if template == "defeasible_nli":
        return _build_defeasible_nli_prompt(row)
    raise ValueError(f"Unsupported reasoning template: {template}")


def add_hf_reasoning_source(
    *,
    output_handle: Any,
    source: ReasoningSource,
    streaming: bool,
    max_records: int | None,
) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install datasets to build the reasoning dialogue corpus.") from exc

    dataset = load_dataset(source.dataset, source.config, split=source.split, streaming=streaming)
    label_names = _class_names(dataset, "label")
    written = 0
    for record_index, row in enumerate(dataset, start=1):
        if max_records is not None and written >= max_records:
            break
        if not isinstance(row, dict):
            continue
        prompt_answer = _row_to_dialogue(row, template=source.template, label_names=label_names)
        if prompt_answer is None:
            continue
        prompt, answer = prompt_answer
        _write_dialogue(
            output_handle,
            source=f"{source.label}:{record_index}",
            prompt=prompt,
            answer=answer,
        )
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reasoning dialogue corpus from HF reasoning/NLI datasets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hf-reasoning-source", action="append", default=[])
    parser.add_argument("--no-default-sources", action="store_true")
    parser.add_argument("--hf-streaming", action="store_true", default=True)
    parser.add_argument("--no-hf-streaming", dest="hf_streaming", action="store_false")
    parser.add_argument("--max-records-per-source", type=int, default=50000)
    parser.add_argument("--skip-failed-sources", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_specs = ([] if args.no_default_sources else list(DEFAULT_REASONING_SOURCES)) + args.hf_reasoning_source
    sources = [parse_reasoning_source(spec) for spec in source_specs]
    source_counts: dict[str, int] = {}
    started = time.perf_counter()

    with output.open("w", encoding="utf-8") as handle:
        for source in sources:
            try:
                count = add_hf_reasoning_source(
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

    meta = {
        "output": str(output),
        "total_documents": sum(source_counts.values()),
        "source_counts": source_counts,
        "reasoning_sources": [asdict(source) for source in sources],
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(output.with_suffix(output.suffix + ".meta.json"), meta)
    print(f"output={output} total_documents={meta['total_documents']:,}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
