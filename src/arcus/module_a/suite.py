from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Iterable

from .schema import ControlType, FactExample, FactKey, Modality, classify_retain_label, parse_forget_label


def stable_surface_id(question: str, answer: str, label: str) -> str:
    payload = f"{label}\n{question}\n{answer}".encode("utf-8")
    return sha256(payload).hexdigest()[:16]


def row_to_example(row: dict, split: str) -> FactExample:
    """Normalize one SUITE-like row.

    The loader intentionally requires explicit `topic`, `label`, `question`, and `answer` fields.
    If the upstream dataset schema changes, update this adapter and its audit instead of silently guessing.
    """
    required = {"topic", "label", "question", "answer"}
    missing = required.difference(row)
    if missing:
        raise KeyError(f"Dataset row missing required fields: {sorted(missing)}")

    topic = str(row["topic"])
    label = str(row["label"])
    question = str(row["question"])
    answer = str(row["answer"])
    surface_id = str(row.get("surface_form_id") or stable_surface_id(question, answer, label))

    if split.startswith("forget"):
        fact_key, modality = parse_forget_label(topic, label)
        if fact_key is None:
            raise ValueError(f"Unrecognized forget label {label!r}; do not infer fact identity from text")
        return FactExample(
            topic=topic,
            question=question,
            answer=answer,
            raw_label=label,
            fact_key=fact_key,
            modality=modality,
            control_type=ControlType.FORGET,
            surface_form_id=surface_id,
            source_split=split,
        )

    control_type, tier = classify_retain_label(label)
    return FactExample(
        topic=topic,
        question=question,
        answer=answer,
        raw_label=label,
        fact_key=None,
        modality=Modality.UNKNOWN,
        control_type=control_type,
        semantic_tier=tier,
        surface_form_id=surface_id,
        source_split=split,
    )


def load_hf_split(dataset_name: str, split: str, topic: str | None = None) -> list[FactExample]:
    """Load a configured Hugging Face dataset split lazily.

    No dataset identifier is hard-coded because Module A must pin the exact dataset/revision used in
    the experiment configuration once the team verifies the canonical source.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    rows = ds if topic is None else (row for row in ds if str(row["topic"]) == topic)
    return [row_to_example(dict(row), split=split) for row in rows]


def group_forget_by_fact(examples: Iterable[FactExample]) -> dict[FactKey, list[FactExample]]:
    grouped: dict[FactKey, list[FactExample]] = defaultdict(list)
    for ex in examples:
        if ex.fact_key is not None:
            grouped[ex.fact_key].append(ex)
    return dict(grouped)


def dataset_audit(examples: Iterable[FactExample]) -> dict:
    examples = list(examples)
    controls: dict[str, int] = defaultdict(int)
    modalities: dict[str, int] = defaultdict(int)
    facts: dict[str, int] = defaultdict(int)
    for ex in examples:
        controls[ex.control_type.value] += 1
        modalities[ex.modality.value] += 1
        if ex.fact_key is not None:
            facts[f"{ex.fact_key.topic}:{ex.fact_key.fact_id}"] += 1
    return {
        "n_examples": len(examples),
        "control_types": dict(sorted(controls.items())),
        "modalities": dict(sorted(modalities.items())),
        "fact_counts": dict(sorted(facts.items())),
    }
