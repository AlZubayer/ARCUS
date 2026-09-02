"""SUITE dataset adapter, written against the inspected upstream schema.

Canonical source, pinned (docs/module_a/06_IMPLEMENTATION_PLAN.md P0):

* ``apeleg/SUITE``            rev 3f5f6b0897dac10baacf1aa8b35319a02abccd23
* ``apeleg/SUITE-rephrasings`` rev 81a52d60ec7d3231169b16a54ad1b2a58221ca6e

Both carry ``topic``/``question``/``answer``/``label``; the rephrasings set additionally
carries 15 wide augmentation columns that this module melts into long form.

The adapter loads a pinned revision, maps rows into :class:`FactExample` without guessing,
preserves source identifiers, and fails closed when a required field cannot be resolved.
It must NOT decide whether the model knows a fact -- that is A0.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .schema import (
    ControlType,
    FactExample,
    FactKey,
    Generator,
    Modality,
    Split,
    normalize_question,
    parse_augmentation,
    parse_label,
    stable_surface_id,
)

ADAPTER_VERSION = "suite_adapter_v2"
SPLIT_POLICY_VERSION = "split_policy_provenance_v1"

SUITE_DATASET_ID = "apeleg/SUITE"
SUITE_REVISION = "3f5f6b0897dac10baacf1aa8b35319a02abccd23"
REPHRASINGS_DATASET_ID = "apeleg/SUITE-rephrasings"
REPHRASINGS_REVISION = "81a52d60ec7d3231169b16a54ad1b2a58221ca6e"

SUITE_SPLITS: tuple[str, ...] = ("forget_train", "forget_eval", "retain_train", "retain_eval")
REPHRASINGS_SPLIT = "forget_eval_rephrasings"

REQUIRED_COLUMNS: tuple[str, ...] = ("topic", "question", "answer", "label")

#: Wide augmentation columns in ``SUITE-rephrasings``, melted into surface forms.
REPHRASING_PARAPHRASE_COLUMNS: tuple[str, ...] = tuple(f"q_gemini{i}" for i in range(1, 11))
REPHRASING_FILL_IN_COLUMNS: tuple[str, ...] = tuple(f"blank_gemini{i}" for i in range(1, 6))
REPHRASING_AUGMENTATION_COLUMNS: tuple[str, ...] = (
    REPHRASING_PARAPHRASE_COLUMNS + REPHRASING_FILL_IN_COLUMNS
)

TOPICS: tuple[str, ...] = (
    "challenger_disaster",
    "salem_witch_trials",
    "steve_jobs_medical",
    "britney_spears_conservatorship",
)


class DatasetSchemaError(RuntimeError):
    """Raised when the upstream dataset does not match the pinned, audited schema."""


@dataclass(frozen=True)
class DroppedRow:
    """A surface form deliberately excluded, with a machine-readable reason.

    Every filtering decision writes a reason (06_IMPLEMENTATION_PLAN.md, coding conventions).
    """

    surface_form_id: str
    raw_label: str
    topic: str
    source_split: str
    reason: str


@dataclass
class SuiteCorpus:
    """Normalized SUITE corpus with splits assigned and drops recorded."""

    examples: list[FactExample] = field(default_factory=list)
    dropped: list[DroppedRow] = field(default_factory=list)
    source_revisions: dict[str, str] = field(default_factory=dict)

    def by_split(self, split: Split) -> list[FactExample]:
        return [ex for ex in self.examples if ex.split is split]

    def forget(self) -> list[FactExample]:
        return [ex for ex in self.examples if ex.is_forget]

    def retain(self) -> list[FactExample]:
        return [ex for ex in self.examples if not ex.is_forget]

    def facts(self) -> dict[FactKey, list[FactExample]]:
        grouped: dict[FactKey, list[FactExample]] = defaultdict(list)
        for ex in self.examples:
            if ex.fact_key is not None and ex.is_forget:
                grouped[ex.fact_key].append(ex)
        return dict(grouped)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_split(dataset_id: str, revision: str, split: str) -> list[dict[str, Any]]:
    """Load one pinned split as plain dicts, preserving row order.

    ``revision`` is required: an unpinned dataset cannot appear in a run manifest.
    """
    if not revision:
        raise DatasetSchemaError(
            f"Refusing to load {dataset_id}:{split} without a pinned revision. "
            "Pin the canonical revision in the experiment config."
        )
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, revision=revision)
    missing = [c for c in REQUIRED_COLUMNS if c not in ds.column_names]
    if missing:
        raise DatasetSchemaError(
            f"{dataset_id}:{split} is missing required columns {missing}; "
            f"found {sorted(ds.column_names)}. Re-run the dataset audit."
        )
    return [dict(row) for row in ds]


def row_to_example(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset: str,
    revision: str,
    source_split: str,
) -> FactExample:
    """Normalize one narrow SUITE row (forget_*/retain_*). Fails closed."""
    missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c] is None]
    if missing:
        raise DatasetSchemaError(
            f"{dataset}:{source_split} row {row_index} missing required fields {missing}"
        )

    topic = str(row["topic"])
    label = str(row["label"])
    question = str(row["question"])
    answer = str(row["answer"])
    parsed = parse_label(topic, label)
    aug = parsed.augmentation

    return FactExample(
        surface_form_id=stable_surface_id(
            dataset=dataset,
            source_split=source_split,
            row_index=row_index,
            label=label,
            augmentation_id=aug.augmentation_id,
            question=question,
            answer=answer,
        ),
        topic=topic,
        question=question,
        answer=answer,
        raw_label=label,
        control_type=parsed.control_type,
        fact_key=parsed.fact_key,
        modality=parsed.modality,
        surface_kind=aug.surface_kind,
        generator=aug.generator,
        augmentation_id=aug.augmentation_id,
        semantic_tier=parsed.semantic_tier,
        domain=parsed.domain,
        linked_fact_key=parsed.linked_fact_key,
        linked_modality=parsed.linked_modality,
        source_dataset=dataset,
        source_revision=revision,
        source_split=source_split,
        source_row_index=row_index,
        source_column="question",
    )


def melt_rephrasing_row(
    row: dict[str, Any],
    *,
    row_index: int,
    dataset: str = REPHRASINGS_DATASET_ID,
    revision: str = REPHRASINGS_REVISION,
    source_split: str = REPHRASINGS_SPLIT,
) -> list[FactExample]:
    """Melt one wide rephrasings row into up to 16 long surface forms.

    All emitted forms share ``fact_key``, ``modality`` and ``answer`` from the parent row
    and differ only in ``augmentation_id`` / ``surface_kind`` / ``source_column``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in row or row[c] is None]
    if missing:
        raise DatasetSchemaError(
            f"{dataset}:{source_split} row {row_index} missing required fields {missing}"
        )

    topic = str(row["topic"])
    label = str(row["label"])
    answer = str(row["answer"])
    parsed = parse_label(topic, label)
    if parsed.control_type is not ControlType.FORGET:
        raise DatasetSchemaError(
            f"{dataset}:{source_split} row {row_index} has non-forget label {label!r}"
        )

    columns: list[tuple[str, str]] = [("question", "original")]
    columns += [(c, c) for c in REPHRASING_AUGMENTATION_COLUMNS if c in row]

    out: list[FactExample] = []
    for column, augmentation_id in columns:
        text = row.get(column)
        if text is None or not str(text).strip():
            continue
        question = str(text)
        aug = parse_augmentation(augmentation_id)
        out.append(
            FactExample(
                surface_form_id=stable_surface_id(
                    dataset=dataset,
                    source_split=source_split,
                    row_index=row_index,
                    label=label,
                    augmentation_id=aug.augmentation_id,
                    question=question,
                    answer=answer,
                ),
                topic=topic,
                question=question,
                answer=answer,
                raw_label=label,
                control_type=ControlType.FORGET,
                fact_key=parsed.fact_key,
                modality=parsed.modality,
                surface_kind=aug.surface_kind,
                generator=aug.generator,
                augmentation_id=aug.augmentation_id,
                source_dataset=dataset,
                source_revision=revision,
                source_split=source_split,
                source_row_index=row_index,
                source_column=column,
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Split assignment
# --------------------------------------------------------------------------------------


def assign_split(example: FactExample) -> tuple[Split | None, str | None]:
    """Provenance-based split policy (``split_policy_provenance_v1``).

    Splits follow generator provenance rather than random rows, so no generated paraphrase
    group can straddle a fold (02_DATA_AND_SPLITS.md section 4).

    * discovery  -- ``forget_train`` (Claude augmentations; direct/reverse only)
    * validation -- ``SUITE-rephrasings`` Gemini augmentations, direct/reverse
    * stress     -- ``SUITE-rephrasings`` indirect modality (a modality absent from
      training entirely, so it can never have informed candidate selection)

    The ``original`` surface in the rephrasings set is textually identical to the
    ``@original`` row already present in ``forget_train``; for direct/reverse it is
    dropped here so discovery and validation share no identical prompt. It is retained
    for indirect, where no training counterpart exists.
    """
    if not example.is_forget:
        return None, None

    if example.source_split == "forget_train":
        return Split.DISCOVERY, None

    if example.source_split == REPHRASINGS_SPLIT:
        if example.modality is Modality.INDIRECT:
            return Split.STRESS, None
        if example.generator is Generator.NONE:
            return None, "duplicate_of_discovery_original"
        return Split.VALIDATION, None

    # forget_eval originals are documented upstream as reference-only; forgetting is
    # measured on the rephrasings (discrepancy D4).
    if example.source_split == "forget_eval":
        return None, "forget_eval_is_reference_only_upstream"

    return None, "unassigned_source_split"


def build_corpus(
    *,
    topics: Sequence[str] | None = None,
    suite_dataset_id: str = SUITE_DATASET_ID,
    suite_revision: str = SUITE_REVISION,
    rephrasings_dataset_id: str = REPHRASINGS_DATASET_ID,
    rephrasings_revision: str = REPHRASINGS_REVISION,
) -> SuiteCorpus:
    """Load, normalize and split the full SUITE corpus for the requested topics."""
    wanted = set(topics) if topics else None
    corpus = SuiteCorpus(
        source_revisions={
            suite_dataset_id: suite_revision,
            rephrasings_dataset_id: rephrasings_revision,
        }
    )

    for split_name in SUITE_SPLITS:
        for idx, row in enumerate(load_split(suite_dataset_id, suite_revision, split_name)):
            if wanted is not None and str(row["topic"]) not in wanted:
                continue
            ex = row_to_example(
                row,
                row_index=idx,
                dataset=suite_dataset_id,
                revision=suite_revision,
                source_split=split_name,
            )
            _place(corpus, ex)

    for idx, row in enumerate(
        load_split(rephrasings_dataset_id, rephrasings_revision, REPHRASINGS_SPLIT)
    ):
        if wanted is not None and str(row["topic"]) not in wanted:
            continue
        for ex in melt_rephrasing_row(
            row,
            row_index=idx,
            dataset=rephrasings_dataset_id,
            revision=rephrasings_revision,
        ):
            _place(corpus, ex)

    _drop_cross_split_duplicate_prompts(corpus)
    return corpus


#: Splits ranked by which copy of a colliding prompt is kept. Discovery wins because it is
#: where candidate selection happens; a validation item identical to a discovery item is by
#: definition not held out, so it is the copy that must go.
_SPLIT_PRIORITY: dict[Split, int] = {Split.DISCOVERY: 0, Split.VALIDATION: 1, Split.STRESS: 2}

DEDUPLICATION_POLICY_VERSION = "exact_cross_split_prompt_dedup_v1"


def _drop_cross_split_duplicate_prompts(corpus: SuiteCorpus) -> None:
    """Enforce "no identical prompt in discovery and validation" (02_DATA section 9).

    The two augmentation generators were run independently, and for fill-in-the-blank
    templates especially, Gemini sometimes reproduced a Claude phrasing verbatim. Those
    collisions are real leakage, so the lower-priority copy is dropped with a reason.

    Only *exact* normalized-prompt collisions are removed here. Near-duplicates are
    reported by the audit rather than silently filtered, because the cutoff for "too
    similar" is a preregistration decision, not an implementation detail.
    """
    best: dict[str, FactExample] = {}
    for ex in corpus.examples:
        if ex.split is None:
            continue
        key = normalize_question(ex.question)
        current = best.get(key)
        if current is None or _dedup_sort_key(ex) < _dedup_sort_key(current):
            best[key] = ex

    kept: list[FactExample] = []
    for ex in corpus.examples:
        if ex.split is None:
            kept.append(ex)
            continue
        key = normalize_question(ex.question)
        winner = best[key]
        if ex.surface_form_id == winner.surface_form_id:
            kept.append(ex)
        else:
            corpus.dropped.append(
                DroppedRow(
                    surface_form_id=ex.surface_form_id,
                    raw_label=ex.raw_label,
                    topic=ex.topic,
                    source_split=ex.source_split or "",
                    reason=(
                        f"identical_prompt_in_{winner.split.value}"
                        if winner.split is not ex.split
                        else "duplicate_prompt_within_split"
                    ),
                )
            )
    corpus.examples = kept


def _dedup_sort_key(example: FactExample) -> tuple[int, str, int, str]:
    """Deterministic winner selection: split priority, then stable source coordinates."""
    return (
        _SPLIT_PRIORITY.get(example.split, 99) if example.split else 99,
        example.source_split or "",
        example.source_row_index if example.source_row_index is not None else 0,
        example.augmentation_id,
    )


def _place(corpus: SuiteCorpus, example: FactExample) -> None:
    split, reason = assign_split(example)
    if reason is not None:
        corpus.dropped.append(
            DroppedRow(
                surface_form_id=example.surface_form_id,
                raw_label=example.raw_label,
                topic=example.topic,
                source_split=example.source_split or "",
                reason=reason,
            )
        )
        return
    # Retain rows carry no forget split; they are indexed by control_type instead.
    corpus.examples.append(example if split is None else _with_split(example, split))


def _with_split(example: FactExample, split: Split) -> FactExample:
    from dataclasses import replace

    return replace(example, split=split)


# --------------------------------------------------------------------------------------
# Grouping helpers
# --------------------------------------------------------------------------------------


def group_by_fact(examples: Iterable[FactExample]) -> dict[FactKey, list[FactExample]]:
    grouped: dict[FactKey, list[FactExample]] = defaultdict(list)
    for ex in examples:
        if ex.fact_key is not None:
            grouped[ex.fact_key].append(ex)
    return dict(grouped)


def group_by_fact_modality(
    examples: Iterable[FactExample],
) -> dict[tuple[FactKey, Modality], list[FactExample]]:
    grouped: dict[tuple[FactKey, Modality], list[FactExample]] = defaultdict(list)
    for ex in examples:
        if ex.fact_key is not None and ex.modality is not None:
            grouped[(ex.fact_key, ex.modality)].append(ex)
    return dict(grouped)


def syntax_controls_by_link(
    examples: Iterable[FactExample],
) -> dict[tuple[FactKey, Modality, str], FactExample]:
    """Index the R3 same-syntax controls by (linked fact, linked modality, augmentation).

    ``retain_train`` Syntax rows reuse the exact question template of a specific forget
    surface form, so this index yields an augmentation-matched control (discrepancy D7).
    """
    index: dict[tuple[FactKey, Modality, str], FactExample] = {}
    for ex in examples:
        if (
            ex.control_type is ControlType.SYNTACTIC
            and ex.linked_fact_key is not None
            and ex.linked_modality is not None
        ):
            index[(ex.linked_fact_key, ex.linked_modality, ex.augmentation_id)] = ex
    return index


def example_to_row(example: FactExample) -> dict[str, Any]:
    """Flatten one example for JSONL persistence, keeping identity fields separate."""
    return {
        "surface_form_id": example.surface_form_id,
        "fact_key": (
            {"topic": example.fact_key.topic, "fact_id": example.fact_key.fact_id}
            if example.fact_key
            else None
        ),
        "topic": example.topic,
        "question": example.question,
        "answer": example.answer,
        "raw_label": example.raw_label,
        "control_type": example.control_type.value,
        "modality": example.modality.value if example.modality else None,
        "surface_kind": example.surface_kind.value,
        "generator": example.generator.value,
        "augmentation_id": example.augmentation_id,
        "semantic_tier": example.semantic_tier,
        "domain": example.domain,
        "linked_fact_key": (
            {"topic": example.linked_fact_key.topic, "fact_id": example.linked_fact_key.fact_id}
            if example.linked_fact_key
            else None
        ),
        "linked_modality": example.linked_modality.value if example.linked_modality else None,
        "split": example.split.value if example.split else None,
        "group_id": example.group_id,
        "source_dataset": example.source_dataset,
        "source_revision": example.source_revision,
        "source_split": example.source_split,
        "source_row_index": example.source_row_index,
        "source_column": example.source_column,
    }
