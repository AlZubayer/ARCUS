"""Dataset audit (Gate G0).

Produces the machine-readable evidence that the pinned SUITE revision really has the
schema the adapter assumes, that fact identity is deterministically reconstructible, and
that discovery/validation/stress folds do not leak into one another.

This runs before any model is loaded. It must be able to fail the run.
"""

from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .schema import (
    ANSWER_NORMALIZATION_VERSION,
    SCHEMA_VERSION,
    ControlType,
    FactExample,
    Split,
    normalize_answer,
    normalize_question,
)
from .suite import (
    ADAPTER_VERSION,
    REPHRASING_AUGMENTATION_COLUMNS,
    REPHRASINGS_DATASET_ID,
    REPHRASINGS_SPLIT,
    SPLIT_POLICY_VERSION,
    SUITE_DATASET_ID,
    SUITE_SPLITS,
    SuiteCorpus,
)

AUDIT_VERSION = "dataset_audit_v1"

#: difflib similarity at or above which two prompts count as near-duplicates.
NEAR_DUPLICATE_THRESHOLD = 0.95


@dataclass(frozen=True)
class AuditFailure:
    """A Gate G0 violation. Any of these stops the run."""

    check: str
    detail: str


def _fact_id(example: FactExample) -> str | None:
    return example.fact_key.id if example.fact_key else None


def answer_degeneracy(examples: Iterable[FactExample]) -> dict[str, Any]:
    """Distinct-answer count per ``(topic, modality)`` over forget facts.

    This is the check that mechanically surfaces discrepancy D3: SUITE reverse questions
    ask for the fact from the answer side, and within a topic they nearly all resolve to
    the same entity. Where ``distinct_answers`` collapses toward 1, a same-topic distractor
    pool cannot separate facts and the factual margin is undefined for that cell.
    """
    per_cell: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for ex in examples:
        if ex.is_forget and ex.fact_key is not None and ex.modality is not None:
            per_cell[(ex.topic, ex.modality.value)][ex.fact_key.fact_id] = ex.answer

    out: dict[str, Any] = {}
    for (topic, modality), answers in sorted(per_cell.items()):
        distinct = {normalize_answer(a) for a in answers.values()}
        n_facts = len(answers)
        ratio = len(distinct) / n_facts if n_facts else 0.0
        out[f"{topic}|{modality}"] = {
            "n_facts": n_facts,
            "distinct_answers": len(distinct),
            "distinct_ratio": round(ratio, 4),
            "usable_for_same_topic_distractors": len(distinct) == n_facts,
            "most_common_answers": [
                {"answer": a, "n_facts": c}
                for a, c in Counter(normalize_answer(v) for v in answers.values()).most_common(3)
            ],
        }
    return out


def duplicate_report(examples: Sequence[FactExample]) -> dict[str, Any]:
    """Exact-duplicate questions and question/answer pairs, and cross-split prompt overlap."""
    by_question: dict[str, list[FactExample]] = defaultdict(list)
    by_qa: dict[tuple[str, str], list[FactExample]] = defaultdict(list)
    for ex in examples:
        by_question[normalize_question(ex.question)].append(ex)
        by_qa[(normalize_question(ex.question), normalize_answer(ex.answer))].append(ex)

    dup_q = {k: v for k, v in by_question.items() if len(v) > 1}
    dup_qa = {k: v for k, v in by_qa.items() if len(v) > 1}

    # Cross-split identical prompts are the leakage condition that matters.
    split_of: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        if ex.split is not None:
            split_of[normalize_question(ex.question)].add(ex.split.value)
    cross_split = {q: sorted(s) for q, s in split_of.items() if len(s) > 1}

    return {
        "n_duplicate_normalized_questions": len(dup_q),
        "n_duplicate_question_answer_pairs": len(dup_qa),
        "n_cross_split_identical_prompts": len(cross_split),
        "cross_split_examples": [
            {"question": q, "splits": s} for q, s in list(cross_split.items())[:10]
        ],
        "duplicate_question_examples": [
            {
                "question": q,
                "n": len(v),
                "labels": sorted({e.raw_label for e in v})[:6],
                "splits": sorted({e.split.value for e in v if e.split}),
            }
            for q, v in list(dup_q.items())[:10]
        ],
    }


def near_duplicate_report(
    examples: Sequence[FactExample],
    *,
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
    max_pairs_reported: int = 25,
) -> dict[str, Any]:
    """Cross-split near-duplicate scan.

    Compared only across discovery/validation/stress and only within a fact, because that
    is where a near-duplicate would actually leak candidate-selection information. Blocking
    per fact keeps this quadratic scan tractable.
    """
    by_fact: dict[str, list[FactExample]] = defaultdict(list)
    for ex in examples:
        if ex.split is not None and ex.fact_key is not None:
            by_fact[ex.fact_key.id].append(ex)

    flagged: list[dict[str, Any]] = []
    scores: list[float] = []
    for fact, group in sorted(by_fact.items()):
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.split is b.split:
                    continue
                ratio = difflib.SequenceMatcher(
                    None, normalize_question(a.question), normalize_question(b.question)
                ).ratio()
                scores.append(ratio)
                if ratio >= threshold:
                    flagged.append(
                        {
                            "fact_key": fact,
                            "ratio": round(ratio, 4),
                            "a": {
                                "surface_form_id": a.surface_form_id,
                                "split": a.split.value,
                                "augmentation_id": a.augmentation_id,
                                "question": a.question,
                            },
                            "b": {
                                "surface_form_id": b.surface_form_id,
                                "split": b.split.value,
                                "augmentation_id": b.augmentation_id,
                                "question": b.question,
                            },
                        }
                    )
    scores.sort()

    def pct(p: float) -> float:
        if not scores:
            return 0.0
        return round(scores[min(len(scores) - 1, int(p * len(scores)))], 4)

    return {
        "threshold": threshold,
        "method": "difflib.SequenceMatcher on normalized questions, cross-split within fact",
        "n_comparisons": len(scores),
        "n_flagged": len(flagged),
        "similarity_percentiles": {
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p99": pct(0.99),
            "max": round(scores[-1], 4) if scores else 0.0,
        },
        "flagged_pairs": flagged[:max_pairs_reported],
    }


def leakage_checks(corpus: SuiteCorpus) -> tuple[dict[str, Any], list[AuditFailure]]:
    """Explicit checks from 02_DATA_AND_SPLITS.md section 9."""
    failures: list[AuditFailure] = []
    examples = corpus.examples

    dups = duplicate_report(examples)
    if dups["n_cross_split_identical_prompts"] > 0:
        failures.append(
            AuditFailure(
                check="no_identical_prompt_across_splits",
                detail=(
                    f"{dups['n_cross_split_identical_prompts']} normalized prompts appear in "
                    "more than one split"
                ),
            )
        )

    # A generated paraphrase group must live entirely inside one split.
    group_splits: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        if ex.split is not None:
            group_splits[ex.group_id].add(ex.split.value)
    straddling = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    if straddling:
        failures.append(
            AuditFailure(
                check="no_surface_group_split_across_folds",
                detail=f"{len(straddling)} groups straddle folds: {list(straddling)[:5]}",
            )
        )

    # No sink-derived annotation may exist in an A0-A3 table.
    sink_tokens = ("sink", "bos_anchor", "anchor_head", "carrier")
    sink_fields = ["raw_label", "domain", "augmentation_id"]
    contaminated = [
        ex.raw_label
        for ex in examples
        if any(
            token in str(getattr(ex, field_name) or "").casefold()
            for field_name in sink_fields
            for token in sink_tokens
        )
    ]
    if contaminated:
        failures.append(
            AuditFailure(
                check="no_sink_derived_fields_in_a0_tables",
                detail=f"{len(contaminated)} rows carry sink-like annotations",
            )
        )

    return {
        "duplicates": dups,
        "n_groups": len(group_splits),
        "n_groups_straddling_folds": len(straddling),
        "sink_annotation_fields_present": bool(contaminated),
        "checked_sink_field_names": sink_fields,
    }, failures


def fact_identity_checks(corpus: SuiteCorpus) -> tuple[dict[str, Any], list[AuditFailure]]:
    """Every forget surface must carry a deterministic fact key and modality."""
    failures: list[AuditFailure] = []
    missing_key = [ex.raw_label for ex in corpus.forget() if ex.fact_key is None]
    missing_mod = [ex.raw_label for ex in corpus.forget() if ex.modality is None]
    if missing_key:
        failures.append(
            AuditFailure("every_forget_row_has_fact_key", f"{len(missing_key)} rows lack a fact key")
        )
    if missing_mod:
        failures.append(
            AuditFailure("every_forget_row_has_modality", f"{len(missing_mod)} rows lack a modality")
        )

    per_topic: dict[str, set[str]] = defaultdict(set)
    for ex in corpus.forget():
        if ex.fact_key is not None:
            per_topic[ex.topic].add(ex.fact_key.fact_id)

    return {
        "n_forget_surface_forms": len(corpus.forget()),
        "facts_per_topic": {t: len(v) for t, v in sorted(per_topic.items())},
        "fact_ids_per_topic": {t: sorted(v, key=_fact_sort) for t, v in sorted(per_topic.items())},
        "n_missing_fact_key": len(missing_key),
        "n_missing_modality": len(missing_mod),
    }, failures


def _fact_sort(fact_id: str) -> tuple[str, int]:
    return fact_id[0], int(fact_id[1:])


def surface_form_census(corpus: SuiteCorpus) -> dict[str, Any]:
    """Counts along every axis the design requires to stay separate."""
    ex = corpus.examples
    per_fact_split: dict[str, Counter] = defaultdict(Counter)
    for e in ex:
        if e.fact_key is not None and e.split is not None:
            per_fact_split[e.fact_key.id][e.split.value] += 1

    counts = {
        split.value: {
            "n": len(corpus.by_split(split)),
            "by_modality": dict(
                Counter(str(e.modality) for e in corpus.by_split(split)).most_common()
            ),
            "by_surface_kind": dict(
                Counter(e.surface_kind.value for e in corpus.by_split(split)).most_common()
            ),
            "by_generator": dict(
                Counter(e.generator.value for e in corpus.by_split(split)).most_common()
            ),
        }
        for split in Split
    }

    modality_by_kind: Counter = Counter()
    for e in ex:
        if e.is_forget and e.modality is not None:
            modality_by_kind[f"{e.modality.value}|{e.surface_kind.value}"] += 1

    return {
        "n_examples": len(ex),
        "n_dropped": len(corpus.dropped),
        "drop_reasons": dict(Counter(d.reason for d in corpus.dropped).most_common()),
        "by_control_type": dict(Counter(e.control_type.value for e in ex).most_common()),
        "by_split": counts,
        "modality_x_surface_kind": dict(sorted(modality_by_kind.items())),
        "surface_forms_per_fact": {
            k: dict(v) for k, v in sorted(per_fact_split.items())
        },
        "semantic_tier_histogram": dict(
            sorted(
                Counter(
                    e.semantic_tier for e in ex if e.control_type is ControlType.SEMANTIC
                ).items(),
                key=lambda kv: (kv[0] is None, kv[0]),
            )
        ),
        "retain_categories": dict(
            Counter(e.control_type.value for e in corpus.retain()).most_common()
        ),
    }


def answer_census(corpus: SuiteCorpus) -> dict[str, Any]:
    """Answer cardinality and length, which drive distractor design and multi-token scoring."""
    answers = [e.answer for e in corpus.forget()]
    word_lengths = [len(a.split()) for a in answers]
    word_lengths.sort()

    def pct(p: float) -> int:
        if not word_lengths:
            return 0
        return word_lengths[min(len(word_lengths) - 1, int(p * len(word_lengths)))]

    return {
        "n_forget_answers": len(answers),
        "n_distinct_normalized_answers": len({normalize_answer(a) for a in answers}),
        "answer_word_length": {
            "min": word_lengths[0] if word_lengths else 0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "max": word_lengths[-1] if word_lengths else 0,
        },
        "fraction_multiword": (
            round(sum(1 for n in word_lengths if n > 1) / len(word_lengths), 4)
            if word_lengths
            else 0.0
        ),
        "degeneracy_by_topic_modality": answer_degeneracy(corpus.examples),
    }


SCAFFOLD_MISMATCHES: list[dict[str, str]] = [
    {
        "id": "D1",
        "severity": "blocking",
        "finding": (
            "The pre-existing adapter rejected 1775 of 1775 forget_train rows: its "
            "re.fullmatch(r'([KM]\\d+)-(direct|reverse|indirect)') grammar has no place for the "
            "'@augmentation' suffix every training row carries (e.g. 'M1-direct@q_claude9')."
        ),
        "resolution": "Rewrote the grammar with an optional @augmentation group; parsed 100% of rows.",
    },
    {
        "id": "D2",
        "severity": "design_conflict",
        "finding": (
            "02_DATA_AND_SPLITS.md lists fill-in as a query modality. The dataset encodes it as "
            "an augmentation ('blank_*') of a direct or reverse question, not as a modality."
        ),
        "resolution": "Added SurfaceKind alongside Modality so neither axis is collapsed.",
    },
    {
        "id": "D3",
        "severity": "scientific",
        "finding": (
            "Reverse-modality answers are near-degenerate within a topic (every challenger "
            "reverse answer is 'Challenger' or 'Challenger disaster'). The design's default "
            "distractor rule would therefore place the correct answer inside D_f, leaving the "
            "factual margin undefined for that cell."
        ),
        "resolution": (
            "answer_degeneracy() reports distinct-answer ratio per (topic, modality); the "
            "distractor builder refuses degenerate cells with reason 'degenerate_answer_pool'."
        ),
    },
    {
        "id": "D4",
        "severity": "design_conflict",
        "finding": (
            "Upstream documents forget_eval as reference-only; forgetting is measured on "
            "SUITE-rephrasings. The design pack did not distinguish the two."
        ),
        "resolution": "forget_eval is dropped with a reason; rephrasings supply validation/stress.",
    },
    {
        "id": "D5",
        "severity": "scientific",
        "finding": (
            "forget_train contains no indirect rows (direct 1475 / reverse 300 / indirect 0), so "
            "minimum_modalities_known >= 2 cannot be satisfied from discovery data alone."
        ),
        "resolution": "Indirect is assigned to the stress split; A0 coverage is computed per split.",
    },
    {
        "id": "D6",
        "severity": "intentional_divergence",
        "finding": (
            "Upstream scores correctness with a Qwen LLM judge over free-form generation."
        ),
        "resolution": (
            "ARCUS uses teacher-forced margins per 02_DATA_AND_SPLITS.md section 5. Generation is "
            "kept as a diagnostic only. Recorded so the numbers are never compared naively."
        ),
    },
    {
        "id": "D7",
        "severity": "opportunity",
        "finding": (
            "retain_train Syntax rows are row-aligned AND augmentation-matched to forget_train "
            "(e.g. 'Syntax-M1-direct@q_claude9' reuses that exact template for another entity)."
        ),
        "resolution": "Used as the same_syntax (R3) pair family via syntax_controls_by_link().",
    },
    {
        "id": "D8",
        "severity": "blocking",
        "finding": "pilot_challenger.yaml carried revision: null and dataset_name: null.",
        "resolution": "Config now pins model, tokenizer and both dataset revisions.",
    },
]


def build_dataset_audit(
    corpus: SuiteCorpus,
    *,
    topics: Sequence[str] | None = None,
    run_near_duplicates: bool = True,
) -> dict[str, Any]:
    """Assemble the full audit document. ``gate_g0_passed`` is the machine-readable verdict."""
    failures: list[AuditFailure] = []

    identity, identity_failures = fact_identity_checks(corpus)
    failures.extend(identity_failures)
    leakage, leakage_failures = leakage_checks(corpus)
    failures.extend(leakage_failures)

    audit: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "split_policy_version": SPLIT_POLICY_VERSION,
        "answer_normalization_version": ANSWER_NORMALIZATION_VERSION,
        "source": {
            "suite_dataset_id": SUITE_DATASET_ID,
            "rephrasings_dataset_id": REPHRASINGS_DATASET_ID,
            "revisions": corpus.source_revisions,
            "suite_splits": list(SUITE_SPLITS),
            "rephrasings_split": REPHRASINGS_SPLIT,
            "raw_field_names": ["topic", "question", "answer", "label"],
            "rephrasings_wide_columns": list(REPHRASING_AUGMENTATION_COLUMNS),
            "topics_requested": list(topics) if topics else "all",
        },
        "parsing_assumptions": {
            "fact_identity": (
                "Read from the label only. fact_key = (topic, {K|M}<n>). Never inferred from "
                "question text, never from answer text."
            ),
            "modality": "Label suffix -direct/-reverse/-indirect.",
            "surface_kind": (
                "Augmentation prefix: 'original' -> original, 'q_*' -> paraphrase, "
                "'blank_*' -> fill_in."
            ),
            "generator": "Augmentation infix: claude (forget_train) or gemini (rephrasings).",
            "split_policy": (
                "Provenance-based. discovery = forget_train; validation = rephrasings Gemini "
                "augmentations (direct/reverse); stress = rephrasings indirect. The rephrasings "
                "'original' surface duplicates the forget_train '@original' row and is dropped "
                "for direct/reverse."
            ),
            "unparseable_label_policy": "Raise LabelParseError. Never guess, never drop silently.",
        },
        "fact_identity": identity,
        "census": surface_form_census(corpus),
        "answers": answer_census(corpus),
        "leakage": leakage,
        "dropped_rows": [
            {
                "surface_form_id": d.surface_form_id,
                "raw_label": d.raw_label,
                "topic": d.topic,
                "source_split": d.source_split,
                "reason": d.reason,
            }
            for d in corpus.dropped
        ],
        "scaffold_mismatches": SCAFFOLD_MISMATCHES,
    }

    if run_near_duplicates:
        audit["near_duplicates"] = near_duplicate_report(corpus.examples)

    audit["gate_g0_failures"] = [{"check": f.check, "detail": f.detail} for f in failures]
    audit["gate_g0_passed"] = not failures
    return audit
