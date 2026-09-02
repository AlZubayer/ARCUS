"""A0: the Known-Fact Core.

Establishes that the base model robustly knows a fact before any mechanistic analysis
touches it. Without this gate, "no causal route found" is indistinguishable from "the
model never knew the fact".

    K_f     = (1/|Q_f|) sum_q 1[correct(q)]
    K_{f,m} = the same, restricted to modality m

Eligibility requires the configured overall accuracy AND the configured modality coverage.
Neither threshold is adjusted here. Excluded facts stay in the artifact with reason codes;
exclusion is never hidden preprocessing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

from ..backend.hf import HFBackend
from ..schema import ControlType, FactExample, FactKey, Modality, Split
from ..scoring import (
    DISTRACTOR_POLICY_VERSION,
    DegenerateDistractorPool,
    DistractorSet,
    build_distractor_set,
    candidate_answers_for,
    score_example,
)

A0_VERSION = "known_fact_core_v1"


class ExclusionReason:
    """Machine-readable reasons a fact or cell leaves the analysis."""

    DEGENERATE_POOL = "degenerate_answer_pool"
    BELOW_ACCURACY = "below_minimum_base_accuracy"
    INSUFFICIENT_MODALITIES = "insufficient_modality_coverage"
    NO_SCOREABLE_SURFACES = "no_scoreable_surfaces"


def build_distractor_sets(
    examples: Sequence[FactExample],
    *,
    all_forget_examples: Sequence[FactExample],
    count: int,
    seed: int,
    allowed_modalities: Iterable[str] | None = None,
) -> tuple[dict[tuple[FactKey, Modality], DistractorSet], list[dict[str, Any]]]:
    """Freeze one distractor pool per (fact, modality) cell, recording every refusal."""
    allowed = set(allowed_modalities) if allowed_modalities else None
    cells: dict[tuple[FactKey, Modality], str] = {}
    for ex in examples:
        if ex.is_forget and ex.fact_key is not None and ex.modality is not None:
            cells.setdefault((ex.fact_key, ex.modality), ex.answer)

    built: dict[tuple[FactKey, Modality], DistractorSet] = {}
    refusals: list[dict[str, Any]] = []
    for (fact_key, modality), answer in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if allowed is not None and modality.value not in allowed:
            refusals.append(
                {
                    "fact_key": fact_key.id,
                    "modality": modality.value,
                    "reason": "modality_not_configured",
                    "detail": f"known_fact_modalities restricts scoring to {sorted(allowed)}",
                }
            )
            continue
        candidates = candidate_answers_for(all_forget_examples, fact_key, modality)
        try:
            built[(fact_key, modality)] = build_distractor_set(
                fact_key, modality, answer, candidates, count=count, seed=seed
            )
        except DegenerateDistractorPool as exc:
            refusals.append(
                {
                    "fact_key": fact_key.id,
                    "modality": modality.value,
                    "reason": ExclusionReason.DEGENERATE_POOL,
                    "detail": str(exc),
                }
            )
    return built, refusals


def score_surfaces(
    backend: HFBackend,
    examples: Sequence[FactExample],
    distractor_sets: dict[tuple[FactKey, Modality], DistractorSet],
    *,
    answer_score: str = "mean_logprob",
    progress_every: int = 100,
) -> list[dict[str, Any]]:
    """Score every surface whose (fact, modality) cell has a usable distractor pool."""
    rows: list[dict[str, Any]] = []
    scoreable = [
        ex
        for ex in examples
        if ex.fact_key is not None
        and ex.modality is not None
        and (ex.fact_key, ex.modality) in distractor_sets
    ]
    for i, ex in enumerate(scoreable, start=1):
        score = score_example(
            backend, ex, distractor_sets[(ex.fact_key, ex.modality)], answer_score=answer_score
        )
        row = score.to_dict()
        row["split"] = ex.split.value if ex.split else None
        row["question"] = ex.question
        rows.append(row)
        if progress_every and i % progress_every == 0:
            print(f"    scored {i}/{len(scoreable)} surfaces", flush=True)
    return rows


def aggregate_facts(
    rows: Sequence[dict[str, Any]],
    *,
    minimum_base_accuracy: float,
    minimum_modalities_known: int,
) -> list[dict[str, Any]]:
    """Compute K_f and K_{f,m} and apply the preregistered eligibility gate."""
    by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['fact_key']['topic']}:{row['fact_key']['fact_id']}"
        by_fact[key].append(row)

    out: list[dict[str, Any]] = []
    for fact, fact_rows in sorted(by_fact.items()):
        by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in fact_rows:
            by_modality[row["modality"]].append(row)

        modality_stats = {}
        for modality, mod_rows in sorted(by_modality.items()):
            correct = sum(1 for r in mod_rows if r["is_correct"])
            margins = [r["factual_margin"] for r in mod_rows]
            modality_stats[modality] = {
                "n_surfaces": len(mod_rows),
                "n_correct": correct,
                "accuracy": correct / len(mod_rows),
                "mean_margin": sum(margins) / len(margins),
                "min_margin": min(margins),
                "max_margin": max(margins),
                "passes_threshold": correct / len(mod_rows) >= minimum_base_accuracy,
            }

        n_correct = sum(1 for r in fact_rows if r["is_correct"])
        overall = n_correct / len(fact_rows)
        margins = [r["factual_margin"] for r in fact_rows]
        modalities_passing = [m for m, s in modality_stats.items() if s["passes_threshold"]]

        reasons: list[str] = []
        if overall < minimum_base_accuracy:
            reasons.append(ExclusionReason.BELOW_ACCURACY)
        if len(modalities_passing) < minimum_modalities_known:
            reasons.append(ExclusionReason.INSUFFICIENT_MODALITIES)

        out.append(
            {
                "fact_key": {
                    "topic": fact.split(":", 1)[0],
                    "fact_id": fact.split(":", 1)[1],
                },
                "fact_id": fact,
                "n_surfaces": len(fact_rows),
                "n_correct": n_correct,
                "K_f": overall,
                "mean_margin": sum(margins) / len(margins),
                "min_margin": min(margins),
                "modalities_scored": sorted(modality_stats),
                "modalities_passing": sorted(modalities_passing),
                "K_f_by_modality": modality_stats,
                "eligible": not reasons,
                "exclusion_reasons": reasons,
                # Reported separately so a coverage failure is never mistaken for the model
                # not knowing the fact.
                "passes_accuracy_only": overall >= minimum_base_accuracy,
            }
        )
    return out


def run_a0(
    backend: HFBackend,
    corpus: Any,
    *,
    minimum_base_accuracy: float,
    minimum_modalities_known: int,
    distractor_count: int,
    seed: int,
    splits: Sequence[str],
    allowed_modalities: Iterable[str] | None = None,
    answer_score: str = "mean_logprob",
) -> dict[str, Any]:
    """Run the full A0 screen and return artifacts plus a summary."""
    wanted_splits = {Split(s) for s in splits}
    scored_examples = [
        ex for ex in corpus.examples if ex.is_forget and ex.split in wanted_splits
    ]
    all_forget = [ex for ex in corpus.examples if ex.is_forget]

    distractor_sets, refusals = build_distractor_sets(
        scored_examples,
        all_forget_examples=all_forget,
        count=distractor_count,
        seed=seed,
        allowed_modalities=allowed_modalities,
    )
    print(
        f"  distractor cells: {len(distractor_sets)} usable, {len(refusals)} refused",
        flush=True,
    )

    rows = score_surfaces(
        backend, scored_examples, distractor_sets, answer_score=answer_score
    )
    facts = aggregate_facts(
        rows,
        minimum_base_accuracy=minimum_base_accuracy,
        minimum_modalities_known=minimum_modalities_known,
    )

    eligible = [f for f in facts if f["eligible"]]
    accuracy_only = [f for f in facts if f["passes_accuracy_only"]]

    core = {
        "a0_version": A0_VERSION,
        "distractor_policy": DISTRACTOR_POLICY_VERSION,
        "thresholds": {
            "minimum_base_accuracy": minimum_base_accuracy,
            "minimum_modalities_known": minimum_modalities_known,
            "distractor_count": distractor_count,
            "note": "Preregistered. Not adjusted after inspecting results.",
        },
        "splits_scored": sorted(s.value for s in wanted_splits),
        "n_surfaces_scored": len(rows),
        "n_facts_screened": len(facts),
        "n_eligible": len(eligible),
        "n_passing_accuracy_only": len(accuracy_only),
        "eligible_fact_ids": [f["fact_id"] for f in eligible],
        "accuracy_only_fact_ids": [f["fact_id"] for f in accuracy_only],
        "refused_cells": refusals,
        "facts": facts,
    }
    return {
        "known_fact_core": core,
        "known_fact_scores": rows,
        "distractor_sets": [d.to_dict() for d in distractor_sets.values()],
    }


def retain_examples_by_ring(corpus: Any, topic: str) -> dict[str, list[FactExample]]:
    """Group retain controls into the R1-R5 rings used for selectivity reporting."""
    rings: dict[str, list[FactExample]] = defaultdict(list)
    for ex in corpus.examples:
        if ex.is_forget or ex.topic != topic:
            continue
        if ex.control_type is ControlType.SEMANTIC:
            tier = ex.semantic_tier
            rings["R2_semantic_neighbor" if tier is not None and tier <= 1 else "R2_semantic_far"].append(ex)
        elif ex.control_type is ControlType.SYNTACTIC:
            rings["R3_same_syntax"].append(ex)
        elif ex.control_type is ControlType.LEXICAL:
            rings["R4_same_lexical"].append(ex)
        elif ex.control_type is ControlType.GENERAL_KNOWLEDGE:
            rings["R5_general_knowledge"].append(ex)
    return dict(rings)
