"""Build and score matched clean/corrupt pairs.

The corrupt margin is the *target* fact's margin evaluated under the corrupt prompt:

    Delta_f = M_f(q+) - M_f(q-)

Both terms use the same answer y_f and the same frozen distractor pool D_f, so Delta
measures how much the corruption removed the fact rather than how well the corrupt prompt
answers its own question.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

from ..backend.hf import HFBackend
from ..pairing import (
    PAIR_POLICY_VERSION,
    CleanCorruptPair,
    build_pairs_for_surface,
    validate_pair,
)
from ..schema import FactExample, FactKey, Modality, Split
from ..scoring import DistractorSet, factual_margin
from ..schema import ControlType
from ..suite import syntax_controls_by_link

PAIRS_STAGE_VERSION = "clean_corrupt_pairs_v1"


def _margin_for(
    backend: HFBackend,
    question: str,
    distractors: DistractorSet,
    cache: dict[str, float],
    *,
    answer_score: str,
) -> float:
    """Score one question against a frozen distractor pool, memoized per prompt."""
    key = f"{distractors.fact_key.id}|{distractors.modality.value}|{question}"
    if key in cache:
        return cache[key]
    prompt = backend.build_prompt(question)
    scores = backend.score_answers(prompt, distractors.answers)
    values = [getattr(s, answer_score) for s in scores]
    margin = factual_margin(values[0], values[1:])
    cache[key] = margin
    return margin


def select_clean_surfaces(
    examples: Sequence[FactExample],
    fact_key: FactKey,
    *,
    modality: Modality,
    split: Split,
    limit: int,
    correct_surface_ids: set[str] | None = None,
    preferred_augmentations: set[str] | None = None,
) -> list[FactExample]:
    """Pick clean surfaces deterministically, so a rerun builds identical pairs.

    Restricted to surfaces A0 scored as correct where that information is available: a
    clean run that never exhibited the fact cannot anchor a restoration experiment, and
    including it would put noise in the denominator of every normalized effect.

    ``preferred_augmentations`` puts surfaces carrying an exactly matched same-syntax twin
    first. That control is the sharpest available test that a route is about the fact
    rather than the question form, and the criterion is control *availability* -- it never
    inspects a margin, an attribution or any model output, so it cannot bias an outcome.
    Ordering stays deterministic within each group.
    """
    candidates = [
        ex
        for ex in examples
        if ex.fact_key == fact_key and ex.modality is modality and ex.split is split
    ]
    if correct_surface_ids is not None:
        preferred = [ex for ex in candidates if ex.surface_form_id in correct_surface_ids]
        candidates = preferred or []
    twins = preferred_augmentations or set()
    return sorted(
        candidates,
        key=lambda ex: (0 if ex.augmentation_id in twins else 1, ex.surface_form_id),
    )[:limit]


def run_pairs(
    backend: HFBackend,
    corpus: Any,
    *,
    eligible_facts: Sequence[str],
    distractor_sets: dict[tuple[FactKey, Modality], DistractorSet],
    families: Sequence[str],
    modality: Modality,
    split: Split,
    clean_surfaces_per_fact: int,
    max_pairs_per_family: int,
    min_abs_delta: float,
    seed: int,
    answer_score: str = "mean_logprob",
    correct_surface_ids: set[str] | None = None,
    prefer_exact_syntax_twin: bool = False,
) -> dict[str, Any]:
    """Build every configured family for every eligible fact, then score and validate."""
    forget_pool = [ex for ex in corpus.examples if ex.is_forget]
    retain_pool = [ex for ex in corpus.examples if not ex.is_forget]
    syntax_index = syntax_controls_by_link(retain_pool)

    # Exact syntactic twins are keyed by Claude augmentation ids and so only exist for
    # discovery surfaces. Held-out validation surfaces fall back to a same-(fact, modality)
    # syntax control, which the pair records as a weaker match rather than an exact one.
    syntax_fallback: dict[tuple[FactKey, Modality], list[FactExample]] = defaultdict(list)
    for ex in retain_pool:
        if (
            ex.control_type is ControlType.SYNTACTIC
            and ex.linked_fact_key is not None
            and ex.linked_modality is not None
        ):
            syntax_fallback[(ex.linked_fact_key, ex.linked_modality)].append(ex)

    eligible = set(eligible_facts)
    cache: dict[str, float] = {}
    pairs: list[CleanCorruptPair] = []

    for fact_id in sorted(eligible):
        topic, short = fact_id.split(":", 1)
        fact_key = FactKey(topic=topic, fact_id=short)
        distractors = distractor_sets.get((fact_key, modality))
        if distractors is None:
            continue

        twin_augmentations = (
            {
                aug
                for (linked_fact, linked_mod, aug) in syntax_index
                if linked_fact == fact_key and linked_mod is modality
            }
            if prefer_exact_syntax_twin
            else set()
        )
        for clean in select_clean_surfaces(
            corpus.examples, fact_key, modality=modality, split=split,
            limit=clean_surfaces_per_fact, correct_surface_ids=correct_surface_ids,
            preferred_augmentations=twin_augmentations,
        ):
            clean_margin = _margin_for(
                backend, clean.question, distractors, cache, answer_score=answer_score
            )
            built = build_pairs_for_surface(
                clean,
                forget_pool=forget_pool,
                retain_pool=retain_pool,
                syntax_index=syntax_index,
                families=families,
                limit=max_pairs_per_family,
                seed=seed,
                syntax_fallback=syntax_fallback,
            )
            for pair in built:
                corrupt_margin = _margin_for(
                    backend, pair.corrupt_question, distractors, cache,
                    answer_score=answer_score,
                )
                pairs.append(
                    validate_pair(
                        pair, clean_margin, corrupt_margin, min_abs_delta=min_abs_delta
                    )
                )
            print(
                f"    {fact_id} {clean.augmentation_id}: {len(built)} pairs "
                f"(clean margin {clean_margin:+.2f})",
                flush=True,
            )

    accepted = [p for p in pairs if p.validation_status == "accepted"]
    by_family: dict[str, list[CleanCorruptPair]] = defaultdict(list)
    for pair in pairs:
        by_family[pair.family].append(pair)

    summary = {
        "stage_version": PAIRS_STAGE_VERSION,
        "pair_policy_version": PAIR_POLICY_VERSION,
        "modality": modality.value,
        "clean_split": split.value,
        "min_abs_delta": min_abs_delta,
        "n_pairs": len(pairs),
        "n_accepted": len(accepted),
        "n_rejected": len(pairs) - len(accepted),
        "rejection_reasons": dict(
            Counter(p.rejection_reason for p in pairs if p.rejection_reason).most_common()
        ),
        "by_family": {
            family: {
                "n": len(group),
                "n_accepted": sum(1 for p in group if p.validation_status == "accepted"),
                "mean_delta": (
                    sum(p.delta for p in group) / len(group) if group else 0.0
                ),
                "median_abs_delta": (
                    sorted(abs(p.delta) for p in group)[len(group) // 2] if group else 0.0
                ),
            }
            for family, group in sorted(by_family.items())
        },
        "n_facts_with_accepted_pairs": len({p.target_fact_key.id for p in accepted}),
        "syntax_match_quality": dict(
            Counter(
                p.constraints.get("syntax_match")
                for p in pairs
                if p.family == "same_syntax"
            ).most_common()
        ),
    }
    return {"pairs": [p.to_dict() for p in pairs], "summary": summary}
