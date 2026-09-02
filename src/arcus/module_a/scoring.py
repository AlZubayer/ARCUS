"""Factual outcome metric and distractor pools.

Canonical definitions (03_EXPERIMENT_PROTOCOL.md section 1):

    s(y | q) = (1/T) sum_t log p(y_t | q, y_<t)
    M_f(q)   = s(y_f | q) - log sum_{y in D_f} exp s(y | q)
    Delta_f  = M_f(q+) - M_f(q-)

Correctness is a forced choice against matched distractors, not a string match on a
generated continuation. Generation is recorded separately as a diagnostic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .backend.base import SequenceScore
from .schema import FactExample, FactKey, Modality, normalize_answer

MARGIN_VERSION = "factual_margin_v1"
CORRECTNESS_VERSION = "correct_over_distractors_v1"
DISTRACTOR_POLICY_VERSION = "same_topic_same_modality_v1"


class DegenerateDistractorPool(ValueError):
    """Raised when a (fact, modality) cell cannot yield distinct distractors.

    SUITE reverse questions ask for a fact from the answer side, so within one topic they
    nearly all resolve to the same entity. Scoring such a cell against a same-topic pool
    would put the correct answer inside D_f and make M_f meaningless, so the cell is
    refused with a reason instead of silently producing a number.
    """


@dataclass(frozen=True)
class DistractorSet:
    """A frozen distractor pool for one (fact, modality) cell."""

    fact_key: FactKey
    modality: Modality
    correct_answer: str
    distractors: tuple[str, ...]
    policy: str = DISTRACTOR_POLICY_VERSION
    n_candidates: int = 0
    excluded_as_synonymous: tuple[str, ...] = ()

    @property
    def answers(self) -> tuple[str, ...]:
        """Correct answer first, then distractors. Order is stable for auditing."""
        return (self.correct_answer,) + self.distractors

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_key": {"topic": self.fact_key.topic, "fact_id": self.fact_key.fact_id},
            "modality": self.modality.value,
            "correct_answer": self.correct_answer,
            "distractors": list(self.distractors),
            "policy": self.policy,
            "n_candidates": self.n_candidates,
            "excluded_as_synonymous": list(self.excluded_as_synonymous),
        }


@dataclass
class FactualScore:
    """One scored surface form, with the raw per-answer scores kept for auditing."""

    surface_form_id: str
    fact_key: FactKey
    modality: Modality
    correct_answer: str
    margin: float
    correct_score: float
    distractor_scores: dict[str, float]
    is_correct: bool
    n_answer_tokens: int
    per_token_logprobs: tuple[float, ...] = ()
    margin_version: str = MARGIN_VERSION
    correctness_version: str = CORRECTNESS_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_form_id": self.surface_form_id,
            "fact_key": {"topic": self.fact_key.topic, "fact_id": self.fact_key.fact_id},
            "modality": self.modality.value,
            "correct_answer": self.correct_answer,
            "answer_scores": {
                self.correct_answer: round(self.correct_score, 6),
                **{k: round(v, 6) for k, v in self.distractor_scores.items()},
            },
            "factual_margin": round(self.margin, 6),
            "is_correct": self.is_correct,
            "n_answer_tokens": self.n_answer_tokens,
            "per_token_logprobs": [round(x, 6) for x in self.per_token_logprobs],
            "margin_version": self.margin_version,
            "correctness_version": self.correctness_version,
            **self.metadata,
        }


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value")
    top = max(values)
    if math.isinf(top) and top < 0:
        return top
    return top + math.log(sum(math.exp(v - top) for v in values))


def factual_margin(correct_score: float, distractor_scores: Sequence[float]) -> float:
    """M_f(q) = s(y_f|q) - log sum_{y in D_f} exp s(y|q)."""
    if not distractor_scores:
        raise ValueError("factual margin requires a non-empty distractor set")
    return correct_score - logsumexp(list(distractor_scores))


def build_distractor_set(
    fact_key: FactKey,
    modality: Modality,
    correct_answer: str,
    candidate_answers: Iterable[str],
    *,
    count: int,
    seed: int,
) -> DistractorSet:
    """Draw ``count`` matched distractors from other facts in the same (topic, modality).

    Any candidate whose normalized form equals the correct answer is dropped: a pool
    containing the target makes the margin uninterpretable. If too few distinct candidates
    survive, the cell is refused rather than padded from a weaker source.
    """
    target = normalize_answer(correct_answer)
    seen: set[str] = {target}
    pool: list[str] = []
    synonymous: list[str] = []
    n_candidates = 0

    for candidate in candidate_answers:
        n_candidates += 1
        norm = normalize_answer(candidate)
        if norm == target:
            synonymous.append(candidate)
            continue
        if norm in seen:
            continue
        seen.add(norm)
        pool.append(candidate)

    if len(pool) < count:
        raise DegenerateDistractorPool(
            f"{fact_key.id}/{modality.value}: only {len(pool)} distinct distractors available "
            f"for the required {count} (from {n_candidates} candidates, "
            f"{len(synonymous)} identical to the correct answer). "
            "This cell cannot support a same-topic factual margin."
        )

    # Deterministic per-cell draw: the same config and seed always yield the same pool.
    rng = random.Random(f"{seed}|{fact_key.id}|{modality.value}")
    chosen = tuple(sorted(rng.sample(pool, count)))
    return DistractorSet(
        fact_key=fact_key,
        modality=modality,
        correct_answer=correct_answer,
        distractors=chosen,
        n_candidates=n_candidates,
        excluded_as_synonymous=tuple(sorted(set(synonymous))),
    )


def candidate_answers_for(
    examples: Iterable[FactExample], fact_key: FactKey, modality: Modality
) -> list[str]:
    """One answer per *other* fact in the same topic and modality.

    Deduplicating by fact keeps the pool one-answer-per-fact, so a fact with many surface
    forms cannot dominate it.
    """
    by_fact: dict[str, str] = {}
    for ex in examples:
        if (
            ex.is_forget
            and ex.fact_key is not None
            and ex.modality is modality
            and ex.topic == fact_key.topic
            and ex.fact_key != fact_key
        ):
            by_fact.setdefault(ex.fact_key.fact_id, ex.answer)
    return [by_fact[k] for k in sorted(by_fact)]


def score_example(
    backend: Any,
    example: FactExample,
    distractors: DistractorSet,
    *,
    answer_score: str = "mean_logprob",
) -> FactualScore:
    """Score one surface form against its frozen distractor pool."""
    prompt = backend.build_prompt(example.question)
    scores: list[SequenceScore] = backend.score_answers(
        prompt, distractors.answers, surface_form_id=example.surface_form_id
    )
    values = [getattr(s, answer_score) for s in scores]
    correct, *rest = values
    margin = factual_margin(correct, rest)

    return FactualScore(
        surface_form_id=example.surface_form_id,
        fact_key=example.fact_key,
        modality=example.modality,
        correct_answer=distractors.correct_answer,
        margin=margin,
        correct_score=correct,
        distractor_scores=dict(zip(distractors.distractors, rest)),
        # Forced choice: the target must beat the aggregated alternatives.
        is_correct=margin > 0.0,
        n_answer_tokens=scores[0].n_answer_tokens,
        per_token_logprobs=scores[0].per_token_logprobs,
        metadata={
            "split": example.split.value if example.split else None,
            "surface_kind": example.surface_kind.value,
            "generator": example.generator.value,
            "augmentation_id": example.augmentation_id,
            "prompt_text_sha256": None,
        },
    )


def delta(clean_margin: float, corrupt_margin: float) -> float:
    """Full clean-corrupt factual effect, Delta_f."""
    return clean_margin - corrupt_margin
