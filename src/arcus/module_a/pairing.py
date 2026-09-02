"""Clean/corrupt pair construction.

A pair changes factual identity while holding nuisance structure fixed. Random token
corruption is a secondary robustness control, never the primary scientific comparison
(02_DATA_AND_SPLITS.md section 7).

Families are kept strictly separate. Pooling them would let a strong same-topic effect
mask the absence of a same-syntax effect, which is exactly the confound A1 exists to rule
out.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from .schema import ControlType, FactExample, FactKey, Modality, normalize_answer

PAIR_POLICY_VERSION = "pair_families_v1"


class PairFamily:
    """The six registered corruption families."""

    SAME_TOPIC_FACT_SWAP = "same_topic_fact_swap"
    SEMANTIC_NEIGHBOR = "semantic_neighbor"
    SAME_SYNTAX = "same_syntax"
    SAME_LEXICAL = "same_lexical_different_meaning"
    CROSS_TOPIC_MATCHED = "cross_topic_matched"
    RANDOM_TOKEN_CONTROL = "random_token_control"


class RejectionReason:
    WEAK_EFFECT = "weak_effect_below_min_abs_delta"
    SAME_ANSWER = "corrupt_answer_equals_clean_answer"
    SAME_FACT = "corrupt_shares_clean_fact_id"
    NO_CANDIDATE = "no_matched_candidate_available"
    CLEAN_NOT_CORRECT = "clean_surface_not_answered_correctly"


@dataclass
class CleanCorruptPair:
    """One matched causal comparison, with every constraint recorded."""

    pair_id: str
    family: str
    target_fact_key: FactKey
    clean_surface_id: str
    clean_question: str
    clean_answer: str
    corrupt_surface_id: str
    corrupt_question: str
    corrupt_answer: str
    corrupt_fact_key: FactKey | None
    modality: Modality
    constraints: dict[str, Any] = field(default_factory=dict)
    clean_margin: float | None = None
    corrupt_margin: float | None = None
    delta: float | None = None
    validation_status: str = "unvalidated"
    rejection_reason: str | None = None
    alignment: dict[str, Any] | None = None
    policy_version: str = PAIR_POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_fact_key"] = {
            "topic": self.target_fact_key.topic,
            "fact_id": self.target_fact_key.fact_id,
        }
        payload["corrupt_fact_key"] = (
            {"topic": self.corrupt_fact_key.topic, "fact_id": self.corrupt_fact_key.fact_id}
            if self.corrupt_fact_key
            else None
        )
        payload["modality"] = self.modality.value
        return payload


def _constraints(
    clean: FactExample, corrupt: FactExample, *, clean_tokens: int = 0, corrupt_tokens: int = 0
) -> dict[str, Any]:
    return {
        "same_modality": clean.modality == corrupt.modality,
        "same_topic": clean.topic == corrupt.topic,
        "same_surface_kind": clean.surface_kind == corrupt.surface_kind,
        "same_augmentation": clean.augmentation_id == corrupt.augmentation_id,
        "same_answer_type": None,
        "clean_char_len": len(clean.question),
        "corrupt_char_len": len(corrupt.question),
        "char_len_delta": len(corrupt.question) - len(clean.question),
        "token_len_delta": corrupt_tokens - clean_tokens if clean_tokens else None,
        "corrupt_control_type": corrupt.control_type.value,
        "corrupt_semantic_tier": corrupt.semantic_tier,
    }


def _pair_id(family: str, clean: FactExample, corrupt: FactExample) -> str:
    return f"{family}|{clean.surface_form_id}|{corrupt.surface_form_id}"


def _make(
    family: str, clean: FactExample, corrupt: FactExample
) -> CleanCorruptPair:
    return CleanCorruptPair(
        pair_id=_pair_id(family, clean, corrupt),
        family=family,
        target_fact_key=clean.fact_key,
        clean_surface_id=clean.surface_form_id,
        clean_question=clean.question,
        clean_answer=clean.answer,
        corrupt_surface_id=corrupt.surface_form_id,
        corrupt_question=corrupt.question,
        corrupt_answer=corrupt.answer,
        corrupt_fact_key=corrupt.fact_key,
        modality=clean.modality,
        constraints=_constraints(clean, corrupt),
    )


def _token_len(text: str) -> int:
    return len(text.split())


def build_same_topic_fact_swap(
    clean: FactExample, pool: Sequence[FactExample], *, limit: int
) -> list[CleanCorruptPair]:
    """Another fact from the same topic, matched on modality and surface kind.

    Candidates are ordered by closeness in length so the corruption changes the fact rather
    than the shape of the question.
    """
    candidates = [
        ex
        for ex in pool
        if ex.fact_key is not None
        and ex.fact_key != clean.fact_key
        and ex.topic == clean.topic
        and ex.modality is clean.modality
        and ex.surface_kind is clean.surface_kind
        and normalize_answer(ex.answer) != normalize_answer(clean.answer)
    ]
    # One surface per competing fact, then nearest length first.
    best: dict[str, FactExample] = {}
    for ex in candidates:
        key = ex.fact_key.fact_id
        current = best.get(key)
        if current is None or abs(_token_len(ex.question) - _token_len(clean.question)) < abs(
            _token_len(current.question) - _token_len(clean.question)
        ):
            best[key] = ex
    ordered = sorted(
        best.values(),
        key=lambda ex: (abs(_token_len(ex.question) - _token_len(clean.question)), ex.surface_form_id),
    )
    return [_make(PairFamily.SAME_TOPIC_FACT_SWAP, clean, ex) for ex in ordered[:limit]]


def build_semantic_neighbor(
    clean: FactExample, retain: Sequence[FactExample], *, limit: int, max_tier: int = 1
) -> list[CleanCorruptPair]:
    """Closely related factual content from the retain rings (semantic tiers 0-1)."""
    candidates = [
        ex
        for ex in retain
        if ex.control_type is ControlType.SEMANTIC
        and ex.topic == clean.topic
        and ex.semantic_tier is not None
        and ex.semantic_tier <= max_tier
    ]
    ordered = sorted(
        candidates,
        key=lambda ex: (abs(_token_len(ex.question) - _token_len(clean.question)), ex.surface_form_id),
    )
    return [_make(PairFamily.SEMANTIC_NEIGHBOR, clean, ex) for ex in ordered[:limit]]


def build_same_syntax(
    clean: FactExample,
    syntax_index: dict[tuple[FactKey, Modality, str], FactExample],
    *,
    fallback_index: dict[tuple[FactKey, Modality], list[FactExample]] | None = None,
    limit: int = 1,
) -> list[CleanCorruptPair]:
    """The retain_train Syntax row that reuses this question's template.

    SUITE ships an augmentation-matched syntactic twin for every *forget_train* surface,
    which differs from the clean prompt only in the entity asked about. That is the
    sharpest available test that a route is about the fact and not the question form.

    The twins are keyed by Claude augmentation ids, so an exact match exists only for
    discovery-split surfaces. For a held-out validation surface (Gemini augmentations) no
    exact twin exists, and the fallback is a same-(fact, modality) syntax control with
    ``template_matched_exactly`` recorded as False. The weaker match is labelled, never
    passed off as the exact one.
    """
    exact = syntax_index.get((clean.fact_key, clean.modality, clean.augmentation_id))
    if exact is not None:
        pair = _make(PairFamily.SAME_SYNTAX, clean, exact)
        pair.constraints["template_matched_exactly"] = True
        pair.constraints["syntax_match"] = "augmentation_matched"
        return [pair]

    candidates = (fallback_index or {}).get((clean.fact_key, clean.modality), [])
    ordered = sorted(
        candidates,
        key=lambda ex: (
            abs(_token_len(ex.question) - _token_len(clean.question)),
            ex.surface_form_id,
        ),
    )
    pairs = []
    for ex in ordered[:limit]:
        pair = _make(PairFamily.SAME_SYNTAX, clean, ex)
        pair.constraints["template_matched_exactly"] = False
        pair.constraints["syntax_match"] = "fact_modality_matched_only"
        pairs.append(pair)
    return pairs


def build_same_lexical(
    clean: FactExample, retain: Sequence[FactExample], *, limit: int
) -> list[CleanCorruptPair]:
    """Forget-set vocabulary used outside the forget topic."""
    candidates = [
        ex for ex in retain if ex.control_type is ControlType.LEXICAL and ex.topic == clean.topic
    ]
    ordered = sorted(candidates, key=lambda ex: ex.surface_form_id)
    return [_make(PairFamily.SAME_LEXICAL, clean, ex) for ex in ordered[:limit]]


def build_cross_topic_matched(
    clean: FactExample, pool: Sequence[FactExample], *, limit: int
) -> list[CleanCorruptPair]:
    """Same modality and surface kind, different topic entirely."""
    candidates = [
        ex
        for ex in pool
        if ex.fact_key is not None
        and ex.topic != clean.topic
        and ex.modality is clean.modality
        and ex.surface_kind is clean.surface_kind
    ]
    best: dict[str, FactExample] = {}
    for ex in candidates:
        best.setdefault(ex.fact_key.id, ex)
    ordered = sorted(
        best.values(),
        key=lambda ex: (abs(_token_len(ex.question) - _token_len(clean.question)), ex.surface_form_id),
    )
    return [_make(PairFamily.CROSS_TOPIC_MATCHED, clean, ex) for ex in ordered[:limit]]


def build_random_token_control(
    clean: FactExample, *, seed: int, limit: int = 1
) -> list[CleanCorruptPair]:
    """Shuffle the clean question's content words.

    A secondary robustness check only. It destroys syntax as well as fact identity, so it
    cannot separate a factual route from a question-form route; that is what the matched
    families are for.
    """
    rng = random.Random(f"{seed}|{clean.surface_form_id}|random_token")
    words = clean.question.split()
    if len(words) < 4:
        return []
    out: list[CleanCorruptPair] = []
    for i in range(limit):
        shuffled = words[:]
        rng.shuffle(shuffled)
        corrupted_text = " ".join(shuffled)
        if corrupted_text == clean.question:
            continue
        synthetic = FactExample(
            surface_form_id=f"{clean.surface_form_id}-randtok{i}",
            topic=clean.topic,
            question=corrupted_text,
            answer=clean.answer,
            raw_label=f"{clean.raw_label}#random_token_control",
            control_type=clean.control_type,
            fact_key=None,
            modality=clean.modality,
            surface_kind=clean.surface_kind,
            augmentation_id=clean.augmentation_id,
            source_dataset="synthetic",
            source_split="random_token_control",
        )
        pair = _make(PairFamily.RANDOM_TOKEN_CONTROL, clean, synthetic)
        pair.constraints["word_order_shuffled"] = True
        pair.constraints["seed"] = seed
        out.append(pair)
    return out


def build_pairs_for_surface(
    clean: FactExample,
    *,
    forget_pool: Sequence[FactExample],
    retain_pool: Sequence[FactExample],
    syntax_index: dict[tuple[FactKey, Modality, str], FactExample],
    families: Iterable[str],
    limit: int,
    seed: int,
    syntax_fallback: dict[tuple[FactKey, Modality], list[FactExample]] | None = None,
) -> list[CleanCorruptPair]:
    """Build every configured family for one clean surface form."""
    wanted = set(families)
    pairs: list[CleanCorruptPair] = []
    if PairFamily.SAME_TOPIC_FACT_SWAP in wanted:
        pairs += build_same_topic_fact_swap(clean, forget_pool, limit=limit)
    if PairFamily.SEMANTIC_NEIGHBOR in wanted:
        pairs += build_semantic_neighbor(clean, retain_pool, limit=limit)
    if PairFamily.SAME_SYNTAX in wanted:
        pairs += build_same_syntax(clean, syntax_index, fallback_index=syntax_fallback)
    if PairFamily.SAME_LEXICAL in wanted:
        pairs += build_same_lexical(clean, retain_pool, limit=limit)
    if PairFamily.CROSS_TOPIC_MATCHED in wanted:
        pairs += build_cross_topic_matched(clean, forget_pool, limit=limit)
    if PairFamily.RANDOM_TOKEN_CONTROL in wanted:
        pairs += build_random_token_control(clean, seed=seed)
    return pairs


def validate_pair(
    pair: CleanCorruptPair,
    clean_margin: float,
    corrupt_margin: float,
    *,
    min_abs_delta: float,
    require_clean_correct: bool = True,
) -> CleanCorruptPair:
    """Score-based acceptance.

    A pair whose |delta| is near zero cannot support any normalized causal metric: every
    one of them divides by that delta. Such pairs are rejected with a reason and kept in
    the artifact rather than dropped.
    """
    pair.clean_margin = clean_margin
    pair.corrupt_margin = corrupt_margin
    pair.delta = clean_margin - corrupt_margin

    # The random-token control keeps the target answer by construction: it shuffles the
    # clean question and asks whether retrieval survives. Only a *different-fact*
    # corruption that happens to share the target answer is disqualifying, because there
    # the corrupt run still elicits the fact and Delta stops meaning anything.
    shares_answer = normalize_answer(pair.corrupt_answer) == normalize_answer(pair.clean_answer)
    if require_clean_correct and clean_margin <= 0.0:
        # A restoration experiment is meaningless if the clean run never exhibited the fact.
        pair.validation_status = "rejected"
        pair.rejection_reason = RejectionReason.CLEAN_NOT_CORRECT
    elif shares_answer and pair.family != PairFamily.RANDOM_TOKEN_CONTROL:
        pair.validation_status = "rejected"
        pair.rejection_reason = RejectionReason.SAME_ANSWER
    elif pair.corrupt_fact_key is not None and pair.corrupt_fact_key == pair.target_fact_key:
        pair.validation_status = "rejected"
        pair.rejection_reason = RejectionReason.SAME_FACT
    elif abs(pair.delta) < min_abs_delta:
        pair.validation_status = "rejected"
        pair.rejection_reason = RejectionReason.WEAK_EFFECT
    else:
        pair.validation_status = "accepted"
        pair.rejection_reason = None
    return pair
