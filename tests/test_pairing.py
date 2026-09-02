"""Pair-builder tests: families stay separate, constraints are recorded, weak pairs die."""

from dataclasses import replace

import pytest

from arcus.module_a.pairing import (
    PairFamily,
    RejectionReason,
    build_cross_topic_matched,
    build_pairs_for_surface,
    build_random_token_control,
    build_same_lexical,
    build_same_syntax,
    build_same_topic_fact_swap,
    build_semantic_neighbor,
    validate_pair,
)
from arcus.module_a.schema import Split
from arcus.module_a.suite import row_to_example, syntax_controls_by_link

TOPIC = "challenger_disaster"
ALL_FAMILIES = [
    PairFamily.SAME_TOPIC_FACT_SWAP,
    PairFamily.SEMANTIC_NEIGHBOR,
    PairFamily.SAME_SYNTAX,
    PairFamily.SAME_LEXICAL,
    PairFamily.CROSS_TOPIC_MATCHED,
    PairFamily.RANDOM_TOKEN_CONTROL,
]


def ex(row, split_name, index=0, split=None):
    out = row_to_example(
        row, row_index=index, dataset="apeleg/SUITE", revision="rev", source_split=split_name
    )
    return replace(out, split=split) if split else out


CLEAN = ex(
    {
        "question": "What was the official mission code for the Challenger's final flight?",
        "answer": "STS-51-L",
        "label": "M1-direct@q_claude9",
        "topic": TOPIC,
    },
    "forget_train",
    split=Split.DISCOVERY,
)

FORGET_POOL = [CLEAN] + [
    ex(
        {
            "question": f"Question number {i} about the disaster with some words?",
            "answer": f"answer {i}",
            "label": f"K{i}-direct@q_claude9",
            "topic": TOPIC,
        },
        "forget_train",
        index=i,
    )
    for i in range(1, 6)
] + [
    ex(
        {
            "question": "Who presided over the trials?",
            "answer": "William Stoughton",
            "label": "K1-direct@q_claude9",
            "topic": "salem_witch_trials",
        },
        "forget_train",
        index=90,
    )
]

RETAIN_POOL = [
    ex(
        {
            "question": "How many main engines were on the Challenger orbiter?",
            "answer": "Three",
            "label": "Semantic-0-retain_challenger_not_disaster",
            "topic": TOPIC,
        },
        "retain_eval",
        index=1,
    ),
    ex(
        {
            "question": "What caused the Columbia disaster?",
            "answer": "Foam strike",
            "label": "Semantic-1-The Columbia Space Shuttle Disaster",
            "topic": TOPIC,
        },
        "retain_eval",
        index=2,
    ),
    ex(
        {
            "question": "A far-away topic question?",
            "answer": "Something",
            "label": "Semantic-12-The Apollo 13 Mission",
            "topic": TOPIC,
        },
        "retain_eval",
        index=3,
    ),
    ex(
        {
            "question": "What British ship gave Challenger Deep its name?",
            "answer": "HMS Challenger",
            "label": "Lexical-Challenger",
            "topic": TOPIC,
        },
        "retain_eval",
        index=4,
    ),
    ex(
        {
            "question": "What was the official name for the Apollo 11 landing site?",
            "answer": "Tranquility Base",
            "label": "Syntax-M1-direct@q_claude9",
            "topic": TOPIC,
        },
        "retain_train",
        index=5,
    ),
]


def test_same_topic_swap_matches_modality_and_surface_kind():
    pairs = build_same_topic_fact_swap(CLEAN, FORGET_POOL, limit=4)
    assert pairs and len(pairs) <= 4
    for pair in pairs:
        assert pair.family == PairFamily.SAME_TOPIC_FACT_SWAP
        assert pair.constraints["same_topic"] is True
        assert pair.constraints["same_modality"] is True
        assert pair.constraints["same_surface_kind"] is True
        assert pair.corrupt_fact_key != CLEAN.fact_key


def test_same_topic_swap_takes_one_surface_per_competing_fact():
    duplicated = FORGET_POOL + [
        replace(FORGET_POOL[1], surface_form_id=f"dup{i}") for i in range(5)
    ]
    pairs = build_same_topic_fact_swap(CLEAN, duplicated, limit=10)
    fact_ids = [p.corrupt_fact_key.fact_id for p in pairs]
    assert len(fact_ids) == len(set(fact_ids))


def test_same_syntax_uses_the_augmentation_matched_template():
    """SUITE ships a syntactic twin per forget surface; it must be matched exactly."""
    index = syntax_controls_by_link(RETAIN_POOL)
    pairs = build_same_syntax(CLEAN, index)
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.family == PairFamily.SAME_SYNTAX
    assert pair.constraints["template_matched_exactly"] is True
    assert pair.constraints["same_augmentation"] is True
    assert pair.corrupt_answer == "Tranquility Base"


def test_same_syntax_returns_nothing_when_no_twin_exists():
    other = replace(CLEAN, augmentation_id="q_claude1")
    assert build_same_syntax(other, syntax_controls_by_link(RETAIN_POOL)) == []


def test_semantic_neighbor_uses_near_tiers_only():
    pairs = build_semantic_neighbor(CLEAN, RETAIN_POOL, limit=5, max_tier=1)
    tiers = {p.constraints["corrupt_semantic_tier"] for p in pairs}
    assert tiers <= {0, 1}
    assert 12 not in tiers


def test_lexical_and_cross_topic_families():
    lexical = build_same_lexical(CLEAN, RETAIN_POOL, limit=4)
    assert lexical and all(p.corrupt_answer == "HMS Challenger" for p in lexical)

    cross = build_cross_topic_matched(CLEAN, FORGET_POOL, limit=4)
    assert cross
    for pair in cross:
        assert pair.constraints["same_topic"] is False
        assert pair.constraints["same_modality"] is True


def test_random_token_control_is_deterministic_and_marked():
    a = build_random_token_control(CLEAN, seed=42)
    b = build_random_token_control(CLEAN, seed=42)
    assert a[0].corrupt_question == b[0].corrupt_question
    assert a[0].corrupt_question != CLEAN.question
    assert a[0].constraints["word_order_shuffled"] is True
    assert sorted(a[0].corrupt_question.split()) == sorted(CLEAN.question.split())


def test_families_are_built_separately_and_never_pooled():
    pairs = build_pairs_for_surface(
        CLEAN,
        forget_pool=FORGET_POOL,
        retain_pool=RETAIN_POOL,
        syntax_index=syntax_controls_by_link(RETAIN_POOL),
        families=ALL_FAMILIES,
        limit=2,
        seed=42,
    )
    families = {p.family for p in pairs}
    assert families == set(ALL_FAMILIES)
    # Every pair carries exactly one family label; nothing is merged.
    assert all(p.family in ALL_FAMILIES for p in pairs)
    assert len({p.pair_id for p in pairs}) == len(pairs)


def test_only_requested_families_are_built():
    pairs = build_pairs_for_surface(
        CLEAN,
        forget_pool=FORGET_POOL,
        retain_pool=RETAIN_POOL,
        syntax_index=syntax_controls_by_link(RETAIN_POOL),
        families=[PairFamily.SAME_SYNTAX],
        limit=2,
        seed=42,
    )
    assert {p.family for p in pairs} == {PairFamily.SAME_SYNTAX}


# -- Validation --------------------------------------------------------------------------


def base_pair():
    return build_same_topic_fact_swap(CLEAN, FORGET_POOL, limit=1)[0]


def test_strong_pair_is_accepted_and_delta_recorded():
    pair = validate_pair(base_pair(), 3.0, -2.0, min_abs_delta=0.5)
    assert pair.validation_status == "accepted"
    assert pair.delta == pytest.approx(5.0)
    assert pair.rejection_reason is None


def test_weak_pair_is_rejected_with_a_reason():
    """Every normalized causal metric divides by delta, so a near-zero delta is unusable."""
    pair = validate_pair(base_pair(), 1.0, 0.9, min_abs_delta=0.5)
    assert pair.validation_status == "rejected"
    assert pair.rejection_reason == RejectionReason.WEAK_EFFECT
    # Rejected pairs keep their scores and stay in the artifact.
    assert pair.delta == pytest.approx(0.1)


def test_pair_sharing_the_clean_answer_is_rejected():
    pair = base_pair()
    pair.corrupt_answer = " sts-51-l. "
    validated = validate_pair(pair, 3.0, -3.0, min_abs_delta=0.5)
    assert validated.rejection_reason == RejectionReason.SAME_ANSWER


def test_pair_sharing_the_clean_fact_is_rejected():
    pair = base_pair()
    pair.corrupt_fact_key = pair.target_fact_key
    validated = validate_pair(pair, 3.0, -3.0, min_abs_delta=0.5)
    assert validated.rejection_reason == RejectionReason.SAME_FACT


def test_negative_delta_is_still_a_valid_pair():
    """Direction is not assumed: a corruption may raise the margin, and that is data."""
    pair = validate_pair(base_pair(), 1.0, 5.0, min_abs_delta=0.5)
    assert pair.validation_status == "accepted"
    assert pair.delta == pytest.approx(-4.0)


def test_clean_surface_the_model_gets_wrong_is_rejected():
    """A restoration experiment needs a clean run that actually exhibited the fact."""
    pair = validate_pair(base_pair(), -0.4, -5.0, min_abs_delta=0.5)
    assert pair.validation_status == "rejected"
    assert pair.rejection_reason == RejectionReason.CLEAN_NOT_CORRECT


def test_random_token_control_may_keep_the_target_answer():
    """It shuffles the clean question and asks whether retrieval survives, so sharing the
    target answer is by construction, not a defect."""
    control = build_random_token_control(CLEAN, seed=42)[0]
    assert control.corrupt_answer == CLEAN.answer
    validated = validate_pair(control, 3.0, -1.0, min_abs_delta=0.5)
    assert validated.validation_status == "accepted"

    # The same-answer rule still disqualifies a different-fact corruption.
    swap = base_pair()
    swap.corrupt_answer = swap.clean_answer
    assert validate_pair(swap, 3.0, -1.0, min_abs_delta=0.5).rejection_reason == (
        RejectionReason.SAME_ANSWER
    )


def test_same_syntax_falls_back_and_labels_the_weaker_match():
    """Exact twins are keyed by Claude augmentation ids, so a Gemini-augmented held-out
    surface has no exact twin and must not be given one."""
    from arcus.module_a.pairing import build_same_syntax

    twin = RETAIN_POOL[-1]
    fallback = {(CLEAN.fact_key, CLEAN.modality): [twin]}
    gemini_surface = replace(CLEAN, augmentation_id="q_gemini3")

    exact = build_same_syntax(CLEAN, syntax_controls_by_link(RETAIN_POOL), fallback_index=fallback)
    assert exact[0].constraints["template_matched_exactly"] is True
    assert exact[0].constraints["syntax_match"] == "augmentation_matched"

    weak = build_same_syntax(
        gemini_surface, syntax_controls_by_link(RETAIN_POOL), fallback_index=fallback
    )
    assert weak[0].constraints["template_matched_exactly"] is False
    assert weak[0].constraints["syntax_match"] == "fact_modality_matched_only"


def test_pair_serializes_identity_fields_separately():
    payload = validate_pair(base_pair(), 3.0, -2.0, min_abs_delta=0.5).to_dict()
    assert payload["target_fact_key"] == {"topic": TOPIC, "fact_id": "M1"}
    assert payload["modality"] == "direct"
    assert payload["family"] in ALL_FAMILIES
    assert payload["clean_surface_id"] and payload["corrupt_surface_id"]
    assert payload["validation_status"] == "accepted"
    assert payload["policy_version"] == "pair_families_v1"
