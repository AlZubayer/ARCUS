"""Tokenization and scoring-math tests against the real Llama-3 tokenizer.

These load the tokenizer only (no weights), so they stay fast. They cover the two silent
failure modes that would corrupt every downstream number: a doubled BOS, and answer-token
indices derived from a boundary the tokenizer actually merges across.
"""

import math

import pytest

from arcus.module_a.config import SUITE_SYSTEM_PROMPT
from arcus.module_a.scoring import (
    DegenerateDistractorPool,
    build_distractor_set,
    candidate_answers_for,
    factual_margin,
    logsumexp,
)
from arcus.module_a.schema import FactKey, Modality

transformers = pytest.importorskip("transformers")

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"

# Real SUITE answers, including the multi-token and punctuation-bearing shapes that make up
# 81% of the forget set.
SUITE_ANSWERS = [
    "STS-51-L",
    "January 28, 1986",
    "73 seconds",
    "Michael Smith",
    "Richard Feynman",
    "Reinforced Carbon-Carbon",
    "record-low temperatures",
    "46,000 feet",
    "Three",
]


@pytest.fixture(scope="module")
def backend():
    """A backend shell with the real tokenizer and no model weights."""
    from transformers import AutoTokenizer

    from arcus.module_a.backend.hf import HFBackend

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    except Exception as exc:  # pragma: no cover - offline / ungated environments
        pytest.skip(f"Llama-3.2 tokenizer unavailable: {exc}")

    be = HFBackend.__new__(HFBackend)
    be.tokenizer = tokenizer
    be.prompt_template = "chat"
    be.system_prompt = SUITE_SYSTEM_PROMPT
    be.add_generation_prompt = True
    be.add_special_tokens = False
    be.raw_completion_template = "Question: {question}\nAnswer:"
    return be


QUESTION = "On what date did the Challenger disaster occur?"


def test_chat_template_emits_exactly_one_bos(backend):
    prompt = backend.build_prompt(QUESTION)
    ids = backend.tokenizer(prompt, add_special_tokens=False).input_ids
    assert ids.count(backend.tokenizer.bos_token_id) == 1
    assert ids[0] == backend.tokenizer.bos_token_id


def test_add_special_tokens_true_would_double_the_bos(backend):
    """Guards the exact mistake add_special_tokens=False exists to prevent."""
    prompt = backend.build_prompt(QUESTION)
    doubled = backend.tokenizer(prompt, add_special_tokens=True).input_ids
    assert doubled.count(backend.tokenizer.bos_token_id) == 2

    backend.add_special_tokens = True
    try:
        with pytest.raises(Exception, match="BOS"):
            backend.tokenize(prompt, "January 28, 1986")
    finally:
        backend.add_special_tokens = False


@pytest.mark.parametrize("answer", SUITE_ANSWERS)
def test_joint_tokenization_extends_the_prompt(backend, answer):
    """Answer indices are only valid if prompt+answer extends the prompt tokenization."""
    prompt = backend.build_prompt(QUESTION)
    example = backend.tokenize(prompt, answer)

    assert example.full_token_ids[: example.prompt_len] == example.prompt_token_ids
    assert example.answer_token_positions == tuple(
        range(example.prompt_len, example.total_len)
    )
    assert len(example.answer_token_ids) == len(example.answer_token_positions) >= 1
    # The answer tokens must decode back to the answer text.
    assert backend.tokenizer.decode(example.answer_token_ids) == answer


def test_multi_token_answers_are_not_scored_as_one_token(backend):
    prompt = backend.build_prompt(QUESTION)
    single = backend.tokenize(prompt, "Three")
    multi = backend.tokenize(prompt, "January 28, 1986")
    assert len(single.answer_token_ids) == 1
    assert len(multi.answer_token_ids) > 1
    assert single.prompt_len == multi.prompt_len


def test_empty_answer_is_refused(backend):
    prompt = backend.build_prompt(QUESTION)
    with pytest.raises(Exception):
        backend.tokenize(prompt, "")


def test_raw_completion_template_is_available_as_a_sensitivity_check(backend):
    backend.prompt_template = "raw_completion"
    try:
        prompt = backend.build_prompt(QUESTION)
        assert prompt == f"Question: {QUESTION}\nAnswer:"
        assert backend.tokenizer.bos_token not in prompt
    finally:
        backend.prompt_template = "chat"


# -- Scoring math (no tokenizer needed) --------------------------------------------------


def test_logsumexp_matches_math():
    values = [-1.0, -2.0, -3.5]
    assert logsumexp(values) == pytest.approx(math.log(sum(math.exp(v) for v in values)))


def test_factual_margin_formula():
    # M_f = s(y_f) - log sum exp s(y in D_f)
    correct, distractors = -0.5, [-2.0, -3.0, -4.0]
    expected = correct - math.log(sum(math.exp(v) for v in distractors))
    assert factual_margin(correct, distractors) == pytest.approx(expected)


def test_margin_is_positive_only_when_target_beats_the_pool():
    assert factual_margin(-0.1, [-5.0, -5.0]) > 0
    assert factual_margin(-5.0, [-0.1, -0.1]) < 0


def test_margin_requires_a_non_empty_pool():
    with pytest.raises(ValueError):
        factual_margin(-1.0, [])


# -- Distractor pools --------------------------------------------------------------------

FACT = FactKey(topic="challenger_disaster", fact_id="M2")


def test_distractor_pool_is_deterministic_for_a_seed():
    candidates = [f"answer {i}" for i in range(12)]
    a = build_distractor_set(FACT, Modality.DIRECT, "January 28, 1986", candidates, count=4, seed=42)
    b = build_distractor_set(FACT, Modality.DIRECT, "January 28, 1986", candidates, count=4, seed=42)
    assert a.distractors == b.distractors
    assert len(a.distractors) == 4
    assert a.answers[0] == "January 28, 1986"


def test_distractor_pool_excludes_the_correct_answer():
    candidates = ["January 28, 1986", " january 28, 1986. ", "July 20, 1969", "a", "b", "c", "d"]
    built = build_distractor_set(
        FACT, Modality.DIRECT, "January 28, 1986", candidates, count=4, seed=1
    )
    assert "January 28, 1986" not in built.distractors
    # Formatting variants of the target are caught by normalization and recorded.
    assert len(built.excluded_as_synonymous) == 2


def test_degenerate_reverse_pool_is_refused_not_padded():
    """The D3 guard: reverse cells collapse to one answer, so the margin is undefined."""
    candidates = ["Challenger"] * 24
    with pytest.raises(DegenerateDistractorPool, match="cannot support"):
        build_distractor_set(
            FACT, Modality.REVERSE, "Challenger", candidates, count=4, seed=42
        )


def test_candidate_answers_take_one_answer_per_other_fact():
    from arcus.module_a.suite import row_to_example

    rows = [
        {"question": f"q{i}", "answer": f"answer {i}", "label": f"K{i}-direct",
         "topic": "challenger_disaster"}
        for i in range(1, 5)
    ]
    # A fact with many surface forms must not dominate the pool.
    rows += [
        {"question": f"dup{j}", "answer": "answer 1", "label": "K1-direct@q_claude%d" % j,
         "topic": "challenger_disaster"}
        for j in range(1, 6)
    ]
    examples = [
        row_to_example(r, row_index=i, dataset="apeleg/SUITE", revision="rev",
                       source_split="forget_train")
        for i, r in enumerate(rows)
    ]
    pool = candidate_answers_for(examples, FactKey("challenger_disaster", "K2"), Modality.DIRECT)
    assert sorted(pool) == ["answer 1", "answer 3", "answer 4"]
    assert pool.count("answer 1") == 1
