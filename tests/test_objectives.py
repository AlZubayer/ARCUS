"""Tests for the pre-answer discriminative-token objective.

The point of J_f is that it never conditions on a correct-answer-specific prefix. These
tests pin that property and the rejection rules, using real SUITE answer strings.
"""

import pytest

from arcus.module_a.objectives import (
    OBJECTIVE_VERSION,
    ObjectiveUndefined,
    RejectReason,
    build_discriminative_spec,
    longest_common_prefix,
    objective_definition,
)


def spec(correct_tokens, distractor_tokens, *, prompt_len=69, strict=True):
    return build_discriminative_spec(
        prompt_len=prompt_len,
        correct_answer="correct",
        correct_tokens=correct_tokens,
        distractor_answers=[f"d{i}" for i in range(len(distractor_tokens))],
        distractor_tokens=distractor_tokens,
        strict=strict,
    )


def test_longest_common_prefix():
    assert longest_common_prefix([[1, 2, 3], [1, 2, 9], [1, 2, 7]]) == (1, 2)
    assert longest_common_prefix([[1, 2], [3, 4]]) == ()
    assert longest_common_prefix([[1, 2], [1, 2, 3]]) == (1, 2)
    assert longest_common_prefix([]) == ()


def test_no_common_prefix_puts_t_star_at_the_first_answer_token():
    """The usual SUITE case: answers diverge immediately."""
    s = spec([100, 200, 300], [[400, 500], [600]])
    assert s.common_prefix_token_ids == ()
    assert s.t_star == 69
    assert s.logit_position == 68
    assert s.correct_token_id == 100
    assert s.distractor_token_ids == (400, 600)


def test_shared_prefix_moves_t_star_past_it():
    """Only the prefix every candidate shares is conditioned on."""
    s = spec([10, 11, 42], [[10, 11, 99], [10, 11, 77]])
    assert s.common_prefix_token_ids == (10, 11)
    assert s.t_star == 71
    assert s.logit_position == 70
    assert s.correct_token_id == 42
    assert s.distractor_token_ids == (99, 77)


def test_prefix_never_includes_a_correct_answer_specific_token():
    """The property the objective exists for.

    Two distractors share a longer prefix with each other than with the correct answer;
    the common prefix must stop at what ALL candidates share, so no token unique to the
    correct answer is ever conditioned on.
    """
    s = spec([10, 42, 43], [[10, 99, 98], [10, 99, 97]])
    assert s.common_prefix_token_ids == (10,)
    assert 42 not in s.common_prefix_token_ids
    assert 43 not in s.common_prefix_token_ids
    assert s.correct_token_id == 42


def test_candidate_that_is_a_strict_prefix_is_rejected():
    """The correct answer ends at the common prefix, so it has no token at t*."""
    with pytest.raises(ObjectiveUndefined, match=RejectReason.CANDIDATE_IS_PREFIX):
        spec([10], [[10, 99], [10, 98]])
    # And the same when a distractor is the one that runs out.
    with pytest.raises(ObjectiveUndefined, match=RejectReason.CANDIDATE_IS_PREFIX):
        spec([10, 42], [[10], [10, 99]])


def test_distractor_sharing_the_correct_token_is_rejected():
    """Otherwise the margin compares the correct token against itself."""
    with pytest.raises(ObjectiveUndefined, match=RejectReason.DISTRACTOR_SHARES_TOKEN):
        spec([10, 42], [[10, 42], [10, 99]])


def test_empty_candidates_are_rejected():
    with pytest.raises(ObjectiveUndefined, match=RejectReason.NO_DISTRACTORS):
        spec([10, 11], [])
    with pytest.raises(ObjectiveUndefined, match=RejectReason.EMPTY_ANSWER):
        spec([], [[1]])
    with pytest.raises(ObjectiveUndefined, match=RejectReason.EMPTY_ANSWER):
        spec([1], [[]])


def test_non_strict_mode_records_warnings_instead_of_raising():
    """Auditing keeps the ill-defined case visible rather than dropping it."""
    s = spec([10, 42], [[10, 42], [10, 99]], strict=False)
    assert s.warnings
    assert RejectReason.DISTRACTOR_SHARES_TOKEN in s.warnings[0]
    assert s.to_dict()["warnings"]


def test_spec_serializes_everything_needed_to_recompute():
    s = spec([10, 11, 42], [[10, 11, 99]])
    payload = s.to_dict()
    for key in (
        "objective_version",
        "t_star",
        "logit_position",
        "prompt_len",
        "common_prefix_token_ids",
        "correct_token_id",
        "distractor_token_ids",
    ):
        assert key in payload
    assert payload["objective_version"] == OBJECTIVE_VERSION
    assert payload["n_common_prefix_tokens"] == 2


def test_objective_definition_documents_the_companion_metric():
    """The sequence metric must not be presented as replaced."""
    d = objective_definition()
    assert d["companion_metric"]["name"] == "factual_margin_v1"
    assert "never replaces it" in d["companion_metric"]["note"]
    assert d["role"] == "A1 route discovery only"
    assert set(d["rejection_reasons"]) == {
        RejectReason.CANDIDATE_IS_PREFIX,
        RejectReason.DISTRACTOR_SHARES_TOKEN,
        RejectReason.NO_DISTRACTORS,
        RejectReason.EMPTY_ANSWER,
    }


# -- Against the real tokenizer ----------------------------------------------------------

transformers = pytest.importorskip("transformers")

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
REVISION = "0cb88a4f764b7a12671c53f0838cd831a0843b95"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    except Exception as exc:  # pragma: no cover - offline environments
        pytest.skip(f"tokenizer unavailable: {exc}")


def test_real_suite_answers_diverge_at_the_first_token(tokenizer):
    """The M2 cell: a date against matched date distractors."""
    correct = "January 28, 1986"
    distractors = ["July 20, 1969", "April 26, 1986", "73 seconds", "Three"]
    ids = [tokenizer(a, add_special_tokens=False).input_ids for a in [correct, *distractors]]

    s = build_discriminative_spec(
        prompt_len=69,
        correct_answer=correct,
        correct_tokens=ids[0],
        distractor_answers=distractors,
        distractor_tokens=ids[1:],
    )
    # These answers share no leading token, so nothing at all is conditioned on.
    assert s.common_prefix_token_ids == ()
    assert s.t_star == 69
    assert s.correct_token_id == ids[0][0]
    assert s.correct_token_id not in s.distractor_token_ids
    # The multi-token continuation "28, 1986" is exactly what J_f avoids scoring.
    assert len(ids[0]) > 1


def test_two_dates_in_the_same_year_still_discriminate(tokenizer):
    """A harder case: answers that agree late must still differ at t*."""
    correct = "January 28, 1986"
    distractors = ["January 12, 1986"]
    ids = [tokenizer(a, add_special_tokens=False).input_ids for a in [correct, *distractors]]
    s = build_discriminative_spec(
        prompt_len=69,
        correct_answer=correct,
        correct_tokens=ids[0],
        distractor_answers=distractors,
        distractor_tokens=ids[1:],
    )
    assert s.correct_token_id != s.distractor_token_ids[0]
    assert s.n_common_prefix_tokens >= 1
    assert s.t_star == 69 + s.n_common_prefix_tokens
