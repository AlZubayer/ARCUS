"""Discovery objective for A1: a pre-answer, answer-prefix-free factual margin.

Why not reuse the full-sequence margin here
-------------------------------------------
``M_f`` scores every answer token teacher-forced, so for a multi-token answer the later
tokens are conditioned on *earlier correct answer tokens*. Attributing against it would
credit components that merely continue a partially-written answer ("January 28, ..." -> 1986)
alongside components that actually retrieve the fact. 81% of SUITE answers are multi-token,
so this is not a corner case.

``J_f`` instead scores the single earliest position at which the correct answer and its
matched distractors disagree, conditioning only on the prompt plus the token prefix that
*every* candidate shares. No correct-answer-specific prefix is ever conditioned on.

``M_f`` is kept and is what exact causal validation uses; ``J_f`` is discovery only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

OBJECTIVE_VERSION = "discriminative_token_margin_v1"


class ObjectiveUndefined(ValueError):
    """Raised when no robust discriminative token exists for a candidate set."""


class RejectReason:
    CANDIDATE_IS_PREFIX = "candidate_is_prefix_of_another"
    DISTRACTOR_SHARES_TOKEN = "distractor_shares_correct_token_at_t_star"
    NO_DISTRACTORS = "no_distractors"
    EMPTY_ANSWER = "empty_answer_tokens"


@dataclass(frozen=True)
class DiscriminativeTokenSpec:
    """Everything needed to recompute ``J_f`` and audit it by hand."""

    t_star: int
    logit_position: int
    common_prefix_token_ids: tuple[int, ...]
    correct_token_id: int
    distractor_token_ids: tuple[int, ...]
    correct_answer: str
    distractor_answers: tuple[str, ...]
    prompt_len: int
    objective_version: str = OBJECTIVE_VERSION
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_common_prefix_tokens(self) -> int:
        return len(self.common_prefix_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_version": self.objective_version,
            "t_star": self.t_star,
            "logit_position": self.logit_position,
            "prompt_len": self.prompt_len,
            "n_common_prefix_tokens": self.n_common_prefix_tokens,
            "common_prefix_token_ids": list(self.common_prefix_token_ids),
            "correct_token_id": self.correct_token_id,
            "distractor_token_ids": list(self.distractor_token_ids),
            "correct_answer": self.correct_answer,
            "distractor_answers": list(self.distractor_answers),
            "warnings": list(self.warnings),
            **self.metadata,
        }


def longest_common_prefix(sequences: Sequence[Sequence[int]]) -> tuple[int, ...]:
    """Longest token prefix shared by every candidate answer."""
    if not sequences:
        return ()
    shortest = min(len(s) for s in sequences)
    prefix: list[int] = []
    for i in range(shortest):
        token = sequences[0][i]
        if all(s[i] == token for s in sequences):
            prefix.append(token)
        else:
            break
    return tuple(prefix)


def build_discriminative_spec(
    *,
    prompt_len: int,
    correct_answer: str,
    correct_tokens: Sequence[int],
    distractor_answers: Sequence[str],
    distractor_tokens: Sequence[Sequence[int]],
    strict: bool = True,
) -> DiscriminativeTokenSpec:
    """Locate ``t*`` and the tokens compared there.

    ``strict`` controls whether an ill-defined case raises (discovery) or is returned with
    warnings (auditing). Either way the reason is recorded; nothing is silently repaired.
    """
    if not distractor_tokens:
        raise ObjectiveUndefined(f"{RejectReason.NO_DISTRACTORS}: {correct_answer!r}")
    if not correct_tokens or any(len(d) == 0 for d in distractor_tokens):
        raise ObjectiveUndefined(f"{RejectReason.EMPTY_ANSWER}: {correct_answer!r}")

    all_candidates = [list(correct_tokens)] + [list(d) for d in distractor_tokens]
    prefix = longest_common_prefix(all_candidates)
    offset = len(prefix)

    warnings: list[str] = []

    # A candidate that ends exactly at the common prefix has no token to compare at t*.
    exhausted = [
        answer
        for answer, tokens in zip(
            [correct_answer, *distractor_answers], all_candidates
        )
        if len(tokens) <= offset
    ]
    if exhausted:
        message = f"{RejectReason.CANDIDATE_IS_PREFIX}: {exhausted}"
        if strict:
            raise ObjectiveUndefined(message)
        warnings.append(message)

    correct_token = all_candidates[0][offset] if len(all_candidates[0]) > offset else -1
    distractor_ids = [
        tokens[offset] if len(tokens) > offset else -1 for tokens in all_candidates[1:]
    ]

    # A distractor sharing the correct token at t* is not discriminated there, so the
    # margin would compare the correct token against itself.
    colliding = [
        answer
        for answer, token in zip(distractor_answers, distractor_ids)
        if token == correct_token
    ]
    if colliding:
        message = f"{RejectReason.DISTRACTOR_SHARES_TOKEN}: {colliding}"
        if strict:
            raise ObjectiveUndefined(message)
        warnings.append(message)

    t_star = prompt_len + offset
    return DiscriminativeTokenSpec(
        t_star=t_star,
        # Token at t* is predicted by the logits one position earlier.
        logit_position=t_star - 1,
        common_prefix_token_ids=prefix,
        correct_token_id=correct_token,
        distractor_token_ids=tuple(distractor_ids),
        correct_answer=correct_answer,
        distractor_answers=tuple(distractor_answers),
        prompt_len=prompt_len,
        warnings=tuple(warnings),
        metadata={
            "n_candidates": len(all_candidates),
            "candidate_token_lengths": [len(c) for c in all_candidates],
        },
    )


def spec_from_backend(
    backend: Any,
    prompt: str,
    correct_answer: str,
    distractor_answers: Sequence[str],
    *,
    strict: bool = True,
) -> DiscriminativeTokenSpec:
    """Build the spec using the backend's joint tokenization.

    Reuses ``HFBackend.tokenize``, which already asserts that ``prompt + answer`` extends
    the prompt tokenization, so answer-token indices here are exact rather than inferred.
    """
    correct = backend.tokenize(prompt, correct_answer)
    distractors = [backend.tokenize(prompt, a) for a in distractor_answers]
    return build_discriminative_spec(
        prompt_len=correct.prompt_len,
        correct_answer=correct_answer,
        correct_tokens=correct.answer_token_ids,
        distractor_answers=list(distractor_answers),
        distractor_tokens=[d.answer_token_ids for d in distractors],
        strict=strict,
    )


def objective_definition() -> dict[str, Any]:
    """The registered definition, persisted as ``discovery_objective.json``."""
    return {
        "objective_version": OBJECTIVE_VERSION,
        "role": "A1 route discovery only",
        "formula": "J_f(q) = z[correct_token @ t*] - logsumexp(z[distractor_tokens @ t*])",
        "logits_from": "position t* - 1 of a single forward pass over prompt + common prefix",
        "t_star": (
            "prompt_len + length of the longest token prefix shared by the correct answer "
            "and every matched distractor"
        ),
        "why_not_full_sequence": (
            "The full-sequence margin teacher-forces every answer token, so later tokens are "
            "conditioned on earlier correct answer tokens. Attribution against it would credit "
            "components that continue a partially written answer alongside components that "
            "retrieve the fact. 81% of SUITE answers are multi-token."
        ),
        "conditioning": (
            "Only the prompt plus the token prefix shared by all candidates. No "
            "correct-answer-specific prefix is conditioned on."
        ),
        "rejection_reasons": {
            RejectReason.CANDIDATE_IS_PREFIX: (
                "A candidate answer ends at or before t*, so it has no token to compare."
            ),
            RejectReason.DISTRACTOR_SHARES_TOKEN: (
                "A distractor has the same token as the correct answer at t*, so the margin "
                "would compare the correct token against itself."
            ),
            RejectReason.NO_DISTRACTORS: "Empty distractor set.",
            RejectReason.EMPTY_ANSWER: "A candidate contributed no answer tokens.",
        },
        "companion_metric": {
            "name": "factual_margin_v1",
            "role": "behavioral evaluation and exact causal validation",
            "note": "Retained unchanged. J_f never replaces it for causal claims.",
        },
        "consistency_check": (
            "Spearman correlation between J_f and M_f across all pilot surfaces is reported "
            "in discovery_objective.json. They measure related but distinct quantities, so "
            "the check is for sane agreement, not equality."
        ),
    }
