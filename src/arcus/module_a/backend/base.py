"""Architecture-independent backend types.

Every intervention names the *semantic tensor* it touches and the *explicit token
positions* it touches (04_BACKEND_AND_INTERVENTIONS.md sections 5 and 6). Nothing here
assumes two sequences align just because they have the same length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, Sequence


class InvalidHookPointError(ValueError):
    """Raised for a layer/component/head coordinate the backend cannot honour.

    Deliberately loud: a silently ignored patch coordinate would produce a null result
    that looks like a scientific finding.
    """


class TokenAlignmentError(ValueError):
    """Raised when target and source token positions cannot be justified."""


class TokenizationBoundaryError(RuntimeError):
    """Raised when tokenizing prompt+answer does not extend the prompt tokenization.

    If this fires, answer-token indices derived from the prompt length would be wrong,
    so scoring must stop rather than fall back to an approximation.
    """


class Component(StrEnum):
    """Canonical semantic hook-point names (04_BACKEND_AND_INTERVENTIONS.md section 5)."""

    RESID_PRE = "resid_pre"
    RESID_POST = "resid_post"
    ATTN_OUT = "attn_out"
    MLP_OUT = "mlp_out"
    # Interfaces defined now, implementations land with head-level and A4 work.
    HEAD_OUT = "head_out"
    Q = "q"
    K = "k"
    V = "v"
    ATTN_PATTERN = "attn_pattern"


#: Components the exact-patching backend implements today.
IMPLEMENTED_COMPONENTS: frozenset[Component] = frozenset(
    {Component.RESID_PRE, Component.RESID_POST, Component.ATTN_OUT, Component.MLP_OUT}
)


class PatchDirection(StrEnum):
    """Which run is the target and which supplies the replacement values."""

    #: Run the clean prompt, write corrupt activations in. Measures necessity.
    CLEAN_TO_CORRUPT = "clean_to_corrupt"
    #: Run the corrupt prompt, restore clean activations. Measures sufficiency.
    CORRUPT_TO_CLEAN = "corrupt_to_clean"
    #: Self-patch controls; must be exact no-ops.
    CLEAN_TO_CLEAN = "clean_to_clean"
    CORRUPT_TO_CORRUPT = "corrupt_to_corrupt"


class PatchMode(StrEnum):
    NODE_SET = "node_set"
    PATH_CONSISTENT = "path_consistent"


class TokenPolicy(StrEnum):
    """Named token-position policies. A patch must state one; there is no default."""

    LAST_PROMPT_TOKEN = "last_prompt_token"
    FINAL_K_PROMPT_TOKENS = "final_k_prompt_tokens"
    ALL_PROMPT_TOKENS = "all_prompt_tokens"
    EXPLICIT_INDICES = "explicit_indices"


@dataclass(frozen=True)
class HookPoint:
    """One semantic intervention location."""

    layer: int
    component: Component
    head: int | None = None

    @property
    def id(self) -> str:
        base = f"L{self.layer}.{self.component.value}"
        return base if self.head is None else f"L{self.layer}.H{self.head}.{self.component.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "component": self.component.value,
            "head": self.head,
        }


@dataclass(frozen=True)
class ModelMetadata:
    """Everything a run manifest needs to reconstruct this backend exactly."""

    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    dtype: str
    device: str
    attn_implementation: str
    seed: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_model: int
    d_head: int
    vocab_size: int
    architecture: str
    hook_map_version: str
    prompt_template_version: str
    chat_template_used: bool
    chat_template_sha256: str | None
    system_prompt: str | None
    add_generation_prompt: bool
    add_special_tokens: bool
    bos_token: str | None
    bos_token_id: int | None
    bos_inserted_by_template: bool
    eos_token: str | None
    pad_token: str | None
    generation_prompt_suffix: str | None
    eval_mode: bool
    resolved_model_commit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass(frozen=True)
class TokenizedExample:
    """A prompt plus one answer, tokenized jointly so answer indices are exact."""

    prompt_text: str
    answer_text: str
    prompt_token_ids: tuple[int, ...]
    full_token_ids: tuple[int, ...]
    answer_token_ids: tuple[int, ...]
    answer_token_positions: tuple[int, ...]
    prompt_text_sha256: str
    surface_form_id: str | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_len(self) -> int:
        return len(self.full_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_text_sha256": self.prompt_text_sha256,
            "surface_form_id": self.surface_form_id,
            "prompt_len": self.prompt_len,
            "total_len": self.total_len,
            "answer_token_ids": list(self.answer_token_ids),
            "answer_token_positions": list(self.answer_token_positions),
        }


@dataclass(frozen=True)
class SequenceScore:
    """Teacher-forced score of one answer under one prompt."""

    answer_text: str
    answer_token_ids: tuple[int, ...]
    answer_token_positions: tuple[int, ...]
    per_token_logprobs: tuple[float, ...]
    sum_logprob: float
    mean_logprob: float

    @property
    def n_answer_tokens(self) -> int:
        return len(self.answer_token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "answer_token_ids": list(self.answer_token_ids),
            "answer_token_positions": list(self.answer_token_positions),
            "per_token_logprobs": [round(x, 6) for x in self.per_token_logprobs],
            "sum_logprob": self.sum_logprob,
            "mean_logprob": self.mean_logprob,
            "n_answer_tokens": self.n_answer_tokens,
        }


@dataclass(frozen=True)
class TokenAlignment:
    """A resolved, explicit mapping of target positions to source positions.

    Persisted with every intervention so no artifact ever leaves it implicit which
    positions were written, or why index N in one sequence was matched to index N in
    another.
    """

    policy: TokenPolicy
    pairs: tuple[tuple[int, int], ...]
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.pairs:
            raise TokenAlignmentError(
                f"Token policy {self.policy.value} resolved to an empty alignment; "
                "an intervention with no positions is not a valid experiment."
            )

    @property
    def target_indices(self) -> tuple[int, ...]:
        return tuple(t for t, _ in self.pairs)

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(s for _, s in self.pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "pairs": [list(p) for p in self.pairs],
            "n_positions": len(self.pairs),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PatchSpec:
    """A complete, auditable description of one exact intervention."""

    hook_points: tuple[HookPoint, ...]
    alignment: TokenAlignment
    direction: PatchDirection
    mode: PatchMode = PatchMode.NODE_SET
    replacement: str = "exact"

    def __post_init__(self) -> None:
        if not self.hook_points:
            raise InvalidHookPointError("PatchSpec requires at least one hook point")
        if self.replacement != "exact":
            raise InvalidHookPointError(
                "Only exact activation replacement is supported. Interpolation is a "
                "dose-response experiment and must be requested explicitly."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_points": [h.to_dict() for h in self.hook_points],
            "alignment": self.alignment.to_dict(),
            "direction": self.direction.value,
            "mode": self.mode.value,
            "replacement": self.replacement,
        }


def resolve_alignment(
    policy: TokenPolicy | str,
    target: TokenizedExample,
    source: TokenizedExample,
    *,
    k: int | None = None,
    pairs: Sequence[tuple[int, int]] | None = None,
) -> TokenAlignment:
    """Turn a named policy into explicit (target, source) index pairs.

    Clean and corrupt prompts routinely differ in token count, so index-wise alignment is
    never assumed. Policies that cannot be justified for a given pair raise.
    """
    policy = TokenPolicy(policy)
    detail: dict[str, Any] = {
        "target_prompt_len": target.prompt_len,
        "source_prompt_len": source.prompt_len,
        "prompt_len_delta": target.prompt_len - source.prompt_len,
    }

    if policy is TokenPolicy.LAST_PROMPT_TOKEN:
        # Well defined regardless of length difference: both sequences have a final
        # prompt token, and it is the position the answer is generated from.
        resolved = ((target.prompt_len - 1, source.prompt_len - 1),)

    elif policy is TokenPolicy.FINAL_K_PROMPT_TOKENS:
        if k is None or k < 1:
            raise TokenAlignmentError("final_k_prompt_tokens requires k >= 1")
        span = min(k, target.prompt_len, source.prompt_len)
        detail["requested_k"] = k
        detail["effective_k"] = span
        resolved = tuple(
            (target.prompt_len - span + i, source.prompt_len - span + i) for i in range(span)
        )

    elif policy is TokenPolicy.ALL_PROMPT_TOKENS:
        if target.prompt_len != source.prompt_len:
            raise TokenAlignmentError(
                "all_prompt_tokens requires equal prompt lengths "
                f"(target {target.prompt_len} vs source {source.prompt_len}). "
                "Use last_prompt_token or final_k_prompt_tokens for unequal prompts."
            )
        resolved = tuple((i, i) for i in range(target.prompt_len))

    elif policy is TokenPolicy.EXPLICIT_INDICES:
        if not pairs:
            raise TokenAlignmentError("explicit_indices requires a non-empty pair list")
        resolved = tuple((int(t), int(s)) for t, s in pairs)

    else:  # pragma: no cover - StrEnum is exhaustive
        raise TokenAlignmentError(f"Unsupported token policy {policy}")

    for t, s in resolved:
        if not 0 <= t < target.total_len:
            raise TokenAlignmentError(
                f"target index {t} outside sequence of length {target.total_len}"
            )
        if not 0 <= s < source.total_len:
            raise TokenAlignmentError(
                f"source index {s} outside sequence of length {source.total_len}"
            )

    return TokenAlignment(policy=policy, pairs=resolved, detail=detail)


def text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class ModelBackend(Protocol):
    """The scientific operations every backend must expose.

    Experiment stages call these; no stage reaches into Hugging Face module paths.
    """

    def metadata(self) -> ModelMetadata: ...

    def tokenize(self, prompt: str, answer: str) -> TokenizedExample: ...

    def score_answers(self, prompt: str, answers: Sequence[str]) -> list[SequenceScore]: ...

    def available_hook_points(self) -> list[HookPoint]: ...

    def capture(
        self, prompt: str, hook_points: Sequence[HookPoint]
    ) -> dict[HookPoint, Any]: ...

    def score_answers_with_patch(
        self,
        prompt: str,
        answers: Sequence[str],
        patch_spec: PatchSpec,
        source_activations: dict[HookPoint, Any],
    ) -> list[SequenceScore]: ...
