"""A1 exact causal validation of candidate circuits on held-out surfaces.

Attribution ranked the candidates. This measures them, with exact multi-node intervention
and the **full-sequence factual margin** M_f -- not the discovery objective. A circuit that
only moves the quantity it was selected against has proved nothing.

    Necessity   N_f(C) = (M_f(q+) - M_f(q+; C <- q-)) / Delta_f
    Sufficiency S_f(C) = (M_f(q-; C <- q+) - M_f(q-)) / Delta_f

Neither is clipped: interventions can overcorrect, and hiding that would misreport the
measurement. Both are undefined when |Delta_f| is below the configured floor.

Selectivity applies the *same* intervention to each retain ring. A circuit that damages
neighbouring facts as much as its target is not fact-selective, however large its necessity.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from ..backend.base import (
    Component,
    HookPoint,
    PatchDirection,
    PatchSpec,
    TokenPolicy,
    resolve_alignment,
)
from ..backend.hf import HFBackend
from ..scoring import DistractorSet, factual_margin
from .patching import normalized_effect

VALIDATION_VERSION = "a1_exact_validation_v1"
NECESSITY_VERSION = "necessity_v1"
SUFFICIENCY_VERSION = "sufficiency_v1"


def hook_point_from_id(object_id: str) -> HookPoint:
    """`L12.H7.head_out` / `L12.mlp_out` -> HookPoint."""
    parts = object_id.split(".")
    layer = int(parts[0][1:])
    if parts[1].startswith("H"):
        return HookPoint(layer=layer, component=Component.HEAD_OUT, head=int(parts[1][1:]))
    return HookPoint(layer=layer, component=Component.MLP_OUT)


def _margin(
    backend: HFBackend,
    prompt: str,
    pool: DistractorSet,
    *,
    answer_score: str,
    patch_spec: PatchSpec | None = None,
    source_activations: dict[HookPoint, Any] | None = None,
) -> float:
    """Full-sequence factual margin, optionally under an exact intervention."""
    if patch_spec is None:
        scores = backend.score_answers(prompt, pool.answers)
    else:
        scores = backend.score_answers_with_patch(
            prompt, pool.answers, patch_spec, source_activations or {}
        )
    values = [getattr(s, answer_score) for s in scores]
    return factual_margin(values[0], values[1:])


def validate_circuit_on_pair(
    backend: HFBackend,
    *,
    object_ids: Sequence[str],
    clean_question: str,
    corrupt_question: str,
    pool: DistractorSet,
    min_abs_delta: float,
    answer_score: str = "mean_logprob",
    final_k: int | None = None,
) -> dict[str, Any]:
    """Exact necessity and sufficiency for one circuit on one held-out pair."""
    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    clean_ex = backend.tokenize(clean_prompt, pool.correct_answer)
    corrupt_ex = backend.tokenize(corrupt_prompt, pool.correct_answer)

    points = tuple(hook_point_from_id(o) for o in object_ids)
    clean_acts = backend.capture(clean_prompt, points)
    corrupt_acts = backend.capture(corrupt_prompt, points)

    # Same end-aligned policy attribution used, so the intervention and the ranking that
    # produced it address the same positions.
    span = final_k or min(clean_ex.prompt_len, corrupt_ex.prompt_len)
    necessity_alignment = resolve_alignment(
        TokenPolicy.FINAL_K_PROMPT_TOKENS, clean_ex, corrupt_ex, k=span
    )
    sufficiency_alignment = resolve_alignment(
        TokenPolicy.FINAL_K_PROMPT_TOKENS, corrupt_ex, clean_ex, k=span
    )

    clean_margin = _margin(backend, clean_prompt, pool, answer_score=answer_score)
    corrupt_margin = _margin(backend, corrupt_prompt, pool, answer_score=answer_score)
    delta = clean_margin - corrupt_margin

    necessity_spec = PatchSpec(
        hook_points=points,
        alignment=necessity_alignment,
        direction=PatchDirection.CLEAN_TO_CORRUPT,
    )
    necessity_margin = _margin(
        backend, clean_prompt, pool, answer_score=answer_score,
        patch_spec=necessity_spec, source_activations=corrupt_acts,
    )

    sufficiency_spec = PatchSpec(
        hook_points=points,
        alignment=sufficiency_alignment,
        direction=PatchDirection.CORRUPT_TO_CLEAN,
    )
    sufficiency_margin = _margin(
        backend, corrupt_prompt, pool, answer_score=answer_score,
        patch_spec=sufficiency_spec, source_activations=clean_acts,
    )

    return {
        "n_objects": len(points),
        "clean_margin": round(clean_margin, 6),
        "corrupt_margin": round(corrupt_margin, 6),
        "full_effect_delta": round(delta, 6),
        "necessity": {
            "direction": PatchDirection.CLEAN_TO_CORRUPT.value,
            "intervened_margin": round(necessity_margin, 6),
            "raw_effect": round(necessity_margin - clean_margin, 6),
            "normalized": normalized_effect(
                clean_margin - necessity_margin, delta, min_abs_delta=min_abs_delta
            ),
            "metric_version": NECESSITY_VERSION,
            "alignment": necessity_alignment.to_dict()["detail"],
        },
        "sufficiency": {
            "direction": PatchDirection.CORRUPT_TO_CLEAN.value,
            "intervened_margin": round(sufficiency_margin, 6),
            "raw_effect": round(sufficiency_margin - corrupt_margin, 6),
            "normalized": normalized_effect(
                sufficiency_margin - corrupt_margin, delta, min_abs_delta=min_abs_delta
            ),
            "metric_version": SUFFICIENCY_VERSION,
            "alignment": sufficiency_alignment.to_dict()["detail"],
        },
        "outcome_metric": "factual_margin_v1 (full teacher-forced sequence)",
        "not_the_discovery_objective": (
            "Validation uses M_f, not the J_f the circuit was selected against. A circuit "
            "that only moves its own selection criterion has proved nothing."
        ),
    }


def selectivity_on_control(
    backend: HFBackend,
    *,
    object_ids: Sequence[str],
    clean_question: str,
    corrupt_question: str,
    pool: DistractorSet,
    answer_score: str = "mean_logprob",
    final_k: int | None = None,
) -> dict[str, Any]:
    """The same intervention applied to a control item, reported as a raw margin change.

    Deliberately raw: a ratio alone can look impressive because the denominator is small.
    """
    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    clean_ex = backend.tokenize(clean_prompt, pool.correct_answer)
    corrupt_ex = backend.tokenize(corrupt_prompt, pool.correct_answer)

    points = tuple(hook_point_from_id(o) for o in object_ids)
    corrupt_acts = backend.capture(corrupt_prompt, points)
    span = final_k or min(clean_ex.prompt_len, corrupt_ex.prompt_len)
    alignment = resolve_alignment(
        TokenPolicy.FINAL_K_PROMPT_TOKENS, clean_ex, corrupt_ex, k=span
    )

    before = _margin(backend, clean_prompt, pool, answer_score=answer_score)
    after = _margin(
        backend, clean_prompt, pool, answer_score=answer_score,
        patch_spec=PatchSpec(
            hook_points=points, alignment=alignment,
            direction=PatchDirection.CLEAN_TO_CORRUPT,
        ),
        source_activations=corrupt_acts,
    )
    return {
        "margin_before": round(before, 6),
        "margin_after": round(after, 6),
        "raw_effect": round(after - before, 6),
        "abs_raw_effect": round(abs(after - before), 6),
    }


def summarise_validation(
    rows: Sequence[dict[str, Any]],
    selectivity: dict[str, list[dict[str, Any]]],
    *,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Per-fact necessity/sufficiency and selectivity by retain ring."""

    def stat(values: Sequence[float]) -> dict[str, Any]:
        clean = [v for v in values if v is not None]
        if not clean:
            return {"n": 0, "mean": None, "median": None}
        return {
            "n": len(clean),
            "mean": round(statistics.fmean(clean), 6),
            "median": round(statistics.median(clean), 6),
            "min": round(min(clean), 6),
            "max": round(max(clean), 6),
        }

    necessity = [r["necessity"]["normalized"] for r in rows]
    sufficiency = [r["sufficiency"]["normalized"] for r in rows]
    target_abs = [abs(r["necessity"]["raw_effect"]) for r in rows]
    target_mean = statistics.fmean(target_abs) if target_abs else 0.0

    rings = {}
    for ring, entries in sorted(selectivity.items()):
        effects = [e["abs_raw_effect"] for e in entries]
        ring_mean = statistics.fmean(effects) if effects else 0.0
        rings[ring] = {
            "n": len(effects),
            "mean_abs_raw_effect": round(ring_mean, 6),
            "median_abs_raw_effect": round(statistics.median(effects), 6) if effects else None,
            "max_abs_raw_effect": round(max(effects), 6) if effects else None,
            "selectivity_ratio": round(target_mean / (epsilon + ring_mean), 4)
            if effects
            else None,
        }

    return {
        "validation_version": VALIDATION_VERSION,
        "n_interventions": len(rows),
        "necessity": stat(necessity),
        "sufficiency": stat(sufficiency),
        "target_mean_abs_raw_effect": round(target_mean, 6),
        "selectivity_by_ring": rings,
        "reporting_rule": (
            "Raw target and retain effects are reported beside every ratio. Normalized "
            "effects are never clipped, and are null when |Delta| is below the floor."
        ),
    }
