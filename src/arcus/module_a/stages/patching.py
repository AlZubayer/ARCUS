"""Exact residual-stream patching: sanity gates, then the restoration/suppression scan.

    Necessity   N_f(C) = (M_f(q+) - M_f(q+; C <- q-)) / Delta_f
    Sufficiency S_f(C) = (M_f(q-; C <- q+) - M_f(q-)) / Delta_f

Neither is clipped to [0, 1]: interventions can be nonlinear or overcorrect, and hiding
that would misrepresent the measurement. Both are undefined when |Delta_f| is below the
configured floor, and are reported as null rather than as an unstable ratio.

The scan output is a restoration/suppression map. It is NOT a circuit: it says where a
single activation carries the factual difference, not that those sites form a validated
causal route.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ..backend.base import (
    Component,
    HookPoint,
    InvalidHookPointError,
    PatchDirection,
    PatchSpec,
    TokenAlignment,
    TokenAlignmentError,
    TokenPolicy,
    resolve_alignment,
)
from ..backend.hf import HFBackend
from ..scoring import DistractorSet, factual_margin

PATCHING_VERSION = "exact_residual_patch_v1"
NECESSITY_VERSION = "necessity_v1"
SUFFICIENCY_VERSION = "sufficiency_v1"


def _margin(
    backend: HFBackend,
    prompt: str,
    distractors: DistractorSet,
    *,
    answer_score: str,
    patch_spec: PatchSpec | None = None,
    source_activations: dict[HookPoint, torch.Tensor] | None = None,
) -> float:
    if patch_spec is None:
        scores = backend.score_answers(prompt, distractors.answers)
    else:
        scores = backend.score_answers_with_patch(
            prompt, distractors.answers, patch_spec, source_activations or {}
        )
    values = [getattr(s, answer_score) for s in scores]
    return factual_margin(values[0], values[1:])


def normalized_effect(
    numerator: float, delta: float, *, min_abs_delta: float
) -> float | None:
    """Divide by Delta_f, or refuse when Delta_f is too small to divide by."""
    if abs(delta) < min_abs_delta:
        return None
    return numerator / delta


def aligned_positions(
    target_prompt_len: int, source_prompt_len: int, offset_from_end: int
) -> tuple[int, int]:
    """Align one prompt position from the end of each sequence.

    Aligning from the end keeps the position semantically comparable when the clean and
    corrupt prompts differ in length: offset 0 is the token the answer is generated from.
    """
    return (target_prompt_len - 1 - offset_from_end, source_prompt_len - 1 - offset_from_end)


def run_sanity_gates(
    backend: HFBackend,
    *,
    clean_question: str,
    corrupt_question: str,
    distractors: DistractorSet,
    components: Sequence[Component],
    layers: Sequence[int],
    answer_score: str = "mean_logprob",
    tolerance: float = 0.0,
    cross_shape_tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Gates A-F on a real clean/corrupt pair, plus a random-location control."""
    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    clean_ex = backend.tokenize(clean_prompt, distractors.correct_answer)
    corrupt_ex = backend.tokenize(corrupt_prompt, distractors.correct_answer)

    points = [HookPoint(layer=layer, component=c) for layer in layers for c in components]
    clean_acts = backend.capture(clean_prompt, points)
    corrupt_acts = backend.capture(corrupt_prompt, points)

    clean_margin = _margin(backend, clean_prompt, distractors, answer_score=answer_score)
    corrupt_margin = _margin(backend, corrupt_prompt, distractors, answer_score=answer_score)

    checks: list[dict[str, Any]] = []

    # Gate A / B: patching a run with its own activations must change nothing.
    #
    # Two variants, because they test different things.
    #
    # The in-situ variant captures with the same batch and sequence shape the scoring
    # forward uses, so the written values are bitwise identical to the endogenous ones.
    # That isolates the patch write itself and is gated at exactly zero.
    #
    # The cross-shape variant captures the way real experiments do (a prompt-only forward,
    # since the corrupt source has a different length and its own answer). cuBLAS and the
    # attention kernels select tiling by tensor shape, so a float32 activation captured at
    # [1, prompt_len] differs in its last bits from the same activation inside a
    # [n_answers, prompt_len + answer_len] forward. Measured on this model: 7.7e-07 from
    # the length change, 2.0e-06 from the batch change, ~1.8e-06 combined -- about ten
    # times float32 epsilon. It is kernel shape-dependence, not a hook defect, so this
    # variant carries a documented tolerance and reports the drift it actually sees.
    for label, prompt, acts, baseline, direction, tokenized, in_situ in (
        ("A_clean_to_clean", clean_prompt, clean_acts, clean_margin,
         PatchDirection.CLEAN_TO_CLEAN, clean_ex, True),
        ("B_corrupt_to_corrupt", corrupt_prompt, corrupt_acts, corrupt_margin,
         PatchDirection.CORRUPT_TO_CORRUPT, corrupt_ex, True),
        ("A_clean_to_clean", clean_prompt, clean_acts, clean_margin,
         PatchDirection.CLEAN_TO_CLEAN, clean_ex, False),
        ("B_corrupt_to_corrupt", corrupt_prompt, corrupt_acts, corrupt_margin,
         PatchDirection.CORRUPT_TO_CORRUPT, corrupt_ex, False),
    ):
        source = (
            backend.capture_during_scoring(prompt, distractors.answers, points)
            if in_situ
            else acts
        )
        limit = tolerance if in_situ else cross_shape_tolerance
        worst = 0.0
        rows = []
        for hp in points:
            spec = PatchSpec(
                hook_points=(hp,),
                alignment=resolve_alignment(
                    TokenPolicy.LAST_PROMPT_TOKEN, tokenized, tokenized
                ),
                direction=direction,
            )
            got = _margin(
                backend, prompt, distractors, answer_score=answer_score,
                patch_spec=spec, source_activations=source,
            )
            diff = abs(got - baseline)
            worst = max(worst, diff)
            rows.append({"hook_point": hp.id, "margin": got, "abs_diff": diff})
        suffix = "_patch_is_a_no_op" if in_situ else "_patch_is_a_no_op_cross_shape_capture"
        checks.append(
            {
                "check": f"{label}{suffix}",
                "capture": "in_situ_same_shape" if in_situ else "prompt_only_forward",
                "baseline_margin": baseline,
                "n_hook_points": len(points),
                "max_abs_margin_diff": worst,
                "tolerance": limit,
                "rationale": (
                    "Bitwise-identical values written back; any drift is a patch bug."
                    if in_situ
                    else "Float32 kernel tiling depends on tensor shape; drift here is "
                    "numerical, and is ~5 orders of magnitude below min_abs_delta."
                ),
                "passed": worst <= limit,
                "rows": rows,
            }
        )

    # Gate C: capture-only neutrality, re-verified on this pair.
    sink: dict[HookPoint, torch.Tensor] = {}
    specs = [(hp, backend.hook_map.capture_hook(hp, sink)) for hp in points]
    baseline_scores = backend._score_batch(
        [backend.tokenize(clean_prompt, a) for a in distractors.answers]
    )
    captured_scores = backend._score_batch(
        [backend.tokenize(clean_prompt, a) for a in distractors.answers], hook_specs=specs
    )
    capture_diff = max(
        abs(getattr(a, answer_score) - getattr(b, answer_score))
        for a, b in zip(baseline_scores, captured_scores)
    )
    checks.append(
        {
            "check": "C_capture_only_is_a_no_op",
            "max_abs_score_diff": capture_diff,
            "tolerance": tolerance,
            "passed": capture_diff <= tolerance,
        }
    )

    # Gate D: invalid coordinates must fail loudly, never silently no-op.
    invalid: list[dict[str, Any]] = []
    n_layers = backend.metadata().n_layers
    for description, thunk in (
        ("negative_layer", lambda: backend.capture(
            clean_prompt, [HookPoint(layer=-1, component=Component.RESID_PRE)])),
        ("layer_past_end", lambda: backend.capture(
            clean_prompt, [HookPoint(layer=n_layers, component=Component.RESID_PRE)])),
        ("unimplemented_component", lambda: backend.capture(
            clean_prompt, [HookPoint(layer=0, component=Component.ATTN_PATTERN)])),
        ("head_on_headless_component", lambda: backend.capture(
            clean_prompt, [HookPoint(layer=0, component=Component.RESID_PRE, head=2)])),
        ("out_of_range_token", lambda: resolve_alignment(
            TokenPolicy.EXPLICIT_INDICES, clean_ex, corrupt_ex,
            pairs=[(clean_ex.total_len + 50, 0)])),
        ("empty_alignment", lambda: TokenAlignment(
            policy=TokenPolicy.EXPLICIT_INDICES, pairs=())),
        ("unequal_all_prompt_tokens", lambda: resolve_alignment(
            TokenPolicy.ALL_PROMPT_TOKENS, clean_ex, corrupt_ex)),
    ):
        try:
            thunk()
            invalid.append({"case": description, "raised": None, "passed": False})
        except (InvalidHookPointError, TokenAlignmentError) as exc:
            invalid.append(
                {"case": description, "raised": type(exc).__name__, "passed": True,
                 "message": str(exc)[:160]}
            )
    checks.append(
        {
            "check": "D_invalid_coordinates_fail_loudly",
            "cases": invalid,
            "passed": all(c["passed"] for c in invalid),
        }
    )

    # Gate F: alignment is explicit and recorded, and unequal prompts still align.
    alignment = resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, clean_ex, corrupt_ex)
    checks.append(
        {
            "check": "F_token_alignment_is_explicit",
            "clean_prompt_len": clean_ex.prompt_len,
            "corrupt_prompt_len": corrupt_ex.prompt_len,
            "prompt_len_delta": clean_ex.prompt_len - corrupt_ex.prompt_len,
            "alignment": alignment.to_dict(),
            "passed": len(alignment.pairs) == 1,
        }
    )

    return {
        "patching_version": PATCHING_VERSION,
        "clean_question": clean_question,
        "corrupt_question": corrupt_question,
        "clean_margin": clean_margin,
        "corrupt_margin": corrupt_margin,
        "delta": clean_margin - corrupt_margin,
        "checks": checks,
        "gate_g2_passed": all(c["passed"] for c in checks),
    }


def scan_pair(
    backend: HFBackend,
    *,
    pair_id: str,
    fact_id: str,
    clean_question: str,
    corrupt_question: str,
    distractors: DistractorSet,
    components: Sequence[Component],
    layers: Sequence[int],
    offsets_from_end: Sequence[int],
    directions: Sequence[PatchDirection],
    min_abs_delta: float,
    answer_score: str = "mean_logprob",
) -> list[dict[str, Any]]:
    """Sweep layers x token positions x components x directions for one pair."""
    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    clean_ex = backend.tokenize(clean_prompt, distractors.correct_answer)
    corrupt_ex = backend.tokenize(corrupt_prompt, distractors.correct_answer)

    points = [HookPoint(layer=layer, component=c) for layer in layers for c in components]
    clean_acts = backend.capture(clean_prompt, points)
    corrupt_acts = backend.capture(corrupt_prompt, points)

    clean_margin = _margin(backend, clean_prompt, distractors, answer_score=answer_score)
    corrupt_margin = _margin(backend, corrupt_prompt, distractors, answer_score=answer_score)
    delta = clean_margin - corrupt_margin

    rows: list[dict[str, Any]] = []
    for direction in directions:
        if direction is PatchDirection.CLEAN_TO_CORRUPT:
            target_prompt, target_ex, source_ex = clean_prompt, clean_ex, corrupt_ex
            source_acts, baseline = corrupt_acts, clean_margin
        else:
            target_prompt, target_ex, source_ex = corrupt_prompt, corrupt_ex, clean_ex
            source_acts, baseline = clean_acts, corrupt_margin

        for offset in offsets_from_end:
            t_idx, s_idx = aligned_positions(target_ex.prompt_len, source_ex.prompt_len, offset)
            if t_idx < 0 or s_idx < 0:
                continue
            alignment = resolve_alignment(
                TokenPolicy.EXPLICIT_INDICES, target_ex, source_ex, pairs=[(t_idx, s_idx)]
            )
            for hp in points:
                spec = PatchSpec(
                    hook_points=(hp,), alignment=alignment, direction=direction
                )
                intervened = _margin(
                    backend, target_prompt, distractors, answer_score=answer_score,
                    patch_spec=spec, source_activations=source_acts,
                )
                raw = intervened - baseline
                if direction is PatchDirection.CLEAN_TO_CORRUPT:
                    # Necessity: how much of the clean-corrupt gap this site destroys.
                    normalized = normalized_effect(
                        clean_margin - intervened, delta, min_abs_delta=min_abs_delta
                    )
                    metric = NECESSITY_VERSION
                else:
                    # Sufficiency: how much of the gap restoring this site recovers.
                    normalized = normalized_effect(
                        intervened - corrupt_margin, delta, min_abs_delta=min_abs_delta
                    )
                    metric = SUFFICIENCY_VERSION

                rows.append(
                    {
                        "pair_id": pair_id,
                        "fact_id": fact_id,
                        "direction": direction.value,
                        "mode": "node_set",
                        "hook_point": hp.id,
                        "layer": hp.layer,
                        "component": hp.component.value,
                        "offset_from_end": offset,
                        "target_token_index": t_idx,
                        "source_token_index": s_idx,
                        "token_policy": TokenPolicy.EXPLICIT_INDICES.value,
                        "alignment_detail": "aligned_from_prompt_end",
                        "baseline_margin": baseline,
                        "clean_margin": clean_margin,
                        "corrupt_margin": corrupt_margin,
                        "intervened_margin": intervened,
                        "full_effect": delta,
                        "raw_effect": raw,
                        "normalized_effect": normalized,
                        "metric_version": metric,
                        "patching_version": PATCHING_VERSION,
                    }
                )
    return rows


def random_location_control(
    backend: HFBackend,
    *,
    clean_question: str,
    corrupt_question: str,
    distractors: DistractorSet,
    layers: Sequence[int],
    offsets_from_end: Sequence[int],
    seed: int,
    n_samples: int,
    min_abs_delta: float,
    answer_score: str = "mean_logprob",
) -> dict[str, Any]:
    """Patch randomly chosen (layer, position) sites as a null comparison.

    Restoration concentrated at particular sites only means something if arbitrary sites do
    not restore equally well.
    """
    import random

    rng = random.Random(f"{seed}|random_location|{clean_question}")
    sites = [(layer, offset) for layer in layers for offset in offsets_from_end]
    sampled = rng.sample(sites, min(n_samples, len(sites)))

    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    clean_ex = backend.tokenize(clean_prompt, distractors.correct_answer)
    corrupt_ex = backend.tokenize(corrupt_prompt, distractors.correct_answer)

    points = [HookPoint(layer=layer, component=Component.RESID_PRE) for layer, _ in sampled]
    clean_acts = backend.capture(clean_prompt, points)
    clean_margin = _margin(backend, clean_prompt, distractors, answer_score=answer_score)
    corrupt_margin = _margin(backend, corrupt_prompt, distractors, answer_score=answer_score)
    delta = clean_margin - corrupt_margin

    effects: list[float] = []
    for layer, offset in sampled:
        hp = HookPoint(layer=layer, component=Component.RESID_PRE)
        t_idx, s_idx = aligned_positions(corrupt_ex.prompt_len, clean_ex.prompt_len, offset)
        if t_idx < 0 or s_idx < 0:
            continue
        spec = PatchSpec(
            hook_points=(hp,),
            alignment=resolve_alignment(
                TokenPolicy.EXPLICIT_INDICES, corrupt_ex, clean_ex, pairs=[(t_idx, s_idx)]
            ),
            direction=PatchDirection.CORRUPT_TO_CLEAN,
        )
        intervened = _margin(
            backend, corrupt_prompt, distractors, answer_score=answer_score,
            patch_spec=spec, source_activations=clean_acts,
        )
        normalized = normalized_effect(
            intervened - corrupt_margin, delta, min_abs_delta=min_abs_delta
        )
        if normalized is not None:
            effects.append(normalized)

    return {
        "check": "random_location_control",
        "n_sites": len(effects),
        "mean_sufficiency": sum(effects) / len(effects) if effects else None,
        "max_sufficiency": max(effects) if effects else None,
        "sampled_sites": [{"layer": layer, "offset_from_end": off} for layer, off in sampled],
        "note": (
            "Restoration is only meaningful if arbitrary sites do not restore equally well."
        ),
    }
