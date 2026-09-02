"""Gate G4: validate attribution before any ranking is trusted.

Attribution is a cheap approximation to a causal effect. Before its ordering is used to
select anything, it has to be checked against the expensive thing it approximates: exact
single-object intervention.

If G4 fails, the milestone stops here and says so. A ranking that does not track exact
causal effect is an attribution pattern, not a route.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

import numpy as np

from ..backend.base import Component, HookPoint
from ..backend.hf import HFBackend
from ..discovery.eap_ig import attribute_pair, resolve_end_aligned
from ..discovery.graph import G0Graph
from ..objectives import spec_from_backend
from ..stages.a1_objective import spearman

G4_VERSION = "a1_gate_g4_v1"


def _hook_point_from_id(object_id: str) -> HookPoint:
    parts = object_id.split(".")
    layer = int(parts[0][1:])
    if parts[1].startswith("H"):
        return HookPoint(layer=layer, component=Component.HEAD_OUT, head=int(parts[1][1:]))
    return HookPoint(layer=layer, component=Component.MLP_OUT)


def exact_single_object_effect(
    backend: HFBackend,
    graph: G0Graph,
    *,
    clean_prompt: str,
    corrupt_prompt: str,
    spec: Any,
    object_ids: Sequence[str],
    alignment_policy: str,
) -> dict[str, float]:
    """Exact effect of replacing one object's output with its corrupt value.

    Run on the clean prompt, one object at a time, everything else endogenous. This is the
    quantity attribution is trying to approximate, measured directly.
    """
    from ..discovery.eap_ig import _capture_nodes

    clean_acts = _capture_nodes(backend, clean_prompt, graph, spec)
    corrupt_acts = _capture_nodes(backend, corrupt_prompt, graph, spec)
    pairs, _ = resolve_end_aligned(
        clean_acts[graph.nodes[0]].shape[1],
        corrupt_acts[graph.nodes[0]].shape[1],
        policy=alignment_policy,
    )
    baseline = backend.score_discriminative_margin(clean_prompt, spec)

    effects: dict[str, float] = {}
    for object_id in object_ids:
        hp = _hook_point_from_id(object_id)
        source = corrupt_acts[hp]
        hook = backend.hook_map.patch_hook(hp, source, pairs)
        value = backend.score_discriminative_margin(
            clean_prompt, spec, hook_specs=[(hp, hook)]
        )
        # Sign convention: attribution is (clean - corrupt) . grad, i.e. how much of the
        # clean-over-corrupt advantage this object carries. Replacing it with the corrupt
        # value should therefore REDUCE J by that amount.
        effects[object_id] = baseline - value
    return effects


def sample_objects(
    vector: np.ndarray,
    object_ids: Sequence[str],
    *,
    n_top: int,
    n_random: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Top-attributed objects, and layer-matched random objects as a control."""
    order = np.argsort(-np.abs(vector))
    top = [object_ids[int(j)] for j in order[:n_top]]

    rng = np.random.default_rng(seed)
    top_layers = [int(o.split(".")[0][1:]) for o in top]
    pool_by_layer: dict[int, list[str]] = {}
    for i, object_id in enumerate(object_ids):
        layer = int(object_id.split(".")[0][1:])
        pool_by_layer.setdefault(layer, []).append(object_id)

    random_objects: list[str] = []
    for layer in top_layers[:n_random]:
        candidates = [o for o in pool_by_layer[layer] if o not in top]
        if candidates:
            random_objects.append(candidates[int(rng.integers(0, len(candidates)))])
    return top, random_objects


def check_sign_and_rank_agreement(
    backend: HFBackend,
    graph: G0Graph,
    *,
    clean_question: str,
    corrupt_question: str,
    correct_answer: str,
    distractors: Sequence[str],
    vector: np.ndarray,
    object_ids: Sequence[str],
    alignment_policy: str,
    n_top: int,
    n_random: int,
    seed: int,
) -> dict[str, Any]:
    """G4-A and G4-B on one pair, plus the G4-D random control."""
    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    spec = spec_from_backend(backend, clean_prompt, correct_answer, list(distractors))

    top, random_objects = sample_objects(
        vector, object_ids, n_top=n_top, n_random=n_random, seed=seed
    )
    index = {o: i for i, o in enumerate(object_ids)}
    sampled = top + random_objects

    effects = exact_single_object_effect(
        backend, graph,
        clean_prompt=clean_prompt, corrupt_prompt=corrupt_prompt, spec=spec,
        object_ids=sampled, alignment_policy=alignment_policy,
    )

    attr_top = [float(vector[index[o]]) for o in top]
    eff_top = [effects[o] for o in top]
    eff_rand = [effects[o] for o in random_objects]

    sign_agree = sum(1 for a, e in zip(attr_top, eff_top) if (a > 0) == (e > 0))
    return {
        "n_top": len(top),
        "n_random": len(random_objects),
        "sign_agreement_top": round(sign_agree / len(top), 4) if top else None,
        "spearman_attr_vs_exact_top": round(spearman(attr_top, eff_top), 4)
        if len(top) > 2
        else None,
        "spearman_abs_top": round(
            spearman([abs(a) for a in attr_top], [abs(e) for e in eff_top]), 4
        )
        if len(top) > 2
        else None,
        "mean_abs_exact_effect_top": round(statistics.fmean([abs(e) for e in eff_top]), 6)
        if eff_top
        else None,
        "mean_abs_exact_effect_random": round(
            statistics.fmean([abs(e) for e in eff_rand]), 6
        )
        if eff_rand
        else None,
        "top_over_random_ratio": (
            round(
                statistics.fmean([abs(e) for e in eff_top])
                / max(statistics.fmean([abs(e) for e in eff_rand]), 1e-9),
                4,
            )
            if eff_top and eff_rand
            else None
        ),
        "rows": [
            {
                "object_id": o,
                "group": "top" if o in top else "random_layer_matched",
                "attribution": round(float(vector[index[o]]), 6),
                "exact_effect": round(effects[o], 6),
                "sign_agrees": (float(vector[index[o]]) > 0) == (effects[o] > 0),
            }
            for o in sampled
        ],
    }


def check_ig_step_sensitivity(
    backend: HFBackend,
    graph: G0Graph,
    *,
    clean_question: str,
    corrupt_question: str,
    correct_answer: str,
    distractors: Sequence[str],
    step_counts: Sequence[int],
    alignment_policy: str,
    reference_steps: int,
    top_k: int = 30,
) -> dict[str, Any]:
    """G4-C: does the conclusion depend on how finely the path is integrated?"""
    clean_prompt = backend.build_prompt(clean_question)
    corrupt_prompt = backend.build_prompt(corrupt_question)
    spec = spec_from_backend(backend, clean_prompt, correct_answer, list(distractors))

    results: dict[int, np.ndarray] = {}
    meta: dict[int, dict[str, Any]] = {}
    for steps in step_counts:
        r = attribute_pair(
            backend, graph,
            clean_prompt=clean_prompt, corrupt_prompt=corrupt_prompt, spec=spec,
            integration_steps=steps, alignment_policy=alignment_policy,
        )
        results[steps] = r.scores
        meta[steps] = {
            "l1_norm": round(float(np.abs(r.scores).sum()), 6),
            "l2_norm": round(float(np.linalg.norm(r.scores)), 6),
            "completeness_ratio": round(r.completeness_ratio, 6)
            if r.completeness_ratio is not None
            else None,
        }

    reference = results.get(reference_steps, results[max(results)])
    ref_top = set(np.argsort(-np.abs(reference))[:top_k].tolist())
    for steps in step_counts:
        this_top = set(np.argsort(-np.abs(results[steps]))[:top_k].tolist())
        meta[steps]["topk_jaccard_vs_reference"] = round(
            len(ref_top & this_top) / len(ref_top | this_top), 4
        )
        meta[steps]["spearman_vs_reference"] = round(
            spearman(results[steps].tolist(), reference.tolist()), 4
        )
        meta[steps]["cosine_vs_reference"] = round(
            float(
                results[steps]
                @ reference
                / max(np.linalg.norm(results[steps]) * np.linalg.norm(reference), 1e-12)
            ),
            6,
        )
    return {
        "reference_steps": reference_steps,
        "top_k": top_k,
        "by_steps": {str(k): v for k, v in sorted(meta.items())},
    }


def summarise_g4(
    *,
    completeness: dict[str, Any],
    agreement: Sequence[dict[str, Any]],
    step_sensitivity: dict[str, Any],
    family_robustness: dict[str, Any],
    firewall: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Collect the criteria and issue one machine-readable verdict."""
    sign = [a["sign_agreement_top"] for a in agreement if a["sign_agreement_top"] is not None]
    ratios = [
        a["top_over_random_ratio"] for a in agreement if a["top_over_random_ratio"] is not None
    ]
    jaccards = [
        v["topk_jaccard_vs_reference"]
        for k, v in step_sensitivity.get("by_steps", {}).items()
        if int(k) >= 8
    ]

    criteria = {
        "completeness": {
            "value": completeness.get("fraction_within_5pct_of_1"),
            "threshold": thresholds["completeness_fraction"],
            "passed": bool(
                (completeness.get("fraction_within_5pct_of_1") or 0)
                >= thresholds["completeness_fraction"]
            ),
            "what_it_shows": "the attribution decomposes the path effect it claims to",
        },
        "A_sign_agreement": {
            "value": round(statistics.fmean(sign), 4) if sign else None,
            "threshold": thresholds["sign_agreement"],
            "passed": bool(sign and statistics.fmean(sign) >= thresholds["sign_agreement"]),
            "what_it_shows": "attribution sign matches the exact intervention's sign",
        },
        "B_top_over_random": {
            "value": round(statistics.fmean(ratios), 4) if ratios else None,
            "threshold": thresholds["top_over_random"],
            "passed": bool(
                ratios and statistics.fmean(ratios) >= thresholds["top_over_random"]
            ),
            "what_it_shows": (
                "top-attributed objects have larger exact effects than layer-matched random "
                "ones"
            ),
        },
        "C_step_stability": {
            "value": round(min(jaccards), 4) if jaccards else None,
            "threshold": thresholds["topk_jaccard"],
            "passed": bool(jaccards and min(jaccards) >= thresholds["topk_jaccard"]),
            "what_it_shows": "the ranking does not depend on the integration step count",
        },
        "E_family_robustness": {
            "value": family_robustness.get("n_families_agreeing"),
            "threshold": thresholds["families_agreeing"],
            "passed": bool(
                (family_robustness.get("n_families_agreeing") or 0)
                >= thresholds["families_agreeing"]
            ),
            "what_it_shows": "conclusions survive a change of corruption family",
        },
        "F_heldout_firewall": {
            "value": firewall.get("n_heldout_surfaces_in_discovery"),
            "threshold": 0,
            "passed": bool(firewall.get("passed")),
            "what_it_shows": "no validation or stress surface entered discovery",
        },
    }
    return {
        "g4_version": G4_VERSION,
        "criteria": criteria,
        "g4_passed": all(c["passed"] for c in criteria.values()),
        "blocking": (
            "If g4_passed is false the milestone stops here. A ranking that does not track "
            "exact causal effect is an attribution pattern, not a route."
        ),
    }
