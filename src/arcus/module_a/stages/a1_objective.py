"""Register the A1 discovery objective and check it against the sequence metric.

J_f and M_f measure related but different things, so the check is for sane agreement, not
equality. What attribution actually depends on is the *direction* of the clean/corrupt
difference, so that is reported separately and is the criterion that matters.
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from ..backend.hf import HFBackend
from ..objectives import ObjectiveUndefined, objective_definition, spec_from_backend
from ..schema import FactKey, Modality
from ..scoring import DistractorSet

OBJECTIVE_STAGE_VERSION = "a1_discovery_objective_v1"


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Rank correlation with tie averaging."""
    if len(a) < 2:
        return float("nan")

    def rank(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)
    ) ** 0.5
    return num / den if den else float("nan")


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 4),
        "median": round(statistics.median(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def check_surface_consistency(
    backend: HFBackend,
    score_rows: Sequence[dict[str, Any]],
    distractor_sets: dict[tuple[FactKey, Modality], DistractorSet],
) -> dict[str, Any]:
    """Compare J_f and M_f over every scored surface, including incorrect ones.

    Restricting to correct surfaces would range-restrict the comparison and make a low rank
    correlation uninterpretable, so the full range is used.
    """
    js: list[float] = []
    ms: list[float] = []
    rejected: list[dict[str, Any]] = []
    prefix_lengths: list[int] = []

    for row in score_rows:
        key = FactKey(**row["fact_key"])
        modality = Modality(row["modality"])
        pool = distractor_sets.get((key, modality))
        if pool is None:
            continue
        prompt = backend.build_prompt(row["question"])
        try:
            spec = spec_from_backend(backend, prompt, pool.correct_answer, list(pool.distractors))
        except ObjectiveUndefined as exc:
            rejected.append(
                {
                    "fact_id": key.id,
                    "modality": modality.value,
                    "surface_form_id": row["surface_form_id"],
                    "reason": str(exc),
                }
            )
            continue
        js.append(backend.score_discriminative_margin(prompt, spec))
        ms.append(row["factual_margin"])
        prefix_lengths.append(spec.n_common_prefix_tokens)

    sign_agree = sum(1 for j, m in zip(js, ms) if (j > 0) == (m > 0))
    return {
        "scope": "all scored surfaces, correct and incorrect, so the range is not restricted",
        "n_scored": len(js),
        "n_rejected": len(rejected),
        "rejected": rejected[:50],
        "J_f": _summary(js),
        "M_f": _summary(ms),
        "spearman_J_vs_M": round(spearman(js, ms), 4),
        "sign_agreement": round(sign_agree / len(js), 4) if js else None,
        "fraction_J_positive": round(sum(1 for j in js if j > 0) / len(js), 4) if js else None,
        "fraction_M_positive": round(sum(1 for m in ms if m > 0) / len(ms), 4) if ms else None,
        "common_prefix_tokens": {
            "mean": round(statistics.fmean(prefix_lengths), 4) if prefix_lengths else None,
            "max": max(prefix_lengths) if prefix_lengths else None,
            "n_zero": prefix_lengths.count(0),
            "note": (
                "Zero means J_f conditions on the prompt alone, with no answer token "
                "conditioned on at all."
            ),
        },
    }


def check_pair_direction_consistency(
    backend: HFBackend,
    pairs: Sequence[dict[str, Any]],
    distractor_sets: dict[tuple[FactKey, Modality], DistractorSet],
    *,
    limit: int = 60,
) -> dict[str, Any]:
    """Do J_f and M_f agree on the sign of the clean/corrupt effect?

    This is the criterion that matters: attribution is computed against the clean-corrupt
    difference, so the two metrics must agree on its direction even if they order individual
    surfaces differently.
    """
    dj: list[float] = []
    dm: list[float] = []
    for pair in list(pairs)[:limit]:
        key = FactKey(**pair["target_fact_key"])
        modality = Modality(pair["modality"])
        pool = distractor_sets.get((key, modality))
        if pool is None:
            continue
        clean_prompt = backend.build_prompt(pair["clean_question"])
        corrupt_prompt = backend.build_prompt(pair["corrupt_question"])
        try:
            clean_spec = spec_from_backend(
                backend, clean_prompt, pool.correct_answer, list(pool.distractors)
            )
            corrupt_spec = spec_from_backend(
                backend, corrupt_prompt, pool.correct_answer, list(pool.distractors)
            )
        except ObjectiveUndefined:
            continue
        dj.append(
            backend.score_discriminative_margin(clean_prompt, clean_spec)
            - backend.score_discriminative_margin(corrupt_prompt, corrupt_spec)
        )
        dm.append(pair["delta"])

    sign_agree = sum(1 for a, b in zip(dj, dm) if (a > 0) == (b > 0))
    return {
        "criterion": (
            "Attribution is computed against the clean-corrupt difference, so agreement on "
            "its sign is what the objective must satisfy."
        ),
        "n_pairs": len(dj),
        "delta_J": _summary(dj),
        "delta_M": _summary(dm),
        "spearman_deltaJ_vs_deltaM": round(spearman(dj, dm), 4),
        "sign_agreement": round(sign_agree / len(dj), 4) if dj else None,
        "n_sign_disagreements": len(dj) - sign_agree,
    }


def build_objective_artifact(
    backend: HFBackend,
    *,
    score_rows: Sequence[dict[str, Any]],
    pairs: Sequence[dict[str, Any]],
    distractor_sets: dict[tuple[FactKey, Modality], DistractorSet],
    pair_limit: int = 60,
) -> dict[str, Any]:
    surface = check_surface_consistency(backend, score_rows, distractor_sets)
    direction = check_pair_direction_consistency(
        backend, pairs, distractor_sets, limit=pair_limit
    )

    notes: list[str] = []
    if surface["sign_agreement"] is not None and surface["sign_agreement"] < 0.95:
        notes.append(
            f"J_f and M_f agree in sign on {surface['sign_agreement']:.1%} of individual "
            f"surfaces ({surface['fraction_J_positive']:.1%} of surfaces have J_f>0 versus "
            f"{surface['fraction_M_positive']:.1%} for M_f). J_f is the more permissive of "
            "the two because it only requires the discriminative token to be right, whereas "
            "M_f requires the whole answer sequence to beat the distractors on average. That "
            "gap is the answer-prefix effect J_f exists to avoid, seen from the other side."
        )
    if direction["sign_agreement"] is not None:
        notes.append(
            f"On clean/corrupt pairs the two agree in sign {direction['sign_agreement']:.1%} "
            f"of the time ({direction['n_sign_disagreements']} disagreements of "
            f"{direction['n_pairs']}). That is the property attribution depends on."
        )

    return {
        "stage_version": OBJECTIVE_STAGE_VERSION,
        "definition": objective_definition(),
        "surface_consistency": surface,
        "pair_direction_consistency": direction,
        "interpretation": notes,
        "verdict_note": (
            "J_f is registered for A1 discovery only. Every causal claim in A1 is validated "
            "with the full-sequence factual margin M_f, which is unchanged from P0-P5."
        ),
    }
