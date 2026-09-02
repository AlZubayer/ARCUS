"""Candidate circuit extraction from attribution vectors.

All three rules are registered here *before* any held-out surface is touched, and the
primary rule is named in config. The others are sensitivity analyses, not alternatives to be
chosen after seeing which one validates best.

Output is a CandidateCircuit. It is never called a fact circuit: promotion requires exact
necessity, sufficiency and selectivity on held-out surfaces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

EXTRACTION_VERSION = "candidate_circuit_v1"


class ExtractionRule:
    ATTRIBUTION_MASS_PREFIX = "attribution_mass_prefix"
    TOP_K = "top_k"
    STABILITY = "stability"

    ALL = (ATTRIBUTION_MASS_PREFIX, TOP_K, STABILITY)


@dataclass
class CandidateCircuit:
    """A candidate only. Nothing here is a causal claim."""

    circuit_id: str
    fact_id: str
    rule: str
    threshold: float
    object_ids: tuple[str, ...]
    signed_weights: tuple[float, ...]
    attribution_mass_captured: float
    total_abs_attribution: float
    surface_stability: dict[str, Any]
    discovery_surface_ids: tuple[str, ...]
    family: str | None = None
    residual_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.object_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "circuit_id": self.circuit_id,
            "extraction_version": EXTRACTION_VERSION,
            "fact_id": self.fact_id,
            "family": self.family,
            "selection_rule": self.rule,
            "selection_value": self.threshold,
            "size": self.size,
            "objects": list(self.object_ids),
            "signed_weights": [round(float(w), 6) for w in self.signed_weights],
            "selected_abs_attribution": round(self.attribution_mass_captured, 6),
            "total_abs_attribution": round(self.total_abs_attribution, 6),
            "mass_fraction_captured": (
                round(self.attribution_mass_captured / self.total_abs_attribution, 6)
                if self.total_abs_attribution > 0
                else None
            ),
            "surface_stability": self.surface_stability,
            "discovery_surface_ids": list(self.discovery_surface_ids),
            "fact_specific_residual_score": (
                round(self.residual_score, 6) if self.residual_score is not None else None
            ),
            "status": "candidate_only",
            "not_a_claim": (
                "Derived from attribution on discovery surfaces. Promotion to a fact route "
                "requires exact necessity, sufficiency and selectivity on held-out surfaces."
            ),
            **self.metadata,
        }


def mean_vector(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    return matrix[list(indices)].mean(axis=0)


def extract_by_mass(vector: np.ndarray, object_ids: Sequence[str], fraction: float):
    """Smallest prefix of |attribution| covering the requested share of total mass."""
    order = np.argsort(-np.abs(vector))
    total = float(np.abs(vector).sum())
    if total <= 0:
        return [], 0.0, total
    running = 0.0
    chosen: list[int] = []
    for j in order:
        chosen.append(int(j))
        running += float(abs(vector[j]))
        if running / total >= fraction:
            break
    return chosen, running, total


def extract_top_k(vector: np.ndarray, object_ids: Sequence[str], k: int):
    order = np.argsort(-np.abs(vector))[:k]
    total = float(np.abs(vector).sum())
    return [int(j) for j in order], float(np.abs(vector[order]).sum()), total


def extract_by_stability(
    matrix: np.ndarray,
    indices: Sequence[int],
    object_ids: Sequence[str],
    *,
    top_k: int,
    fraction: float,
):
    """Objects appearing in the top-k of at least ``fraction`` of a fact's surfaces.

    This is the rule that most directly encodes the A1 hypothesis: a formulation-invariant
    core should be stable across surfaces, not merely strong on one of them.
    """
    counts: dict[int, int] = defaultdict(int)
    for i in indices:
        for j in np.argsort(-np.abs(matrix[i]))[:top_k]:
            counts[int(j)] += 1
    needed = max(2, int(np.ceil(fraction * len(indices))))
    chosen = sorted(
        (j for j, c in counts.items() if c >= needed),
        key=lambda j: -abs(float(mean_vector(matrix, indices)[j])),
    )
    avg = mean_vector(matrix, indices)
    total = float(np.abs(avg).sum())
    return chosen, float(np.abs(avg[chosen]).sum()) if chosen else 0.0, total


def surface_stability(
    matrix: np.ndarray, indices: Sequence[int], chosen: Sequence[int], *, top_k: int
) -> dict[str, Any]:
    """How consistently the selected objects rank highly across the fact's own surfaces."""
    if not chosen:
        return {"n_surfaces": len(indices), "mean_fraction_in_topk": None}
    per_surface = []
    for i in indices:
        ranked = set(int(j) for j in np.argsort(-np.abs(matrix[i]))[:top_k])
        per_surface.append(len(set(chosen) & ranked) / len(chosen))
    signs = np.sign(matrix[list(indices)][:, list(chosen)])
    sign_consistency = float(np.abs(signs.mean(axis=0)).mean()) if len(indices) > 1 else 1.0
    return {
        "n_surfaces": len(indices),
        "top_k_used": top_k,
        "mean_fraction_in_topk": round(float(np.mean(per_surface)), 6),
        "min_fraction_in_topk": round(float(np.min(per_surface)), 6),
        "sign_consistency": round(sign_consistency, 6),
        "note": (
            "sign_consistency is the mean |average sign| over the selected objects; 1.0 "
            "means every surface agreed on the direction of every selected object."
        ),
    }


def extract_circuits(
    matrix: np.ndarray,
    object_ids: Sequence[str],
    fact_groups: dict[str, list[int]],
    rows: Sequence[dict[str, Any]],
    *,
    rules: Sequence[str],
    mass_fraction: float,
    top_k: int,
    stability_fraction: float,
    residual_matrix: np.ndarray | None = None,
    family: str | None = None,
) -> list[CandidateCircuit]:
    """Extract one candidate per (fact, rule). All rules run; none is picked after the fact."""
    circuits: list[CandidateCircuit] = []
    for fact, indices in sorted(fact_groups.items()):
        if not indices:
            continue
        avg = mean_vector(matrix, indices)
        surfaces = tuple(
            str(rows[i].get("surface_form_id") or rows[i]["item_id"]) for i in indices
        )

        for rule in rules:
            if rule == ExtractionRule.ATTRIBUTION_MASS_PREFIX:
                chosen, captured, total = extract_by_mass(avg, object_ids, mass_fraction)
                threshold = mass_fraction
            elif rule == ExtractionRule.TOP_K:
                chosen, captured, total = extract_top_k(avg, object_ids, top_k)
                threshold = float(top_k)
            elif rule == ExtractionRule.STABILITY:
                chosen, captured, total = extract_by_stability(
                    matrix, indices, object_ids, top_k=top_k, fraction=stability_fraction
                )
                threshold = stability_fraction
            else:
                raise ValueError(f"Unknown extraction rule {rule!r}")

            residual_score = None
            if residual_matrix is not None and chosen:
                residual_avg = mean_vector(residual_matrix, indices)
                residual_score = float(np.abs(residual_avg[chosen]).sum()) / max(
                    float(np.abs(residual_avg).sum()), 1e-12
                )

            circuits.append(
                CandidateCircuit(
                    circuit_id=f"{fact}|{rule}|{family or 'any'}",
                    fact_id=fact,
                    rule=rule,
                    threshold=threshold,
                    object_ids=tuple(object_ids[j] for j in chosen),
                    signed_weights=tuple(float(avg[j]) for j in chosen),
                    attribution_mass_captured=captured,
                    total_abs_attribution=total,
                    surface_stability=surface_stability(
                        matrix, indices, chosen, top_k=top_k
                    ),
                    discovery_surface_ids=surfaces,
                    family=family,
                    residual_score=residual_score,
                    metadata={
                        "n_discovery_surfaces": len(indices),
                        "layers_touched": sorted(
                            {int(object_ids[j].split(".")[0][1:]) for j in chosen}
                        ),
                    },
                )
            )
    return circuits


def extraction_policy(
    *, primary_rule: str, mass_fraction: float, top_k: int, stability_fraction: float
) -> dict[str, Any]:
    return {
        "extraction_version": EXTRACTION_VERSION,
        "primary_rule": primary_rule,
        "rules": {
            ExtractionRule.ATTRIBUTION_MASS_PREFIX: {
                "value": mass_fraction,
                "definition": "smallest prefix of |attribution| covering this share of total mass",
            },
            ExtractionRule.TOP_K: {
                "value": top_k,
                "definition": "top k objects by |attribution| of the fact's mean vector",
            },
            ExtractionRule.STABILITY: {
                "value": stability_fraction,
                "definition": (
                    "objects in the top-k of at least this fraction of the fact's discovery "
                    "surfaces"
                ),
            },
        },
        "registered_before_heldout": True,
        "note": (
            "All three rules are registered in config before any held-out surface is "
            "touched. The non-primary rules are sensitivity analyses, not alternatives to "
            "select among after seeing which validates best."
        ),
    }
