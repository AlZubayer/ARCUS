"""Within-fact vs matched-control route similarity, raw and backbone-residual.

The question is not "are two attribution vectors similar" -- almost any two factual
questions share a large generic retrieval component. It is whether vectors from the *same
fact* are more similar to each other than to matched controls, and whether that survives
removing the generic component.

Symmetry rule
-------------
Every cosine compares two vectors that have had the **same** background subtracted.
Residualising only one side would shrink the similarity mechanically, because the shared
backbone is exactly the part being removed, and would inflate every distinctness figure.

The background is always estimated from facts excluding *both* vectors in the comparison:

* within fact f            -> background from pilot facts other than f
* fact f vs control item c -> background from pilot facts other than f (c is never a
                              pilot fact, so nothing of c is in it)
* fact f vs fact g         -> background from pilot facts other than f and g

Control classes are reported separately throughout. Pooling them would let a strong
same-topic effect hide the absence of a same-syntax effect.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

SIMILARITY_VERSION = "route_similarity_cosine_v2"
BACKBONE_VERSION = "leave_facts_out_mean_v1"


class Representation:
    RAW = "raw_attribution"
    SUBTRACT = "residual_backbone_subtracted"
    PROJECT = "residual_backbone_projected_out"


@dataclass
class VectorSet:
    """Attribution vectors plus the metadata needed to group them."""

    matrix: np.ndarray
    rows: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.rows)

    def fact_indices(self, item_class: str) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            if row["item_class"] == item_class and row.get("fact_id"):
                groups[row["fact_id"]].append(i)
        return dict(groups)

    def class_indices(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            groups[row["item_class"]].append(i)
        return dict(groups)


def subtract_backbone(matrix: np.ndarray, backbone: np.ndarray) -> np.ndarray:
    """a_tilde = a - a_bar. The form the brief specifies; the primary representation."""
    return matrix - backbone[None, :]


def project_out_backbone(matrix: np.ndarray, backbone: np.ndarray) -> np.ndarray:
    """Secondary variant: remove the backbone DIRECTION rather than its mean vector.

    Cosine is scale-invariant, so if every surface loads on the same backbone with a
    different magnitude, plain subtraction leaves a residual still dominated by it.
    Projection removes the direction outright. Reported alongside the primary subtraction,
    never instead of it.
    """
    norm = np.linalg.norm(backbone)
    if norm < 1e-12:
        return matrix.copy()
    unit = backbone / norm
    return matrix - np.outer(matrix @ unit, unit)


TRANSFORMS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    Representation.RAW: lambda m, b: m,
    Representation.SUBTRACT: subtract_backbone,
    Representation.PROJECT: project_out_backbone,
}


class BackboneEstimator:
    """Generic factual-retrieval component, estimated from facts only.

    The pool is the target facts. A background is never estimated from a fact that appears
    on either side of the comparison it is used for.
    """

    def __init__(self, matrix: np.ndarray, fact_groups: dict[str, list[int]]) -> None:
        self.matrix = matrix
        self.fact_groups = fact_groups
        self.dim = matrix.shape[1]

    def excluding(self, facts: Sequence[str]) -> np.ndarray:
        exclude = set(facts)
        rows = [i for f, idx in self.fact_groups.items() if f not in exclude for i in idx]
        if not rows:
            return np.zeros(self.dim)
        return self.matrix[rows].mean(axis=0)

    def pooled(self) -> np.ndarray:
        rows = [i for idx in self.fact_groups.values() for i in idx]
        return self.matrix[rows].mean(axis=0) if rows else np.zeros(self.dim)


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine between every row of a and every row of b."""
    na = np.linalg.norm(a, axis=1, keepdims=True)
    nb = np.linalg.norm(b, axis=1, keepdims=True)
    ua = a / np.where(na < 1e-12, 1.0, na)
    ub = b / np.where(nb < 1e-12, 1.0, nb)
    return np.clip(ua @ ub.T, -1.0, 1.0)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "sd": None}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), 6),
        "median": round(statistics.median(values), 6),
        "sd": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def bootstrap_ci(
    values: Sequence[float], *, seed: int, n_samples: int = 2000, alpha: float = 0.05
) -> dict[str, Any]:
    if len(values) < 2:
        return {"lo": None, "hi": None, "n_samples": 0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = arr[rng.integers(0, len(arr), size=(n_samples, len(arr)))].mean(axis=1)
    return {
        "lo": round(float(np.quantile(means, alpha / 2)), 6),
        "hi": round(float(np.quantile(means, 1 - alpha / 2)), 6),
        "n_samples": n_samples,
    }


def permutation_p_value(
    within: Sequence[float], between: Sequence[float], *, seed: int, n_samples: int = 5000
) -> float | None:
    """One-sided: how often does a label shuffle produce a gap this large?

    Note the unit is the surface *pair*, not the fact, so this is a within-corpus shuffle
    test and not a population claim about facts. The per-fact breakdown carries that.
    """
    if not within or not between:
        return None
    rng = np.random.default_rng(seed)
    pooled = np.array(list(within) + list(between), dtype=np.float64)
    n_within = len(within)
    observed = float(np.mean(within) - np.mean(between))
    count = 0
    for _ in range(n_samples):
        rng.shuffle(pooled)
        if float(pooled[:n_within].mean() - pooled[n_within:].mean()) >= observed:
            count += 1
    return round((count + 1) / (n_samples + 1), 6)


def analyse(
    vectors: VectorSet,
    *,
    target_class: str,
    control_classes: Sequence[str],
    family: str | None,
    seed: int,
    representation: str,
) -> dict[str, Any]:
    """Within-fact vs each control class, under one representation.

    Both sides of every cosine get the same background subtracted, estimated from facts
    appearing on neither side.
    """
    transform = TRANSFORMS[representation]
    matrix = vectors.matrix

    fact_groups = {
        fact: [i for i in idx if family is None or vectors.rows[i].get("family") == family]
        for fact, idx in vectors.fact_indices(target_class).items()
    }
    fact_groups = {f: idx for f, idx in fact_groups.items() if len(idx) >= 2}
    estimator = BackboneEstimator(matrix, fact_groups)
    classes = vectors.class_indices()

    within_by_fact: dict[str, list[float]] = {}
    for fact, idx in fact_groups.items():
        block = transform(matrix[idx], estimator.excluding([fact]))
        sim = _cos(block, block)
        within_by_fact[fact] = [
            float(sim[a, b]) for a in range(len(idx)) for b in range(a + 1, len(idx))
        ]
    within_all = [v for values in within_by_fact.values() for v in values]

    between: dict[str, dict[str, Any]] = {}
    for control in control_classes:
        values: list[float] = []

        if control == "same_topic_different_fact":
            # Pilot facts are same-topic controls for each other, the tightest form of this
            # comparison. Their shared background excludes both of them.
            facts = sorted(fact_groups)
            for n, fa in enumerate(facts):
                for fb in facts[n + 1 :]:
                    backbone = estimator.excluding([fa, fb])
                    sim = _cos(
                        transform(matrix[fact_groups[fa]], backbone),
                        transform(matrix[fact_groups[fb]], backbone),
                    )
                    values += [float(x) for x in sim.ravel()]

        control_rows = classes.get(control, [])
        if control_rows:
            for fact, idx in fact_groups.items():
                # The control is never a pilot fact, so excluding the target alone already
                # excludes both sides of the comparison.
                backbone = estimator.excluding([fact])
                sim = _cos(
                    transform(matrix[idx], backbone),
                    transform(matrix[control_rows], backbone),
                )
                values += [float(x) for x in sim.ravel()]

        gap = (
            round(statistics.fmean(within_all) - statistics.fmean(values), 6)
            if values and within_all
            else None
        )
        between[control] = {
            "similarity": _stats(values),
            "distinctness_D": gap,
            "bootstrap_ci_of_between_mean": bootstrap_ci(values, seed=seed),
            "permutation_p_within_gt_between": permutation_p_value(
                within_all, values, seed=seed
            ),
        }

    return {
        "similarity_version": SIMILARITY_VERSION,
        "representation": representation,
        "backbone_version": BACKBONE_VERSION if representation != Representation.RAW else None,
        "family": family,
        "n_facts": len(fact_groups),
        "n_target_vectors": sum(len(v) for v in fact_groups.values()),
        "symmetry": (
            "Both sides of every cosine had the same background subtracted, estimated from "
            "facts appearing on neither side."
        ),
        "within_fact": {
            "pooled": _stats(within_all),
            "bootstrap_ci": bootstrap_ci(within_all, seed=seed),
            "by_fact": {
                fact: _stats(values) for fact, values in sorted(within_by_fact.items())
            },
        },
        "between_by_control_class": between,
        "controls_never_pooled": True,
        "note": (
            "Cosine of signed 700-dimensional attribution vectors. Distinctness D is "
            "within-fact mean minus that control class's mean; it is descriptive. The "
            "permutation p-value shuffles surface-pair labels, so it is a within-corpus "
            "test, not a population claim about facts."
        ),
    }


def backbone_report(
    vectors: VectorSet,
    *,
    target_class: str,
    family: str | None,
    object_ids: Sequence[str],
) -> dict[str, Any]:
    """How large is the generic component, and what is in it?"""
    fact_groups = {
        fact: [i for i in idx if family is None or vectors.rows[i].get("family") == family]
        for fact, idx in vectors.fact_indices(target_class).items()
    }
    fact_groups = {f: idx for f, idx in fact_groups.items() if len(idx) >= 2}
    estimator = BackboneEstimator(vectors.matrix, fact_groups)
    pooled = estimator.pooled()

    cosines = []
    for idx in fact_groups.values():
        for i in idx:
            v = vectors.matrix[i]
            cosines.append(
                float(v @ pooled / max(np.linalg.norm(v) * np.linalg.norm(pooled), 1e-12))
            )

    order = np.argsort(-np.abs(pooled))[:20]
    return {
        "backbone_version": BACKBONE_VERSION,
        "estimator": "mean attribution vector over surfaces of the other target facts",
        "why_leave_out": (
            "Estimating the background from a fact on either side of a comparison would "
            "subtract part of the signal being measured."
        ),
        "n_facts": len(fact_groups),
        "family": family,
        "pooled_backbone_l2": round(float(np.linalg.norm(pooled)), 6),
        "cosine_target_vs_pooled_backbone": _stats(cosines),
        "interpretation": (
            "Cosine of each target vector with the pooled backbone. High values mean the raw "
            "vectors are dominated by a component shared across facts, which is why the "
            "residual comparison matters more than the raw one."
        ),
        "top_backbone_objects": [
            {"object_id": object_ids[int(j)], "score": round(float(pooled[int(j)]), 6)}
            for j in order
        ],
        "per_fact_backbone_l2": {
            fact: round(float(np.linalg.norm(estimator.excluding([fact]))), 6)
            for fact in sorted(fact_groups)
        },
    }


def top_component_stability(
    vectors: VectorSet,
    *,
    target_class: str,
    family: str | None,
    object_ids: Sequence[str],
    top_k: int = 20,
) -> dict[str, Any]:
    """How often does the same object land in a fact's top-k across its formulations?"""
    out: dict[str, Any] = {}
    for fact, idx in sorted(vectors.fact_indices(target_class).items()):
        idx = [i for i in idx if family is None or vectors.rows[i].get("family") == family]
        if len(idx) < 2:
            continue
        counts: dict[int, int] = defaultdict(int)
        for i in idx:
            for j in np.argsort(-np.abs(vectors.matrix[i]))[:top_k]:
                counts[int(j)] += 1
        n = len(idx)
        shared_all = [j for j, c in counts.items() if c == n]
        shared_two_thirds = [j for j, c in counts.items() if c >= max(2, int(0.667 * n))]
        out[fact] = {
            "n_surfaces": n,
            "top_k": top_k,
            "n_distinct_objects_in_any_topk": len(counts),
            "n_in_every_surface": len(shared_all),
            "n_in_two_thirds": len(shared_two_thirds),
            "expected_overlap_if_random": round(top_k / max(1, len(object_ids)), 6),
            "objects_in_every_surface": [object_ids[j] for j in sorted(shared_all)][:30],
        }
    return out
