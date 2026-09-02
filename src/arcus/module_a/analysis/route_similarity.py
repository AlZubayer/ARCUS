"""Within-fact vs matched-control route similarity, raw and backbone-residual.

The question this answers is not "are two attribution vectors similar" -- almost any two
factual questions will share a large generic retrieval component. It is whether vectors from
the *same fact* are more similar to each other than to matched controls, and whether that
survives removing the generic component.

Control classes are reported separately throughout. Pooling them would let a strong
same-topic effect hide the absence of a same-syntax effect.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

SIMILARITY_VERSION = "route_similarity_cosine_v1"
BACKBONE_VERSION = "leave_one_fact_out_mean_v1"


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    """All-pairs cosine similarity of the signed attribution vectors."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    unit = matrix / safe
    sim = unit @ unit.T
    return np.clip(sim, -1.0, 1.0)


@dataclass
class VectorSet:
    """Attribution vectors plus the metadata needed to group them."""

    matrix: np.ndarray
    rows: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, **criteria: Any) -> list[int]:
        out = []
        for i, row in enumerate(self.rows):
            if all(row.get(k) == v for k, v in criteria.items()):
                out.append(i)
        return out

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


def leave_one_fact_out_backbone(
    vectors: VectorSet, fact_groups: dict[str, list[int]]
) -> dict[str, np.ndarray]:
    """Generic factual-retrieval component estimated from OTHER facts only.

    Estimating the backbone from the target fact itself would subtract part of the very
    signal being measured, so each fact gets a background built from every other fact's
    surfaces and never from its own.
    """
    backbones: dict[str, np.ndarray] = {}
    for fact in fact_groups:
        other = [i for f, idx in fact_groups.items() if f != fact for i in idx]
        if not other:
            backbones[fact] = np.zeros(vectors.matrix.shape[1])
            continue
        backbones[fact] = vectors.matrix[other].mean(axis=0)
    return backbones


def global_backbone(vectors: VectorSet, indices: Sequence[int]) -> np.ndarray:
    return vectors.matrix[list(indices)].mean(axis=0)


def subtract_backbone(matrix: np.ndarray, backbone: np.ndarray) -> np.ndarray:
    """Residual attribution: a_tilde = a - a_bar. The form the brief specifies."""
    return matrix - backbone[None, :]


def project_out_backbone(matrix: np.ndarray, backbone: np.ndarray) -> np.ndarray:
    """Secondary variant: remove the backbone DIRECTION rather than its mean vector.

    Cosine similarity is scale-invariant, so if every surface loads on the same backbone
    with a different magnitude, plain subtraction leaves a residual still dominated by it.
    Projection removes the direction outright. Reported alongside the primary subtraction,
    never instead of it.
    """
    norm = np.linalg.norm(backbone)
    if norm < 1e-12:
        return matrix.copy()
    unit = backbone / norm
    return matrix - np.outer(matrix @ unit, unit)


def _pair_values(sim: np.ndarray, rows_a: Sequence[int], rows_b: Sequence[int], *, same: bool):
    if same:
        return [
            float(sim[i, j])
            for n, i in enumerate(rows_a)
            for j in rows_a[n + 1 :]
        ]
    return [float(sim[i, j]) for i in rows_a for j in rows_b]


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
    """One-sided: how often does a label shuffle produce a gap this large?"""
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
    label: str,
    matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    """Within-fact vs each control class, for one representation of the vectors."""
    work = vectors.matrix if matrix is None else matrix
    sim = cosine_matrix(work)

    targets = {
        fact: [i for i in idx if family is None or vectors.rows[i].get("family") == family]
        for fact, idx in vectors.fact_indices(target_class).items()
    }
    targets = {f: idx for f, idx in targets.items() if len(idx) >= 2}

    classes = vectors.class_indices()

    # Same-fact similarity, per fact and pooled over facts.
    within_by_fact: dict[str, list[float]] = {
        fact: _pair_values(sim, idx, idx, same=True) for fact, idx in targets.items()
    }
    within_all = [v for values in within_by_fact.values() for v in values]

    # Between-fact similarity, computed SEPARATELY per control class.
    between: dict[str, dict[str, Any]] = {}
    for control in control_classes:
        if control == "same_topic_different_fact":
            # Other pilot facts count as same-topic controls for each other too, which is
            # the tightest available version of this comparison.
            values: list[float] = []
            facts = sorted(targets)
            for n, fa in enumerate(facts):
                for fb in facts[n + 1 :]:
                    values += _pair_values(sim, targets[fa], targets[fb], same=False)
            values += [
                float(sim[i, j])
                for fact_idx in targets.values()
                for i in fact_idx
                for j in classes.get(control, [])
            ]
        else:
            values = [
                float(sim[i, j])
                for fact_idx in targets.values()
                for i in fact_idx
                for j in classes.get(control, [])
            ]
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
        "representation": label,
        "family": family,
        "n_facts": len(targets),
        "n_target_vectors": sum(len(v) for v in targets.values()),
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
            "within-fact mean minus that control class's mean; it is descriptive, and the "
            "permutation p-value is one-sided for within > between."
        ),
    }


def top_component_stability(
    vectors: VectorSet, *, target_class: str, family: str | None, top_k: int = 20
) -> dict[str, Any]:
    """How often does the same object land in a fact's top-k across its formulations?"""
    out: dict[str, Any] = {}
    object_ids = vectors.rows[0]["_object_ids"] if vectors.rows else []
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
            "jaccard_expected_if_random": round(top_k / max(1, len(object_ids) or 700), 6),
            "objects_in_every_surface": [
                object_ids[j] if object_ids else str(j) for j in sorted(shared_all)
            ][:30],
        }
    return out
