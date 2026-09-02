"""A1 route discovery: compute and persist full signed attribution vectors.

Targets and all five control classes are attributed identically -- same objective, same G0
graph, same integration path -- because route similarity between a target and a control is
only interpretable if the two vectors were built the same way.

Nothing here selects, thresholds, or interprets. Attribution is candidate discovery; Gate
G4 checks it against exact interventions before any ranking is used.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..backend.hf import HFBackend
from ..discovery.controls import AttributionItem, ControlClass
from ..discovery.eap_ig import ATTRIBUTION_VERSION, attribute_pair
from ..discovery.graph import G0Graph
from ..objectives import ObjectiveUndefined, spec_from_backend

DISCOVERY_STAGE_VERSION = "a1_discovery_v1"

TARGET_CLASS = "target_fact"


def run_attribution(
    backend: HFBackend,
    graph: G0Graph,
    items: Sequence[AttributionItem],
    *,
    integration_steps: int,
    alignment_policy: str,
    progress_every: int = 10,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Attribute every item. Returns the stacked matrix, an index, and any skips."""
    vectors: list[np.ndarray] = []
    index: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    started = time.time()

    for i, item in enumerate(items, start=1):
        clean_prompt = backend.build_prompt(item.clean_question)
        corrupt_prompt = backend.build_prompt(item.corrupt_question)
        try:
            spec = spec_from_backend(
                backend, clean_prompt, item.correct_answer, list(item.distractors)
            )
        except ObjectiveUndefined as exc:
            skipped.append({**item.to_dict(), "reason": str(exc)[:200]})
            continue

        try:
            result = attribute_pair(
                backend,
                graph,
                clean_prompt=clean_prompt,
                corrupt_prompt=corrupt_prompt,
                spec=spec,
                integration_steps=integration_steps,
                alignment_policy=alignment_policy,
                metadata={"item_id": item.item_id, "item_class": item.item_class},
            )
        except ValueError as exc:  # exact-length policy refusing an unequal pair
            skipped.append({**item.to_dict(), "reason": str(exc)[:200]})
            continue

        vectors.append(result.scores)
        index.append(
            {
                "vector_index": len(vectors) - 1,
                **item.to_dict(),
                "objective": spec.to_dict(),
                **result.summary(),
            }
        )
        if progress_every and i % progress_every == 0:
            rate = (time.time() - started) / i
            print(
                f"    {i}/{len(items)} attributed ({rate:.1f}s each, "
                f"{rate * (len(items) - i) / 60:.1f} min left)",
                flush=True,
            )

    matrix = np.vstack(vectors) if vectors else np.zeros((0, len(graph)))
    return matrix, index, skipped


def save_attribution(
    out_dir: Path,
    matrix: np.ndarray,
    index: Sequence[dict[str, Any]],
    graph: G0Graph,
    *,
    name: str = "vectors",
) -> dict[str, Any]:
    """Persist the FULL signed matrix, not a top-k.

    Row order matches the index; column order is the graph's fixed node order, so a vector
    is meaningless without both and they are written together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npz"
    np.savez_compressed(
        path,
        scores=matrix.astype(np.float32),
        object_ids=np.array(graph.object_ids),
        item_ids=np.array([row["item_id"] for row in index]),
    )
    return {
        "path": str(path),
        "shape": list(matrix.shape),
        "dtype": "float32",
        "row_order": "matches the index file, one row per attributed item",
        "column_order": graph.describe()["node_order"],
        "truncation": "none; every one of the graph's objects is stored with its sign",
    }


def completeness_report(index: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Does sum(attr) recover the measured path effect?

    This is a validity check on the attribution itself, independent of anything it is later
    used to conclude.
    """
    ratios = [
        row["completeness_ratio"]
        for row in index
        if row.get("completeness_ratio") is not None
    ]
    if not ratios:
        return {"n": 0}
    arr = np.array(ratios, dtype=np.float64)
    within = float(np.mean(np.abs(arr - 1.0) <= 0.05))
    return {
        "n": len(ratios),
        "mean": round(float(arr.mean()), 6),
        "median": round(float(np.median(arr)), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "fraction_within_5pct_of_1": round(within, 4),
        "criterion": "Integrated gradients should give sum(attr) == J(alpha=1) - J(alpha=0).",
        "passed": bool(within >= 0.95),
    }


def objective_health(index: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Distribution of J at both path endpoints, and how often the clean end is positive.

    J_f subtracts a logsumexp over the distractor tokens, which adds about log(n_distractors)
    nats of headroom, so J_f < 0 at the clean end does NOT mean the model prefers a
    distractor -- only that the correct token's lead is smaller than that offset. Recorded
    so the numbers are not over-read.
    """
    j_clean = np.array([row["j_clean"] for row in index], dtype=np.float64)
    j_base = np.array([row["j_corrupt_baseline"] for row in index], dtype=np.float64)
    effect = np.array([row["path_effect"] for row in index], dtype=np.float64)
    return {
        "n": len(index),
        "j_clean": {
            "mean": round(float(j_clean.mean()), 4),
            "min": round(float(j_clean.min()), 4),
            "max": round(float(j_clean.max()), 4),
            "fraction_positive": round(float((j_clean > 0).mean()), 4),
        },
        "j_corrupt_baseline": {
            "mean": round(float(j_base.mean()), 4),
            "min": round(float(j_base.min()), 4),
            "max": round(float(j_base.max()), 4),
        },
        "path_effect": {
            "mean": round(float(effect.mean()), 4),
            "min": round(float(effect.min()), 4),
            "max": round(float(effect.max()), 4),
            "fraction_positive": round(float((effect > 0).mean()), 4),
        },
        "note": (
            "J_f subtracts logsumexp over the distractor tokens, worth about "
            "log(n_distractors) nats, so J_f < 0 at the clean end does not mean a distractor "
            "is preferred. The quantity attribution decomposes is path_effect."
        ),
    }


def alignment_report(index: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How much of the corpus is exactly length-matched, for the sensitivity check."""
    deltas = [row["alignment"]["prompt_len_delta"] for row in index]
    exact = [row for row in index if row["alignment"]["exact_length_match"]]
    return {
        "policy": index[0]["alignment"]["policy"] if index else None,
        "n_vectors": len(index),
        "n_exact_length": len(exact),
        "fraction_exact_length": round(len(exact) / len(index), 4) if index else None,
        "prompt_len_delta": {
            "mean_abs": round(float(np.mean(np.abs(deltas))), 4) if deltas else None,
            "max_abs": int(np.max(np.abs(deltas))) if deltas else None,
        },
        "exact_length_item_ids": [row["item_id"] for row in exact],
        "caveat": (
            "Where lengths differ the question span is offset by prompt_len_delta. The "
            "exact-length subset is analysed separately as a registered sensitivity check."
        ),
    }


def class_counts(index: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in index:
        counts[row["item_class"]] = counts.get(row["item_class"], 0) + 1
    return {
        "by_class": dict(sorted(counts.items())),
        "classes_never_pooled": list(ControlClass.ALL),
        "target_class": TARGET_CLASS,
    }


def discovery_summary(
    index: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    graph: G0Graph,
    *,
    integration_steps: int,
    alignment_policy: str,
) -> dict[str, Any]:
    return {
        "stage_version": DISCOVERY_STAGE_VERSION,
        "attribution_version": ATTRIBUTION_VERSION,
        "graph": graph.describe(),
        "integration_steps": integration_steps,
        "alignment_policy": alignment_policy,
        "n_vectors": len(index),
        "n_skipped": len(skipped),
        "skipped": list(skipped)[:50],
        "counts": class_counts(index),
        "completeness": completeness_report(index),
        "objective_health": objective_health(index),
        "alignment": alignment_report(index),
        "not_a_claim": (
            "These are attribution vectors: candidate rankings only. No causal statement "
            "follows from them until Gate G4 and exact intervention."
        ),
    }
