from __future__ import annotations

import numpy as np


def normalized_causal_effect(clean: float, corrupted: float, intervened: float) -> float:
    """Fraction of the clean-vs-corrupted factual effect recovered by an intervention."""
    denom = clean - corrupted
    if np.isclose(denom, 0.0):
        return float("nan")
    return float((intervened - corrupted) / denom)


def necessity(clean: float, corrupted: float, clean_with_circuit_corrupted: float) -> float:
    denom = clean - corrupted
    if np.isclose(denom, 0.0):
        return float("nan")
    return float((clean - clean_with_circuit_corrupted) / denom)


def sufficiency(clean: float, corrupted: float, corrupted_with_circuit_restored: float) -> float:
    return normalized_causal_effect(clean, corrupted, corrupted_with_circuit_restored)


def causal_selectivity(target_effect: float, retain_effects: list[float], eps: float = 1e-6) -> float:
    if not retain_effects:
        return float("nan")
    return abs(target_effect) / (eps + float(np.mean(np.abs(retain_effects))))


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return float("nan")
    return float(np.dot(a, b) / denom)


def route_distinctness(within_fact: list[float], between_fact: list[float]) -> float:
    if not within_fact or not between_fact:
        return float("nan")
    return float(np.mean(within_fact) - np.mean(between_fact))


def attribution_weighted_sink_participation(
    edge_scores: dict[str, float], sink_edge_ids: set[str]
) -> float:
    total = sum(abs(v) for v in edge_scores.values())
    if np.isclose(total, 0.0):
        return float("nan")
    inside = sum(abs(score) for edge, score in edge_scores.items() if edge in sink_edge_ids)
    return float(inside / total)


def causal_sink_mediation(total_circuit_effect: float, effect_after_sink_portion_removed: float) -> float:
    """Descriptive mediation fraction; interpretation requires matched retain controls."""
    if np.isclose(total_circuit_effect, 0.0):
        return float("nan")
    return float((total_circuit_effect - effect_after_sink_portion_removed) / total_circuit_effect)
