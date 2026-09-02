from __future__ import annotations

from dataclasses import dataclass

from .interfaces import EdgeRef


@dataclass(frozen=True)
class RankedEdge:
    edge: EdgeRef
    score: float


@dataclass
class FactCircuit:
    fact_id: str
    edges: list[RankedEdge]
    discovery_surface_ids: list[str]
    metadata: dict


def top_k_edges(scores: dict[EdgeRef, float], k: int) -> list[RankedEdge]:
    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [RankedEdge(edge=edge, score=float(score)) for edge, score in ranked[:k]]


def weighted_edge_vector(circuit: FactCircuit, vocabulary: list[str]) -> list[float]:
    by_id = {item.edge.id: item.score for item in circuit.edges}
    return [by_id.get(edge_id, 0.0) for edge_id in vocabulary]


def cumulative_abs_fraction(edges: list[RankedEdge], k: int) -> float:
    total = sum(abs(item.score) for item in edges)
    if total == 0.0:
        return 0.0
    return sum(abs(item.score) for item in edges[:k]) / total


def smallest_prefix_for_fraction(edges: list[RankedEdge], target_fraction: float) -> list[RankedEdge]:
    if not 0 < target_fraction <= 1:
        raise ValueError("target_fraction must lie in (0, 1]")
    for k in range(1, len(edges) + 1):
        if cumulative_abs_fraction(edges, k) >= target_fraction:
            return edges[:k]
    return list(edges)
