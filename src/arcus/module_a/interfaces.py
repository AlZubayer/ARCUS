from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .schema import FactExample


@dataclass(frozen=True, order=True)
class NodeRef:
    layer: int
    kind: str  # residual_pre | residual_mid | residual_post | head | mlp | q | k | v
    index: int | None = None
    position: int | None = None

    @property
    def id(self) -> str:
        bits = [f"L{self.layer}", self.kind]
        if self.index is not None:
            bits.append(str(self.index))
        if self.position is not None:
            bits.append(f"P{self.position}")
        return ":".join(bits)


@dataclass(frozen=True)
class EdgeRef:
    src: NodeRef
    dst: NodeRef

    @property
    def id(self) -> str:
        return f"{self.src.id}->{self.dst.id}"


@dataclass
class ModelRun:
    answer_score: float
    hidden: dict[NodeRef, np.ndarray]
    attention: dict[tuple[int, int], np.ndarray]
    metadata: dict


class ModelBackend(Protocol):
    """Exact model execution and intervention interface.

    Implement this before fast attribution. Every intervention must specify the exact tensor, layer,
    token positions, source run, and patch semantics in metadata.
    """

    def score(self, example: FactExample, distractor_answers: Sequence[str]) -> float: ...

    def run_with_cache(self, example: FactExample, nodes: Sequence[NodeRef]) -> ModelRun: ...

    def patch_nodes(
        self,
        target: FactExample,
        source: FactExample,
        nodes: Sequence[NodeRef],
        distractor_answers: Sequence[str],
    ) -> float: ...

    def project_representation(
        self,
        target: FactExample,
        node: NodeRef,
        basis: np.ndarray,
        distractor_answers: Sequence[str],
    ) -> float: ...


class RouteDiscoverer(Protocol):
    """Attribution is candidate discovery only; exact patching validates returned candidates."""

    def rank_edges(
        self,
        backend: ModelBackend,
        clean: FactExample,
        corrupted: FactExample,
        distractor_answers: Sequence[str],
    ) -> dict[EdgeRef, float]: ...


class SinkMapper(Protocol):
    """Independent A4 mapper; must not be consumed by A1 discovery code."""

    def map(self, backend: ModelBackend, examples: Sequence[FactExample]) -> list[dict]: ...
