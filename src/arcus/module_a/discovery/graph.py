"""The G0 computational graph.

G0 is the coarse, component-level decomposition the design pack asks for first
(01_SYSTEM_DESIGN.md section 5). Its node set is the *additive* basis of the residual
stream:

    final_resid = embed + sum_L sum_H W_O^(L,H) head_out(L,H) + sum_L mlp_out(L)

For Llama-3.2-3B that is 28 x 24 head outputs plus 28 MLP outputs = 700 objects.

Residual-stream objects are deliberately NOT separate nodes. Under this decomposition the
attribution of ``resid_pre(L)`` is exactly the sum of the attributions of every head and
MLP upstream of layer L, so adding it as its own node would double-count the same
contribution. It is derived from the head/MLP attributions instead.

Q/K/V and attention-pattern edges are out of scope here on purpose: the question G0
answers is whether a stable component-level route exists at all, before any finer
decomposition is worth doing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from ..backend.base import Component, HookPoint

GRAPH_VERSION = "g0_head_mlp_v1"


@dataclass(frozen=True)
class G0Graph:
    """A fixed, ordered node set. The index order defines the attribution vector layout."""

    n_layers: int
    n_heads: int
    nodes: tuple[HookPoint, ...]
    version: str = GRAPH_VERSION

    @classmethod
    def build(cls, n_layers: int, n_heads: int) -> "G0Graph":
        nodes: list[HookPoint] = []
        for layer in range(n_layers):
            for head in range(n_heads):
                nodes.append(
                    HookPoint(layer=layer, component=Component.HEAD_OUT, head=head)
                )
            nodes.append(HookPoint(layer=layer, component=Component.MLP_OUT))
        return cls(n_layers=n_layers, n_heads=n_heads, nodes=tuple(nodes))

    @classmethod
    def from_backend(cls, backend: Any) -> "G0Graph":
        meta = backend.metadata()
        return cls.build(meta.n_layers, meta.n_heads)

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[HookPoint]:
        return iter(self.nodes)

    @property
    def object_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    def index_of(self, node: HookPoint) -> int:
        return self.nodes.index(node)

    @property
    def index(self) -> dict[str, int]:
        return {node.id: i for i, node in enumerate(self.nodes)}

    def head_nodes(self) -> tuple[HookPoint, ...]:
        return tuple(n for n in self.nodes if n.component is Component.HEAD_OUT)

    def mlp_nodes(self) -> tuple[HookPoint, ...]:
        return tuple(n for n in self.nodes if n.component is Component.MLP_OUT)

    def nodes_in_layer(self, layer: int) -> tuple[HookPoint, ...]:
        return tuple(n for n in self.nodes if n.layer == layer)

    def layer_of(self, index: int) -> int:
        return self.nodes[index].layer

    def describe(self) -> dict[str, Any]:
        return {
            "graph_version": self.version,
            "granularity": "G0",
            "n_nodes": len(self.nodes),
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_head_nodes": len(self.head_nodes()),
            "n_mlp_nodes": len(self.mlp_nodes()),
            "node_order": "layer-major; within a layer, heads 0..H-1 then mlp_out",
            "components": ["head_out", "mlp_out"],
            "residual_objects": (
                "Derived, not separate. resid_pre(L) attribution equals the sum of the "
                "attributions of every head and MLP upstream of L under this additive "
                "decomposition, so a separate node would double-count."
            ),
            "excluded": {
                "q/k/v/attn_pattern": (
                    "Out of scope for G0. Finer decomposition is only worth doing once a "
                    "stable component-level route is established."
                )
            },
        }

    def sum_by_layer(self, vector: Sequence[float]) -> dict[int, float]:
        """Aggregate a node vector to per-layer totals.

        This is how residual-stream attribution is *derived* rather than measured
        separately.
        """
        out: dict[int, float] = {layer: 0.0 for layer in range(self.n_layers)}
        for node, value in zip(self.nodes, vector):
            out[node.layer] += float(value)
        return out
