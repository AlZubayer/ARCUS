from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


@dataclass(frozen=True)
class SinkCodeSource:
    """Provenance for the two sink codebases supplied by the research team."""

    name: str
    repository: str
    branch: str
    role: str


WITHOUT_PLUMBING_SOURCE = SinkCodeSource(
    name="A Sink Without the Plumbing",
    repository="https://github.com/AlZubayer/MechanisticAccountofSinks.git",
    branch="sink-inheritance-foundation",
    role="sink inheritance, route probes, and causal-function controls",
)

DIFFERENT_PLUMBING_SOURCE = SinkCodeSource(
    name="Same Sink, Different Plumbing",
    repository="https://github.com/AlZubayer/MechanisticAccountofSinks.git",
    branch="main",
    role="GPT-2 anchor/route decomposition and reproduction intervention battery",
)

REFERENCE_SOURCES = (WITHOUT_PLUMBING_SOURCE, DIFFERENT_PLUMBING_SOURCE)


class LegacyGPT2Intervention(StrEnum):
    """Named GPT-2 interventions from the reproduced sink battery.

    These names are recorded for provenance. They are not assumed to have direct
    analogues in RoPE models such as Llama. Any cross-architecture analogue must
    be separately defined and causally validated.
    """

    NULLIFY_QUERY_BIAS = "nullify_query_bias"
    REMOVE_FIRST_POSITION_ENCODING = "remove_first_position_encoding"
    SWAP_EFFECTIVE_POSITIONAL_ENCODING = "swap_effective_positional_encoding"
    SWAP_POSITION_EMBEDDING_CONTROL = "swap_position_embedding_control"
    ZERO_TOKEN0_EMBEDDING_CONTROL = "zero_token0_embedding_control"
    REMOVE_FIRST_LAYER_MLP = "remove_first_layer_mlp"
    REMOVE_POSITIONAL_ENCODING = "remove_positional_encoding"
    ZERO_MASSIVE_WK_COLUMNS = "zero_massive_wk_columns"
    ZERO_RANDOM_WK_COLUMNS_CONTROL = "zero_random_wk_columns_control"
    DELETE_ANCHOR_ATTENTION = "delete_anchor_attention"
    RELOCATE_ANCHOR_ATTENTION = "relocate_anchor_attention"


def position0_sink_strength(
    attention: np.ndarray,
    *,
    query_start_fraction: float = 0.5,
) -> float:
    """Mean attention received by key position 0 from late queries.

    The last two dimensions must be ``[..., query_position, key_position]``.
    This mirrors the registered sink-strength convention used by the supplied
    GPT-2 work: position-0 attention from second-half queries. Layer/head
    aggregation is deliberately left to the caller so every reported aggregate
    keeps its layer/head scope explicit.
    """

    if not 0.0 <= query_start_fraction < 1.0:
        raise ValueError("query_start_fraction must lie in [0, 1)")

    arr = np.asarray(attention, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError("attention must have at least query and key dimensions")
    if arr.shape[-2] == 0 or arr.shape[-1] == 0:
        raise ValueError("attention query/key dimensions must be non-empty")

    start = int(np.floor(arr.shape[-2] * query_start_fraction))
    start = min(start, arr.shape[-2] - 1)
    return float(np.nanmean(arr[..., start:, 0]))


def gpt2_anchor_query_routes(
    content_query: np.ndarray,
    query_bias: np.ndarray,
    anchor_key: np.ndarray,
    *,
    score_scale: float = 1.0,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Decompose the two query-side routes into a shared anchor key.

    For the GPT-2 score terms that survive the target-wise softmax comparison,
    the supplied reproduction distinguishes:

    * route B: ``(x_i W_Q) · k_0`` -- activation/context dependent;
    * route A: ``b_Q · k_0`` -- fixed query-bias route.

    Returns ``(route_b, route_a, route_a_plus_b)``. This is a score-space
    decomposition, not a claim that softmax attention or causal effects add
    linearly.
    """

    q = np.asarray(content_query, dtype=np.float64)
    b = np.asarray(query_bias, dtype=np.float64)
    k = np.asarray(anchor_key, dtype=np.float64)

    if q.shape[-1] != k.shape[-1]:
        raise ValueError("content_query and anchor_key must share the head dimension")
    if b.shape != k.shape:
        raise ValueError("query_bias and anchor_key must have identical shape")

    route_b = np.sum(q * k, axis=-1) * score_scale
    route_a = float(np.dot(b, k) * score_scale)
    combined = route_b + route_a
    return route_b, route_a, combined


def descriptive_bias_route_fraction(
    route_b: np.ndarray,
    route_a: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Absolute score-share diagnostic for route A.

    This is intentionally labelled *descriptive*. Attention is normalized by a
    softmax and intervention effects are not additive, so this quantity must not
    be reported as a causal mediation fraction.
    """

    b = np.asarray(route_b, dtype=np.float64)
    numerator = np.abs(route_a)
    denominator = numerator + np.abs(b) + eps
    return numerator / denominator
