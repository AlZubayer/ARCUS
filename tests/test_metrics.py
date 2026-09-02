"""Causal-metric tests: sign conventions, no clipping, and refusal near zero Delta."""

import math

import pytest

from arcus.module_a.metrics import (
    attribution_weighted_sink_participation,
    causal_selectivity,
    cosine_similarity,
    necessity,
    route_distinctness,
    sufficiency,
)
from arcus.module_a.stages.patching import aligned_positions, normalized_effect


def test_causal_metrics():
    assert math.isclose(necessity(10.0, 2.0, 6.0), 0.5)
    assert math.isclose(sufficiency(10.0, 2.0, 6.0), 0.5)
    assert causal_selectivity(4.0, [1.0, 1.0]) > 3.9


def test_route_distinctness():
    assert math.isclose(route_distinctness([0.8, 0.7], [0.2, 0.3]), 0.5)


def test_necessity_endpoints():
    """0 means the intervention explained none of the gap; 1 means all of it."""
    clean, corrupt = 4.0, -1.0
    assert necessity(clean, corrupt, clean) == pytest.approx(0.0)
    assert necessity(clean, corrupt, corrupt) == pytest.approx(1.0)


def test_sufficiency_endpoints():
    clean, corrupt = 4.0, -1.0
    assert sufficiency(clean, corrupt, corrupt) == pytest.approx(0.0)
    assert sufficiency(clean, corrupt, clean) == pytest.approx(1.0)


def test_effects_outside_unit_interval_are_not_clipped():
    """Interventions can overcorrect; clipping would misreport the measurement."""
    clean, corrupt = 4.0, -1.0
    assert necessity(clean, corrupt, -6.0) == pytest.approx(2.0)
    assert sufficiency(clean, corrupt, 9.0) == pytest.approx(2.0)
    # A backfiring intervention yields a negative effect, and that is data.
    assert sufficiency(clean, corrupt, -3.0) == pytest.approx(-0.4)


def test_metrics_are_undefined_when_the_gap_vanishes():
    assert math.isnan(necessity(1.0, 1.0, 0.5))
    assert math.isnan(sufficiency(1.0, 1.0, 0.5))


def test_normalized_effect_refuses_a_near_zero_denominator():
    """Every normalized causal metric divides by Delta_f, so a small Delta is refused."""
    assert normalized_effect(1.0, 4.0, min_abs_delta=0.5) == pytest.approx(0.25)
    assert normalized_effect(1.0, 0.2, min_abs_delta=0.5) is None
    assert normalized_effect(1.0, -0.2, min_abs_delta=0.5) is None
    # Sign of Delta does not matter, only magnitude.
    assert normalized_effect(1.0, -4.0, min_abs_delta=0.5) == pytest.approx(-0.25)


def test_selectivity_reports_nan_without_retain_controls():
    assert math.isnan(causal_selectivity(4.0, []))


def test_cosine_similarity_edge_cases():
    import numpy as np

    assert cosine_similarity(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert math.isnan(cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 0.0])))


def test_attribution_weighted_participation():
    scores = {"a": 2.0, "b": -1.0, "c": 1.0}
    assert attribution_weighted_sink_participation(scores, {"a"}) == pytest.approx(0.5)
    assert attribution_weighted_sink_participation(scores, set()) == pytest.approx(0.0)


def test_positions_align_from_the_prompt_end():
    """Offset 0 is the token the answer is generated from, whatever the prompt length."""
    assert aligned_positions(20, 15, 0) == (19, 14)
    assert aligned_positions(20, 15, 3) == (16, 11)
    # Equal-length prompts reduce to index-wise alignment, but only because they match.
    assert aligned_positions(12, 12, 2) == (9, 9)
