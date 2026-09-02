import math

from arcus.module_a.metrics import (
    attribution_weighted_sink_participation,
    causal_selectivity,
    necessity,
    route_distinctness,
    sufficiency,
)


def test_causal_metrics():
    assert math.isclose(necessity(10.0, 2.0, 6.0), 0.5)
    assert math.isclose(sufficiency(10.0, 2.0, 6.0), 0.5)
    assert causal_selectivity(4.0, [1.0, 1.0]) > 3.9


def test_route_distinctness():
    assert math.isclose(route_distinctness([0.8, 0.7], [0.2, 0.3]), 0.5)


def test_sink_participation():
    scores = {"a": 3.0, "b": -1.0}
    assert math.isclose(attribution_weighted_sink_participation(scores, {"a"}), 0.75)
