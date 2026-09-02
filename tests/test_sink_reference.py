import numpy as np

from arcus.module_a.sink_reference import (
    descriptive_bias_route_fraction,
    gpt2_anchor_query_routes,
    position0_sink_strength,
)


def test_position0_sink_strength_uses_late_queries() -> None:
    attn = np.zeros((4, 4), dtype=float)
    attn[0, 0] = 1.0
    attn[1, 0] = 1.0
    attn[2, 0] = 0.25
    attn[3, 0] = 0.75

    assert np.isclose(position0_sink_strength(attn), 0.5)


def test_gpt2_anchor_query_route_decomposition() -> None:
    content_query = np.array([[1.0, 2.0], [3.0, -1.0]])
    query_bias = np.array([0.5, 1.0])
    anchor_key = np.array([2.0, 1.0])

    route_b, route_a, combined = gpt2_anchor_query_routes(
        content_query,
        query_bias,
        anchor_key,
    )

    np.testing.assert_allclose(route_b, np.array([4.0, 5.0]))
    assert np.isclose(route_a, 2.0)
    np.testing.assert_allclose(combined, np.array([6.0, 7.0]))


def test_bias_route_fraction_is_descriptive_score_share() -> None:
    route_b = np.array([2.0, -6.0])
    fraction = descriptive_bias_route_fraction(route_b, route_a=2.0)
    np.testing.assert_allclose(fraction, np.array([0.5, 0.25]), rtol=1e-8)
