"""head_out semantics and the G0 graph.

The invariant that matters: attn_out is the post-projection sum over heads, head_out is one
head's pre-projection output, and the two must not be conflated
(04_BACKEND_AND_INTERVENTIONS.md section 9). Llama has attention_bias=False, so the
decomposition is exact, not approximate, and can be asserted rather than eyeballed.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from arcus.module_a.backend.base import (  # noqa: E402
    Component,
    HookPoint,
    InvalidHookPointError,
    PatchDirection,
    PatchSpec,
    TokenPolicy,
    resolve_alignment,
)
from arcus.module_a.backend.hook_maps import LlamaHookMap  # noqa: E402
from arcus.module_a.discovery.graph import G0Graph  # noqa: E402

N_LAYERS, N_HEADS, D_MODEL = 4, 4, 32
D_HEAD = D_MODEL // N_HEADS


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS,
        num_key_value_heads=2,
        max_position_embeddings=128,
        attention_bias=False,
    )
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def hook_map(model):
    return LlamaHookMap(model)


@pytest.fixture(scope="module")
def ids():
    return torch.tensor([[5, 9, 14, 21, 33, 42]])


# -- Graph -------------------------------------------------------------------------------


def test_graph_is_the_additive_basis():
    g = G0Graph.build(28, 24)
    assert len(g) == 28 * 24 + 28 == 700
    assert len(g.head_nodes()) == 672
    assert len(g.mlp_nodes()) == 28
    assert len(set(g.object_ids)) == len(g)
    # Ordering is layer-major with mlp_out last in each layer; index order defines the
    # attribution vector layout and must be stable.
    assert g.object_ids[0] == "L0.H0.head_out"
    assert g.object_ids[24] == "L0.mlp_out"
    assert g.object_ids[-1] == "L27.mlp_out"
    assert G0Graph.build(28, 24).object_ids == g.object_ids


def test_graph_documents_why_residual_objects_are_derived():
    described = G0Graph.build(4, 4).describe()
    assert "double-count" in described["residual_objects"]
    assert described["granularity"] == "G0"
    assert "q/k/v/attn_pattern" in described["excluded"]


def test_sum_by_layer_derives_residual_attribution():
    g = G0Graph.build(2, 2)  # 2 layers x (2 heads + 1 mlp) = 6 nodes
    per_layer = g.sum_by_layer([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    assert per_layer == {0: 6.0, 1: 60.0}


# -- head_out semantics ------------------------------------------------------------------


def test_attn_out_equals_sum_of_projected_head_outputs(model, hook_map, ids):
    """The decomposition the G0 node set relies on. Exact because there is no o_proj bias."""
    layer = 2
    points = [
        HookPoint(layer=layer, component=Component.HEAD_OUT, head=h) for h in range(N_HEADS)
    ]
    points.append(HookPoint(layer=layer, component=Component.ATTN_OUT))

    sink = {}
    handles = [hook_map.register(p, hook_map.capture_hook(p, sink)) for p in points]
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    attn_out = sink[HookPoint(layer=layer, component=Component.ATTN_OUT)]
    rebuilt = torch.zeros_like(attn_out)
    for h in range(N_HEADS):
        hp = HookPoint(layer=layer, component=Component.HEAD_OUT, head=h)
        weight = hook_map.head_output_weight(hp)  # [d_model, d_head]
        rebuilt = rebuilt + sink[hp] @ weight.T

    assert model.model.layers[layer].self_attn.o_proj.bias is None
    assert torch.allclose(rebuilt, attn_out, atol=1e-5), (
        f"max diff {(rebuilt - attn_out).abs().max().item():.3e}"
    )


def test_head_out_has_the_head_dimension_not_the_model_dimension(model, hook_map, ids):
    hp = HookPoint(layer=1, component=Component.HEAD_OUT, head=2)
    sink = {}
    handle = hook_map.register(hp, hook_map.capture_hook(hp, sink))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
    assert sink[hp].shape == (1, ids.shape[1], D_HEAD)
    assert hook_map.describe(hp)["shape_template"] == ["batch", "seq", "d_head"]


def test_different_heads_capture_different_slices(model, hook_map, ids):
    points = [HookPoint(layer=1, component=Component.HEAD_OUT, head=h) for h in range(N_HEADS)]
    sink = {}
    handles = [hook_map.register(p, hook_map.capture_hook(p, sink)) for p in points]
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    assert not torch.equal(sink[points[0]], sink[points[1]])


# -- head_out interventions --------------------------------------------------------------


def _capture(model, hook_map, points, ids):
    sink = {}
    handles = [hook_map.register(p, hook_map.capture_hook(p, sink)) for p in points]
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()
    return sink


def test_patching_one_head_leaves_other_heads_bitwise_unchanged(model, hook_map, ids):
    """Ablating one head must preserve the others (design section 9)."""
    target = HookPoint(layer=1, component=Component.HEAD_OUT, head=1)
    others = [
        HookPoint(layer=1, component=Component.HEAD_OUT, head=h)
        for h in range(N_HEADS)
        if h != 1
    ]
    before = _capture(model, hook_map, [target, *others], ids)

    source = torch.zeros_like(before[target])
    pairs = tuple((i, i) for i in range(ids.shape[1]))

    after = {}
    handles = [hook_map.register(p, hook_map.capture_hook(p, after)) for p in others]
    handles.append(hook_map.register(target, hook_map.patch_hook(target, source, pairs)))
    handles.append(hook_map.register(target, hook_map.capture_hook(target, after)))
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    for other in others:
        assert torch.equal(after[other], before[other]), f"{other.id} was disturbed"


def test_head_self_patch_is_an_exact_no_op(model, hook_map, ids):
    hp = HookPoint(layer=2, component=Component.HEAD_OUT, head=3)
    before = _capture(model, hook_map, [hp], ids)[hp]
    pairs = tuple((i, i) for i in range(ids.shape[1]))

    with torch.no_grad():
        baseline = model(input_ids=ids).logits
    handle = hook_map.register(hp, hook_map.patch_hook(hp, before, pairs))
    try:
        with torch.no_grad():
            patched = model(input_ids=ids).logits
    finally:
        handle.remove()
    assert torch.equal(baseline, patched)


def test_zeroing_a_head_changes_the_output(model, hook_map, ids):
    hp = HookPoint(layer=1, component=Component.HEAD_OUT, head=0)
    captured = _capture(model, hook_map, [hp], ids)[hp]
    pairs = tuple((i, i) for i in range(ids.shape[1]))

    with torch.no_grad():
        baseline = model(input_ids=ids).logits
    handle = hook_map.register(hp, hook_map.patch_hook(hp, torch.zeros_like(captured), pairs))
    try:
        with torch.no_grad():
            ablated = model(input_ids=ids).logits
    finally:
        handle.remove()
    assert not torch.equal(baseline, ablated)


def test_head_capture_is_numerically_neutral(model, hook_map, ids):
    points = [
        HookPoint(layer=layer, component=Component.HEAD_OUT, head=h)
        for layer in range(N_LAYERS)
        for h in range(N_HEADS)
    ]
    with torch.no_grad():
        baseline = model(input_ids=ids).logits
    sink = {}
    handles = [hook_map.register(p, hook_map.capture_hook(p, sink)) for p in points]
    try:
        with torch.no_grad():
            instrumented = model(input_ids=ids).logits
    finally:
        for h in handles:
            h.remove()
    assert len(sink) == len(points)
    assert torch.equal(baseline, instrumented)


# -- Coordinate validation ---------------------------------------------------------------


def test_head_out_requires_a_head_index(hook_map):
    with pytest.raises(InvalidHookPointError, match="requires a head index"):
        hook_map.validate(HookPoint(layer=0, component=Component.HEAD_OUT))


@pytest.mark.parametrize("head", [-1, N_HEADS, 999])
def test_out_of_range_head_raises(hook_map, head):
    with pytest.raises(InvalidHookPointError, match="outside"):
        hook_map.validate(HookPoint(layer=0, component=Component.HEAD_OUT, head=head))


def test_head_index_on_headless_component_still_raises(hook_map):
    with pytest.raises(InvalidHookPointError, match="no head axis"):
        hook_map.validate(HookPoint(layer=0, component=Component.MLP_OUT, head=1))


def test_all_hook_points_expands_heads(hook_map):
    points = hook_map.all_hook_points([Component.HEAD_OUT, Component.MLP_OUT])
    assert len(points) == N_LAYERS * (N_HEADS + 1)
    heads = [p for p in points if p.component is Component.HEAD_OUT]
    assert len(heads) == N_LAYERS * N_HEADS
    assert all(p.head is not None for p in heads)


def test_patch_spec_over_head_nodes_serializes_head_ids():
    from arcus.module_a.backend.base import TokenizedExample

    example = TokenizedExample(
        prompt_text="p",
        answer_text="a",
        prompt_token_ids=(1, 2, 3),
        full_token_ids=(1, 2, 3, 4),
        answer_token_ids=(4,),
        answer_token_positions=(3,),
        prompt_text_sha256="x",
    )
    spec = PatchSpec(
        hook_points=(
            HookPoint(layer=3, component=Component.HEAD_OUT, head=7),
            HookPoint(layer=3, component=Component.MLP_OUT),
        ),
        alignment=resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, example, example),
        direction=PatchDirection.CORRUPT_TO_CLEAN,
    )
    ids_out = [h["id"] for h in spec.to_dict()["hook_points"]]
    assert ids_out == ["L3.H7.head_out", "L3.mlp_out"]
