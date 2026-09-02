"""Intervention-mechanics gates on a tiny randomly-initialised Llama.

These validate code, not scientific claims about LLM factual storage
(07_ACCEPTANCE_TESTS.md, "Synthetic/tiny-model test suite"). The model is built from a
config, so the suite runs offline in seconds and covers Fixtures 1 and 2.
"""

import zlib

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
    TokenAlignment,
    TokenAlignmentError,
    TokenizedExample,
    TokenPolicy,
    resolve_alignment,
)
from arcus.module_a.backend.hf import HFBackend  # noqa: E402
from arcus.module_a.backend.hook_maps import LlamaHookMap  # noqa: E402

N_LAYERS = 4
D_MODEL = 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=D_MODEL,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


@pytest.fixture(scope="module")
def hook_map(model):
    return LlamaHookMap(model)


@pytest.fixture(scope="module")
def backend(model, hook_map):
    """A backend shell wired to the tiny model, bypassing the Hub."""
    be = HFBackend.__new__(HFBackend)
    be.model = model
    be.hook_map = hook_map
    be.device = torch.device("cpu")
    be.add_special_tokens = False
    be.prompt_template = "raw_completion"
    be.raw_completion_template = "{question}"
    be.tokenizer = _FakeTokenizer()
    return be


class _FakeTokenizer:
    """Deterministic whitespace tokenizer: each word maps to a stable id.

    Real tokenizer behaviour (BOS handling, boundary merges) is covered against the actual
    Llama tokenizer in test_backend_scoring.py; here the point is patch mechanics.
    """

    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    pad_token = "<pad>"
    eos_token = "<eos>"
    bos_token = "<bos>"
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        ids = [3 + (zlib.crc32(w.encode()) % 100) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def toks(backend, prompt, answer):
    return backend.tokenize(prompt, answer)


PROMPT_A = "alpha beta gamma delta "
PROMPT_B = "epsilon zeta eta theta iota "
ANSWER = "kappa lambda"


ALL_POINTS = [
    HookPoint(layer=layer, component=component)
    for layer in range(N_LAYERS)
    for component in (
        Component.RESID_PRE,
        Component.RESID_POST,
        Component.ATTN_OUT,
        Component.MLP_OUT,
    )
]


# -- Gate C: capture-only must not move the output --------------------------------------


def test_capture_hooks_do_not_change_logits(backend):
    """Capture clones and returns None, so any drift at all is a bug, not noise."""
    example = toks(backend, PROMPT_A, ANSWER)
    baseline = backend._score_batch([example])[0]

    sink = {}
    specs = [(hp, backend.hook_map.capture_hook(hp, sink)) for hp in ALL_POINTS]
    instrumented = backend._score_batch([example], hook_specs=specs)[0]

    assert len(sink) == len(ALL_POINTS)
    assert instrumented.per_token_logprobs == baseline.per_token_logprobs
    assert instrumented.mean_logprob == baseline.mean_logprob


def test_hook_map_self_test_resid_post_equals_next_resid_pre(backend):
    """resid_post(L) and resid_pre(L+1) are the same tensor by construction."""
    captured = backend.capture(PROMPT_A, ALL_POINTS)
    for layer in range(N_LAYERS - 1):
        post = captured[HookPoint(layer=layer, component=Component.RESID_POST)]
        nxt = captured[HookPoint(layer=layer + 1, component=Component.RESID_PRE)]
        assert torch.equal(post, nxt), f"resid_post({layer}) != resid_pre({layer + 1})"


# -- Gates A and B: self-patch must be an exact no-op ------------------------------------


@pytest.mark.parametrize(
    "component",
    [Component.RESID_PRE, Component.RESID_POST, Component.ATTN_OUT, Component.MLP_OUT],
)
@pytest.mark.parametrize("layer", range(N_LAYERS))
def test_self_patch_is_an_exact_no_op(backend, layer, component):
    """Writing an activation back over itself changes nothing, at every layer."""
    hp = HookPoint(layer=layer, component=component)
    example = toks(backend, PROMPT_A, ANSWER)
    baseline = backend._score_batch([example])[0]

    source = backend.capture(PROMPT_A, [hp])
    alignment = resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, example, example)
    spec = PatchSpec(
        hook_points=(hp,), alignment=alignment, direction=PatchDirection.CLEAN_TO_CLEAN
    )
    patched = backend.score_answers_with_patch(PROMPT_A, [ANSWER], spec, source)[0]

    assert patched.mean_logprob == baseline.mean_logprob
    assert patched.per_token_logprobs == baseline.per_token_logprobs


def test_self_patch_no_op_over_all_prompt_positions(backend):
    """The no-op must hold when every prompt position is written, not just the last one."""
    hp = HookPoint(layer=2, component=Component.RESID_PRE)
    example = toks(backend, PROMPT_A, ANSWER)
    baseline = backend._score_batch([example])[0]

    source = backend.capture(PROMPT_A, [hp])
    alignment = resolve_alignment(TokenPolicy.ALL_PROMPT_TOKENS, example, example)
    spec = PatchSpec(
        hook_points=(hp,), alignment=alignment, direction=PatchDirection.CLEAN_TO_CLEAN
    )
    patched = backend.score_answers_with_patch(PROMPT_A, [ANSWER], spec, source)[0]
    assert patched.per_token_logprobs == baseline.per_token_logprobs


# -- A real patch must actually change something, and only what it names -----------------


def test_cross_patch_changes_the_score(backend):
    hp = HookPoint(layer=1, component=Component.RESID_PRE)
    target = toks(backend, PROMPT_A, ANSWER)
    source_ex = toks(backend, PROMPT_B, ANSWER)
    baseline = backend._score_batch([target])[0]

    source = backend.capture(PROMPT_B, [hp])
    alignment = resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, target, source_ex)
    spec = PatchSpec(
        hook_points=(hp,), alignment=alignment, direction=PatchDirection.CLEAN_TO_CORRUPT
    )
    patched = backend.score_answers_with_patch(PROMPT_A, [ANSWER], spec, source)[0]
    assert patched.mean_logprob != baseline.mean_logprob


def test_patch_touches_only_the_registered_positions(backend):
    """Everything upstream, and every unwritten position, must stay bitwise identical."""
    hp = HookPoint(layer=2, component=Component.RESID_PRE)
    target = toks(backend, PROMPT_A, ANSWER)
    source_ex = toks(backend, PROMPT_B, ANSWER)
    patch_pos = target.prompt_len - 1

    before = backend.capture(PROMPT_A, ALL_POINTS)
    source = backend.capture(PROMPT_B, [hp])
    alignment = resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, target, source_ex)

    sink = {}
    specs = [(p, backend.hook_map.capture_hook(p, sink)) for p in ALL_POINTS]
    specs.append((hp, backend.hook_map.patch_hook(hp, source[hp], alignment.pairs)))
    backend._score_batch([toks(backend, PROMPT_A, ANSWER)], hook_specs=specs)

    # Layers strictly before the patch site are untouched.
    for point in ALL_POINTS:
        if point.layer < hp.layer:
            assert torch.equal(sink[point][:, : target.prompt_len], before[point][:, : target.prompt_len]), (
                f"{point.id} changed but sits upstream of {hp.id}"
            )

    # At the patch site itself, only the written column differs.
    observed = sink[hp][0]
    original = before[hp][0]
    for pos in range(target.prompt_len):
        if pos == patch_pos:
            continue
        assert torch.equal(observed[pos], original[pos]), f"position {pos} changed unexpectedly"


def test_patching_does_not_mutate_the_captured_source(backend):
    hp = HookPoint(layer=1, component=Component.RESID_POST)
    source = backend.capture(PROMPT_B, [hp])
    snapshot = source[hp].clone()

    target = toks(backend, PROMPT_A, ANSWER)
    source_ex = toks(backend, PROMPT_B, ANSWER)
    spec = PatchSpec(
        hook_points=(hp,),
        alignment=resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, target, source_ex),
        direction=PatchDirection.CORRUPT_TO_CLEAN,
    )
    backend.score_answers_with_patch(PROMPT_A, [ANSWER], spec, source)
    assert torch.equal(source[hp], snapshot)


# -- Gate D: invalid coordinates must fail loudly ----------------------------------------


@pytest.mark.parametrize(
    "hook_point",
    [
        HookPoint(layer=-1, component=Component.RESID_PRE),
        HookPoint(layer=N_LAYERS, component=Component.RESID_PRE),
        HookPoint(layer=999, component=Component.MLP_OUT),
        HookPoint(layer=0, component=Component.HEAD_OUT),
        HookPoint(layer=0, component=Component.ATTN_PATTERN),
        HookPoint(layer=0, component=Component.RESID_PRE, head=3),
    ],
)
def test_invalid_hook_coordinates_raise(hook_map, hook_point):
    with pytest.raises(InvalidHookPointError):
        hook_map.validate(hook_point)


def test_capture_rejects_invalid_coordinate_before_running(backend):
    with pytest.raises(InvalidHookPointError):
        backend.capture(PROMPT_A, [HookPoint(layer=N_LAYERS, component=Component.RESID_PRE)])


def test_patch_without_captured_source_raises(backend):
    hp = HookPoint(layer=0, component=Component.RESID_PRE)
    target = toks(backend, PROMPT_A, ANSWER)
    spec = PatchSpec(
        hook_points=(hp,),
        alignment=resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, target, target),
        direction=PatchDirection.CLEAN_TO_CORRUPT,
    )
    with pytest.raises(InvalidHookPointError):
        backend.score_answers_with_patch(PROMPT_A, [ANSWER], spec, {})


def test_out_of_range_token_index_raises(backend):
    target = toks(backend, PROMPT_A, ANSWER)
    with pytest.raises(TokenAlignmentError):
        resolve_alignment(
            TokenPolicy.EXPLICIT_INDICES, target, target, pairs=[(target.total_len + 5, 0)]
        )


def test_empty_alignment_raises():
    with pytest.raises(TokenAlignmentError):
        TokenAlignment(policy=TokenPolicy.EXPLICIT_INDICES, pairs=())


def test_patch_spec_requires_hook_points():
    example = TokenizedExample(
        prompt_text="p",
        answer_text="a",
        prompt_token_ids=(1, 2),
        full_token_ids=(1, 2, 3),
        answer_token_ids=(3,),
        answer_token_positions=(2,),
        prompt_text_sha256="x",
    )
    alignment = resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, example, example)
    with pytest.raises(InvalidHookPointError):
        PatchSpec(hook_points=(), alignment=alignment, direction=PatchDirection.CLEAN_TO_CLEAN)


def test_interpolated_replacement_is_refused():
    example = TokenizedExample(
        prompt_text="p",
        answer_text="a",
        prompt_token_ids=(1, 2),
        full_token_ids=(1, 2, 3),
        answer_token_ids=(3,),
        answer_token_positions=(2,),
        prompt_text_sha256="x",
    )
    with pytest.raises(InvalidHookPointError):
        PatchSpec(
            hook_points=(HookPoint(layer=0, component=Component.RESID_PRE),),
            alignment=resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, example, example),
            direction=PatchDirection.CLEAN_TO_CLEAN,
            replacement="interpolate",
        )


# -- Gate F: alignment is explicit -------------------------------------------------------


def test_all_prompt_tokens_refuses_unequal_prompts(backend):
    """Index N must never be patched to index N just because both sequences have one."""
    target = toks(backend, PROMPT_A, ANSWER)
    source = toks(backend, PROMPT_B, ANSWER)
    assert target.prompt_len != source.prompt_len
    with pytest.raises(TokenAlignmentError):
        resolve_alignment(TokenPolicy.ALL_PROMPT_TOKENS, target, source)


def test_last_prompt_token_handles_unequal_lengths(backend):
    target = toks(backend, PROMPT_A, ANSWER)
    source = toks(backend, PROMPT_B, ANSWER)
    alignment = resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, target, source)
    assert alignment.pairs == ((target.prompt_len - 1, source.prompt_len - 1),)
    assert alignment.detail["prompt_len_delta"] == target.prompt_len - source.prompt_len


def test_final_k_aligns_from_the_end_and_clamps(backend):
    target = toks(backend, PROMPT_A, ANSWER)
    source = toks(backend, PROMPT_B, ANSWER)
    alignment = resolve_alignment(TokenPolicy.FINAL_K_PROMPT_TOKENS, target, source, k=3)
    assert alignment.pairs[-1] == (target.prompt_len - 1, source.prompt_len - 1)
    assert len(alignment.pairs) == 3

    wide = resolve_alignment(TokenPolicy.FINAL_K_PROMPT_TOKENS, target, source, k=999)
    assert wide.detail["effective_k"] == min(target.prompt_len, source.prompt_len)


def test_final_k_requires_positive_k(backend):
    target = toks(backend, PROMPT_A, ANSWER)
    with pytest.raises(TokenAlignmentError):
        resolve_alignment(TokenPolicy.FINAL_K_PROMPT_TOKENS, target, target, k=0)


# -- Gate E: direction is recorded -------------------------------------------------------


def test_patch_spec_serializes_direction_and_alignment(backend):
    target = toks(backend, PROMPT_A, ANSWER)
    source = toks(backend, PROMPT_B, ANSWER)
    spec = PatchSpec(
        hook_points=(HookPoint(layer=1, component=Component.RESID_PRE),),
        alignment=resolve_alignment(TokenPolicy.LAST_PROMPT_TOKEN, target, source),
        direction=PatchDirection.CORRUPT_TO_CLEAN,
    )
    payload = spec.to_dict()
    assert payload["direction"] == "corrupt_to_clean"
    assert payload["alignment"]["policy"] == "last_prompt_token"
    assert payload["alignment"]["n_positions"] == 1
    assert payload["hook_points"][0]["id"] == "L1.resid_pre"
    assert payload["replacement"] == "exact"


# -- Teacher-forced scoring mechanics ----------------------------------------------------


def test_multi_token_answer_indexing(backend):
    single = toks(backend, PROMPT_A, "kappa")
    multi = toks(backend, PROMPT_A, "kappa lambda mu")
    assert single.prompt_len == multi.prompt_len
    assert len(single.answer_token_positions) == 1
    assert len(multi.answer_token_positions) == 3
    assert multi.answer_token_positions == tuple(
        range(multi.prompt_len, multi.prompt_len + 3)
    )
    assert multi.full_token_ids[: multi.prompt_len] == multi.prompt_token_ids


def test_mean_logprob_is_the_length_normalized_sum(backend):
    score = backend._score_batch([toks(backend, PROMPT_A, "kappa lambda mu")])[0]
    assert score.n_answer_tokens == 3
    assert score.sum_logprob == pytest.approx(sum(score.per_token_logprobs))
    assert score.mean_logprob == pytest.approx(score.sum_logprob / 3)
    assert all(lp <= 0 for lp in score.per_token_logprobs)


def test_batching_matches_one_at_a_time(backend):
    """Right padding must not change any answer's score."""
    answers = ["kappa", "kappa lambda", "kappa lambda mu nu"]
    examples = [toks(backend, PROMPT_A, a) for a in answers]
    batched = backend._score_batch(examples)
    for example, got in zip(examples, batched):
        alone = backend._score_batch([example])[0]
        assert got.mean_logprob == pytest.approx(alone.mean_logprob, abs=1e-6)
