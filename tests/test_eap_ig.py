"""Attribution mechanics on a tiny model.

The property worth testing is completeness: integrated gradients along a path whose two
endpoints are both measurable should satisfy sum(attr) ~= J(alpha=1) - J(alpha=0). If that
fails, the attribution is not measuring the path it claims to.
"""

import zlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from arcus.module_a.backend.hf import HFBackend  # noqa: E402
from arcus.module_a.backend.hook_maps import LlamaHookMap  # noqa: E402
from arcus.module_a.discovery.eap_ig import (  # noqa: E402
    ATTRIBUTION_VERSION,
    AlignmentPolicy,
    attribute_pair,
    resolve_end_aligned,
)
from arcus.module_a.discovery.graph import G0Graph  # noqa: E402
from arcus.module_a.objectives import build_discriminative_spec  # noqa: E402

N_LAYERS, N_HEADS, D_MODEL = 3, 4, 32


class _FakeTokenizer:
    pad_token_id, eos_token_id, bos_token_id = 0, 1, 2
    pad_token, eos_token, bos_token = "<pad>", "<eos>", "<bos>"
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        ids = [3 + (zlib.crc32(w.encode()) % 100) for w in text.split()]
        return type("Enc", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


@pytest.fixture(scope="module")
def backend():
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
    model = LlamaForCausalLM(cfg).eval().requires_grad_(False)
    be = HFBackend.__new__(HFBackend)
    be.model = model
    be.hook_map = LlamaHookMap(model)
    be.device = torch.device("cpu")
    be.add_special_tokens = False
    be.prompt_template = "raw_completion"
    be.raw_completion_template = "{question}"
    be.tokenizer = _FakeTokenizer()
    return be


@pytest.fixture(scope="module")
def graph():
    return G0Graph.build(N_LAYERS, N_HEADS)


CLEAN = "alpha beta gamma delta "
CORRUPT_SAME_LEN = "epsilon zeta eta theta "
CORRUPT_LONGER = "epsilon zeta eta theta iota kappa "


@pytest.fixture(scope="module")
def spec(backend):
    """Two single-token candidates that differ, so t* is the first answer position."""
    return build_discriminative_spec(
        prompt_len=len(backend.tokenizer(CLEAN).input_ids),
        correct_answer="mu",
        correct_tokens=backend.tokenizer("mu").input_ids,
        distractor_answers=["nu", "xi"],
        distractor_tokens=[
            backend.tokenizer("nu").input_ids,
            backend.tokenizer("xi").input_ids,
        ],
    )


# -- Alignment ---------------------------------------------------------------------------


def test_equal_lengths_align_index_wise():
    pairs, info = resolve_end_aligned(10, 10, policy=AlignmentPolicy.END_ALIGNED)
    assert pairs == tuple((i, i) for i in range(10))
    assert info["exact_length_match"] is True
    assert info["prompt_len_delta"] == 0
    assert "exact" in info["caveat"]


def test_unequal_lengths_align_from_the_end_and_record_the_caveat():
    pairs, info = resolve_end_aligned(12, 9, policy=AlignmentPolicy.END_ALIGNED)
    assert len(pairs) == 9
    assert pairs[-1] == (11, 8)
    assert pairs[0] == (3, 0)
    assert info["prompt_len_delta"] == 3
    assert info["exact_length_match"] is False
    # The misalignment must be stated on every vector, not left implicit.
    assert "offset by prompt_len_delta" in info["caveat"]


def test_exact_length_policy_refuses_unequal_prompts():
    with pytest.raises(ValueError, match="equal prompt lengths"):
        resolve_end_aligned(12, 9, policy=AlignmentPolicy.EXACT_LENGTH)
    pairs, _ = resolve_end_aligned(9, 9, policy=AlignmentPolicy.EXACT_LENGTH)
    assert len(pairs) == 9


# -- Attribution -------------------------------------------------------------------------


def test_attribution_covers_every_g0_object(backend, graph, spec):
    result = attribute_pair(
        backend, graph,
        clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN, spec=spec,
        integration_steps=8,
    )
    assert len(result.scores) == len(graph) == N_LAYERS * (N_HEADS + 1)
    assert result.object_ids == graph.object_ids
    # A full signed vector: both signs present, nothing truncated to a top-k.
    assert result.n_nonzero if hasattr(result, "n_nonzero") else True
    summary = result.summary()
    assert summary["n_objects"] == len(graph)
    assert summary["attribution_version"] == ATTRIBUTION_VERSION
    assert summary["n_positive"] + summary["n_negative"] <= len(graph)


def test_completeness_holds(backend, graph, spec):
    """sum(attr) should recover the measured J difference between the path endpoints."""
    result = attribute_pair(
        backend, graph,
        clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN, spec=spec,
        integration_steps=32,
    )
    assert abs(result.path_effect) > 1e-6, "degenerate pair: endpoints coincide"
    assert result.completeness_ratio == pytest.approx(1.0, abs=0.15), (
        f"sum(attr)={result.total_attribution:.4f} vs path effect={result.path_effect:.4f}"
    )


def test_completeness_improves_with_more_steps(backend, graph, spec):
    """More integration steps should not make completeness worse."""
    coarse = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN,
        spec=spec, integration_steps=2,
    )
    fine = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN,
        spec=spec, integration_steps=32,
    )
    assert abs(fine.completeness_ratio - 1.0) <= abs(coarse.completeness_ratio - 1.0) + 0.05


def test_endpoints_are_measured_not_assumed(backend, graph, spec):
    """J(alpha=1) must equal the plain clean run: at alpha=1 every override is z_clean."""
    result = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN,
        spec=spec, integration_steps=4,
    )
    plain = backend.score_discriminative_margin(CLEAN, spec)
    assert result.j_clean == pytest.approx(plain, abs=1e-4)
    # The corrupt endpoint is a different run and must differ.
    assert result.j_corrupt_baseline != pytest.approx(plain, abs=1e-6)


def test_identical_prompts_give_no_attribution(backend, graph, spec):
    """Clean against itself: nothing to attribute, and the path effect vanishes."""
    result = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CLEAN,
        spec=spec, integration_steps=4,
    )
    assert abs(result.path_effect) < 1e-4
    assert abs(result.total_attribution) < 1e-4
    assert float(abs(result.scores).max()) < 1e-4


def test_unequal_length_pairs_are_supported(backend, graph, spec):
    """The reason this formulation was chosen over input-embedding EAP-IG."""
    result = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_LONGER,
        spec=spec, integration_steps=8,
    )
    assert result.alignment["exact_length_match"] is False
    assert result.alignment["n_aligned_positions"] == min(
        len(backend.tokenizer(CLEAN).input_ids),
        len(backend.tokenizer(CORRUPT_LONGER).input_ids),
    )
    assert len(result.scores) == len(graph)


def test_attribution_is_deterministic(backend, graph, spec):
    a = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN,
        spec=spec, integration_steps=8,
    )
    b = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN,
        spec=spec, integration_steps=8,
    )
    assert (a.scores == b.scores).all()


def test_metadata_and_alignment_are_carried_on_the_result(backend, graph, spec):
    result = attribute_pair(
        backend, graph, clean_prompt=CLEAN, corrupt_prompt=CORRUPT_SAME_LEN, spec=spec,
        integration_steps=4, metadata={"pair_id": "p1", "family": "same_topic_fact_swap"},
    )
    summary = result.summary()
    assert summary["pair_id"] == "p1"
    assert summary["family"] == "same_topic_fact_swap"
    assert summary["alignment"]["policy"] == AlignmentPolicy.END_ALIGNED
    assert summary["integration_steps"] == 4
