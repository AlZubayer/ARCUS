"""Gate G1: instrumentation parity.

Proves that observing the model does not change it, before any causal experiment runs.

Two different tolerances apply here, and the difference is deliberate:

* Capture hooks clone and return ``None``, so hook-only execution must reproduce the
  baseline **exactly**. Any nonzero drift is a bug in the hook map, not noise.
* The backend-vs-plain-HF scoring check compares two mathematically equivalent float32
  reduction orders (a batched log_softmax over gathered rows versus one row at a time), so
  it carries a small documented tolerance. Observed drift is reported next to it.
"""

from __future__ import annotations

from typing import Any

import torch

from ..backend.base import Component, HookPoint
from ..backend.hf import HFBackend

PARITY_VERSION = "instrumentation_parity_v1"

#: Float32 reduction-order slack for the backend-vs-reference scoring comparison.
#: Five orders of magnitude below the factual margins any result is interpreted from,
#: and ~25x above the largest drift observed on the pilot model.
SCORING_PARITY_TOLERANCE = 1e-5

ALL_COMPONENTS = (
    Component.RESID_PRE,
    Component.RESID_POST,
    Component.ATTN_OUT,
    Component.MLP_OUT,
)


def _uninstrumented_scores(backend: HFBackend, prompt: str, answer: str) -> dict[str, Any]:
    """Score one answer with plain Hugging Face calls, bypassing the backend's scorer.

    This is the independent reference the backend's teacher-forced path is checked against.
    """
    tok = backend.tokenizer
    prompt_ids = tok(prompt, add_special_tokens=backend.add_special_tokens).input_ids
    full_ids = tok(prompt + answer, add_special_tokens=backend.add_special_tokens).input_ids
    ids = torch.tensor([full_ids], device=backend.device)

    with torch.no_grad():
        logits = backend.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]

    per_token: list[float] = []
    for position in range(len(prompt_ids), len(full_ids)):
        row = torch.log_softmax(logits[position - 1].float(), dim=-1)
        per_token.append(float(row[full_ids[position]].item()))

    total = sum(per_token)
    return {
        "answer": answer,
        "prompt_token_ids": prompt_ids,
        "answer_token_ids": full_ids[len(prompt_ids) :],
        "answer_token_positions": list(range(len(prompt_ids), len(full_ids))),
        "per_token_logprobs": per_token,
        "sum_logprob": total,
        "mean_logprob": total / len(per_token),
    }


def check_scoring_parity(
    backend: HFBackend,
    prompt: str,
    answers: list[str],
    *,
    tolerance: float = SCORING_PARITY_TOLERANCE,
) -> dict[str, Any]:
    """Backend teacher-forced scoring vs an independent plain-HF computation."""
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for answer in answers:
        reference = _uninstrumented_scores(backend, prompt, answer)
        got = backend.score_answers(prompt, [answer])[0]
        diff = abs(got.mean_logprob - reference["mean_logprob"])
        worst = max(worst, diff)
        rows.append(
            {
                "answer": answer,
                "n_answer_tokens": got.n_answer_tokens,
                "answer_token_ids": list(got.answer_token_ids),
                "answer_token_positions": list(got.answer_token_positions),
                "backend_mean_logprob": got.mean_logprob,
                "reference_mean_logprob": reference["mean_logprob"],
                "abs_diff": diff,
                "indices_match": list(got.answer_token_positions)
                == reference["answer_token_positions"],
                "token_ids_match": list(got.answer_token_ids) == reference["answer_token_ids"],
                "per_token_logprobs": [round(x, 6) for x in got.per_token_logprobs],
            }
        )

    return {
        "check": "backend_scoring_matches_uninstrumented_hf",
        "max_abs_mean_logprob_diff": worst,
        "tolerance": tolerance,
        "rationale": (
            "Batched log_softmax over gathered rows and per-row log_softmax are the same "
            "computation in different float32 reduction orders, so this comparison carries "
            "a documented tolerance. Answer token ids and positions must match exactly."
        ),
        "passed": worst <= tolerance
        and all(r["indices_match"] and r["token_ids_match"] for r in rows),
        "rows": rows,
    }


def check_capture_parity(backend: HFBackend, prompt: str, answer: str) -> dict[str, Any]:
    """Hook-only execution must reproduce the baseline exactly."""
    example = backend.tokenize(prompt, answer)
    baseline = backend._score_batch([example])[0]

    hook_points = backend.available_hook_points(list(ALL_COMPONENTS))
    sink: dict[HookPoint, torch.Tensor] = {}
    specs = [(hp, backend.hook_map.capture_hook(hp, sink)) for hp in hook_points]
    instrumented = backend._score_batch([example], hook_specs=specs)[0]

    per_token = max(
        abs(a - b)
        for a, b in zip(baseline.per_token_logprobs, instrumented.per_token_logprobs)
    )
    mean_diff = abs(baseline.mean_logprob - instrumented.mean_logprob)

    return {
        "check": "capture_only_hooks_are_numerically_neutral",
        "n_hook_points": len(hook_points),
        "hook_points_fired": len(sink),
        "baseline_mean_logprob": baseline.mean_logprob,
        "instrumented_mean_logprob": instrumented.mean_logprob,
        "max_abs_per_token_diff": per_token,
        "abs_mean_logprob_diff": mean_diff,
        "tolerance": 0.0,
        "rationale": "Capture clones and returns None, so any nonzero drift is a bug.",
        "passed": per_token == 0.0 and mean_diff == 0.0 and len(sink) == len(hook_points),
    }


def check_hook_map_self_consistency(backend: HFBackend, prompt: str) -> dict[str, Any]:
    """resid_post(L) must be the same tensor as resid_pre(L+1)."""
    n_layers = backend.metadata().n_layers
    points = [HookPoint(layer=layer, component=Component.RESID_PRE) for layer in range(n_layers)]
    points += [HookPoint(layer=layer, component=Component.RESID_POST) for layer in range(n_layers)]
    captured = backend.capture(prompt, points)

    worst = 0.0
    mismatches: list[int] = []
    for layer in range(n_layers - 1):
        post = captured[HookPoint(layer=layer, component=Component.RESID_POST)]
        nxt = captured[HookPoint(layer=layer + 1, component=Component.RESID_PRE)]
        diff = float((post - nxt).abs().max().item())
        worst = max(worst, diff)
        if diff != 0.0:
            mismatches.append(layer)

    return {
        "check": "resid_post_L_equals_resid_pre_L_plus_1",
        "n_layers": n_layers,
        "max_abs_diff": worst,
        "mismatched_layers": mismatches,
        "tolerance": 0.0,
        "passed": worst == 0.0,
    }


def check_shapes(backend: HFBackend, prompt: str) -> dict[str, Any]:
    """Captured tensors must match the hook map's declared shape template."""
    meta = backend.metadata()
    prompt_len = len(
        backend.tokenizer(prompt, add_special_tokens=backend.add_special_tokens).input_ids
    )
    points = [HookPoint(layer=0, component=c) for c in ALL_COMPONENTS]
    captured = backend.capture(prompt, points)

    rows = []
    ok = True
    for hp, tensor in captured.items():
        expected = (1, prompt_len, meta.d_model)
        matches = tuple(tensor.shape) == expected
        ok = ok and matches
        rows.append(
            {
                "hook_point": hp.id,
                "shape": list(tensor.shape),
                "expected": list(expected),
                "matches": matches,
                "declared": backend.hook_map.describe(hp)["shape_template"],
            }
        )
    return {"check": "captured_shapes_match_hook_map", "rows": rows, "passed": ok}


def run_parity(
    backend: HFBackend,
    *,
    prompt_question: str,
    correct_answer: str,
    wrong_answer: str,
    multi_token_answer: str,
) -> dict[str, Any]:
    """Run every Gate G1 check and return one auditable report."""
    prompt = backend.build_prompt(prompt_question)
    meta = backend.metadata()

    checks = [
        check_scoring_parity(backend, prompt, [correct_answer, wrong_answer, multi_token_answer]),
        check_capture_parity(backend, prompt, correct_answer),
        check_hook_map_self_consistency(backend, prompt),
        check_shapes(backend, prompt),
    ]

    # The scientific sanity case from 07_ACCEPTANCE_TESTS.md Gate G1: a correct answer
    # should outscore a matched wrong one, with the raw per-token values on record.
    scores = backend.score_answers(prompt, [correct_answer, wrong_answer])
    sanity = {
        "check": "correct_answer_outscores_matched_wrong_answer",
        "question": prompt_question,
        "correct": scores[0].to_dict(),
        "wrong": scores[1].to_dict(),
        "margin_mean_logprob": scores[0].mean_logprob - scores[1].mean_logprob,
        "passed": scores[0].mean_logprob > scores[1].mean_logprob,
        "note": "Diagnostic only; A0 uses the full distractor-pool margin.",
    }
    checks.append(sanity)

    return {
        "parity_version": PARITY_VERSION,
        "model": meta.to_dict(),
        "hook_map": backend.describe_hook_map(),
        "prompt": {
            "question": prompt_question,
            "rendered": prompt,
            "sha256": backend.tokenize(prompt, correct_answer).prompt_text_sha256,
            "token_ids": list(backend.tokenize(prompt, correct_answer).prompt_token_ids),
            "n_prompt_tokens": backend.tokenize(prompt, correct_answer).prompt_len,
        },
        "checks": checks,
        "gate_g1_passed": all(c["passed"] for c in checks),
    }
