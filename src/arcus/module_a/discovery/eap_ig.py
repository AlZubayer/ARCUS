"""EAP-IG-style node attribution over G0 component outputs (``eap_ig_node_v1``).

Definition
----------
Integrated gradients along a straight line in the joint component-output space, run on the
clean prompt with corrupt values substituted at aligned token positions:

    for k in 1..m,  alpha_k = (k - 0.5) / m:
        override every G0 node u at aligned positions with
            z_clean(u) + (1 - alpha_k) * (z_corrupt(u) - z_clean(u))
        forward -> J_f ; backward -> g_k(u) = dJ_f / d(override value)

    attr(u) = (z_clean(u) - z_corrupt(u)) . (1/m) sum_k g_k(u)

alpha = 1 is the clean run. alpha = 0 is the clean run with every component output replaced
by its corrupt value at the aligned positions. Both endpoints are directly measurable, so
the path has a **completeness check**:

    sum_u attr(u)  ~=  J(alpha=1) - J(alpha=0)

which is reported as a Gate G4 criterion. Cost is m forward+backward passes per vector, not
per node, because all 700 nodes are hooked at once and their gradients read together.

Why not textbook EAP-IG
-----------------------
The usual formulation interpolates *input embeddings*, which requires clean and corrupt to
have equal sequence length. Only 63 of 409 accepted pairs do. Interpolating component
outputs at aligned positions instead handles unequal lengths, keeps one backward pass per
step, and retains a measurable completeness property.

Attribution is candidate discovery. It is never causal proof: Gate G4 checks it against
exact single-node interventions before any ranking is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from ..backend.base import Component, HookPoint
from ..backend.hf import HFBackend
from .graph import G0Graph

ATTRIBUTION_VERSION = "eap_ig_node_v1"


class AlignmentPolicy:
    END_ALIGNED = "end_aligned_common_suffix"
    EXACT_LENGTH = "exact_length_only"


@dataclass
class AttributionResult:
    """One signed attribution vector plus everything needed to audit it."""

    object_ids: tuple[str, ...]
    scores: np.ndarray
    j_clean: float
    j_corrupt_baseline: float
    integration_steps: int
    alignment: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def path_effect(self) -> float:
        """J at the clean endpoint minus J at the fully-corrupt-at-aligned-positions end."""
        return self.j_clean - self.j_corrupt_baseline

    @property
    def total_attribution(self) -> float:
        return float(self.scores.sum())

    @property
    def completeness_ratio(self) -> float | None:
        """sum(attr) / (J(1) - J(0)). Integrated gradients should give ~1."""
        if abs(self.path_effect) < 1e-9:
            return None
        return self.total_attribution / self.path_effect

    def summary(self) -> dict[str, Any]:
        order = np.argsort(-np.abs(self.scores))
        return {
            "attribution_version": ATTRIBUTION_VERSION,
            "n_objects": len(self.object_ids),
            "integration_steps": self.integration_steps,
            "j_clean": round(self.j_clean, 6),
            "j_corrupt_baseline": round(self.j_corrupt_baseline, 6),
            "path_effect": round(self.path_effect, 6),
            "total_attribution": round(self.total_attribution, 6),
            "completeness_ratio": (
                round(self.completeness_ratio, 6)
                if self.completeness_ratio is not None
                else None
            ),
            "l1_norm": round(float(np.abs(self.scores).sum()), 6),
            "l2_norm": round(float(np.linalg.norm(self.scores)), 6),
            "n_positive": int((self.scores > 0).sum()),
            "n_negative": int((self.scores < 0).sum()),
            "top_5": [
                {"object_id": self.object_ids[i], "score": round(float(self.scores[i]), 6)}
                for i in order[:5]
            ],
            "alignment": self.alignment,
            **self.metadata,
        }


def resolve_end_aligned(
    target_prompt_len: int, source_prompt_len: int, *, policy: str
) -> tuple[tuple[tuple[int, int], ...], dict[str, Any]]:
    """Align the two prompts' component outputs position by position.

    ``end_aligned_common_suffix`` substitutes over the whole overlapping suffix. When the
    prompts differ in length the question span is misaligned by that difference; the caveat
    is recorded on every vector rather than left implicit, and the exact-length subset is
    analysed separately as a sensitivity check.
    """
    span = min(target_prompt_len, source_prompt_len)
    delta = target_prompt_len - source_prompt_len

    if policy == AlignmentPolicy.EXACT_LENGTH and delta != 0:
        raise ValueError(
            f"exact_length_only requires equal prompt lengths, got "
            f"{target_prompt_len} vs {source_prompt_len}"
        )

    pairs = tuple(
        (target_prompt_len - span + i, source_prompt_len - span + i) for i in range(span)
    )
    return pairs, {
        "policy": policy,
        "target_prompt_len": target_prompt_len,
        "source_prompt_len": source_prompt_len,
        "prompt_len_delta": delta,
        "n_aligned_positions": len(pairs),
        "exact_length_match": delta == 0,
        "caveat": (
            "Positions are aligned from the prompt end. When lengths differ the question "
            "span is offset by prompt_len_delta; the exact-length subset is reported "
            "separately."
            if delta != 0
            else "Lengths match, so alignment is index-wise and exact."
        ),
    }


class _NodeOverride:
    """Holds the leaf tensor written into one node, so its gradient can be read back."""

    __slots__ = ("hook_point", "columns", "value", "target_idx")

    def __init__(
        self,
        hook_point: HookPoint,
        value: torch.Tensor,
        target_idx: torch.Tensor,
        columns: slice | None,
    ) -> None:
        self.hook_point = hook_point
        self.value = value
        self.target_idx = target_idx
        self.columns = columns


def _capture_nodes(
    backend: HFBackend, prompt: str, graph: G0Graph, spec: Any
) -> dict[HookPoint, torch.Tensor]:
    """Capture every G0 node over prompt + the objective's common answer prefix."""
    token_ids = list(
        backend.tokenizer(prompt, add_special_tokens=backend.add_special_tokens).input_ids
    ) + list(spec.common_prefix_token_ids)

    sink: dict[HookPoint, torch.Tensor] = {}
    handles = [
        backend.hook_map.register(node, backend.hook_map.capture_hook(node, sink))
        for node in graph.nodes
    ]
    ids, mask = backend._batch([token_ids])
    try:
        with torch.no_grad():
            backend.model(input_ids=ids, attention_mask=mask)
    finally:
        for handle in handles:
            handle.remove()
    return sink


def _override_hook(override: _NodeOverride, backend: HFBackend):
    """Write the (differentiable) override value into this node's aligned positions."""
    hook_map = backend.hook_map
    hp = override.hook_point

    if hp.component is Component.HEAD_OUT:
        columns = override.columns

        def head_hook(module, args, kwargs):  # noqa: ANN001
            full = args[0]
            patched = full.clone()
            head = patched[..., columns].clone()
            head[:, override.target_idx] = override.value
            patched[..., columns] = head
            return (patched,) + tuple(args[1:]), kwargs

        return head_hook

    def post_hook(module, args, kwargs, output):  # noqa: ANN001
        from ..backend.hook_maps import rewrap, unwrap

        tensor = unwrap(output).clone()
        tensor[:, override.target_idx] = override.value
        return rewrap(output, tensor)

    return post_hook


def attribute_pair(
    backend: HFBackend,
    graph: G0Graph,
    *,
    clean_prompt: str,
    corrupt_prompt: str,
    spec: Any,
    integration_steps: int = 16,
    alignment_policy: str = AlignmentPolicy.END_ALIGNED,
    metadata: dict[str, Any] | None = None,
) -> AttributionResult:
    """Compute the signed G0 attribution vector for one clean/corrupt pair."""
    hook_map = backend.hook_map

    clean_acts = _capture_nodes(backend, clean_prompt, graph, spec)
    corrupt_acts = _capture_nodes(backend, corrupt_prompt, graph, spec)

    clean_len = clean_acts[graph.nodes[0]].shape[1]
    corrupt_len = corrupt_acts[graph.nodes[0]].shape[1]
    pairs, alignment = resolve_end_aligned(clean_len, corrupt_len, policy=alignment_policy)
    target_idx = torch.tensor([t for t, _ in pairs], dtype=torch.long, device=backend.device)
    source_idx = torch.tensor([s for _, s in pairs], dtype=torch.long, device=backend.device)

    # z_clean and z_corrupt restricted to the aligned positions.
    z_clean: dict[HookPoint, torch.Tensor] = {}
    z_corrupt: dict[HookPoint, torch.Tensor] = {}
    for node in graph.nodes:
        z_clean[node] = clean_acts[node][0].index_select(0, target_idx).float()
        z_corrupt[node] = corrupt_acts[node][0].index_select(0, source_idx).float()

    diff = {node: z_clean[node] - z_corrupt[node] for node in graph.nodes}

    def run_at(alpha: float, *, differentiable: bool) -> tuple[float, dict[HookPoint, torch.Tensor]]:
        overrides: list[_NodeOverride] = []
        for node in graph.nodes:
            value = z_corrupt[node] + alpha * diff[node]
            value = value.to(clean_acts[node].dtype).detach()
            if differentiable:
                value.requires_grad_(True)
            overrides.append(
                _NodeOverride(
                    hook_point=node,
                    value=value,
                    target_idx=target_idx,
                    columns=hook_map.head_slice(node)
                    if node.component is Component.HEAD_OUT
                    else None,
                )
            )
        hook_specs = [(o.hook_point, _override_hook(o, backend)) for o in overrides]

        if not differentiable:
            with torch.no_grad():
                value = backend.discriminative_margin_tensor(
                    clean_prompt, spec, hook_specs=hook_specs
                )
            return float(value.item()), {}

        j = backend.discriminative_margin_tensor(clean_prompt, spec, hook_specs=hook_specs)
        grads = torch.autograd.grad(j, [o.value for o in overrides], allow_unused=True)
        return float(j.item()), {
            o.hook_point: (g if g is not None else torch.zeros_like(o.value))
            for o, g in zip(overrides, grads)
        }

    # Path endpoints, measured rather than assumed, so completeness can be checked.
    j_clean, _ = run_at(1.0, differentiable=False)
    j_corrupt_baseline, _ = run_at(0.0, differentiable=False)

    accum = {node: torch.zeros_like(z_clean[node]) for node in graph.nodes}
    for k in range(1, integration_steps + 1):
        alpha = (k - 0.5) / integration_steps
        _, grads = run_at(alpha, differentiable=True)
        for node in graph.nodes:
            accum[node] += grads[node].float()

    scores = np.zeros(len(graph), dtype=np.float64)
    for i, node in enumerate(graph.nodes):
        avg_grad = accum[node] / integration_steps
        scores[i] = float((diff[node] * avg_grad).sum().item())

    return AttributionResult(
        object_ids=graph.object_ids,
        scores=scores,
        j_clean=j_clean,
        j_corrupt_baseline=j_corrupt_baseline,
        integration_steps=integration_steps,
        alignment=alignment,
        metadata=metadata or {},
    )
