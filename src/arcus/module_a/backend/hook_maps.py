"""Llama hook map: semantic component names to concrete modules and tensor axes.

Verified against transformers 4.57 for meta-llama/Llama-3.2-3B-Instruct:

* ``layers[L]`` forward-pre  -- ``args[0]``   is the residual stream entering layer L
* ``layers[L]`` forward      -- output        is the residual stream leaving layer L
* ``layers[L].self_attn``    -- ``output[0]`` is the attention block output AFTER o_proj
* ``layers[L].mlp``          -- output        is the MLP output

``attn_out`` is therefore the combined post-projection block output, never a per-head
tensor; the two are not conflated (04_BACKEND_AND_INTERVENTIONS.md section 9).

Return shapes have changed across transformers releases (a decoder layer used to return a
tuple and now returns a bare tensor), so every accessor handles both and the map records
which convention it observed.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .base import Component, HookPoint, InvalidHookPointError

HOOK_MAP_VERSION = "llama_hook_map_v1"

#: All implemented components are [batch, sequence, d_model] with no head axis.
SHAPE_TEMPLATES: dict[Component, dict[str, Any]] = {
    Component.RESID_PRE: {
        "shape_template": ["batch", "seq", "d_model"],
        "sequence_axis": 1,
        "feature_axis": 2,
        "head_axis": None,
        "site": "layers[L] forward_pre_hook args[0]",
        "semantics": "residual stream entering layer L, before input_layernorm",
    },
    Component.RESID_POST: {
        "shape_template": ["batch", "seq", "d_model"],
        "sequence_axis": 1,
        "feature_axis": 2,
        "head_axis": None,
        "site": "layers[L] forward_hook output",
        "semantics": "residual stream leaving layer L; equals resid_pre(L+1)",
    },
    Component.ATTN_OUT: {
        "shape_template": ["batch", "seq", "d_model"],
        "sequence_axis": 1,
        "feature_axis": 2,
        "head_axis": None,
        "site": "layers[L].self_attn forward_hook output[0]",
        "semantics": "attention block output after o_proj, summed over heads",
    },
    Component.MLP_OUT: {
        "shape_template": ["batch", "seq", "d_model"],
        "sequence_axis": 1,
        "feature_axis": 2,
        "head_axis": None,
        "site": "layers[L].mlp forward_hook output",
        "semantics": "MLP block output, before the residual addition",
    },
    Component.HEAD_OUT: {
        "shape_template": ["batch", "seq", "d_head"],
        "sequence_axis": 1,
        "feature_axis": 2,
        "head_axis": "sliced out of the o_proj input, not a separate axis",
        "site": "layers[L].self_attn.o_proj forward_pre_hook args[0][..., H*d_head:(H+1)*d_head]",
        "semantics": (
            "one head's value-weighted output BEFORE the output projection. Its "
            "contribution to the residual stream is W_O[:, H*d_head:(H+1)*d_head] @ head_out. "
            "Distinct from attn_out, which is the post-projection sum over all heads."
        ),
    },
}


def unwrap(output: Any) -> torch.Tensor:
    """Read the hidden-state tensor out of a module output that may be wrapped."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise InvalidHookPointError(f"Cannot locate a hidden-state tensor in {type(output).__name__}")


def rewrap(output: Any, tensor: torch.Tensor) -> Any:
    """Rebuild a module output structure around a replaced hidden-state tensor."""
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor,) + tuple(output[1:])
    if isinstance(output, list):
        return [tensor] + list(output[1:])
    raise InvalidHookPointError(f"Cannot rebuild output of type {type(output).__name__}")


class LlamaHookMap:
    """Resolves :class:`HookPoint` coordinates against a concrete Llama model."""

    version = HOOK_MAP_VERSION

    def __init__(self, model: Any) -> None:
        self.model = model
        self.layers = model.model.layers
        self.n_layers = len(self.layers)
        config = model.config
        self.n_heads = int(config.num_attention_heads)
        self.d_model = int(config.hidden_size)
        self.d_head = self.d_model // self.n_heads

    # -- validation -------------------------------------------------------------------

    def validate(self, hook_point: HookPoint) -> None:
        """Reject an unusable coordinate loudly, before any forward pass runs."""
        if hook_point.component not in SHAPE_TEMPLATES:
            raise InvalidHookPointError(
                f"Component {hook_point.component!r} is declared but not implemented. "
                f"Implemented: {sorted(c.value for c in SHAPE_TEMPLATES)}"
            )
        if not isinstance(hook_point.layer, int) or isinstance(hook_point.layer, bool):
            raise InvalidHookPointError(f"layer must be an int, got {hook_point.layer!r}")
        if not 0 <= hook_point.layer < self.n_layers:
            raise InvalidHookPointError(
                f"layer {hook_point.layer} outside [0, {self.n_layers}) for this model"
            )
        if hook_point.component is Component.HEAD_OUT:
            if hook_point.head is None:
                raise InvalidHookPointError("head_out requires a head index")
            if not isinstance(hook_point.head, int) or isinstance(hook_point.head, bool):
                raise InvalidHookPointError(f"head must be an int, got {hook_point.head!r}")
            if not 0 <= hook_point.head < self.n_heads:
                raise InvalidHookPointError(
                    f"head {hook_point.head} outside [0, {self.n_heads}) for this model"
                )
        elif hook_point.head is not None:
            raise InvalidHookPointError(
                f"Component {hook_point.component.value} has no head axis; "
                f"head={hook_point.head} is not meaningful here."
            )

    def describe(self, hook_point: HookPoint) -> dict[str, Any]:
        self.validate(hook_point)
        return {
            **SHAPE_TEMPLATES[hook_point.component],
            "hook_map_version": self.version,
            **hook_point.to_dict(),
        }

    def all_hook_points(self, components: list[Component] | None = None) -> list[HookPoint]:
        wanted = components or [c for c in SHAPE_TEMPLATES if c is not Component.HEAD_OUT]
        points: list[HookPoint] = []
        for layer in range(self.n_layers):
            for component in wanted:
                if component is Component.HEAD_OUT:
                    points += [
                        HookPoint(layer=layer, component=component, head=h)
                        for h in range(self.n_heads)
                    ]
                else:
                    points.append(HookPoint(layer=layer, component=component))
        return points

    def head_output_weight(self, hook_point: HookPoint) -> torch.Tensor:
        """The o_proj columns this head's output is multiplied by.

        Llama has ``attention_bias=False``, so the residual contribution of a head is
        exactly ``head_out @ W_O_slice.T`` and the per-head decomposition of ``attn_out``
        is exact rather than approximate.
        """
        self.validate(hook_point)
        o_proj = self.layers[hook_point.layer].self_attn.o_proj
        return o_proj.weight[:, self.head_slice(hook_point)]

    # -- module resolution ------------------------------------------------------------

    def module_for(self, hook_point: HookPoint) -> Any:
        self.validate(hook_point)
        layer = self.layers[hook_point.layer]
        if hook_point.component in (Component.RESID_PRE, Component.RESID_POST):
            return layer
        if hook_point.component is Component.ATTN_OUT:
            return layer.self_attn
        if hook_point.component is Component.MLP_OUT:
            return layer.mlp
        if hook_point.component is Component.HEAD_OUT:
            return layer.self_attn.o_proj
        raise InvalidHookPointError(f"No module for {hook_point.component!r}")

    def is_pre_hook(self, hook_point: HookPoint) -> bool:
        return hook_point.component in (Component.RESID_PRE, Component.HEAD_OUT)

    def head_slice(self, hook_point: HookPoint) -> slice:
        """Columns of the o_proj input belonging to this head."""
        start = hook_point.head * self.d_head
        return slice(start, start + self.d_head)

    # -- hook factories ---------------------------------------------------------------

    def capture_hook(self, hook_point: HookPoint, sink: dict[HookPoint, torch.Tensor]) -> Callable:
        """Read-only hook. Clones so later in-place ops cannot corrupt the record.

        A capture hook returns ``None`` everywhere, so instrumentation cannot perturb the
        forward pass; the parity gate asserts exactly that.
        """
        if hook_point.component is Component.HEAD_OUT:
            columns = self.head_slice(hook_point)

            def head_capture(module, args, kwargs):  # noqa: ANN001
                # o_proj is called positionally with the concatenated per-head outputs.
                sink[hook_point] = args[0][..., columns].detach().clone()
                return None

            return head_capture

        if self.is_pre_hook(hook_point):

            def pre_hook(module, args, kwargs):  # noqa: ANN001
                sink[hook_point] = unwrap(args[0] if args else kwargs["hidden_states"]).detach().clone()
                return None

            return pre_hook

        def post_hook(module, args, kwargs, output):  # noqa: ANN001
            sink[hook_point] = unwrap(output).detach().clone()
            return None

        return post_hook

    def patch_hook(
        self,
        hook_point: HookPoint,
        source: torch.Tensor,
        pairs: tuple[tuple[int, int], ...],
    ) -> Callable:
        """Exact replacement hook: writes source positions into target positions.

        ``source`` is the captured activation for this hook point from the source run,
        shaped [1, seq_source, d_model]; it is broadcast across the target batch. Only the
        listed target positions are written, and every other position, layer and component
        is left endogenous.
        """
        target_idx = torch.tensor([t for t, _ in pairs], dtype=torch.long)
        source_idx = torch.tensor([s for _, s in pairs], dtype=torch.long)

        def apply(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[-1] != source.shape[-1]:
                raise InvalidHookPointError(
                    f"{hook_point.id}: feature dim {tensor.shape[-1]} does not match source "
                    f"{source.shape[-1]}"
                )
            if int(target_idx.max()) >= tensor.shape[1]:
                raise InvalidHookPointError(
                    f"{hook_point.id}: target index {int(target_idx.max())} outside sequence "
                    f"of length {tensor.shape[1]}"
                )
            if int(source_idx.max()) >= source.shape[1]:
                raise InvalidHookPointError(
                    f"{hook_point.id}: source index {int(source_idx.max())} outside captured "
                    f"sequence of length {source.shape[1]}"
                )
            # Out-of-place so the source run's cached tensors are never mutated.
            patched = tensor.clone()
            values = source.to(device=tensor.device, dtype=tensor.dtype)[
                0, source_idx.to(source.device)
            ]
            patched[:, target_idx.to(tensor.device)] = values
            return patched

        if hook_point.component is Component.HEAD_OUT:
            columns = self.head_slice(hook_point)

            def head_patch(module, args, kwargs):  # noqa: ANN001
                # Write only this head's columns of the o_proj input. Every other head's
                # slice is left untouched, so ablating one head cannot disturb the others.
                full = args[0]
                patched = full.clone()
                patched[..., columns] = apply(full[..., columns])
                return (patched,) + tuple(args[1:]), kwargs

            return head_patch

        if self.is_pre_hook(hook_point):

            def pre_hook(module, args, kwargs):  # noqa: ANN001
                if args:
                    return (apply(unwrap(args[0])),) + tuple(args[1:]), kwargs
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = apply(unwrap(kwargs["hidden_states"]))
                return args, kwargs

            return pre_hook

        def post_hook(module, args, kwargs, output):  # noqa: ANN001
            return rewrap(output, apply(unwrap(output)))

        return post_hook

    def register(self, hook_point: HookPoint, fn: Callable) -> Any:
        module = self.module_for(hook_point)
        if self.is_pre_hook(hook_point):
            return module.register_forward_pre_hook(fn, with_kwargs=True)
        return module.register_forward_hook(fn, with_kwargs=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hook_map_version": self.version,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_model": self.d_model,
            "d_head": self.d_head,
            "components": {c.value: SHAPE_TEMPLATES[c] for c in SHAPE_TEMPLATES},
        }
