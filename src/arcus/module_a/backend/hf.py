"""Deterministic Hugging Face backend with exact intervention semantics.

Implements P1 (teacher-forced sequence scoring) and the residual-stream portion of P2
(exact activation capture and replacement).

Everything model-specific stops here. No experiment stage touches ``model.model.layers``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Sequence

import torch

from .base import (
    Component,
    HookPoint,
    InvalidHookPointError,
    ModelMetadata,
    PatchSpec,
    SequenceScore,
    TokenizationBoundaryError,
    TokenizedExample,
    text_sha256,
)
from .hook_maps import LlamaHookMap

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class HFBackend:
    """A pinned, eval-mode, gradient-free decoder-only backend."""

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        dtype: str = "float32",
        device: str = "cuda",
        attn_implementation: str = "eager",
        seed: int = 42,
        system_prompt: str | None = None,
        prompt_template: str = "chat",
        prompt_template_version: str = "llama3_chat_suite_v1",
        add_generation_prompt: bool = True,
        add_special_tokens: bool = False,
        raw_completion_template: str = "Question: {question}\nAnswer:",
        trust_remote_code: bool = False,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if dtype not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(DTYPES)}")

        self.model_id = model_id
        self.revision = revision
        self.tokenizer_id = tokenizer_id or model_id
        self.tokenizer_revision = tokenizer_revision or revision
        self.dtype_name = dtype
        self.attn_implementation = attn_implementation
        self.seed = seed
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.prompt_template_version = prompt_template_version
        self.add_generation_prompt = add_generation_prompt
        self.add_special_tokens = add_special_tokens
        self.raw_completion_template = raw_completion_template

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is unavailable")
        self.device = torch.device(device)

        # Determinism: a mechanistic run must be byte-reproducible.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)  # cuBLAS reductions stay fast but fixed
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_id,
            revision=self.tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=DTYPES[dtype],
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

        self.hook_map = LlamaHookMap(self.model)
        self._metadata = self._build_metadata()

    # -- metadata ---------------------------------------------------------------------

    def _build_metadata(self) -> ModelMetadata:
        cfg = self.model.config
        tok = self.tokenizer
        chat_template = getattr(tok, "chat_template", None)
        uses_chat = self.prompt_template == "chat" and bool(chat_template)

        probe = self.build_prompt("PROBE")
        bos = tok.bos_token
        bos_in_template = bool(bos) and probe.startswith(bos)

        n_heads = int(getattr(cfg, "num_attention_heads"))
        d_model = int(getattr(cfg, "hidden_size"))
        return ModelMetadata(
            model_id=self.model_id,
            model_revision=self.revision,
            tokenizer_id=self.tokenizer_id,
            tokenizer_revision=self.tokenizer_revision,
            dtype=self.dtype_name,
            device=str(self.device),
            attn_implementation=self.attn_implementation,
            seed=self.seed,
            n_layers=int(getattr(cfg, "num_hidden_layers")),
            n_heads=n_heads,
            n_kv_heads=int(getattr(cfg, "num_key_value_heads", n_heads)),
            d_model=d_model,
            d_head=d_model // n_heads,
            vocab_size=int(getattr(cfg, "vocab_size")),
            architecture=type(self.model).__name__,
            hook_map_version=self.hook_map.version,
            prompt_template_version=self.prompt_template_version,
            chat_template_used=uses_chat,
            chat_template_sha256=text_sha256(chat_template) if chat_template else None,
            system_prompt=self.system_prompt,
            add_generation_prompt=self.add_generation_prompt,
            add_special_tokens=self.add_special_tokens,
            bos_token=bos,
            bos_token_id=tok.bos_token_id,
            bos_inserted_by_template=bos_in_template,
            eos_token=tok.eos_token,
            pad_token=tok.pad_token,
            generation_prompt_suffix=probe.split("PROBE")[-1] if "PROBE" in probe else None,
            eval_mode=not self.model.training,
        )

    def metadata(self) -> ModelMetadata:
        return self._metadata

    # -- prompting --------------------------------------------------------------------

    def build_prompt(self, question: str) -> str:
        """Render one question under the registered prompt policy."""
        if self.prompt_template == "raw_completion":
            return self.raw_completion_template.format(question=question)

        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": question})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=self.add_generation_prompt
        )

    # -- tokenization -----------------------------------------------------------------

    def tokenize(
        self, prompt: str, answer: str, *, surface_form_id: str | None = None
    ) -> TokenizedExample:
        """Jointly tokenize prompt and answer so answer indices are exact.

        The answer's token ids are derived from a single joint tokenization, never from
        tokenizing the answer alone: BPE merges across the boundary would otherwise shift
        every position. If the joint tokenization does not extend the prompt tokenization,
        this raises rather than guessing an offset.
        """
        prompt_ids = tuple(
            self.tokenizer(prompt, add_special_tokens=self.add_special_tokens).input_ids
        )
        full_ids = tuple(
            self.tokenizer(prompt + answer, add_special_tokens=self.add_special_tokens).input_ids
        )

        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise TokenizationBoundaryError(
                "Tokenizing prompt+answer did not extend the prompt tokenization; the "
                "tokenizer merged across the boundary, so answer-token indices cannot be "
                f"derived from the prompt length. prompt_ids={prompt_ids[-5:]} "
                f"full_ids={full_ids[len(prompt_ids) - 5 : len(prompt_ids) + 3]}"
            )
        if len(full_ids) == len(prompt_ids):
            raise TokenizationBoundaryError(
                f"Answer {answer!r} contributed no tokens under the registered prompt policy"
            )

        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None and prompt_ids.count(bos_id) > 1:
            raise TokenizationBoundaryError(
                f"Prompt contains {prompt_ids.count(bos_id)} BOS tokens. The chat template "
                "already emits one; add_special_tokens must stay False."
            )

        positions = tuple(range(len(prompt_ids), len(full_ids)))
        return TokenizedExample(
            prompt_text=prompt,
            answer_text=answer,
            prompt_token_ids=prompt_ids,
            full_token_ids=full_ids,
            answer_token_ids=full_ids[len(prompt_ids) :],
            answer_token_positions=positions,
            prompt_text_sha256=text_sha256(prompt),
            surface_form_id=surface_form_id,
        )

    # -- forward passes ---------------------------------------------------------------

    def _pad_id(self) -> int:
        for candidate in (self.tokenizer.pad_token_id, self.tokenizer.eos_token_id, 0):
            if candidate is not None:
                return int(candidate)
        return 0

    def _batch(self, sequences: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad a batch. Padding trails the real tokens, so causal positions are
        unaffected and masked logits are never read."""
        width = max(len(s) for s in sequences)
        pad = self._pad_id()
        ids = torch.full((len(sequences), width), pad, dtype=torch.long)
        mask = torch.zeros((len(sequences), width), dtype=torch.long)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        return ids.to(self.device), mask.to(self.device)

    @torch.no_grad()
    def forward_logits(self, token_ids: Sequence[int]) -> torch.Tensor:
        ids, mask = self._batch([token_ids])
        return self.model(input_ids=ids, attention_mask=mask).logits[0]

    @contextlib.contextmanager
    def _hooks(self, handles: list[Any]) -> Iterator[None]:
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def _score_batch(
        self,
        examples: Sequence[TokenizedExample],
        *,
        hook_specs: Sequence[tuple[HookPoint, Any]] = (),
    ) -> list[SequenceScore]:
        """Teacher-forced scoring of a batch that shares one prompt.

        s(y|q) = (1/T) sum_t log p(y_t | q, y_<t): only answer-token conditionals count,
        never the prompt's own likelihood.
        """
        ids, mask = self._batch([e.full_token_ids for e in examples])
        handles = [self.hook_map.register(hp, fn) for hp, fn in hook_specs]
        with self._hooks(handles):
            logits = self.model(input_ids=ids, attention_mask=mask).logits

        scores: list[SequenceScore] = []
        for i, example in enumerate(examples):
            positions = example.answer_token_positions
            # Token at position t is predicted by the logits at t-1.
            rows = logits[i, [p - 1 for p in positions], :].float()
            logprobs = torch.log_softmax(rows, dim=-1)
            targets = torch.tensor(example.answer_token_ids, device=logprobs.device)
            per_token = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            total = float(per_token.sum().item())
            scores.append(
                SequenceScore(
                    answer_text=example.answer_text,
                    answer_token_ids=example.answer_token_ids,
                    answer_token_positions=positions,
                    per_token_logprobs=tuple(float(x) for x in per_token.tolist()),
                    sum_logprob=total,
                    mean_logprob=total / len(positions),
                )
            )
        return scores

    def score_answers(
        self, prompt: str, answers: Sequence[str], *, surface_form_id: str | None = None
    ) -> list[SequenceScore]:
        """Score several candidate answers under one prompt."""
        if not answers:
            raise ValueError("score_answers requires at least one answer")
        examples = [self.tokenize(prompt, a, surface_form_id=surface_form_id) for a in answers]
        return self._score_batch(examples)

    @torch.no_grad()
    def generate_greedy(self, prompt: str, max_new_tokens: int = 24) -> str:
        """Greedy continuation. A diagnostic field only, never the correctness criterion."""
        ids, mask = self._batch(
            [self.tokenizer(prompt, add_special_tokens=self.add_special_tokens).input_ids]
        )
        out = self.model.generate(
            input_ids=ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=self._pad_id(),
        )
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)

    # -- activation capture -----------------------------------------------------------

    def available_hook_points(self, components: Sequence[Component] | None = None) -> list[HookPoint]:
        return self.hook_map.all_hook_points(list(components) if components else None)

    @torch.no_grad()
    def capture(
        self, prompt: str, hook_points: Sequence[HookPoint], *, answer: str | None = None
    ) -> dict[HookPoint, torch.Tensor]:
        """Capture activations over the prompt (optionally prompt+answer).

        Prompt-position activations are identical with or without a trailing answer,
        because the model is causal; ``answer`` is accepted for answer-span captures.
        """
        for hp in hook_points:
            self.hook_map.validate(hp)
        token_ids = self.tokenizer(
            prompt if answer is None else prompt + answer,
            add_special_tokens=self.add_special_tokens,
        ).input_ids

        sink: dict[HookPoint, torch.Tensor] = {}
        handles = [
            self.hook_map.register(hp, self.hook_map.capture_hook(hp, sink)) for hp in hook_points
        ]
        ids, mask = self._batch([token_ids])
        with self._hooks(handles):
            self.model(input_ids=ids, attention_mask=mask)

        missing = [hp.id for hp in hook_points if hp not in sink]
        if missing:
            raise InvalidHookPointError(f"Hook points never fired: {missing}")
        return sink

    # -- exact patching ---------------------------------------------------------------

    def score_answers_with_patch(
        self,
        prompt: str,
        answers: Sequence[str],
        patch_spec: PatchSpec,
        source_activations: dict[HookPoint, torch.Tensor],
        *,
        surface_form_id: str | None = None,
    ) -> list[SequenceScore]:
        """Run the target prompt while replacing exactly the specified activations.

        The rest of the target execution stays endogenous: only the listed hook points and
        only the listed token positions are written.
        """
        if not answers:
            raise ValueError("score_answers_with_patch requires at least one answer")

        for hp in patch_spec.hook_points:
            self.hook_map.validate(hp)
            if hp not in source_activations:
                raise InvalidHookPointError(
                    f"No source activation captured for {hp.id}; capture it before patching."
                )

        examples = [self.tokenize(prompt, a, surface_form_id=surface_form_id) for a in answers]
        pairs = patch_spec.alignment.pairs
        prompt_len = examples[0].prompt_len
        for target_idx, _ in pairs:
            if target_idx >= prompt_len and patch_spec.alignment.policy.value.endswith(
                "prompt_token"
            ):
                raise InvalidHookPointError(
                    f"Alignment writes position {target_idx} but the prompt is {prompt_len} "
                    "tokens; a prompt-token policy must not reach into the answer span."
                )

        hook_specs = [
            (hp, self.hook_map.patch_hook(hp, source_activations[hp], pairs))
            for hp in patch_spec.hook_points
        ]
        return self._score_batch(examples, hook_specs=hook_specs)

    # -- introspection ----------------------------------------------------------------

    def describe_hook_map(self) -> dict[str, Any]:
        return self.hook_map.to_dict()


def backend_from_config(config: Any) -> HFBackend:
    """Build a backend from a :class:`ModuleAConfig`."""
    return HFBackend(
        model_id=config.model.name,
        revision=config.model.revision,
        tokenizer_id=config.model.resolved_tokenizer_name,
        tokenizer_revision=config.model.resolved_tokenizer_revision,
        dtype=config.model.dtype,
        device=config.model.device,
        attn_implementation=config.model.attn_implementation,
        seed=config.experiment.seed,
        system_prompt=config.prompt.system_prompt,
        prompt_template=config.prompt.template,
        prompt_template_version=config.prompt.template_version,
        add_generation_prompt=config.prompt.add_generation_prompt,
        add_special_tokens=config.prompt.add_special_tokens,
        raw_completion_template=config.prompt.raw_completion_template,
        trust_remote_code=config.model.trust_remote_code,
    )
