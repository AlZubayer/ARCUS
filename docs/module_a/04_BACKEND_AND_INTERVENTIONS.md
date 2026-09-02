# Module A — Backend and Exact-Intervention Contract

## 1. Why this file exists

Module A lives or dies on intervention correctness. A visually convincing circuit plot is useless if the scoring or patching backend is wrong.

Therefore the first real implementation milestone is a deterministic model backend with exact, testable intervention semantics.

Do **not** implement EAP-IG before the backend can reproduce clean scores and exact clean/corrupt patch effects on a small controlled case.

---

## 2. `ModelBackend` minimum API

The concrete backend may use Hugging Face + PyTorch, NNsight, TransformerLens, or another framework, but it must expose the same scientific operations.

Recommended protocol:

```python
class ModelBackend(Protocol):
    def metadata(self) -> ModelMetadata: ...

    def tokenize(self, prompt: str, answer: str | None = None) -> TokenizedExample: ...

    def score_answers(
        self,
        prompt: str,
        answers: Sequence[str],
    ) -> SequenceScoreBatch: ...

    def available_hook_points(self) -> list[HookPoint]: ...

    def capture(
        self,
        example: TokenizedExample,
        hook_points: Sequence[HookPoint],
    ) -> ActivationBundle: ...

    def run_with_patch(
        self,
        target: TokenizedExample,
        source: TokenizedExample,
        patch_spec: PatchSpec,
        answers: Sequence[str],
    ) -> SequenceScoreBatch: ...

    def run_with_ablation(
        self,
        target: TokenizedExample,
        ablation_spec: AblationSpec,
        answers: Sequence[str],
    ) -> SequenceScoreBatch: ...
```

Later optional methods:

```text
capture_attention_components
run_post_softmax_attention_edit
run_weight_intervention
jacobian_vector_product
integrated_gradient_path
```

---

## 3. Teacher-forced sequence scoring

For prompt tokens `x` and answer tokens `y_1,...,y_T`, score only answer-token conditional probabilities.

Do not include prompt-token likelihood in the answer score.

If the tokenizer merges boundary tokens differently for `prompt` versus `prompt + answer`, the backend must derive answer-token indices from a single joint tokenization strategy and test this behavior.

Return:

```text
answer_text
answer_token_ids
answer_token_positions
per_token_logprobs
sum_logprob
mean_logprob
```

The factual scorer can then construct margins without rerunning tokenization logic.

### BOS and chat templates

Record explicitly:

```text
add_special_tokens
BOS inserted? yes/no
chat template used? yes/no
generation prompt suffix
```

This is especially important because A4 distinguishes a literal BOS anchor from a first-content-position anchor.

---

## 4. Determinism

Primary mechanistic runs should use:

```text
model.eval()
dropout disabled
gradients disabled except attribution stages
fixed seed
fixed dtype
fixed attention implementation when possible
no stochastic generation for factual score
```

Record whether the model uses eager, SDPA, or Flash Attention. Some hook-level interventions may require eager attention; if the implementation changes, score parity must be checked.

A backend is accepted only when base logits/scores agree before and after instrumentation within a documented numeric tolerance.

---

## 5. Hook-point semantics

Every hook must specify **what tensor is being patched**, not just where in Python it was intercepted.

Canonical semantic names:

```text
resid_pre(L)
q(L,H)
k(L,H)
v(L,H)
attn_pattern(L,H)
head_out(L,H)
attn_out(L)
mlp_in(L)
mlp_out(L)
resid_post(L)
```

A backend maps these names to model-specific modules/tensors.

Store tensor shape and axis semantics in the hook map.

Example:

```json
{
  "name": "head_out",
  "layer": 12,
  "head_axis": 2,
  "sequence_axis": 1,
  "feature_axis": 3,
  "shape_template": ["batch", "seq", "head", "d_head"]
}
```

Never rely on an undocumented reshape.

---

## 6. Patch semantics

A `PatchSpec` must contain:

```text
hook point
target token policy
source token policy
batch mapping
replacement mode
normalization handling
```

### Replacement mode

Primary mode is exact activation replacement:

$$
h^{target}_{I}\leftarrow h^{source}_{J}.
$$

Do not interpolate unless the experiment explicitly requests a dose-response analysis.

### Token alignment

Clean and corrupt prompts can differ in token count. The backend must not assume index-wise alignment.

Use explicit policies such as:

```text
last_prompt_token -> last_prompt_token
subject_span -> matched subject_span
relation_span -> matched relation_span
semantic role mapping
explicit index lists
```

If an alignment cannot be justified, that pair is invalid for token-specific patching.

---

## 7. Necessity intervention

Input:

```text
clean target run q+
corrupt source run q-
candidate circuit C
```

Operation:

```text
run q+ while replacing only C's specified activations with values from q-
```

The rest of the clean execution remains endogenous.

Do not patch every residual stream position merely because the candidate contains a layer-level object. The circuit must define token scope.

---

## 8. Sufficiency intervention

Input:

```text
corrupt target run q-
clean source run q+
candidate circuit C
```

Operation:

```text
run q- while restoring only C from q+
```

If a multi-edge path is patched, patch sites must be applied in causal order within one forward pass so downstream nodes receive the resulting endogenous computation.

Avoid independently patching downstream nodes with clean values if the scientific claim is that upstream restoration is sufficient. Those are different experiments.

The validator should therefore support two explicit modes:

```text
node-set restoration
path-consistent restoration
```

Report which was used.

---

## 9. Head-output interventions

For a head `H`:

- `head_out` means the per-head value-weighted output before concatenation/output projection when available;
- `attn_out` means the combined attention block output after output projection.

Do not conflate them.

Ablating one head should preserve other heads. Verify the reconstruction:

$$
\text{attn\_out} \approx \sum_h W_O^{(h)}\,\text{head\_out}_h
$$

under the backend's tensor convention.

---

## 10. Attention-pattern edits for A4

Post-softmax sink edits should be isolated from A1-A3.

### Delete anchor mass

For query position `i`, anchor `a`:

1. set `alpha_{i,a}=0`;
2. renormalize remaining allowed keys to preserve row sum 1;
3. verify numerical row-sum conservation.

### Relocate anchor mass

Move anchor mass to a registered target position `b`:

$$
\alpha'_{i,b}=\alpha_{i,b}+\alpha_{i,a},
$$

$$
\alpha'_{i,a}=0.
$$

Other entries remain unchanged.

Check causal-mask validity and probability conservation for every modified row.

These edits reproduce the semantics of the supplied sink work; implementation details must be adapted to the actual attention backend.

---

## 11. Architecture-specific sink interventions

### GPT-2-like absolute-position/query-bias models

Where the actual model exposes the same objects, A4 may implement:

- query-bias nullification;
- positional embedding removal/swap;
- effective positional encoding interventions;
- selected key-projection coordinate interventions;
- matched random-coordinate controls.

Record the exact upstream sink-code commit used to define these semantics.

### RoPE / Llama-like models

Do not invent GPT-2 analogues.

Start with architecture-agnostic measurements:

```text
anchor received attention
key/query geometry
anchor value/output contribution
post-softmax delete/relocate
position/token substitution controls
```

Only add a route-level intervention after identifying an actual parameterized mechanism in that architecture.

---

## 12. Representation projection intervention

For basis `B` with orthonormal columns and target hidden vector `h`:

$$
h' = h - BB^\top h.
$$

Implementation requirements:

- verify `B.T @ B ~= I`;
- state whether `h` was centered before projection;
- state whether basis was estimated on normalized activations;
- patch only the registered token positions;
- compare with equal-rank random-subspace controls;
- compare with retain-derived subspace controls when appropriate.

A projection test without random/equal-rank controls can overstate specificity because removing any sufficiently large subspace may hurt performance.

---

## 13. Representation patching

Two distinct tests:

### Full-state patch

$$
h_{corrupt}\leftarrow h_{clean}.
$$

Tests whether the location carries enough causal information to restore the factual effect.

### Subspace-only patch

$$
h_{corrupt}'=h_{corrupt}+BB^\top(h_{clean}-h_{corrupt}).
$$

Tests whether the candidate fact subspace specifically accounts for restoration.

The second is stronger evidence for the proposed subspace; the first is a useful upper bound.

---

## 14. Attribution backend requirements

Only after exact patching works, add EAP-IG or equivalent.

Minimum tests:

- sign convention documented;
- baseline/corrupt state documented;
- integration path documented;
- number of IG steps configurable;
- convergence/sensitivity to step count checked on a pilot;
- top attributed objects validated by exact single-object patching;
- random objects of matched layer/type tested as controls.

Do not choose the final circuit solely from attribution percentile without checking exact causal fidelity.

---

## 15. Backend acceptance checklist

Before any real A1 result can be trusted:

```text
[ ] base scoring matches uninstrumented model
[ ] answer-token indexing has unit tests
[ ] capture-only hooks do not alter logits
[ ] patching one tensor changes only the intended semantic object
[ ] clean->clean patch is a no-op within tolerance
[ ] corrupt->corrupt patch is a no-op within tolerance
[ ] whole-state clean->corrupt patch produces the expected direction of change on a controlled case
[ ] head ablation preserves non-target heads
[ ] token alignment is explicit
[ ] manifests record hook-map version
```

Until these pass, do not interpret route-discovery outputs scientifically.