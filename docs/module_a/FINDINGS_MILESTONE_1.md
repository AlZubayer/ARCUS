# Module A — Milestone 1 findings

Scope: P0–P5 (dataset pinning and audit, deterministic backend, instrumentation parity,
Known-Fact Core, matched clean/corrupt pairs, exact residual-stream patching).

**Not** in scope, deliberately: EAP/EAP-IG, circuit extraction, sink mapping, route×sink
mediation, unlearning. Attribution built on an unverified intervention backend is
scientifically backwards, so the backend came first.

Nothing here is a causal-circuit claim.

---

## 1. Pinned provenance

| Object | Identifier | Revision |
|---|---|---|
| Dataset | `apeleg/SUITE` | `3f5f6b0897dac10baacf1aa8b35319a02abccd23` |
| Paraphrases | `apeleg/SUITE-rephrasings` | `81a52d60ec7d3231169b16a54ad1b2a58221ca6e` |
| Model + tokenizer | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Sink reference (A4/A5 only) | `MechanisticAccountofSinks` `sink-inheritance-foundation` | `db114c9c5eb6ffc5de13e444c783408ea7401c62` |
| Sink reference (A4/A5 only) | `MechanisticAccountofSinks` `main` | `9ab67e914464b13863b67527d8ea14068ee9ff10` |

Run settings: float32, eager attention, `model.eval()`, gradients disabled, seed 42,
TF32 off. float32 rather than bfloat16 because patch effects can be small and bfloat16
carries roughly three decimal digits; bfloat16 remains available as a registered
sensitivity check.

Prompt policy `llama3_chat_suite_v1`: the Llama-3 chat template with the upstream SUITE
system prompt, `add_generation_prompt=True`, `add_special_tokens=False`. Position 0 is a
literal `<|begin_of_text|>` (id 128000) emitted once by the template — recorded because A4
must later separate a literal-BOS anchor from a first-content-position anchor.

Neither sink repository was read during this milestone. Only their commit SHAs were
recorded, in `artifacts/reference/sink_sources_manifest.json`.

---

## 2. Discrepancies found while verifying the design pack against the data

### D1 — The existing adapter had never parsed the dataset *(blocking, fixed)*

`suite.py:row_to_example` raised on **1775 of 1775** `forget_train` rows. Its
`re.fullmatch(r"([KM]\d+)-(direct|reverse|indirect)")` grammar has no place for the
`@augmentation` suffix every training row carries (`M1-direct@q_claude9`). The scaffold
was written against a guessed schema.

Fixed: one fail-closed regex per label family, with an optional augmentation group. 100%
of rows now parse; unknown labels raise rather than letting fact identity be inferred
from question text.

### D2 — Fill-in is a surface form, not a modality *(design conflict, resolved)*

`02_DATA_AND_SPLITS.md` §3 lists fill-in beside direct/reverse/indirect as a modality.
The data encodes it as an *augmentation* (`blank_*`) of a direct or reverse question.
Resolved by adding `SurfaceKind` alongside `Modality` so neither axis is collapsed.

### D3 — Reverse-modality answers are degenerate in every topic *(scientific)*

The reverse question asks for a fact from the answer side, and within a topic they all
resolve to the same entity:

| Topic | Facts | Distinct reverse answers |
|---|---|---|
| `britney_spears_conservatorship` | 25 | **1** (`britney spears`) |
| `salem_witch_trials` | 25 | **1** (`salem witch trials`) |
| `steve_jobs_medical` | 25 | **1** (`steve jobs`) |
| `challenger_disaster` | 25 | **2** (`challenger` ×16, `challenger disaster` ×9) |

Direct and indirect are 25/25 distinct in all four topics.

A reverse question therefore identifies the **topic**, not the fact. Under the design's
default distractor rule (other facts' answers, same topic and answer type) the correct
answer lands inside `D_f`, and `M_f` becomes meaningless. The distractor builder raises
`DegenerateDistractorPool` for these cells and A0 records all 25 challenger refusals with
reason `degenerate_answer_pool`. This is the metric refusing to produce a number, not a
weakened threshold.

**Consequence for the design pack:** `minimum_modalities_known: 2` can only be met by
{direct, indirect}, and indirect exists only in the stress split. See §5.

### D4 — `forget_eval` is reference-only upstream *(design conflict, resolved)*

The SUITE card states `forget_eval` is not evaluated directly; forgetting is measured on
the rephrasings. ARCUS now drops those 300 rows with an explicit reason and takes
validation/stress from `SUITE-rephrasings`.

### D5 — `forget_train` contains no indirect rows *(scientific)*

Direct 1475 / reverse 300 / **indirect 0**. Indirect is inherently a held-out modality;
it cannot inform candidate selection because it is absent from the discovery split
entirely.

### D6 — Upstream scores with an LLM judge *(intentional divergence, recorded)*

SUITE evaluates free-form generations with a Qwen judge. ARCUS uses teacher-forced
margins per `02_DATA_AND_SPLITS.md` §5 and keeps greedy generation as a diagnostic field
only. Recorded so the two sets of numbers are never compared naively.

### D7 — SUITE ships an exactly matched same-syntax control *(opportunity, used)*

`retain_train` Syntax rows are row-aligned *and augmentation-matched* to `forget_train`:
`Syntax-M1-direct@q_claude9` reuses that exact template for a different entity. The
design pack anticipated a same-syntax ring but not at this granularity. It is now the
`same_syntax` pair family — the sharpest available test that a route is about the fact
rather than the question form.

### D8 — Config carried null source identifiers *(blocking, fixed)*

`pilot_challenger.yaml` had `revision: null` and `dataset_name: null`, violating P0. The
config now pins model, tokenizer and both dataset revisions, and rejects a floating
revision outright.

### D9 — Gemini regenerated 152 prompts verbatim from the Claude set *(leakage, fixed)*

The two augmentation generators ran independently, and for fill-in templates especially,
Gemini sometimes reproduced a Claude phrasing exactly. Across all topics 152 normalized
prompts appeared in both discovery and validation — real leakage under
`02_DATA_AND_SPLITS.md` §9 (32 of them in the challenger topic alone).

The held-out copy is dropped with reason `identical_prompt_in_discovery`; discovery wins
because a validation item identical to a discovery item is by definition not held out.
Only *exact* collisions are removed. Near-duplicates (187 pairs at ≥0.95 similarity of
124,491 cross-split comparisons, max 0.9969 after deduplication) are reported in the
audit rather than silently filtered, because the cutoff for "too similar" is a
preregistration decision.

---

## 3. Split policy

Provenance-based, not random rows, so no generated paraphrase group straddles a fold:

| Split | Source | Challenger `n` |
|---|---|---|
| discovery | `forget_train` (Claude augmentations, direct+reverse) | 450 |
| validation | `SUITE-rephrasings` Gemini augmentations (direct+reverse) | 718 |
| stress | `SUITE-rephrasings` indirect | 400 |

Gate G0 passes for the challenger topic and for all four topics.

---

## 4. Instrumentation parity (Gate G1)

| Check | Result |
|---|---|
| Backend scoring vs independent plain-HF computation | PASS — max diff 3.8e-07 (tol 1e-5) |
| Capture-only hooks numerically neutral | **PASS — exactly 0.0** across 112 hook points |
| `resid_post(L) ≡ resid_pre(L+1)` | **PASS — exactly 0.0** across 28 layers |
| Captured shapes match the hook map | PASS |
| Correct answer outscores a matched wrong one | PASS |

The two tolerances differ deliberately. Capture clones and returns `None`, so any drift
at all is a hook-map bug and the gate sits at exactly zero. The scoring comparison
contrasts a batched `log_softmax` over gathered rows with a per-row one — the same
computation in a different float32 reduction order — so it carries a documented tolerance
with the observed drift reported beside it.

---

## 5. A0 — Known-Fact Core

Challenger, 750 surfaces over 25 facts, at the preregistered thresholds
(`K_f ≥ 0.80`, `≥ 2` modalities). **No threshold was adjusted.**

**12 facts eligible:** M1 M2 K3 K4 K5 K8 K9 K10 K11 K12 K18 K20 — nine at `K_f = 1.00`.
Comfortably above the 5-fact pilot minimum, so no topic switch was needed.

The gradient is sensible rather than saturated: M2 (the disaster date) has a mean margin
of +4.08, while K2 ("record-low temperatures") scores 0.00 with a mean margin of −4.02.
The model genuinely does not know the tail of this topic, which is exactly what A0 exists
to establish before anything is read into a missing route.

Two recorded decisions:

* All 25 reverse cells refused as `degenerate_answer_pool` (D3).
* With reverse unusable, A0 scores validation **and** stress, because direct and indirect
  are the only fact-discriminative modalities and indirect lives only in stress. This is
  knowledge *screening*, not candidate selection — A1 still reads the discovery split
  alone, so no route evidence can come from a surface used here.

Excluded facts remain in the artifact with reason codes, and accuracy-only eligibility is
reported separately so a coverage failure is never mistaken for ignorance of the fact.

---


## 6. Clean/corrupt pairs

432 pairs over the 12 eligible facts; **409 accepted**. The only rejections are 23 weak
effects below `min_abs_delta = 0.5`, all retained in the artifact with reasons.

`Delta_f = M_f(q+) - M_f(q-)` uses the target fact's answer and its frozen distractor pool
under *both* prompts, so it measures how much the corruption removed the fact, not how
well the corrupt prompt answers its own question.

| Family | n | accepted | mean Δ |
|---|---|---|---|
| `same_lexical_different_meaning` | 96 | 91 | +3.73 |
| `cross_topic_matched` | 96 | 93 | +3.56 |
| `same_topic_fact_swap` | 96 | 94 | +3.53 |
| `semantic_neighbor` | 96 | 95 | +3.29 |
| `random_token_control` | 24 | 22 | +1.73 |
| `same_syntax` | 24 | 14 | +1.30 |

The same-syntax control preserves markedly more of the factual margin than any unrelated
corruption. That is a descriptive observation, not a conclusion — but it is exactly the
structure that pooling control families would have hidden, and it is why the design's
insistence on separate rings matters.

Three problems the first pair run exposed, all fixed and all worth recording:

* `random_token_control` was rejected 24/24 for "sharing the clean answer". It keeps the
  target answer *by construction*. The same-answer rule now applies only to different-fact
  corruptions, where a shared answer genuinely makes Δ meaningless.
* `same_syntax` produced **zero** pairs. SUITE's augmentation-matched twins (D7) are keyed
  by *Claude* augmentation ids, so an exact twin exists only for discovery surfaces; a
  held-out Gemini-augmented surface has none. **This qualifies D7:** the exactly matched
  same-syntax control is available for discovery surfaces only. Rather than quietly
  substituting a different template, the builder falls back to a same-(fact, modality)
  control and labels it `fact_modality_matched_only`. All 24 challenger pairs are
  fallbacks.
* Clean surfaces were being chosen without regard to whether the model answers them, so
  some anchored on surfaces with negative margins. A restoration experiment cannot rest on
  a clean run that never exhibited the fact; clean surfaces are now restricted to those A0
  scored correct.

---

## 7. Exact patching (Gate G2)

On the strongest pair (K8: clean +2.751, corrupt −6.137, **Δ = +8.888**):

| Gate | Result |
|---|---|
| A clean→clean, in-situ capture | **exactly 0.0** |
| B corrupt→corrupt, in-situ capture | **exactly 0.0** |
| A clean→clean, cross-shape capture | 3.05e-06 (tol 1e-3) |
| B corrupt→corrupt, cross-shape capture | 9.77e-06 (tol 1e-3) |
| C capture-only | **exactly 0.0** |
| D invalid coordinates fail loudly | 7/7 raised |
| F token alignment explicit and recorded | PASS |

The first run **failed** A and B at ~1e-6, and the cause was worth chasing rather than
absorbing into a tolerance. `capture()` runs `[1, prompt_len]` while scoring runs
`[n_answers, prompt_len + answer_len]`, and cuBLAS selects tiling by tensor shape.
Measured directly on this model: the length change alone moves a float32 activation by
**7.7e-07**, the batch change by **2.0e-06**, ~1.8e-06 combined — about ten times float32
epsilon.

So there are now two variants. The in-situ one captures at the shape scoring actually
uses, making written values bitwise identical, and gates the *patch write itself* at zero.
The cross-shape one reflects how real experiments capture (the corrupt source has a
different length and its own answer) and carries a documented tolerance. The residual
drift is ~5 orders of magnitude below `min_abs_delta`.

Gate D covers negative and past-the-end layers, unimplemented components, a head index on
a headless component, an out-of-range token index, an empty alignment, and
`all_prompt_tokens` on unequal prompts. Each raises a typed error *before* any forward
pass.

---

## 8. Restoration/suppression map

28 layers × 8 token offsets × 2 directions × 4 facts = **1792 interventions**, every one
with a defined normalized effect.

Sufficiency at the last prompt token, averaged over facts:

```
L0–L12   ~0.00
L13      +0.055
L14      +0.086
L15      +0.352   <- onset
L16–L22  +0.37 to +0.40
L23–L27  +0.31 to +0.40
```

Position specificity (mean over layers and facts):

| offset from prompt end | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| mean sufficiency | **+0.178** | +0.001 | −0.002 | +0.003 | +0.046 | +0.011 | +0.001 | +0.007 |

Best single site +0.642 (K4, `L17.resid_pre`, offset 0). Random `(layer, position)`
controls restore **+0.006 to +0.063** on average (max +0.387).

**What this is:** the backend sanity result `07_ACCEPTANCE_TESTS.md` Gate G2 asks for — a
late residual patch at the final token moves the factual score toward the clean run, and
arbitrary sites do not.

**What this is not:** a circuit. The artifact is labelled `restoration_suppression_map`
and carries an explicit `not_a_claim` field. Two cautions in particular:

* A single site restores only ~40% of the effect, so the factual difference is **not**
  carried by any one location.
* Concentration at the last token in late layers is partly what one would expect from
  transplanting the model's near-final summary of the prompt. It says information is
  available there; it says nothing yet about which components computed it.

---

## 9. Unresolved issues and next step

Open:

1. `same_syntax` currently has only fallback-quality matches on held-out surfaces (§6).
   The exact-template control is available if clean surfaces are drawn from discovery
   instead — a deliberate trade against held-out generalization that the team should
   settle before A1.
2. Only `resid_pre` was scanned. `attn_out`, `mlp_out` and per-head outputs are
   implemented (the first two) or interfaced (the rest) but not swept.
3. The scan covers 4 facts of the 12 eligible, on one clean surface each.
4. `bfloat16` parity, the `raw_completion` prompt policy, and multi-seed replication are
   registered but unrun.
5. Sufficiency plateaus near +0.40 rather than approaching 1.0; whether the remainder
   lives in other positions, other components, or distributed redundancy is exactly the
   A1/A2 question.

**Exact next step for A1:** implement `RouteDiscoverer` (EAP-IG or a validated
equivalent) behind its interface, on the **discovery split only**, at graph granularity G0
(residual + `attn_out` + `mlp_out`), persisting full signed attribution vectors rather than
a top-K circuit. Before any of it is treated as a ranking, run the Gate G4 checks: sign
convention, IG step-count sensitivity, exact single-object patch effects for the top
attributed objects, and matched random objects as a control. The exact-intervention
backend those checks require now exists and is gated.

The A1–A3 firewall holds: neither sink repository has been read, and only their commit
SHAs are recorded.
