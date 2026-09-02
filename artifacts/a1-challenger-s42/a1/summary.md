# A1 — blind fact-route discovery, Challenger pilot

Run `a1-challenger-s42`. Model `meta-llama/Llama-3.2-3B-Instruct` @ `0cb88a4f`, float32,
eager, seed 42. Dataset `apeleg/SUITE` @ `3f5f6b08` + `apeleg/SUITE-rephrasings` @ `81a52d60`.
Inherited setup frozen in `artifacts/freeze/p0_p5_freeze.json`.

**No sink information was used.** Neither reference repository was read; only their commit
SHAs are recorded. The P5 restoration map did not restrict the search space — attribution
ran over all 28 layers.

---

## Headline

Two findings that point in different directions, and both are reported.

1. **Attribution vectors are fact-distinctive.** Within-fact route similarity is
   **+0.672**, every control class sits at **+0.17 to +0.24**, and the gap survives
   removing the generic factual backbone (it slightly *widens*). All five control classes
   separate, all permutation p ≈ 0.0002.

2. **The circuits those vectors yield are not fact-selective.** On held-out surfaces the
   candidate circuits have strong necessity (0.74) and sufficiency (0.89), but the same
   intervention damages matched control facts almost as much as the target. Every
   selectivity ratio lies between **0.33 and 1.89** — nowhere near what a fact-specific
   route would produce, and *below 1* for semantic neighbours.

So: a route signature that distinguishes facts, attached to a causal mechanism that does
not. The fact-selective causal route RQ-A1 asks about is **not** established here.

---

## What was analysed

| | |
|---|---|
| Facts | 4 pilot (M2, K10, K8, K5 — top by A0 reliability, all `K_f = 1.00`) |
| Discovery surfaces | 6 per fact, discovery split only |
| Attribution vectors | 95 over 700 G0 objects (28 layers × 24 heads + 28 MLPs) |
| Control classes | 5, never pooled: same-topic (12), cross-topic (12), same-syntax (8), same-lexical (8), semantic-neighbour (7) |
| Pairs | 1222 accepted of 1296 attempted, all six families |
| Held-out validation | validation-split surfaces, never touched during discovery |

Discovery objective `J_f` (`discriminative_token_margin_v1`) scores the earliest token
where the correct answer and its distractors diverge. The shared answer prefix was empty on
750/750 surfaces, so it conditions on the prompt alone — no answer-prefix contamination.
Validation uses the full-sequence margin `M_f`, not `J_f`.

---

## Gate G4 — passed, with a caveat

| Criterion | Value | Threshold |
|---|---|---|
| Completeness (`Σattr` vs measured path effect) | **1.00** | ≥ 0.95 |
| A sign agreement vs exact intervention | 0.75 | ≥ 0.75 |
| B top-attributed vs layer-matched random | **31.0×** | ≥ 2.0 |
| C step-count stability (top-30 Jaccard) | **1.00** | ≥ 0.5 |
| E corruption-family robustness | 3 of 3 | ≥ 2 |
| F held-out firewall | 0 leaks | 0 |

Per fact, the aggregate hides a real split:

| fact | sign | ρ(attr, exact) | \|effect\| top | random | ratio |
|---|---|---|---|---|---|
| M2 | 0.75 | +0.78 | 3.35 | 0.20 | 16.6 |
| K10 | 0.92 | +0.09 | 2.57 | 0.43 | 6.0 |
| K8 | 0.83 | +0.60 | 2.13 | 0.03 | 75.3 |
| K5 | **0.50** | **−0.09** | 2.19 | 0.08 | 26.3 |

Attribution reliably identifies a **set** of causally important objects — 6× to 75× the
exact effect of layer-matched random objects, for every fact. Its **ordering inside that
set** is unreliable: K5 is at chance on sign with a slightly negative rank correlation.
Attribution is therefore used only as a screening device; per-object weights inside a
circuit are not interpreted, and every causal claim rests on the exact validation below.

---

## Route similarity

Cosine of signed 700-dim vectors. Both sides of every comparison had the same background
subtracted, estimated from facts appearing on neither side.

| representation | within | same-topic | semantic | syntax | lexical | cross-topic |
|---|---|---|---|---|---|---|
| raw | **+0.685** | +0.280 | +0.375 | +0.294 | +0.411 | +0.333 |
| residual, subtracted (primary) | **+0.672** | +0.244 | +0.224 | +0.190 | +0.170 | +0.207 |
| residual, projected out (secondary) | **+0.620** | +0.145 | +0.176 | +0.130 | +0.205 | +0.151 |

Distinctness D (within − control) is +0.27…+0.41 raw and +0.43…+0.50 residual. Every
permutation p ≈ 0.0002 (floor of 5000 shuffles).

**The generic backbone is real and large**: the mean cosine between a target vector and the
pooled backbone is **+0.645**. Removing it barely moves within-fact similarity
(+0.685 → +0.672) while control similarity falls. So the generic retrieval component is not
what within-fact similarity is made of.

Top-component stability: 14–20 objects appear in the top-20 of ≥⅔ of a fact's surfaces,
against an expected random overlap of 0.029.

### Extended to all 12 eligible Challenger facts

Re-run over 12 facts / 72 vectors / 180 within-fact pairs (`vectors12`), which
**strengthens** the result rather than diluting it:

| representation | within | same-topic | semantic | syntax | lexical |
|---|---|---|---|---|---|
| raw | +0.669 | +0.327 | +0.362 | +0.281 | +0.391 |
| residual, subtracted | **+0.598** | +0.067 | +0.062 | +0.056 | +0.034 |
| residual, projected out | +0.561 | +0.051 | +0.051 | +0.039 | +0.069 |

Distinctness grows from +0.43…+0.50 (4 facts) to **+0.53…+0.56** (12 facts), with control
similarity dropping to near zero. Part of that is a better backbone estimate: with 12 facts
the leave-two-out background is built from 10 facts rather than 2. The 4-fact figures above
remain the primary, since they are the ones the exact validation was run against.

---

## Exact validation on held-out surfaces

Full-sequence margin `M_f`. Necessity and sufficiency are not clipped.

| rule | \|C\| | necessity | sufficiency | target \|effect\| |
|---|---|---|---|---|
| `attribution_mass_prefix` (primary, 85% mass) | 121 | **0.736** | **0.887** | 4.04 |
| `top_k` | 30 | 0.520 | 0.579 | 2.81 |
| `stability` | 17 | 0.514 | 0.555 | 2.76 |

### Selectivity — this is where it fails

Selectivity ratio = target mean \|effect\| ÷ ring mean \|effect\|. Each control is scored on
**its own** clean prompt against **its own** pool, with 8 items per ring.

| ring | mass-prefix (\|C\|=121) | stability (\|C\|=17) |
|---|---|---|
| R1 same-topic different fact | 1.16 | 1.44 |
| R2 semantic neighbour | **0.85** | **0.91** |
| R3 same syntax | 1.11 | 1.75 |
| R4 same lexical | **0.45** | **0.33** |
| R5 cross-topic | 1.89 | 1.50 |

A fact-selective circuit should move its target by orders of magnitude more than a
neighbouring fact. Nothing here exceeds 1.9, and two rings sit below 1.0 — the intervention
hurts semantic neighbours and lexical controls *more* than the target.

The compact `stability` circuit is somewhat more selective (R1 1.44, R3 1.75) but pays for
it with roughly half the necessity and sufficiency. That is a size/selectivity trade-off
with no selective sweet spot found, not a better circuit.

**R4 caveat:** the lexical ring is character-manipulation questions ("in the word
Challenger, what letter follows the double l?"), a different task type with large fragile
margins (baselines +4.4 to +11.3). Its extreme ratio should not be read as evidence about
factual routes. R1, R2, R3 and R5 carry the conclusion, and they agree.

---

## Verdict

**Generic factual backbone plus a fact-distinctive but causally non-selective signature.**

Against the brief's four options:

- *strongly fact-specific route* — **not supported.** Selectivity fails on every ring.
- *no stable route* — **too strong.** Within-fact similarity is well above every control,
  survives backbone removal, and is stable across three corruption families.
- *generic backbone only* — **incomplete.** A large backbone exists (cosine +0.645), but
  removing it leaves within-fact similarity essentially intact.
- *generic backbone + fact-selective branch* — **closest, with one word changed.** The
  branch is fact-*distinctive* in attribution space but not fact-*selective* in causal
  effect. Those are different claims and only the first is supported.

The dissociation is the finding: attribution can tell facts apart while the components it
points at cannot be intervened on selectively. On this model, topic, granularity and
corruption policy, ARCUS's premise of a fact-selective causal route is **not** established.

---

## Caveats

1. **One topic, one model.** Similarity was extended to all 12 eligible facts; exact
   validation was run on 4. Nothing here generalises beyond Challenger on Llama-3.2-3B.
2. **8 controls per ring.** Enough to see a 0.33–1.89 spread, not enough for a tight
   interval on any single ratio.
3. **G0 granularity only.** A fact-selective route could exist at Q/K/V or
   attention-pattern granularity, which was not searched.
4. **End-aligned substitution.** Only 12 of 95 vectors are exactly length-matched (mean
   \|Δ\| 2.3 tokens); where lengths differ the question span is offset. The exact-length
   sensitivity check is thin.
5. **Attribution ordering is unreliable within a circuit** (G4-A, K5 at chance).
6. **Reverse modality excluded** by `reverse_degenerate_v1` — it identifies the topic, not
   the fact, in all four SUITE topics.
7. Circuit sizes are large (121 of 700 objects for the primary rule). Necessity and
   sufficiency at that size are partly a statement about how much of the model was patched.

---

## Next step for A3

A3 should not assume a validated circuit exists, because A1 did not produce one. The
useful question it inherits is the dissociation above: **is the fact-distinctive signal
that route similarity detects present in representation space, and is it causally
localisable there when the component-level circuit was not?**

Concretely: collect hidden states at the `last_prompt_token` across formulations of each
Known-Fact-Core fact, estimate a candidate subspace by centred SVD leaving out validation
surfaces, then run both causal tests — projection-out `h − BBᵀh` and subspace-only
restoration `h_corrupt + BBᵀ(h_clean − h_corrupt)` — against equal-rank random-subspace
controls and the same five retain rings used here. Selectivity, not decodability, is the
criterion, and it is the criterion A1 failed at component granularity.

A4 and sink analysis remain out of scope until A1–A3 are frozen.
