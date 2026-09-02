# Module A — Acceptance Tests and Scientific Gates

## Purpose

This file defines what must be true before each Module A stage can be treated as scientifically usable.

Passing unit tests is necessary but not sufficient. Module A also needs **scientific acceptance tests** that catch wrong intervention semantics, leakage, and over-interpretation.

---

## Gate G0 — Data readiness

Before any model run:

```text
[ ] canonical dataset source and revision pinned
[ ] upstream schema recorded
[ ] every atomic fact has deterministic fact_key
[ ] modality stored separately from fact identity
[ ] surface forms grouped for leakage-safe splitting
[ ] duplicate and near-duplicate audit completed
[ ] distractor pools frozen by rule
[ ] clean/corrupt pair families labeled
[ ] no sink-derived fields appear in A0-A3 input tables
```

### Fail condition

If fact identity must be inferred from prompt text with an undocumented heuristic, stop and fix the adapter.

---

## Gate G1 — Scoring/backend parity

Required automated tests:

```text
[ ] answer-token positions match manual tokenization examples
[ ] single-token answer scoring correct
[ ] multi-token answer scoring correct
[ ] capture-only hooks preserve logits within tolerance
[ ] uninstrumented HF logits == backend logits within tolerance
[ ] BOS/special-token policy is explicit
[ ] model is in eval mode
[ ] run manifest is complete
```

### Scientific sanity case

Construct at least one prompt where the correct answer has clearly higher sequence score than a matched wrong answer and inspect the raw per-token log probabilities.

### Fail condition

No attribution or patching experiment is allowed until parity passes.

---

## Gate G2 — Exact patching correctness

For one controlled clean/corrupt pair and one hook point:

```text
[ ] clean -> clean patch is a no-op
[ ] corrupt -> corrupt patch is a no-op
[ ] patch only modifies registered tensor/token positions
[ ] token alignment is recorded
[ ] score is recomputed from the intervened forward pass
```

### Whole-state direction test

At a late residual location, a full clean-state patch into corrupt should move the factual score toward the clean run for at least a controlled known example.

This is not expected at every layer; it is a backend sanity check using a location where restoration should be possible.

### Random control

Patch an equal-dimensional random or unrelated location. It should not systematically reproduce the targeted restoration across the pilot.

### Fail condition

If exact patching cannot pass self-patch/no-op tests, stop. Do not compensate by using attribution-only analyses.

---

## Gate G3 — Known-Fact Core

For each pilot fact:

```text
[ ] overall base-model reliability passes configured threshold
[ ] configured modality coverage passes
[ ] factual margin is non-degenerate
[ ] enough discovery surfaces remain
[ ] enough held-out validation surfaces remain
```

Excluded facts remain in the audit with reason codes.

### Fail condition

If too few facts qualify, switch model/topic or report that the chosen model does not robustly know the pilot set. Do not weaken thresholds merely to produce circuits.

---

## Gate G4 — Attribution discovery sanity

Before treating A1 candidate rankings seriously:

```text
[ ] attribution method/config fully recorded
[ ] sign convention tested
[ ] integration step count sensitivity measured
[ ] top objects compared with exact single-object patch effects
[ ] random matched objects tested
[ ] full attribution vector persisted
```

Attribution and exact effect need not correlate perfectly, but there must be enough agreement to justify using attribution as a ranking heuristic.

### Fail condition

If top attributions are consistently causally inert while random objects are equally effective, replace/debug the discovery method.

---

## Gate G5 — Fact-route validation

A fact circuit may be promoted from `CandidateCircuit` to `ValidatedCircuit` only if all preregistered criteria pass.

Minimum evidence categories:

```text
[ ] non-trivial exact necessity and/or sufficiency
[ ] held-out surface-form generalization
[ ] target effect exceeds same-topic retain effect
[ ] target effect exceeds same-syntax control effect
[ ] target effect exceeds same-lexical control effect where available
[ ] effect not explained solely by one discovery prompt
[ ] uncertainty reported
```

The exact numeric thresholds belong in config/preregistration, not this design file.

### Stronger route-identity evidence

Within-fact route similarity should exceed matched between-fact similarity for the primary control classes.

### Fail condition

If exact validation fails, call the discovered structure an attribution pattern, not a fact route.

---

## Gate G6 — Representation localization

A representation candidate becomes a `ValidatedLocus` only if:

```text
[ ] candidate estimated without validation surfaces
[ ] held-out same-fact separability reported
[ ] projection/removal causes selective target change OR
[ ] clean->corrupt patch restores target effect
[ ] equal-rank random-subspace control run
[ ] matched retain controls run
[ ] rank reported
[ ] token policy reported
```

Preferred strong criterion: both projection and patch evidence point to the same location/subspace.

### Fail condition

High probe accuracy with no selective causal effect is **decodability only**, not localization.

---

## Gate G7 — Freeze barrier

Before sink analysis:

```text
[ ] Known-Fact Core hash frozen
[ ] A1 candidate/attribution hashes frozen
[ ] A2 validated circuit hashes frozen
[ ] A3 representation result hashes frozen
[ ] config hash frozen
[ ] code commit frozen
```

A4/A5 run manifests must reference the freeze artifact.

### Fail condition

If A1-A3 are changed after seeing sink results, the new run must get a new lineage and be labeled exploratory unless independently revalidated.

---

## Gate G8 — Sink mapping

A4 must first reproduce expected measurement invariants on the chosen model.

Architecture-independent checks:

```text
[ ] attention row sums valid
[ ] anchor position/object explicit
[ ] delete edit conserves probability after renormalization
[ ] relocate edit conserves probability
[ ] pattern strength reported separately from causal effect
[ ] projected value/output contribution measured where possible
```

Architecture-specific checks:

- GPT-2-specific route probes only when corresponding parameters exist;
- RoPE models do not receive renamed GPT-2 EPE/query-bias interventions.

### Fail condition

High received attention alone cannot promote a head to a causal sink carrier.

---

## Gate G9 — Route x sink mediation

Required comparisons:

```text
[ ] validated fact circuit fixed before overlap computation
[ ] sink map fixed independently
[ ] structural overlap reported
[ ] exact sink-subset intervention reported
[ ] size/importance-matched non-sink subset control reported
[ ] target-fact effect reported
[ ] matched retain-fact effects reported
```

### Positive interpretation

A sink-related route is implicated only when the sink-associated part of the validated fact circuit has a selective causal effect beyond matched non-sink controls.

### Negative interpretation

If overlap exists but causal mediation does not, report **structural overlap without functional mediation**.

If neither exists, report that the tested factual route does not materially use the sink subsystem.

---

## Gate G10 — Statistical validity

Before a primary result is reported:

```text
[ ] primary experimental unit is fact
[ ] repeated prompt rows are not treated as independent facts
[ ] per-fact values available
[ ] uncertainty interval method documented
[ ] matched permutation/bootstrap procedure documented
[ ] all preregistered facts included or exclusions explained
[ ] no threshold selected solely because it maximizes headline effect
```

---

## Synthetic/tiny-model test suite

Before expensive real-model runs, create at least one tiny controlled fixture.

The fixture does not need to mimic factual memory realistically. It exists to verify intervention mechanics.

Useful fixtures:

### Fixture 1 — Identity patch

A tiny network where replacing hidden state at a known layer deterministically changes output. Verify exact patch math.

### Fixture 2 — Known two-path computation

Construct a small module with two additive paths, one designated as target path. Ensure the validator recovers expected necessity/sufficiency under ablation/patching.

### Fixture 3 — Synthetic subspace

Generate states where a known rank-r subspace carries the output-relevant signal. Verify SVD candidate estimation, projection-out effect, subspace patch restoration, and random-subspace controls.

These fixtures validate code, not scientific assumptions about LLM factual storage.

---

## First pilot definition of done

The first Module A pilot is complete when, for at least the configured set of robustly known facts from one topic:

```text
[ ] G0-G3 pass
[ ] A1 candidate routes generated blindly
[ ] at least one candidate receives exact A2 evaluation
[ ] all pilot facts receive either a validated-route result or explicit failure reason
[ ] representation candidates evaluated causally
[ ] A1-A3 freeze artifact written
[ ] independent sink map produced
[ ] route x sink mediation evaluated, including null results
[ ] machine-readable artifacts sufficient to regenerate tables
```

The pilot does **not** require a positive sink result to be considered successful. Its purpose is to determine whether the core ARCUS premise survives causal testing.