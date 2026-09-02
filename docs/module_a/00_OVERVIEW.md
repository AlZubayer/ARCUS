# Module A — Design Overview

## Purpose

Module A is the **mechanistic discovery and validation layer** of ARCUS. It does not unlearn anything yet.

Its job is to answer three questions in strict order:

1. **Fact route:** does a robustly known atomic fact have a fact-selective causal computation anywhere in the model?
2. **Representation locus:** can we identify a hidden-state location/subspace whose intervention causally changes retrieval of that fact?
3. **Sink intersection:** only after 1 and 2 are frozen, does the validated factual computation causally interact with the model's attention-sink / anchor-routing system?

The code must allow a negative answer at every stage.

---

## Core scientific hypothesis

For fact `f` and query realization `q`, the preferred working hypothesis is

$$
C_{f,q}=C_f^{core}\cup C_{f,q}^{readout},
$$

where:

- `C_f^core` is a formulation-invariant causal core associated with factual identity;
- `C_{f,q}^readout` is query/modality-specific computation needed to express the answer.

This is a hypothesis, not an assumption. Valid alternatives are:

- one compact fact-specific circuit;
- a shared factual-retrieval backbone plus a small fact-specific branch;
- several redundant routes;
- no stable fact-specific route at the tested granularity.

---

## Stage graph

```text
SUITE / fact dataset
      |
      v
A0  Dataset audit + Known-Fact Core
      |
      v
A1  Blind route candidate discovery
      |
      v
A2  Exact causal validation
      |
      +-------------------+
      |                   |
      v                   v
A3  Representation      freeze A1-A3
    localization           |
      |                    v
      |                  A4 Independent
      |                     sink mapping
      |                    |
      +----------+---------+
                 v
              A5 Route x sink
                 causal mediation
                 |
                 v
              A6 Statistics,
                 artifacts, report
```

**Hard firewall:** A1-A3 may not use sink-head identities, BOS-derived priors, sink-paper carrier heads, or sink-specific route scores as discovery features.

---

## What counts as evidence

### Not enough

- high attention weight;
- high probe accuracy;
- attribution score alone;
- activation magnitude alone;
- a circuit that only works on the prompt used to discover it;
- a sink-like visual pattern.

### Required for a fact-route claim

- held-out surface-form generalization;
- exact intervention evidence;
- necessity and/or sufficiency relative to the full clean-corrupt effect;
- selectivity against matched retain controls;
- stability across at least the preregistered query modalities.

### Required for a representation-locus claim

- decodability can be used for candidate search;
- but at least one causal intervention must change factual retrieval selectively;
- preferred evidence combines projection/removal and clean-to-corrupt representation patching.

### Required for a sink-intersection claim

- sink map produced independently in A4;
- structural overlap reported separately from causal mediation;
- a matched non-sink intervention control;
- retain-fact effects reported alongside target-fact effects.

---

## Design invariants

1. **Fact identity != prompt identity.** Store them separately everywhere.
2. **Modality != fact identity.** Direct, reverse, indirect, paraphrase, and fill-in realizations must not be collapsed.
3. **Attribution != causality.** EAP-IG or related methods rank candidates only.
4. **Decodability != storage.** A probe is a screening tool, not a localization claim.
5. **Sink pattern != sink circuit != sink function.** A4 records these separately.
6. **No post-hoc widening.** If a preregistered circuit/sink definition fails, record the failure before changing the definition.
7. **Primary unit is the fact.** Token rows and paraphrase rows are repeated observations, not independent facts.
8. **Every run is reconstructible.** Model revision, tokenizer revision, dataset revision, seed, config hash, and intervention specification are mandatory metadata.

---

## Module boundaries

Module A should expose the following conceptual components:

```text
DatasetAdapter
  -> PairBuilder
  -> ModelBackend
  -> FactualScorer
  -> ActivationCache
  -> RouteDiscoverer
  -> ExactCircuitValidator
  -> RepresentationLocalizer
  -> SinkMapper
  -> RouteSinkAnalyzer
  -> StatisticsReporter
```

The model backend owns all model-specific hook names and tensor shapes. Scientific stages must call backend interfaces rather than reach directly into Hugging Face module paths.

---

## Read these files in order

1. `00_OVERVIEW.md` — scientific contract and stage graph.
2. `01_SYSTEM_DESIGN.md` — component interfaces and dependency rules.
3. `02_DATA_AND_SPLITS.md` — SUITE normalization, controls, splits, corruptions.
4. `03_EXPERIMENT_PROTOCOL.md` — A0-A6 experimental procedure and equations.
5. `04_BACKEND_AND_INTERVENTIONS.md` — exact model/backend semantics.
6. `05_ARTIFACTS_AND_METRICS.md` — persisted artifacts and metric definitions.
7. `06_IMPLEMENTATION_PLAN.md` — coding milestones and order of work.
8. `07_ACCEPTANCE_TESTS.md` — tests and scientific gates before a real run.

Existing background documents remain authoritative context:

- `docs/MODULE_A_METHODOLOGY.md`
- `docs/SINK_REFERENCE_INTEGRATION.md`

If a design file conflicts with those documents, stop and resolve the conflict explicitly rather than silently choosing one.