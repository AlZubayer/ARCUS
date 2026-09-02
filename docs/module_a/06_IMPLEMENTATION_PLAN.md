# Module A — Implementation Plan

## Goal

Give the coding agent an execution order that minimizes the chance of producing attractive but scientifically invalid mechanistic results.

The implementation order is intentionally **backend-first, attribution-second, sink-last**.

---

## P0 — Repository and dataset pinning

### Tasks

- pin the canonical SUITE dataset/repository source and revision;
- inspect and record the exact upstream schema;
- update `DatasetAdapter` instead of guessing fields;
- generate a dataset-audit artifact;
- pin one first model and exact revision;
- record tokenizer revision and prompt-format policy.

### Deliverables

```text
src/arcus/module_a/suite.py        # verified adapter
artifacts/.../a0/dataset_audit.json
configs/module_a/pilot_*.yaml      # no null source identifiers
```

### Definition of done

- all atomic facts are deterministically indexed;
- modalities and surface forms are separate;
- duplicates/leakage checks run;
- one pilot topic can be loaded end-to-end.

---

## P1 — Deterministic `HFBackend`

### Tasks

Implement:

```text
model/tokenizer loading
joint prompt+answer tokenization
teacher-forced answer scoring
hook-map registry
activation capture
run manifest metadata
```

Start with residual-stream, `attn_out`, and `mlp_out`; add per-head outputs next.

### Tests

- scoring agrees with direct Hugging Face logits;
- capture-only hooks are numerically neutral;
- answer-token indices are correct for single- and multi-token answers;
- BOS/special-token policy is recorded.

### Definition of done

`arcus-module-a` can score a real pilot fact and dump activations without changing base logits.

---

## P2 — Pairing and exact patching

### Tasks

- implement `PairBuilder`;
- produce matched clean/corrupt pairs;
- implement exact residual-stream replacement;
- implement `attn_out` and `mlp_out` patching;
- implement per-head output patching;
- support explicit token-position policies.

### Required sanity tests

```text
clean -> clean patch = no-op
corrupt -> corrupt patch = no-op
full clean-state -> corrupt patch restores score on a controlled case
full corrupt-state -> clean patch reduces score on a controlled case
random layer patch is not systematically equivalent to the intended patch
```

### Definition of done

The backend can calculate exact raw and normalized causal effects for manually specified components.

---

## P3 — A0 Known-Fact Core

### Tasks

- score all pilot fact formulations;
- build matched distractor pools;
- compute factual margins;
- apply configured eligibility thresholds;
- freeze the first pilot fact set.

### Definition of done

At least the configured number of pilot facts pass robust base-knowledge criteria. If not, change model/topic transparently rather than weakening thresholds silently.

---

## P4 — A1 route candidate discovery

### Tasks

- implement `RouteDiscoverer` behind its interface;
- initial method: EAP-IG or a validated equivalent;
- start with coarse graph G0/G1;
- save full signed attribution vectors;
- compute within-fact and control-specific route similarity;
- extract candidate circuits using configured deterministic rules.

### Required validation before scaling

For the top attributed objects on a few pairs:

- exact single-object patch effect should correlate directionally with attribution;
- random matched objects should be weaker on average;
- IG step-count sensitivity should be reported.

### Definition of done

Candidate circuits exist for pilot facts, but no causal-route claim is made yet.

---

## P5 — A2 exact circuit validation

### Tasks

- exact candidate-circuit necessity;
- exact sufficiency;
- matched retain controls;
- held-out-surface validation;
- iterative pruning/minimality trace;
- uncertainty estimates.

### Decision gate

Proceed to A3 only for facts whose circuits meet the preregistered validation contract.

Do not broaden the circuit until it passes by construction. If candidate discovery is poor, fix discovery and register a new analysis lineage.

---

## P6 — A3 representation localization

### Tasks

- collect candidate hidden states across formulations;
- implement centered SVD/PCA candidate subspace estimation;
- rank sweep from config;
- implement projection-out intervention;
- equal-rank random-subspace control;
- full-state clean->corrupt representation patch;
- subspace-only clean->corrupt patch;
- matched retain effects;
- held-out validation.

### Definition of done

A representation candidate is only promoted to a causal locus when intervention effects are selective and held-out.

---

## P7 — Freeze A1-A3

Write `fact_discovery_freeze.json` containing hashes of all fact-route and representation artifacts.

After this point:

- A4 may use sink-paper information;
- A1/A3 must not be silently recomputed based on sink results.

---

## P8 — A4 sink/anchor mapping

### First implementation target

Reuse semantics from the supplied sink repositories through explicit adapters.

For GPT-2/reference reproduction:

```text
received sink attention
anchor identity perturbations
post-softmax delete
post-softmax relocate
query-bias intervention where valid
EPE/position interventions where valid
key-coordinate intervention + random control
```

For non-GPT-2 architectures:

```text
start with architecture-agnostic anchor measurements
then add only mechanisms supported by that architecture
```

### Definition of done

A4 emits a frozen sink map without using fact labels to choose carrier heads.

---

## P9 — A5 fact-route x sink mediation

### Tasks

- intersect validated circuit objects with sink map;
- compute structural overlap;
- build sink-associated and matched non-sink circuit subsets;
- exact interventions on both;
- target and retain effects;
- per-fact mediation estimates.

### Decision outcomes

```text
positive: fact circuit causally uses sink-related route
mixed: only subset of facts/modalities
negative: no meaningful causal sink involvement
```

All three are acceptable outcomes.

---

## P10 — A6 statistics and report generation

### Tasks

- bootstrap/permutation statistics;
- per-fact tables;
- automatic figure data tables;
- one command that regenerates the primary Module A report from frozen artifacts.

Suggested CLI shape:

```bash
arcus-module-a run-a0 --config ...
arcus-module-a run-a1 --run ...
arcus-module-a run-a2 --run ...
arcus-module-a run-a3 --run ...
arcus-module-a freeze-fact-discovery --run ...
arcus-module-a run-a4 --run ...
arcus-module-a run-a5 --run ...
arcus-module-a report --run ...
```

Do not require notebooks for the canonical experiment.

---

## Recommended first-agent work session

The coding agent should start here, in order:

```text
1. Read AGENTS.md
2. Read docs/module_a/00_OVERVIEW.md
3. Read docs/module_a/01_SYSTEM_DESIGN.md
4. Read docs/module_a/02_DATA_AND_SPLITS.md
5. Read docs/module_a/04_BACKEND_AND_INTERVENTIONS.md
6. Inspect current source interfaces/tests
7. Pin canonical SUITE source
8. Implement P0
9. Implement P1
10. Implement only residual-stream patching from P2
```

The first checkpoint to send the research team should contain:

- dataset audit;
- exact model/tokenizer revision;
- known answer scoring examples;
- parity test results;
- clean/corrupt pair examples;
- one verified residual-stream patch experiment.

Do **not** spend the first session implementing plots, sink visualizations, or a full EAP graph.

---

## Coding conventions for scientific code

- no hidden global config;
- no notebook-only logic;
- no silent fallback from exact intervention to approximation;
- no implicit token-position broadcasting;
- no recomputing upstream stages without lineage metadata;
- every random process receives an explicit seed;
- every filtering decision writes a reason;
- every scientific metric has a version string;
- failed candidates remain in artifacts.

---

## Parallelizable work after P2

Once the exact backend is trusted, team members can work in parallel:

```text
Track 1: A0/data controls
Track 2: A1 attribution discovery
Track 3: A3 representation candidate utilities
Track 4: A4 sink adapter porting
```

But A4 results must remain inaccessible to A1/A3 candidate-selection logic until the freeze barrier is created.