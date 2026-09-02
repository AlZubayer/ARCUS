# Coding-agent handoff: ARCUS Module A

## Goal

Build Module A as a reproducible mechanistic experiment, **not as an unlearning method yet**.

Research questions:

- **RQ-A1:** Does factual identity predict a stable causal subcircuit above and beyond lexical, syntactic, entity, topic, and query-modality similarity?
- **RQ-A2:** Is there a hidden-state location/subspace that is invariant across formulations of the same fact and causally relevant to factual recall?
- **RQ-A3:** After RQ-A1/RQ-A2 are validated blindly, does the fact-selective circuit overlap or interact with attention-sink / anchor routing?

## Start here: design pack

Before writing code, read these files in order:

1. `docs/module_a/00_OVERVIEW.md`
2. `docs/module_a/01_SYSTEM_DESIGN.md`
3. `docs/module_a/02_DATA_AND_SPLITS.md`
4. `docs/module_a/03_EXPERIMENT_PROTOCOL.md`
5. `docs/module_a/04_BACKEND_AND_INTERVENTIONS.md`
6. `docs/module_a/05_ARTIFACTS_AND_METRICS.md`
7. `docs/module_a/06_IMPLEMENTATION_PLAN.md`
8. `docs/module_a/07_ACCEPTANCE_TESTS.md`

Background/context files:

- `docs/MODULE_A_METHODOLOGY.md`
- `docs/SINK_REFERENCE_INTEGRATION.md`

The design pack is the implementation contract. If the current scaffold conflicts with it, preserve the scientific rule and refactor the scaffold explicitly.

## Non-negotiable scientific rules

1. **Do not use sink heads to seed A1 discovery.** Sink mapping is A4.
2. **Attribution is discovery, not causal evidence.** Exact patching/ablation is required for route claims.
3. **Decodability is not storage/localization.** Probe accuracy must be followed by projection or patch interventions.
4. Every fact-level claim must generalize to **held-out surface forms**.
5. Controls must separately test semantic neighbors, same syntax, same lexical tokens, topic sharing, and unrelated general knowledge.
6. Preserve `fact_id`, `topic`, `modality`, `surface_form_id`, and `control_type` as separate fields in every artifact.
7. Cache model revision, tokenizer revision, dataset revision, config hash, seed, prompt text hash, and intervention specification.
8. Never change scoring, token aggregation, corruption policy, graph granularity, or circuit thresholds after inspecting results without making the change explicit and versioned.
9. A null result is valid. Do not force “one fact = one path”.
10. Sink **pattern**, **circuit**, and **function** are separate measurements.
11. Do not call a high-decoding representation a fact locus without a causal intervention.
12. Do not run A4/A5 until the A1-A3 freeze artifact exists.

## Recommended implementation order

1. Pin and audit the canonical SUITE source/revision.
2. Implement a Hugging Face backend for deterministic teacher-forced sequence scoring.
3. Implement activation capture for residual stream, head outputs, MLP outputs, queries/keys/values, and attention probabilities as needed.
4. Implement exact residual activation patching on clean/corrupted pairs.
5. Add head- and MLP-level patching.
6. Build the Known-Fact Core and freeze the first pilot facts.
7. Implement attribution candidate ranking behind `RouteDiscoverer` (EAP-IG or a carefully validated equivalent).
8. Extract compact candidate circuits and validate necessity, sufficiency, selectivity, and held-out-form invariance.
9. Implement representation localization and causal projection/patch tests.
10. Freeze A1-A3 artifacts.
11. Only then implement sink/anchor mapping and route×sink mediation.
12. Add permutation/bootstrap statistics and report generation.

## First-pilot definition of done

For at least 5 robustly-known facts from one SUITE topic:

- base model answers held-out configured modalities reliably;
- candidate circuit is discovered with no sink information;
- exact interventions establish non-trivial causal evidence;
- within-fact route similarity is compared against matched controls;
- causal selectivity is quantified on retain neighbors;
- representation localization is tested by causal intervention, not decoding alone;
- A1-A3 are frozen before sink analysis;
- sink intersection/mediation is reported, including a null result if absent.

## Milestone 1 status: complete

P0-P2 (dataset audit, deterministic backend, exact residual patching), plus A0 and the
pair builder, are implemented and gated. See `docs/module_a/FINDINGS_MILESTONE_1.md` for
results, the nine design/data discrepancies found while verifying the design pack against
the real dataset, and the exact next step for A1.

Gates passing: G0 (data readiness), G1 (scoring/backend parity), G2 (exact patching),
G3 (Known-Fact Core, 12 eligible challenger facts at unchanged thresholds).

Neither sink repository has been read. Only their commit SHAs are recorded, in
`artifacts/reference/sink_sources_manifest.json`.

## First coding task

Implement **P0 + P1 + the residual-patching portion of P2** from `docs/module_a/06_IMPLEMENTATION_PLAN.md`.

The first checkpoint should include:

- canonical dataset/revision and schema audit;
- exact model/tokenizer revision;
- teacher-forced scoring examples with token indices;
- instrumentation parity tests;
- several matched clean/corrupt pairs;
- one verified residual-stream patch experiment.

Do not begin with EAP-IG, sink visualizations, or paper figures. A fast attribution implementation without a trustworthy exact-intervention backend is scientifically backwards for this project.

## Sink implementations supplied by the research team

Use these as the authoritative historical references for A4/A5:

```bash
# A Sink Without the Plumbing
git clone --branch sink-inheritance-foundation --single-branch \
  https://github.com/AlZubayer/MechanisticAccountofSinks.git Sink-KD

# Same Sink, Different Plumbing / reproduction
git clone https://github.com/AlZubayer/MechanisticAccountofSinks.git Sink-Repro
```

Read `docs/SINK_REFERENCE_INTEGRATION.md` before porting anything. The sink repositories are **not** an allowed source of candidate heads for A1-A3. First discover and validate the fact circuit blindly; only then run the independent sink mapper and test intersection/mediation.

Do not assume GPT-2-specific mechanisms transfer to Llama/RoPE models. Query-bias, EPE, positional-identity, and massive-coordinate interventions need architecture-specific analogues and separate causal validation.
