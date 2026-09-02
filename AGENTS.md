# Coding-agent handoff: ARCUS Module A

## Goal

Build Module A as a reproducible mechanistic experiment, **not as an unlearning method yet**.

Research questions:

- **RQ-A1:** Does factual identity predict a stable causal subcircuit above and beyond lexical, syntactic, entity, topic, and query-modality similarity?
- **RQ-A2:** Is there a hidden-state location/subspace that is invariant across formulations of the same fact and causally relevant to factual recall?
- **RQ-A3:** After RQ-A1/RQ-A2 are validated blindly, does the fact-selective circuit overlap or interact with attention-sink / anchor routing?

## Non-negotiable scientific rules

1. **Do not use sink heads to seed A1 discovery.** Sink mapping is A4.
2. **Attribution is discovery, not causal evidence.** Exact patching/ablation is required for A2 claims.
3. **Decodability is not storage/localization.** Probe accuracy must be followed by projection or patch interventions.
4. Every fact-level claim must generalize to **held-out surface forms**.
5. Controls must separately test semantic neighbors, same syntax, same lexical tokens, topic sharing, and unrelated general knowledge.
6. Preserve `fact_id`, `topic`, `modality`, `surface_form_id`, and `control_type` as separate fields in every artifact.
7. Cache model revision, tokenizer revision, dataset revision, config hash, seed, prompt text hash, and intervention specification.
8. Never change scoring, token aggregation, corruption policy, graph granularity, or circuit thresholds after inspecting results without making the change explicit and versioned.
9. A null result is valid. Do not force “one fact = one path”.
10. Sink **pattern**, **circuit**, and **function** are separate measurements.

## Recommended implementation order

1. Implement/verify the SUITE loader and write a dataset-audit artifact.
2. Implement a Hugging Face backend for deterministic teacher-forced sequence scoring.
3. Implement activation capture for residual stream, head outputs, MLP outputs, queries/keys/values, and attention probabilities.
4. Implement exact residual activation patching on clean/corrupted pairs.
5. Add head- and MLP-level patching.
6. Implement attribution candidate ranking behind `RouteDiscoverer` (EAP-IG or a carefully validated equivalent).
7. Extract compact candidate circuits and validate necessity, sufficiency, selectivity, and held-out-form invariance.
8. Implement representation localization and causal projection/patch tests.
9. Only then implement sink/anchor mapping and route×sink mediation.
10. Add permutation/bootstrap statistics and preregistered report generation.

## First-pilot definition of done

For at least 5 robustly-known facts from one SUITE topic:

- base model answers held-out direct/reverse/indirect formulations reliably;
- candidate circuit is discovered with no sink information;
- exact interventions establish non-trivial necessity and sufficiency;
- within-fact route similarity exceeds matched-control similarity;
- causal selectivity is quantified on retain neighbors;
- representation localization survives a causal intervention test;
- sink intersection/mediation is reported, including a null result if absent.

## First coding task

Implement `HFBackend` + dataset audit + exact residual-stream patching before EAP-IG. A fast attribution implementation without a trustworthy exact-intervention backend is scientifically backwards for this project.
