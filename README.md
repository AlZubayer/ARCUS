# ARCUS

ARCUS is a mechanistic unlearning research project. The repository is intentionally being built in modules so that each scientific claim is testable independently.

## Module A — Fact Route & Representation Localization

Module A answers three questions **in this order**:

1. **Fact-route existence:** Does a fact-selective causal route exist anywhere in the model?
2. **Representation localization:** Can we identify a hidden-state location/subspace associated with the fact in a way that is causal, not merely decodable?
3. **Sink intersection:** Only after 1–2 are validated, does the fact-selective route intersect the model's attention-sink / anchor routing system?

The ordering is a scientific constraint: sink information must not seed route discovery. Otherwise we risk finding a sink-related pathway simply because that is where we chose to search.

## Primary dataset

Primary target: **SUITE / “Forget Narrowly, Retain Broadly”**.

Module A treats each atomic fact as a unit of analysis and keeps query modality, surface form, topic, and retain-control type separate. Direct, reverse, indirect, paraphrased, semantic-neighbor, syntactic, lexical, and general-knowledge controls are used to distinguish factual identity from surface similarity.

## Core hypothesis

We do **not** assume that one fact has one unique end-to-end path. The intended hypothesis is weaker and falsifiable:

> A known fact may have a stable, fact-selective causal core that is reused across different formulations, together with query-specific readout paths.

Possible valid outcomes include a fact-specific route, shared retrieval backbone plus fact-specific branch, multiple redundant routes, formulation-specific routes, or no sufficiently selective route at the tested granularity.

## Pipeline

```text
A0  Dataset audit + Known-Fact Core
A1  Blind causal-route candidate discovery
A2  Exact causal validation: necessity / sufficiency / selectivity
A3  Representation-space localization + causal representation tests
A4  Independent attention-sink / anchor mapping
A5  Fact-route × sink intersection and mediation
A6  Statistical report and artifacts
```

## Repository layout

```text
configs/module_a/       experiment configuration
src/arcus/module_a/     Module A implementation
docs/                    scientific methodology
scripts/                 thin reproducible entry points
tests/                   offline unit tests
.github/workflows/       CI
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python -m arcus.module_a.cli validate-config configs/module_a/pilot_challenger.yaml
pytest -q
```

The initial scaffold deliberately leaves model-specific causal-intervention code behind explicit interfaces. Attribution scores are candidate-discovery signals; exact intervention is required before the code calls a route causal.

See [`AGENTS.md`](AGENTS.md) and [`docs/MODULE_A_METHODOLOGY.md`](docs/MODULE_A_METHODOLOGY.md) before implementing model backends.
