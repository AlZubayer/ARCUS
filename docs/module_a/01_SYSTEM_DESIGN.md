# Module A — System Design

## 1. Architectural objective

Module A must separate **scientific meaning** from **model-specific mechanics**. The experimental protocol should be identical whether the backend is GPT-2, Llama, or another decoder-only transformer; only the backend adapter and architecture-specific sink probes should change.

The top-level pipeline therefore operates on typed scientific objects rather than raw module names.

---

## 2. Core data objects

### `FactKey`

Stable identity of an atomic fact.

Recommended fields:

```python
@dataclass(frozen=True)
class FactKey:
    topic: str
    fact_id: str
```

Never encode query modality or surface form in `fact_id`.

### `FactExample`

One surface realization.

Required conceptual fields:

```text
fact_key
question
answer
modality
surface_form_id
split
control_type | null
source_row_id
metadata
```

### `CleanCorruptPair`

A matched causal comparison:

```text
pair_id
fact_key
clean_example_id
corrupt_example_id
corruption_type
match_constraints
validation_status
```

### `HookPoint`

Architecture-independent name for a causal intervention location:

```text
layer
component     # resid_pre, attn_out, mlp_out, head_out, q, k, v, pattern, ...
head | null
token_policy
```

### `CandidateCircuit`

A sparse set of attributed graph objects plus the discovery metadata used to obtain it.

### `ValidatedCircuit`

A candidate circuit plus exact necessity, sufficiency, selectivity, held-out invariance, and uncertainty estimates.

### `RepresentationLocus`

```text
fact_key
layer
component
token_policy
basis
rank
candidate_score
projection_effect
patch_effect
retain_effects
heldout_effects
```

### `SinkHeadRecord`

Keep sink description decomposed into:

```text
pattern evidence
anchor identity evidence
route evidence
functional intervention evidence
```

Do not store a single boolean `is_sink` as the only representation.

---

## 3. Components and interfaces

### 3.1 `DatasetAdapter`

Responsibilities:

- load a pinned dataset revision;
- map upstream rows into `FactExample` without guessing;
- preserve source identifiers;
- emit a schema audit;
- fail closed when required fields cannot be resolved.

It must **not** decide whether the model knows a fact.

### 3.2 `PairBuilder`

Responsibilities:

- create matched clean/corrupt pairs;
- enforce corruption constraints;
- record exactly what changed;
- reject pairs that accidentally preserve the same answer/fact;
- expose separate pair families for fact identity, syntax, lexical, semantic-neighbor, and random controls.

### 3.3 `ModelBackend`

Owns all architecture-specific mechanics:

- tokenizer and prompt formatting;
- teacher-forced sequence scoring;
- activation capture;
- exact activation replacement;
- component ablation;
- head-output capture/patching;
- Q/K/V/pattern access where available;
- model revision metadata;
- deterministic generation/scoring settings.

No experiment-stage file should hard-code `model.model.layers[...]` or equivalent paths.

### 3.4 `FactualScorer`

Computes one canonical factual outcome from model logits.

For correct answer `y_f` and distractor set `D_f`:

$$
M_f(q)=s(y_f\mid q)-\log\sum_{y\in D_f}\exp s(y\mid q).
$$

The scorer must return both the aggregate margin and raw per-answer sequence scores for auditing.

### 3.5 `ActivationCache`

Stores activations indexed by:

```text
run_id / example_id / hook_point / tensor_role
```

It should support chunked disk-backed storage. Real-model runs must not assume all activations fit in RAM.

### 3.6 `RouteDiscoverer`

Input:

- clean/corrupt pairs;
- factual outcome metric;
- allowed graph granularity.

Output:

- ranked edges/nodes with signed attribution;
- complete discovery configuration;
- no causal-language labels.

Initial implementation target: EAP-IG or a validated equivalent.

### 3.7 `ExactCircuitValidator`

Takes a candidate circuit and performs exact interventions.

Must implement at least:

- clean -> corrupted circuit patch for necessity;
- corrupted -> clean circuit patch for sufficiency;
- matched retain/control interventions;
- held-out surface-form validation.

### 3.8 `RepresentationLocalizer`

Two-phase design:

1. candidate search: cross-formulation low-rank structure / probes / subspace statistics;
2. causal validation: projection/removal and representation patching.

A successful probe alone may only produce `RepresentationCandidate`, not `RepresentationLocus`.

### 3.9 `SinkMapper`

Runs only after A1-A3 outputs are frozen.

Backend-specific implementations may use the supplied sink repositories, but output must conform to a common sink record schema.

### 3.10 `RouteSinkAnalyzer`

Consumes only:

- `ValidatedCircuit` from A2;
- frozen representation results from A3;
- independent sink map from A4.

Produces:

- structural overlap;
- attribution-weighted overlap;
- exact sink-portion mediation;
- matched non-sink mediation control;
- retain effects.

---

## 4. Dependency firewall

Allowed dependencies:

```text
A0 -> A1 -> A2 -> A3
                 |
                 v
           freeze fact results
                 |
A0 ------------> A4
A2 + A3 + A4 --> A5
A0..A5 --------> A6
```

Forbidden dependencies:

```text
A4 sink heads -> A1 search space      # forbidden
A4 sink scores -> A3 candidate layers # forbidden
A5 result -> redefine A1 circuit      # forbidden without a new registered analysis
```

The easiest implementation is to make A4 require a frozen A1-A3 manifest hash. A5 should refuse to run if the hashes do not match the artifacts it consumes.

---

## 5. Graph granularity strategy

Do not begin at Q/K/V edge granularity for the entire model. Use a staged refinement:

```text
Stage G0: layer residual + attn_out + mlp_out
Stage G1: per-head attn output + MLP output
Stage G2: selected path edges between validated G1 components
Stage G3: Q/K/V/attention-pattern analysis only inside the selected region
```

Reason: the scientific question is whether a selective causal route exists, not whether we can generate the largest possible attribution tensor.

A1 should first identify a compact region, then refine it.

---

## 6. Token-position policy

Every intervention must state which token positions are modified. Supported policies should be explicit objects, for example:

```text
answer_last_prompt_token
subject_span
relation_span
all_prompt_tokens
final_k_prompt_tokens
anchor_position
explicit_indices
```

Do not silently patch all sequence positions because tensor shapes happen to match.

For cross-formulation analyses, prefer position policies that remain semantically comparable when token counts differ.

---

## 7. Reproducibility contract

Every stage writes a manifest containing:

```text
run_id
parent_run_ids
model_id
model_revision
tokenizer_id
tokenizer_revision
dataset_id
dataset_revision
config_path
config_sha256
code_commit_sha
seed
dtype
device
software versions
prompt-template version
scoring version
corruption-policy version
hook-map version
```

A scientific artifact without a manifest is incomplete and must not be used downstream.

---

## 8. Recommended source layout

The existing `src/arcus/module_a/` package can grow toward:

```text
module_a/
  config.py
  schema.py
  suite.py
  scoring.py
  pairing.py
  backend/
    base.py
    hf.py
    hook_maps.py
  cache.py
  discovery/
    base.py
    eap_ig.py
  validation/
    patching.py
    circuits.py
  representation/
    candidates.py
    causal.py
  sink/
    base.py
    gpt2.py
    sink_references.py
  intersection.py
  statistics.py
  artifacts.py
  pipeline.py
  cli.py
```

Do not refactor immediately just to match this tree. Migrate when the first backend becomes real so changes are driven by working interfaces, not aesthetics.