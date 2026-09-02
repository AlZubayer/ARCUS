# Module A — Artifacts, Schemas, and Metrics

## 1. Principle

Every scientific stage must emit a machine-readable artifact that can be inspected independently of plots or notebooks.

Plots are views over artifacts; they are not the primary record.

---

## 2. Run directory

Recommended layout:

```text
artifacts/<run_id>/
  manifest.json
  config_resolved.yaml
  a0/
    dataset_audit.json
    normalized_examples.jsonl
    fact_index.json
    split_assignments.jsonl
    distractor_sets.jsonl
    clean_corrupt_pairs.jsonl
    known_fact_scores.jsonl
    known_fact_core.json
  a1/
    attribution_rows.jsonl
    attribution_vectors/
    candidate_circuits.jsonl
    route_similarity.jsonl
  a2/
    exact_interventions.jsonl
    validated_circuits.jsonl
    pruning_trace.jsonl
  a3/
    representation_candidates.jsonl
    representation_interventions.jsonl
    validated_loci.jsonl
    bases/
  freeze/
    fact_discovery_freeze.json
  a4/
    sink_head_records.jsonl
    sink_interventions.jsonl
    validated_sink_carriers.jsonl
  a5/
    route_sink_overlap.jsonl
    route_sink_mediation.jsonl
  a6/
    statistics.json
    summary_tables/
    figures/
```

Large tensors should be stored in `.npy`, `.npz`, safetensors, or chunked array formats with references from JSONL rather than embedded into JSON.

---

## 3. `manifest.json`

Required fields:

```json
{
  "run_id": "...",
  "created_at": "...",
  "code_commit_sha": "...",
  "model": {
    "id": "...",
    "revision": "...",
    "tokenizer_id": "...",
    "tokenizer_revision": "...",
    "dtype": "...",
    "attention_backend": "..."
  },
  "dataset": {
    "id": "...",
    "revision": "...",
    "source_hash": "..."
  },
  "config_sha256": "...",
  "seed": 0,
  "software": {},
  "parent_run_ids": [],
  "stage": "a0"
}
```

No downstream stage should accept an artifact whose manifest is absent or whose parent hashes do not match the configured inputs.

---

## 4. A0 schemas

### `known_fact_scores.jsonl`

One row per surface form:

```json
{
  "fact_key": {"topic": "...", "fact_id": "..."},
  "surface_form_id": "...",
  "modality": "...",
  "split": "validation",
  "correct_answer": "...",
  "answer_scores": {"answer A": -1.2, "answer B": -3.5},
  "factual_margin": 2.3,
  "is_correct": true
}
```

### `clean_corrupt_pairs.jsonl`

```json
{
  "pair_id": "...",
  "target_fact_key": {},
  "clean_surface_id": "...",
  "corrupt_surface_id": "...",
  "corrupt_fact_key": {},
  "corruption_type": "same_topic_fact_swap",
  "constraints": {
    "same_modality": true,
    "same_topic": true,
    "same_answer_type": true,
    "token_length_delta": 2
  }
}
```

---

## 5. A1 schemas

### `attribution_rows.jsonl`

One row per attributed object per pair:

```json
{
  "pair_id": "...",
  "fact_key": {},
  "surface_form_id": "...",
  "object_id": "L12.H7.head_out",
  "object_type": "head_out",
  "layer": 12,
  "head": 7,
  "signed_score": 0.031,
  "abs_score": 0.031,
  "discovery_method": "eap_ig",
  "method_params": {}
}
```

### `candidate_circuits.jsonl`

```json
{
  "circuit_id": "...",
  "fact_key": {},
  "selection_rule": "attribution_mass_prefix",
  "selection_value": 0.85,
  "objects": ["..."],
  "discovery_surface_ids": ["..."],
  "total_abs_attribution": 1.0,
  "selected_abs_attribution": 0.86
}
```

---

## 6. A2 schemas

### `exact_interventions.jsonl`

Every exact intervention is one auditable record:

```json
{
  "intervention_id": "...",
  "circuit_id": "...",
  "pair_id": "...",
  "direction": "clean_to_corrupt",
  "mode": "path_consistent",
  "target_surface_id": "...",
  "source_surface_id": "...",
  "patched_objects": ["..."],
  "baseline_margin": 2.1,
  "source_margin": -0.4,
  "intervened_margin": 0.5,
  "full_effect": 2.5,
  "raw_effect": -1.6,
  "normalized_effect": 0.64
}
```

Do not clip `normalized_effect` to `[0,1]`.

### `validated_circuits.jsonl`

Summaries by fact and circuit:

```text
necessity mean/CI
sufficiency mean/CI
selectivity by control ring
held-out-form effects
modalities covered
object count
pruned object count
validation status
```

---

## 7. A3 schemas

### `representation_candidates.jsonl`

```text
fact_key
location_id
layer
component
token_policy
candidate_method
rank
train_surface_ids
heldout_surface_ids
candidate separation metrics
basis_file
```

### `representation_interventions.jsonl`

```text
fact_key
location_id
surface_form_id
intervention_type        # projection_out / full_patch / subspace_patch / random_subspace
rank
target_margin_before
target_margin_after
raw_target_effect
retain_class
retain_effect
basis_file
```

### `validated_loci.jsonl`

A locus is validated only when causal criteria are met. Include all failed candidates as separate records rather than deleting them.

---

## 8. A4/A5 schemas

### `sink_head_records.jsonl`

One row per layer/head/anchor definition:

```text
layer
head
anchor_type
anchor_position_policy
received_attention
projected_value_norm
projected_output_norm
anchor_identity_stability
delete_effect
relocate_effect
architecture_route_probes
pattern_pass
function_pass
carrier_status
```

### `route_sink_mediation.jsonl`

```text
fact_key
validated_circuit_id
sink_map_id
sink_objects
non_sink_matched_objects
full_circuit_effect
sink_subset_effect
matched_non_sink_effect
retain_effects
mediation_fraction
```

---

## 9. Canonical metrics

### Factual margin

$$
M_f(q)=s(y_f\mid q)-\log\sum_{y\in D_f}\exp s(y\mid q).
$$

### Full clean-corrupt effect

$$
\Delta_f=M_f(q^+)-M_f(q^-).
$$

### Necessity

$$
N_f(C)=\frac{M_f(q^+)-M_f(q^+;C\leftarrow q^-)}{\Delta_f}.
$$

### Sufficiency

$$
S_f(C)=\frac{M_f(q^-;C\leftarrow q^+)-M_f(q^-)}{\Delta_f}.
$$

### Route similarity

$$
\operatorname{RouteSim}(q_i,q_j)=\cos(a_i,a_j).
$$

### Route distinctness

$$
D_f^{(c)}=S_{within}(f)-S_{between}^{(c)}(f).
$$

### Causal selectivity

For control class `c`:

$$
\operatorname{Sel}_f^{(c)}=\frac{|E_f|}{\epsilon+\mathbb E_{r\in R_f^{(c)}}|E_r|}.
$$

Always report `E_f` and retain-effect distribution, not just the ratio.

### Representation projection selectivity

Same form as causal selectivity, where `E_f` is the factual-margin change under projection/removal.

### Structural sink overlap

$$
I_f^{attr}=\frac{\sum_{e\in C_f\cap S}|a_f(e)|}{\sum_{e\in C_f}|a_f(e)|}.
$$

### Sink mediation

$$
\operatorname{Med}_f=\frac{|E_f(C_f^{sink})|}{\epsilon+|E_f(C_f)|}.
$$

Again, report raw effects and matched non-sink controls.

---

## 10. Statistical outputs

The `statistics.json` file should preserve both aggregate and per-fact information.

Minimum outputs:

```text
number of eligible facts
number of validated fact circuits
route distinctness by control class
bootstrap CI of median/mean per-fact effects
permutation p-value for within vs matched-between route similarity
necessity/sufficiency distributions
selectivity by retain ring
representation causal-effect distributions
fraction of validated circuits with sink structural overlap
fraction with positive causal sink mediation
```

Do not make the headline result depend only on the fraction of heads/edges passing an arbitrary threshold.

---

## 11. Figure inputs

Figures should be generated from machine-readable summary tables.

Recommended first figures:

1. **Route similarity matrix** ordered by fact, modality, and control class.
2. **Within-vs-between route distinctness** per fact.
3. **Circuit validation scatter**: necessity vs sufficiency, color by selectivity.
4. **Representation locus map**: layer x component with causal projection/patch effect.
5. **Sink map**: pattern strength vs causal delete/relocate effect.
6. **Route x sink mediation**: full circuit effect vs sink-subset effect.

Every plotted point should be traceable back to an artifact row ID.

---

## 12. Versioning metrics

Metric definitions are part of the experimental protocol. Give each metric implementation a version string, e.g.

```text
factual_margin_v1
necessity_v1
sufficiency_v1
route_similarity_cosine_v1
representation_projection_v1
sink_mediation_v1
```

If the formula changes, increment the version and do not overwrite old artifacts.