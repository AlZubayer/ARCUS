# Module A — Data, Splits, and Corruption Design

## 1. Objective

The data layer must let us distinguish **fact identity** from nuisance similarity:

- wording;
- syntax;
- entity/topic;
- answer type;
- lexical overlap;
- query direction/modality.

The route-discovery claim is only meaningful if the same factual association is observed across held-out realizations while matched non-target controls do not produce the same causal pattern.

---

## 2. Canonical dataset handling

The primary dataset is the selected **Forget Narrowly, Retain Broadly / SUITE** resource.

Implementation rule:

> Do not guess the upstream field names from paper/project-page descriptions. Pin the actual dataset/repository revision, inspect the schema, and write one explicit adapter.

The adapter should produce the internal `FactExample` schema defined in `01_SYSTEM_DESIGN.md`.

Required audit outputs:

```text
source location/revision
row count
field names and dtypes
unique topics
unique atomic fact IDs
query modalities
surface-form counts per fact
answer cardinality
retain/control categories
missing values
duplicate questions
duplicate question-answer pairs
cross-split near duplicates
```

The run must stop if fact identity cannot be deterministically reconstructed from source metadata.

---

## 3. Fact identity and surface identity

Define:

```text
fact_key       = stable atomic association
modality       = direct / reverse / indirect / fill-in / other source-defined mode
surface_form   = one concrete wording
```

A surface form is not a new fact.

Use a deterministic `surface_form_id`, e.g. a hash of source row ID plus normalized prompt/answer, only for bookkeeping. Do not infer `fact_id` from text similarity.

---

## 4. Split policy

Module A uses at least three logical partitions:

### Discovery split

Used for:

- Known-Fact screening candidate estimates;
- route attribution/candidate discovery;
- representation candidate search.

### Validation split

Held out from route and representation candidate selection. Used for:

- exact circuit validation;
- route invariance;
- representation intervention validation.

### Stress split

Used only after the initial result is frozen. Contains difficult modalities or transformations not used in candidate selection.

The exact source modalities assigned to each split must be configuration-controlled after the canonical dataset is audited.

**Non-negotiable:** split by surface-realization groups, not random rows, when generated variants are near-duplicates.

---

## 5. Known-Fact Core

A fact is eligible for mechanistic analysis only if the base model robustly knows it.

For fact `f` with validation queries `Q_f`, define

$$
K_f=\frac{1}{|Q_f|}\sum_{q\in Q_f}\mathbf{1}[\text{correct}(q)].
$$

The pipeline should also record per-modality accuracy:

$$
K_{f,m}=\frac{1}{|Q_{f,m}|}\sum_{q\in Q_{f,m}}\mathbf{1}[\text{correct}(q)].
$$

Eligibility should require both:

- overall accuracy above the configured threshold;
- minimum coverage across configured modalities.

Do not lower the threshold after seeing which facts produce convenient circuits without recording a new analysis version.

### Correctness

Use source-provided exact answers where possible, with a documented normalization function for benign formatting variation. For ambiguous/free-form outputs, prefer teacher-forced answer scoring over generated-string matching for the mechanistic outcome.

---

## 6. Distractor-answer design

The factual margin requires plausible alternatives `D_f`.

Priority order:

1. answers to other atomic facts with the same answer type/topic;
2. source-provided alternatives if present;
3. manually/algorithmically constructed matched distractors with frozen rules;
4. generic random answers only as a weak control.

Avoid distractors that are trivially separable by format or length.

For reverse queries, the answer object may change type. Build distractor pools per query family rather than reusing one pool blindly.

---

## 7. Clean/corrupt pair construction

A clean/corrupt pair should change factual identity while preserving nuisance structure as much as practical.

For target fact `f`, define a pair:

```text
clean   = query that elicits f
corrupt = matched query that elicits g != f
```

Preferred constraints, in order:

```text
same modality
same topic or semantic neighborhood when possible
same answer type
similar token length
similar syntactic template
minimal lexical difference consistent with changing fact identity
```

The pair builder must store which constraints were satisfied.

### Pair families

Keep separate categories rather than pooling everything:

- `same_topic_fact_swap`
- `semantic_neighbor`
- `same_syntax`
- `same_lexical_different_meaning`
- `cross_topic_matched`
- `random_token_control`

Random corruption is useful as a robustness check but should not be the primary scientific comparison.

---

## 8. Retain/control sets

For each target fact, build multiple retain rings.

### R0 — Same-fact held-out surfaces

Not retain examples in the unlearning sense; used to test invariance of the discovered route.

### R1 — Same topic, different atomic fact

Most important locality control.

### R2 — Semantic neighbor

Closely related factual content where available.

### R3 — Same syntax/template

Controls whether the route is actually a question-form circuit.

### R4 — Same lexical token, different meaning

Controls entity/name/token-trigger explanations.

### R5 — General knowledge / unrelated

Measures broad utility effects.

All reported selectivity metrics should be broken down by ring before any aggregate is shown.

---

## 9. Leakage prevention

Before model runs, generate `dataset_audit.json` with explicit checks:

```text
no identical prompt in discovery and validation
no identical normalized prompt in discovery and validation
no duplicate generated paraphrase group split across folds
no target answer accidentally present as the corrupt pair's correct answer
fact IDs disjoint where required
no sink-derived annotations in A0-A3 tables
```

If the dataset contains provenance groups for generated paraphrases, use them. If not, create conservative grouping using source metadata; do not invent semantic group labels with an LLM during the primary analysis unless that process is frozen and separately validated.

---

## 10. Pilot selection

The first pilot should be intentionally small and auditable.

Recommended protocol:

```text
one topic
5-10 robustly known atomic facts
all available query modalities for those facts
multiple held-out surface forms
all available same-topic retain facts
```

The goal of the pilot is to verify the causal measurement stack, not estimate population-wide prevalence.

Expansion to all topics happens only after:

- deterministic scoring is validated;
- exact patching passes synthetic sanity checks;
- split leakage audit passes;
- at least one positive and one negative-control circuit behaves as expected.

---

## 11. Data artifacts

Expected outputs from A0:

```text
artifacts/<run_id>/
  manifest.json
  dataset_audit.json
  normalized_examples.jsonl
  fact_index.json
  split_assignments.jsonl
  distractor_sets.jsonl
  clean_corrupt_pairs.jsonl
  known_fact_scores.jsonl
  known_fact_core.json
```

Every downstream artifact must reference `surface_form_id`, `fact_key`, and `pair_id` rather than copying prompt strings as the only identifier.