# Module A — Experimental Protocol

## 1. Canonical factual score

For prompt `q` and answer sequence `y=(y_1,...,y_T)`, use normalized teacher-forced sequence log probability

$$
s(y\mid q)=\frac{1}{T}\sum_{t=1}^{T}\log p(y_t\mid q,y_{<t}).
$$

Given target answer `y_f` and matched distractors `D_f`, define

$$
M_f(q)=s(y_f\mid q)-\log\sum_{y\in D_f}\exp s(y\mid q).
$$

For clean/corrupt pair `(q^+,q^-)`, define the total factual effect

$$
\Delta_f(q)=M_f(q^+)-M_f(q^-).
$$

Every normalized causal metric below is undefined when `|Delta_f|` is below a configured minimum effect. Do not divide by a near-zero denominator and report unstable fractions.

---

## 2. A0 — Known-Fact Core

For each fact:

1. score all preregistered validation surfaces;
2. compute correctness and factual margin by modality;
3. require configured overall and modality-specific reliability;
4. freeze the eligible fact list before A1.

Store excluded facts and reasons. Exclusion is not a hidden preprocessing step.

---

## 3. A1 — Blind route candidate discovery

### 3.1 Computational graph

Represent the model as

$$
G=(V,E).
$$

Start coarse. Recommended first graph:

```text
residual stream -> attention-head outputs -> residual stream
residual stream -> MLP outputs -> residual stream
```

Refine only after a coarse region has evidence.

### 3.2 Candidate attribution

For each discovery clean/corrupt pair, estimate a signed attribution score

$$
a_{f,q}(e)
$$

for candidate edge/node `e` using EAP-IG or a validated equivalent.

Persist the **full ranked vector**, not only the selected top-K circuit.

### 3.3 Cross-formulation invariance

For normalized attribution vectors `a_i,a_j`, compute

$$
S(q_i,q_j)=\cos(a_i,a_j).
$$

Within-fact similarity:

$$
S_{within}(f)=\mathbb E_{i\neq j, q_i,q_j\in Q_f}S(q_i,q_j).
$$

Control-specific between-fact similarity:

$$
S_{between}^{(c)}(f)=\mathbb E_{q\in Q_f,r\in R_f^{(c)}}S(q,r).
$$

Route distinctness for control class `c`:

$$
D_f^{(c)}=S_{within}(f)-S_{between}^{(c)}(f).
$$

Do not collapse control classes until after per-class results are inspected.

### 3.4 Candidate circuit extraction

Candidate extraction must be deterministic from the attribution vector and config.

Support at least:

- top-K edges;
- smallest prefix explaining configured absolute attribution mass;
- threshold by absolute attribution score.

Primary extraction rule must be preregistered. Others are sensitivity analyses.

Output is `CandidateCircuit`, never `FactCircuit` yet.

---

## 4. A2 — Exact causal validation

Exact interventions are the evidence stage.

### 4.1 Necessity

Patch the candidate circuit in the clean run with activation values from the matched corrupt run.

$$
N_f(C)=\frac{M_f(q^+)-M_f(q^+;C\leftarrow q^-)}{\Delta_f(q)}.
$$

Interpretation:

- `0`: patching candidate circuit explains none of the clean-corrupt factual difference;
- `1`: it explains approximately the entire difference;
- values outside `[0,1]` are possible and must not be clipped because interventions can be nonlinear or overcorrect.

### 4.2 Sufficiency

Start from corrupt run and patch only the candidate circuit from clean:

$$
S_f(C)=\frac{M_f(q^-;C\leftarrow q^+)-M_f(q^-)}{\Delta_f(q)}.
$$

### 4.3 Selectivity

Apply the same intervention to matched retain/control examples.

For retain class `c`, define

$$
R_f^{(c)}(C)=\mathbb E_{r\in R_f^{(c)}}|\Delta M_r(C)|.
$$

A descriptive selectivity ratio is

$$
\operatorname{Sel}_f^{(c)}(C)=\frac{|\Delta M_f(C)|}{\epsilon+R_f^{(c)}(C)}.
$$

Always report raw target and retain effects alongside ratios.

### 4.4 Held-out invariance

Circuit selection uses discovery surfaces only. Necessity/sufficiency/selectivity are rerun on held-out validation surfaces.

A route can be called **fact-selective** only if the effect generalizes beyond the surfaces used to discover it.

### 4.5 Minimality / pruning

After a candidate passes validation, iteratively remove the least important edge/component and rerun the exact validation score.

Stop when further removal violates the registered validation floor.

This produces a compact validated circuit while preserving a record of the larger candidate.

---

## 5. A3 — Representation-space localization

A3 is independent of sink information.

### 5.1 Candidate locations

Search representation objects already supported by the backend, for example:

```text
resid_pre
resid_post
attn_out
mlp_out
selected head outputs
```

Search all layers initially or use a route-derived **non-sink** restriction after A2. A2 causal route information is allowed; A4 sink information is not.

### 5.2 Candidate subspace

For fact `f`, location `l`, and discovery states `h_l(q_i)`, form a centered matrix and compute SVD/PCA candidate basis

$$
X_f^{(l)}=[h_l(q_i)-\bar h_f^{(l)}]_i.
$$

Let

$$
B_{f,r}^{(l)}
$$

be the top `r` right-singular directions, with rank selected only from the registered candidate ranks.

Alternative candidate methods may later be added, but the first implementation should remain simple enough to audit.

### 5.3 Representation separability

Measure whether held-out same-fact states are more similar to the candidate subspace than matched controls.

This is candidate evidence only.

### 5.4 Causal projection test

Project out the candidate subspace:

$$
h'_l=h_l-BB^\top h_l.
$$

Measure target and retain factual-margin changes.

### 5.5 Causal patch test

Patch the full candidate representation or the fact-subspace component from clean into corrupt.

Strong evidence requires target restoration on held-out surfaces with limited matched-retain effect.

### 5.6 Reported locus

A locus record includes:

```text
fact_key
layer
component
token_policy
rank
candidate separation
projection effect
patch effect
retain effects
held-out generalization
```

Do not use language such as "the fact is stored exactly here" unless evidence establishes exclusivity, which Module A does not assume.

---

## 6. Freeze barrier before sink analysis

Before A4 begins, write

```text
fact_discovery_freeze.json
```

containing hashes of:

- Known-Fact Core;
- candidate attribution artifacts;
- validated circuits;
- representation candidates/loci;
- configs and code commit.

A4/A5 must reference this freeze artifact. Any later change to A1-A3 creates a new analysis lineage.

---

## 7. A4 — Independent sink/anchor map

A4 uses no target-fact labels when defining sink carriers.

For each layer/head and candidate anchor object, measure where supported:

```text
received attention
anchor key/value norms
projected anchor value/output contribution
anchor identity stability
post-softmax delete effect
post-softmax relocate effect
architecture-specific route interventions
uncertainty across examples
```

Use `docs/SINK_REFERENCE_INTEGRATION.md` for semantics imported from the two sink codebases.

A head can have a strong sink pattern without passing the functional carrier criteria.

---

## 8. A5 — Route x sink intersection

### 8.1 Structural overlap

Let validated fact circuit be `C_f` and independently validated sink-related set be `S`.

Report simple and attribution-weighted overlap, e.g.

$$
I_f^{attr}=\frac{\sum_{e\in C_f\cap S}|a_f(e)|}{\sum_{e\in C_f}|a_f(e)|}.
$$

This is descriptive.

### 8.2 Causal mediation

Partition the fact circuit into sink-associated and non-sink portions:

$$
C_f=C_f^{sink}\cup C_f^{other}.
$$

Intervene exactly on `C_f^sink` and compare with size/attribution-matched subsets of `C_f^other`.

One descriptive mediation fraction is

$$
\operatorname{Med}_f=\frac{|\Delta M_f(C_f^{sink})|}{\epsilon+|\Delta M_f(C_f)|}.
$$

Report the denominator and raw effects. Do not interpret the ratio alone.

### 8.3 Anchor-route analysis

If the architecture exposes a justified decomposition of anchor routing, test whether fact retrieval changes:

- contextual/query-dependent route strength;
- fixed/source-agnostic route strength;
- sink mass;
- projected output.

For GPT-2, use the supplied two-route decomposition only as defined in the reproduction work. Do not transfer it to architectures without the same parameters.

---

## 9. A6 — Statistics

Primary experimental unit: **fact**.

Recommended analyses:

- bootstrap confidence intervals over facts;
- paired bootstrap or permutation tests for within-fact vs matched-control similarity;
- per-fact effect distributions;
- multiple discovery seeds / corruption samples;
- effect sizes, not only p-values;
- no token-row-level pseudo-replication.

All thresholds and primary outcomes must be written into config before the full run.

---

## 10. Stage-level stop rules

Stop or narrow the claim if:

- A0: too few facts are robustly known;
- A1: within-fact routes are not more stable than matched controls;
- A2: attribution candidates fail exact necessity/sufficiency/selectivity;
- A3: representation candidates are decodable but causal interventions fail;
- A4: the model has no validated sink carrier under the chosen definition;
- A5: route and sink show no meaningful causal intersection.

A null result is a result. Module A exists partly to decide whether the sink-based ARCUS premise is worth pursuing.