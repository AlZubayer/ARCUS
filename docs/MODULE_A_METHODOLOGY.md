# ARCUS Module A — Scientific Methodology

## 1. Scope

Module A is a **measurement and causal-localization module**. It does not perform unlearning yet.

It asks, in order:

1. Does a fact-selective causal route exist?
2. Can the associated fact be localized in representation space in a causally meaningful way?
3. Does the validated route intersect attention-sink / anchor routing?

The phrase **unique route** is deliberately avoided as an assumption. Valid outcomes include:

- a fact-specific route;
- a shared factual-retrieval backbone plus fact-specific branches;
- multiple redundant routes;
- formulation-specific routes with no stable factual core;
- no sufficiently selective route at the tested granularity.

---

## 2. A0 — Dataset audit and Known-Fact Core

Let each atomic fact be indexed by `fact_id = (topic, label_base)`. Query modality and surface form are separate variables.

For fact `f`, let `Q_f^test` contain held-out formulations. Define base-model knowledge reliability

$$
K_f = \frac{1}{|Q_f^{test}|}\sum_{q\in Q_f^{test}} \mathbf 1[\text{model answers }q\text{ correctly}].
$$

Only facts satisfying the preregistered `minimum_base_accuracy` and modality-coverage threshold enter mechanistic analysis. This prevents “no route found” from being confused with “the model did not robustly know the fact”.

Splits are made by **surface realization**, not random row, so near-duplicate paraphrases cannot leak between discovery and validation.

---

## 3. Factual outcome metric

For a query `q` with correct answer sequence `y_f=(y_1,...,y_T)`, use teacher-forced normalized sequence log probability

$$
s(y_f\mid q)=\frac{1}{T}\sum_{t=1}^{T}\log p(y_t\mid q,y_{<t}).
$$

Given matched distractor answers `D_f`, define the factual margin

$$
M_f(q)=s(y_f\mid q)-\log\sum_{y\in D_f}\exp s(y\mid q).
$$

A clean/corrupted pair `(q^+,q^-)` defines the full factual effect

$$
\Delta_f(q)=M_f(q^+)-M_f(q^-).
$$

Corruptions should preserve modality, syntax, and topic whenever possible while changing fact identity. Random-token corruption is a control, not the default scientific comparison.

---

## 4. A1 — Blind causal-route candidate discovery

Represent the transformer as a directed computational graph

$$
G=(V,E),
$$

with nodes such as attention-head outputs, MLP outputs, residual-stream states, and optionally finer query/key/value objects. Edges represent transmitted activation.

For each fact and discovery surface form, estimate an edge attribution vector

$$
a_{f,q}\in\mathbb R^{|E|}.
$$

The initial implementation may use EAP-IG or a comparably validated attribution-patching approximation. This stage only ranks **candidate** edges.

For two surface forms, define route similarity from normalized attribution vectors, e.g.

$$
S(q_i,q_j)=\cos(a_{f,q_i},a_{f,q_j}).
$$

Then

$$
S_{within}(f)=\mathbb E_{q_i,q_j\in Q_f}S(q_i,q_j),
$$

and matched between-fact similarities are computed separately for same-topic, semantic-neighbor, same-syntax, same-lexical, and unrelated controls.

A basic route-distinctness statistic is

$$
D_f=S_{within}(f)-S_{between}(f).
$$

The preferred structural hypothesis is

$$
C_{f,q}=C_f^{core}\cup C_{f,q}^{readout},
$$

where a stable factual core is shared across modalities while some downstream readout is query-dependent.

---

## 5. A2 — Exact causal validation

Candidate circuits must be validated with exact interventions.

### Necessity

Let `C_f` be a candidate circuit. Corrupt/ablate it in the clean execution:

$$
N_f(C_f)=\frac{M_f(q^+)-M_f(q^+;C_f\leftarrow corrupt)}{\Delta_f(q)}.
$$

### Sufficiency

Start from the corrupted execution and restore only `C_f`:

$$
S_f(C_f)=\frac{M_f(q^-;C_f\leftarrow clean)-M_f(q^-)}{\Delta_f(q)}.
$$

### Causal selectivity

For matched retain/control queries `R_f`, define

$$
\operatorname{Selectivity}(f,C_f)=\frac{|\Delta_f(C_f)|}{\epsilon+\mathbb E_{r\in R_f}|\Delta_r(C_f)|}.
$$

A causal-route claim requires held-out-form necessity/sufficiency plus selectivity over matched controls. Attribution overlap alone is insufficient.

---

## 6. A3 — Representation-space localization

At candidate locations `l`, collect hidden states for multiple formulations of the same known fact:

$$
H_f^{(l)}=[h_l(q_1),...,h_l(q_n)].
$$

A baseline low-rank estimate uses centered SVD

$$
H_f^{(l)}-\bar h_f^{(l)}=U\Sigma V^\top,
$$

with candidate basis `B_f^{(l)}=V_{1:r}`.

This is **not** yet evidence that the fact “resides” there. Two causal tests are required.

### Projection-out test

$$
h'_l=h_l-B_fB_f^\top h_l.
$$

The intervention should reduce target-fact margin more than matched retain margins on held-out formulations.

### Representation patch test

Patch the candidate representation from a clean fact run into a matched corrupted run. Restoration of factual margin provides stronger evidence that the representation participates causally in retrieval.

Module A should report a **causal fact-representation locus** as a tuple such as

`(layer, component, token_policy, rank, necessity, sufficiency, selectivity)`

rather than claiming a single storage neuron unless the evidence genuinely supports that granularity.

---

## 7. A4 — Independent attention-sink / anchor mapping

This stage runs **only after A1–A3 are frozen**.

For each layer/head and candidate anchor (BOS and/or first position), measure:

- received attention mass;
- anchor identity stability across content;
- projected value/output contribution;
- post-softmax delete effect;
- relocation effect;
- route-level sensitivity where architecture-specific interventions are justified.

A high attention column is not automatically functionally important. Sink pattern, implementation, and causal function are stored as separate fields.

---

## 8. A5 — Fact-route × sink intersection

Let `C_f` be the validated fact circuit and `S_sink` the independently mapped sink-related edge set.

An attribution-weighted descriptive overlap is

$$
I_f^{attr}=\frac{\sum_{e\in C_f\cap S_{sink}}|a_f(e)|}{\sum_{e\in C_f}|a_f(e)|}.
$$

This is descriptive only. The key experiment is **causal sink mediation**: intervene on the sink-associated portion of the validated fact circuit and measure what fraction of the circuit’s factual effect disappears, while repeating the same intervention on matched retained facts.

Possible outcomes are all valid:

- strong sink mediation;
- sink mediation only for a subset of fact types/modalities;
- overlap without mediation;
- essentially no sink intersection.

A null result here means the sink-based ARCUS mechanism must be reconsidered rather than forced.

---

## 9. Statistics and preregistration

Primary experimental unit is the **fact**, not the token row.

Report:

- per-fact distributions;
- bootstrap confidence intervals across facts/surface forms as appropriate;
- permutation tests comparing within-fact route similarity with matched controls;
- multiple-seed replication for discovery thresholds and corruption sampling;
- all preregistered thresholds and any post-hoc changes explicitly.

No claim should depend only on pooled token-level significance.

---

## 10. Module A success criterion

A strong positive result is not “we found a pretty attention path.” It is:

> Factual identity predicts a stable causal core across held-out formulations; intervening on that core selectively changes retrieval; a corresponding representation locus passes causal projection/patch tests; and only then a measurable fraction of that computation is shown to be mediated by the independently identified sink-routing system.

If any stage fails, the failure determines which ARCUS assumption must be revised.
