# Sink-code integration for Module A

This document records the two sink codebases supplied by the research team and defines how they are allowed to enter ARCUS Module A.

## Upstream sources

### A Sink Without the Plumbing

```bash
git clone --branch sink-inheritance-foundation --single-branch \
  https://github.com/AlZubayer/MechanisticAccountofSinks.git Sink-KD
```

Role in ARCUS: reproduce the sink-inheritance measurements and the distinction between sink pattern, parameter-level route, and causal function. This code is relevant to **A4/A5 only**. It must not seed A1 fact-route discovery.

### Same Sink, Different Plumbing

```bash
git clone https://github.com/AlZubayer/MechanisticAccountofSinks.git Sink-Repro
```

The public main branch contains the original/reproduction analysis entry points including `datasets_loader.py`, `experiments_single_input.py`, `experiments_statistical.py`, and `intervention_analysis.py`.

Role in ARCUS: supply the verified GPT-2 anchor/route intervention battery and the route decomposition used to distinguish a fixed query-bias route from an activation-dependent contextual query route into the same first-position key.

## Scientific firewall

Module A has two logically separate discovery problems:

1. **Fact route discovery**: find a fact-selective causal subcircuit without using sink information.
2. **Sink map discovery**: independently characterize the model's anchor/sink mechanism.

Only after both are validated may A5 compute overlap/mediation. Do not use heads identified by the sink papers as the initial search space for A1. Doing so would make a positive route×sink result partly circular.

## What should be reused

The sink code is most valuable for preserving exact experimental definitions and controls:

- position-0 / anchor received-attention measurements;
- registered layer/head scopes;
- anchor identity perturbations;
- GPT-2 query-bias intervention;
- effective positional-identity swaps/removal;
- key-projection coordinate interventions and matched random controls;
- post-softmax delete and relocate interventions;
- probability-conservation checks;
- causal loss/behavior comparisons, not attention visualization alone.

ARCUS should preserve these semantics where the architecture supports them.

## What must NOT be generalized blindly

The GPT-2 circuit is architecture specific. GPT-2 has learned absolute positional embeddings and a learned query bias. Llama-family models use RoPE and generally do not expose the same query-bias/EPE mechanism. Therefore:

- `nullify_query_bias` is not a valid Llama analogue when no query bias exists;
- GPT-2 EPE interventions cannot simply be renamed as RoPE interventions;
- massive-coordinate identities are checkpoint/run specific;
- position 0 must not automatically be called a literal BOS token;
- a visually strong sink must not be labelled functionally important without intervention evidence.

The supplied sink work itself motivates this caution: the stable phenomenon is anchoring, while the supporting route can differ across models or training runs.

## GPT-2 route decomposition used as a reference

For one head, after dropping target-wise constants that cancel in the softmax comparison, the score to target position `j` can be organized as

$$
(x_i W_Q)\,(x_j W_K)^\top
+
b_Q\,(x_j W_K)^\top.
$$

At the first-position anchor `k_0`, ARCUS records the two query-side terms as

$$
R_B(i)=(x_iW_Q)\cdot k_0
$$

and

$$
R_A=b_Q\cdot k_0.
$$

`R_B` is contextual/activation dependent; `R_A` is fixed across source positions for a given head. Both address the same anchor key in the GPT-2 reproduction. This decomposition is implemented in `src/arcus/module_a/sink_reference.py` as a diagnostic only; it is not treated as a universal sink mechanism.

## A4 implementation target

For each candidate architecture, A4 should output one record per layer/head containing at minimum:

- anchor position/object definition;
- mean received attention under a frozen query-position convention;
- anchor key/value norms where available;
- projected value contribution;
- delete effect;
- relocate effect;
- anchor-identity perturbation effect;
- architecture-specific route probes;
- uncertainty across examples/seeds.

A sink candidate becomes a **causal sink carrier** only when the intervention battery supports that interpretation.

## A5 implementation target

A5 receives two independently produced objects:

- validated fact circuit `C_f` from A1/A2;
- validated sink carrier set `S` from A4.

It reports both structural overlap and causal mediation. Structural overlap alone is insufficient. The decisive test intervenes on the sink-associated portion of the already validated fact circuit and asks how much of the fact margin is lost relative to matched non-sink portions and retain controls.

Possible outcomes are all valid:

1. strong route×sink mediation;
2. mediation only for a subset of facts or retrieval modalities;
3. no meaningful sink involvement.

Outcome 3 falsifies the sink-dependent version of ARCUS for that model and should not be hidden by broadening the sink definition post hoc.

## Coding-agent rule

Before porting implementation details from either upstream repository, record the upstream branch and commit SHA in the run manifest. Prefer small, explicit adapters over copying entire scripts. Preserve the upstream intervention semantics and add unit tests for any rewritten metric.
