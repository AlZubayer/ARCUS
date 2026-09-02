# P0-P5 freeze

Frozen 2026-09-02T17:15:00Z at commit `170af0028acae35a27047a28430677efb6f1e225` (tree dirty).

Pin every assumption A1 inherits so an A1 result can be attributed to A1's own choices rather than to a shifted backend, threshold or fact list.

**This file is never edited in place.** A change requires an entry in 'amendments' with a reason, the fields affected, and the commit that made it.

## Pinned revisions

| Object | Identifier | Revision |
|---|---|---|
| Dataset | `apeleg/SUITE` | `3f5f6b0897dac10baacf1aa8b35319a02abccd23` |
| Paraphrases | `apeleg/SUITE-rephrasings` | `81a52d60ec7d3231169b16a54ad1b2a58221ca6e` |
| Model | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Tokenizer | `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |

Run settings: float32, eager attention, seed 42, hook map `llama_hook_map_v1`.

## Policies

- Prompt: `llama3_chat_suite_v1`, `add_special_tokens=False`, BOS from template: True
- Scoring: `answer_score_mean_logprob_v1`, margin `factual_margin_v1`, correctness `correct_over_distractors_v1`
- Distractors: `same_topic_same_modality_v1`, count 4
- Corruption: `pair_families_v1`, families same_topic_fact_swap, semantic_neighbor, same_syntax, same_lexical_different_meaning, cross_topic_matched, random_token_control
- Pair acceptance: `|delta| >= 0.5` and the clean surface must be answered correctly

## Known-Fact Core

12 of 25 facts eligible over 750 scored surfaces, splits ['stress', 'validation'].

| Fact | K_f | mean margin | min margin | modalities |
|---|---|---|---|---|
| `M2` | 1.00 | +4.08 | +2.43 | direct, indirect |
| `K10` | 1.00 | +3.63 | +2.86 | direct, indirect |
| `K8` | 1.00 | +3.32 | +2.08 | direct, indirect |
| `K5` | 1.00 | +3.31 | +2.25 | direct, indirect |
| `K12` | 1.00 | +2.73 | +1.78 | direct, indirect |
| `K11` | 1.00 | +2.12 | +1.42 | direct, indirect |
| `M1` | 1.00 | +2.00 | +0.89 | direct, indirect |
| `K20` | 1.00 | +1.73 | +0.38 | direct, indirect |
| `K9` | 1.00 | +1.61 | +0.67 | direct, indirect |
| `K4` | 0.93 | +1.04 | -0.82 | direct, indirect |
| `K3` | 0.81 | +0.66 | -0.67 | direct, indirect |
| `K18` | 0.81 | +0.60 | -0.79 | direct, indirect |

Excluded: 13 facts, 25 refused cells.

## Reverse-modality exclusion (`reverse_degenerate_v1`)

Reverse-modality cells are excluded from every fact-selective claim. A reverse question asks for the fact from the answer side, and within a topic the answers collapse to a single entity, so a same-topic distractor pool contains the correct answer and the factual margin is undefined.

| Cell | facts | distinct answers | ratio |
|---|---|---|---|
| `challenger_disaster|direct` | 25 | 25 | 1.00 |
| `challenger_disaster|indirect` | 25 | 25 | 1.00 |
| `challenger_disaster|reverse` **degenerate** | 25 | 2 | 0.08 |

No alternate distractor pool is constructed to keep reverse in the primary analysis. It is retained only as a labelled topic-level diagnostic.

## Pair accounting (attempted vs accepted)

| Family | attempted | accepted | rate | mean Δ before | mean Δ after |
|---|---|---|---|---|---|
| `cross_topic_matched` | 96 | 93 | 0.97 | +3.56 | +3.68 |
| `random_token_control` | 24 | 22 | 0.92 | +1.73 | +1.88 |
| `same_lexical_different_meaning` | 96 | 91 | 0.95 | +3.73 | +3.93 |
| `same_syntax` | 24 | 14 | 0.58 | +1.30 | +2.15 |
| `same_topic_fact_swap` | 96 | 94 | 0.98 | +3.53 | +3.60 |
| `semantic_neighbor` | 96 | 95 | 0.99 | +3.29 | +3.32 |

## Tolerances

| Gate | tolerance | observed |
|---|---|---|
| capture-only parity | 0.0 | 0.0 |
| scoring parity | 1e-05 | 3.823910708078415e-07 |
| self-patch, in-situ capture | 0.0 | 0.0 |
| self-patch, cross-shape capture | documented | 9.77217749209558e-06 |

## What A1 may not change

- model/tokenizer revision
- dataset revisions
- prompt policy
- full-sequence factual margin definition
- Known-Fact Core membership
- reverse-modality exclusion rule
- seed

## What A1 introduces

- pre-answer discriminative-token objective (discovery only)
- discovery-split clean/corrupt pairs
- G0 graph with per-head outputs
- attribution vectors

No sink repository is read during A1. Only the recorded SHAs are carried in run manifests.
