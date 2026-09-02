# ARCUS Module A — handover

Working document for resuming after an interruption. Rewritten at the end of every step.
Not a report: for results see `docs/module_a/FINDINGS_MILESTONE_1.md` and (when written)
`artifacts/<run_id>/a1/summary.md`.

---

## Where we are

**Milestone 2 — A1 blind fact-route discovery. Step 6 of 12 (attribution running).**

Milestone 1 (P0–P5) is complete and pushed: commits `b94fa99..92275bb` on `main`, both
remotes in sync.

## The question A1 answers

Across different ways of asking the same fact, does the model repeatedly use the same
internal causal components, above and beyond generic factual retrieval, syntax, topic, or
lexical overlap?

A null answer is a valid result. It must not be reinterpreted or rescued.

## Non-negotiables for this milestone

- **Sink firewall.** Do not read `Sink-KD` or `Sink-Repro`. Only their recorded SHAs are
  used. No sink head selection, no BOS overlay, nothing called sink-mediated.
- **Do not let the P5 restoration map restrict the search space.** A1 runs all 28 layers.
- **Held-out firewall.** Candidate circuits come from *discovery*-split surfaces only.
  Validation and stress surfaces are untouched until exact validation (Step 10).
- Attribution is candidate discovery, never causal proof.
- Control families are never pooled.

## Confirmed decisions (do not re-litigate)

| Decision | Value |
|---|---|
| IG alignment | End-aligned common suffix (primary) + exact-length subset as sensitivity |
| Pilot facts | M2, K10, K8, K5 — top 4 by A0 reliability, all `K_f = 1.00` |
| Scaling | Extend to all 12 eligible Challenger facts if G4 + exact validation are sane |
| Primary corruption family | `same_topic_fact_swap`; robustness on `semantic_neighbor`, `same_syntax` |
| Push target | Both remotes: `origin` (AlZubayer) and `naveed` (SyedNaveedMahmood) |

---

## Step status

| # | Step | State |
|---|---|---|
| 0 | `handover.md` | done |
| 1 | Freeze P0–P5 | done — `artifacts/freeze/p0_p5_freeze.json` |
| 2 | Formal reverse-modality exclusion | done — `reverse_degenerate_v1` inside the freeze |
| 3 | Pre-answer discovery objective `J_f` | done — `discovery_objective.json` |
| 4 | Discovery-split pairs + accounting | done — 414/432 accepted, 24/24 exact syntax twins |
| 5 | G0 graph and `head_out` hook | done — 700 nodes, decomposition exact to 9.8e-07 |
| 6 | Attribution `eap_ig_node_v1` | **running** — engine committed, corpus run in flight |
| 7 | **Gate G4** (blocks everything downstream) | not started |
| 8 | Route similarity, raw + residual | not started |
| 9 | Candidate circuits | not started |
| 10 | Exact validation on held-out surfaces | not started |
| 11 | Artifacts, tables, summary | not started |
| 12 | Scale to 12 facts, report | not started |

## Commits so far this milestone

| Commit | Step |
|---|---|
| `170af00` | freeze code + `handover.md` |
| `4b5e190` | freeze artifact at `170af00` |
| `cd24c04` | pre-answer discovery objective |
| `80fc645` | discovery-split pairs with accounting |
| `967efa2` | G0 graph and per-head outputs |
| `05efda3` | EAP-IG node attribution engine |
| `7e5c551` | lint cleanup |

## Results established so far

- **`J_f` is fully answer-prefix-free**: common answer prefix is empty on 750/750
  challenger surfaces, so it conditions on the prompt alone. 0 surfaces rejected.
- **`J_f` vs `M_f`**: Spearman 0.49 / 74.3% sign agreement across all 750 surfaces;
  **100% sign agreement (60/60)** on the clean/corrupt deltas, which is the property
  attribution depends on.
- **Pairs**: 414 accepted of 432. `same_syntax` remains the outlier — acceptance 0.79,
  mean Δ +1.72 pre-filter vs +2.88…+3.55 for other families — now with an *exactly*
  augmentation-matched template in all 24 cases.
- **G0**: 700 nodes. `attn_out == Σ_h W_O^h head_out_h` verified on the real model to
  9.8e-07 relative (float32 roundoff); `o_proj.bias` is None at every layer checked.
- **Attribution completeness** on the real model: **1.0004 at m=16**, 1.0018 at m=8.
  ~6.7 s per 700-object vector.

## Gotchas discovered (do not rediscover)

1. Re-running the A0 gate on the discovery split yields **0 eligible facts** — that split
   has one modality and the gate needs two. Eligibility must be read from the freeze.
   `a1-pairs` does this; do not "fix" the A0 shortfall message.
2. `J_f` subtracts a logsumexp over 4 distractors (~log 4 = 1.39 nats of headroom), so
   `J_f < 0` on a clean run does **not** mean a distractor is preferred. Do not filter on it.
3. Attribution `.npz` files are un-ignored explicitly in `.gitignore`; they are the
   primary A1 record, not a regenerable cache.
4. Bash heredocs longer than roughly 200 lines get truncated by the tool. Write a patch
   script to the scratchpad and run it instead.

---

## Environment

- Repo: `e:\arcus\ARCUS`, branch `main`, remotes `origin` + `naveed`.
- Python 3.12.3, torch 2.13.0+cu130, transformers 4.57.6, RTX 3090 24 GB.
- Package installed editable: `pip install -e .`; CLI entry point `arcus-module-a`.
- `pydantic`, `pytest`, `ruff` installed.
- Git identity is set locally; pushing to `naveed` needs the SyedNaveedMahmood token:
  ```bash
  GH_TOKEN=$(gh auth token --user SyedNaveedMahmood)
  git -c credential.helper= \
      -c credential.helper='!f() { echo username=SyedNaveedMahmood; echo password='"$GH_TOKEN"'; }; f' \
      push naveed main
  ```

## Pinned provenance

| Object | Revision |
|---|---|
| `apeleg/SUITE` | `3f5f6b0897dac10baacf1aa8b35319a02abccd23` |
| `apeleg/SUITE-rephrasings` | `81a52d60ec7d3231169b16a54ad1b2a58221ca6e` |
| `meta-llama/Llama-3.2-3B-Instruct` | `0cb88a4f764b7a12671c53f0838cd831a0843b95` |
| Sink-KD (`sink-inheritance-foundation`) | `db114c9c5eb6ffc5de13e444c783408ea7401c62` |
| Sink-Repro (`main`) | `9ab67e914464b13863b67527d8ea14068ee9ff10` |

Run settings: float32, eager attention, eval mode, seed 42, TF32 off.
Prompt policy `llama3_chat_suite_v1` (Llama-3 chat template + upstream SUITE system prompt,
`add_special_tokens=False`; template emits one `<|begin_of_text|>`).

## Existing artifacts

| Path | Contents |
|---|---|
| `artifacts/dataset_audit/` | Gate G0, challenger + all-topics |
| `artifacts/parity-20260902T162133Z-s42/` | Gate G1 instrumentation parity |
| `artifacts/a0-challenger-s42/a0/` | Known-Fact Core, distractor sets, 409 accepted pairs |
| `artifacts/a0-challenger-s42/a2/` | Gate G2 sanity + 1792-intervention restoration map |
| `artifacts/a0-all-topics-s42/a0/` | A0 sweep over all four topics |
| `artifacts/reference/sink_sources_manifest.json` | Sink SHAs + firewall declaration |

## Facts established (do not re-derive)

- **G0 graph = 700 nodes**: 28 layers × 24 heads (head_dim 128) + 28 MLPs.
- `attention_bias=False`, so `attn_out(L) = Σ_h W_O^h · head_out(L,h)` is **exact** — use it
  as a hook-map invariant test.
- Surfaces per eligible fact: 15 discovery/direct, 13–15 validation/direct, 16 stress/indirect.
- Only 63/409 accepted pairs are length-matched (21 same-topic across 9 facts).
- Eligible facts ranked by A0 reliability: M2 +4.08, K10 +3.63, K8 +3.32, K5 +3.31,
  K12 +2.73, K11 +2.12, M1 +2.00, K20 +1.73, K9 +1.61 (all `K_f`=1.00), then K4 0.93,
  K3 0.81, K18 0.81.
- Challenger control pools: 175 syntactic retain rows (Claude-augmentation keyed),
  50 semantic tier-0/1, 100 lexical, 150 GK.
- **Milestone-1 pairs used *validation* clean surfaces.** A1 needs *discovery*-split pairs.
  Bonus: discovery surfaces do have exactly augmentation-matched syntax twins, which
  validation surfaces do not (the D7 caveat in the milestone-1 findings).

## Known open issues carried forward

1. `same_syntax` pairs built on validation surfaces are fallback-quality only; discovery
   surfaces fix this.
2. Only `resid_pre` was scanned in P5; `attn_out`/`mlp_out` implemented but unswept,
   `head_out` not yet implemented.
3. bfloat16 parity, `raw_completion` prompt policy, and multi-seed replication registered
   but unrun.

---

## To resume

```bash
cd e:/arcus/ARCUS
git log --oneline -8
pytest -q
cat handover.md
```

Run ids in play:

| Run id | Contents |
|---|---|
| `a0-challenger-s42` | frozen A0 core, distractor pools, milestone-1 pairs and patch scan |
| `a0-discovery-s42` | discovery-surface scores (clean-anchor correctness filter only) |
| `a1-challenger-s42` | A1 objective, pairs, attribution vectors |

Command sequence to reproduce A1 from scratch:

```bash
arcus-module-a freeze-p5     --config configs/module_a/pilot_challenger.yaml --a0-run a0-challenger-s42
arcus-module-a run-a0        --config configs/module_a/a1_challenger.yaml --run a0-discovery-s42
arcus-module-a a1-objective  --config configs/module_a/a1_challenger.yaml --a0-run a0-challenger-s42 --run a1-challenger-s42
arcus-module-a a1-pairs      --config configs/module_a/a1_challenger.yaml --run a1-challenger-s42                              --correctness-run a0-discovery-s42 --distractors-run a0-challenger-s42
arcus-module-a a1-attribute  --config configs/module_a/a1_challenger.yaml --run a1-challenger-s42                              --distractors-run a0-challenger-s42
```

Then continue at the first step marked `not started` in the table above.
