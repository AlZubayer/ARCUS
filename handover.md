# ARCUS Module A — handover

Working document for resuming after an interruption. Rewritten at the end of every step.
Not a report: for results see `docs/module_a/FINDINGS_MILESTONE_1.md` and (when written)
`artifacts/<run_id>/a1/summary.md`.

---

## Where we are

**Milestone 2 — A1 blind fact-route discovery. COMPLETE (all 12 steps).**

Result: **generic factual backbone + a fact-distinctive but causally NON-selective
signature.** RQ-A1's fact-selective causal route is not established. Full write-up:
`artifacts/a1-challenger-s42/a1/summary.md`.

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
| 2 | Reverse-modality exclusion | done — `reverse_degenerate_v1` |
| 3 | Pre-answer objective `J_f` | done — prefix empty on 750/750 |
| 4 | Discovery-split pairs | done — 1222/1296, 70/72 exact syntax twins |
| 5 | G0 graph + `head_out` | done — 700 nodes, decomposition exact |
| 6 | Attribution `eap_ig_node_v1` | done — 95 vectors, completeness 1.0004 |
| 7 | **Gate G4** | **PASSED** — all six criteria |
| 8 | Route similarity raw + residual | done — within +0.672 vs controls +0.17…+0.24 |
| 9 | Candidate circuits | done — 3 rules × 4 facts |
| 10 | Exact validation | done — N 0.74, S 0.89, **selectivity fails** |
| 11 | Artifacts, tables, summary | done — `summary.md`, `tables/` |
| 12 | Scale to 12 facts | done — similarity strengthens (D +0.53…+0.56) |

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
| `dd806f9` | attribution over targets + all five control classes |
| `75c64fd` | fix: symmetric backbone removal on both sides of every cosine |
| `535f1c5` | Gate G4 passes, with the per-fact caveat |
| `e873db2` | exact validation, circuit extraction, A1 result |

## A1 result (final)

**Two findings that point in different directions. Both are reported.**

1. Attribution vectors **are** fact-distinctive. Within-fact route similarity +0.672 vs
   every control class +0.17…+0.24 after backbone removal; all p ≈ 0.0002. Extending to
   12 facts strengthens it (D +0.53…+0.56, controls near zero).
2. The circuits those vectors yield are **not** fact-selective. Necessity 0.74,
   sufficiency 0.89 on held-out surfaces, but every selectivity ratio sits between 0.33
   and 1.89, and *below 1* for semantic neighbours and lexical controls.

Gate G4 passed on all six criteria (completeness 1.00, sign 0.75, top-over-random 31×,
step stability 1.00, 3/3 families, 0 firewall leaks). Per fact, attribution reliably finds
a *set* of causally important objects (6–75× random) but its *ordering* within that set is
unreliable (K5 at chance) — so it is used only as a screening device.

## Earlier milestone facts (still valid)

- `J_f` conditions on the prompt alone: shared answer prefix empty on 750/750 surfaces.
  vs `M_f`: Spearman 0.49 / 74% sign agreement on surfaces, **100% (60/60)** on
  clean/corrupt deltas — the property attribution actually depends on.
- G0 = 700 nodes; `attn_out == Σ_h W_O^h head_out_h` exact to 9.8e-07 on the real model.
- Attribution completeness 1.0004 at m=16; ~7 s per 700-object vector.
- `same_syntax` is the outlier family throughout: acceptance 0.71, mean Δ +1.37 vs
  +2.9…+4.2 elsewhere, now with exactly matched templates.

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
5. Residualising only one side of a similarity comparison inflates distinctness badly
   (D went 0.78 -> 0.48 when fixed). Both sides must have the same background subtracted,
   estimated from facts on neither side.
6. Selectivity rings must sample one pair per *distinct* control fact and score each
   control on its OWN prompt and pool. Sampling by pair id gave four pairs of one fact.
7. The R4 lexical ring is character-manipulation questions, not factual retrieval. Large
   fragile margins; do not let it carry a conclusion.
8. `arcus-module-a a1-similarity` overwrites `route_similarity_*.jsonl`. The 12-fact
   outputs are kept as `*_12facts.*`; re-run with `--name vectors` to restore the primary.

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

### Next milestone: A3 representation localization

A3 must not assume a validated circuit exists — A1 did not produce one. The question it
inherits is the dissociation above: **is the fact-distinctive signal that route similarity
detects present in representation space, and causally localisable there when the
component-level circuit was not?**

Concretely: collect hidden states at `last_prompt_token` across formulations of each
Known-Fact-Core fact; estimate a candidate subspace by centred SVD leaving out validation
surfaces; run both causal tests — projection-out `h − BBᵀh` and subspace-only restoration
`h_corrupt + BBᵀ(h_clean − h_corrupt)` — against equal-rank random-subspace controls and
the same five retain rings. Selectivity, not decodability, is the criterion, and it is
exactly what A1 failed at component granularity.

A4 and any sink analysis stay out of scope until A1–A3 are frozen.
