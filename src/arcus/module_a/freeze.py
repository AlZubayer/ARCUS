"""Freeze the validated P0-P5 setup before A1 begins.

The freeze pins every assumption A1 inherits, so that a later A1 result can be attributed to
A1's own choices rather than to a quietly shifted backend, threshold or fact list.

The file is never edited in place. A change requires an entry in ``amendments`` carrying a
reason, the fields affected, and the commit that made it, so the lineage stays readable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import git_state, sha256_file, software_versions, utc_now
from .config import ModuleAConfig, config_sha256

FREEZE_VERSION = "p0_p5_freeze_v1"


def _hash_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "present": False, "sha256": None}
    return {"path": str(path), "present": True, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pair_accounting(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Attempted vs accepted counts per family.

    Milestone 1 persisted every attempted pair with its rejection reason, so the accounting
    is recoverable here rather than lost to a filter.
    """
    from collections import Counter, defaultdict
    import statistics

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        by_family[row["family"]].append(row)

    out: dict[str, Any] = {}
    for family, group in sorted(by_family.items()):
        accepted = [r for r in group if r["validation_status"] == "accepted"]
        deltas_all = [r["delta"] for r in group if r.get("delta") is not None]
        deltas_acc = [r["delta"] for r in accepted if r.get("delta") is not None]
        out[family] = {
            "n_attempted": len(group),
            "n_accepted": len(accepted),
            "acceptance_rate": round(len(accepted) / len(group), 4) if group else 0.0,
            "rejection_reasons": dict(
                Counter(r["rejection_reason"] for r in group if r.get("rejection_reason")).most_common()
            ),
            "delta_before_filtering": {
                "mean": round(statistics.fmean(deltas_all), 4) if deltas_all else None,
                "median": round(statistics.median(deltas_all), 4) if deltas_all else None,
                "n": len(deltas_all),
            },
            "delta_after_filtering": {
                "mean": round(statistics.fmean(deltas_acc), 4) if deltas_acc else None,
                "median": round(statistics.median(deltas_acc), 4) if deltas_acc else None,
                "n": len(deltas_acc),
            },
        }
    return out


def build_freeze(
    config: ModuleAConfig,
    config_path: str | Path,
    *,
    artifact_root: str | Path,
    a0_run_id: str,
    parity_run_id: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """Assemble the P0-P5 freeze record."""
    root = Path(artifact_root)
    topic = topic or config.dataset.topic
    a0_dir = root / a0_run_id / "a0"
    a2_dir = root / a0_run_id / "a2"

    core = _load_json(a0_dir / f"known_fact_core_{topic}.json") or {}
    audit = _load_json(root / "dataset_audit" / f"dataset_audit_{topic}.json") or {}
    pairs = _read_jsonl(a0_dir / f"clean_corrupt_pairs_{topic}.jsonl")
    sanity = _load_json(a2_dir / "patch_sanity.json") or {}

    parity_path = None
    if parity_run_id:
        parity_path = root / parity_run_id / "parity" / "instrumentation_parity.json"
    else:
        candidates = sorted(root.glob("parity-*/parity/instrumentation_parity.json"))
        parity_path = candidates[-1] if candidates else None
    parity = _load_json(parity_path) if parity_path else None

    def check(report: dict[str, Any] | None, name: str) -> dict[str, Any] | None:
        if not report:
            return None
        for entry in report.get("checks", []):
            if entry["check"] == name:
                return entry
        return None

    capture_check = check(parity, "capture_only_hooks_are_numerically_neutral")
    scoring_check = check(parity, "backend_scoring_matches_uninstrumented_hf")
    resid_check = check(parity, "resid_post_L_equals_resid_pre_L_plus_1")

    facts = core.get("facts", [])
    eligible = [f for f in facts if f.get("eligible")]
    excluded = [f for f in facts if not f.get("eligible")]

    degeneracy = (audit.get("answers", {}) or {}).get("degeneracy_by_topic_modality", {})
    degenerate_cells = sorted(
        cell for cell, info in degeneracy.items() if not info.get("usable_for_same_topic_distractors")
    )

    return {
        "freeze_version": FREEZE_VERSION,
        "created_at": utc_now(),
        "scope": "P0-P5: dataset audit, deterministic backend, parity, Known-Fact Core, pairs, exact patching",
        "purpose": (
            "Pin every assumption A1 inherits so an A1 result can be attributed to A1's own "
            "choices rather than to a shifted backend, threshold or fact list."
        ),
        "amendment_policy": (
            "This file is never edited in place. A change requires an entry in 'amendments' "
            "with a reason, the fields affected, and the commit that made it."
        ),
        "amendments": [],
        "code": git_state("."),
        "software": software_versions(),
        "config": {
            "path": str(config_path),
            "sha256": config_sha256(config_path),
            "config_version": config.config_version,
            "seed": config.experiment.seed,
        },
        "seeds": {
            "experiment_seed": config.experiment.seed,
            "distractor_draw": "seeded per cell by (seed, fact_key, modality)",
            "random_token_control": "seeded per surface by (seed, surface_form_id)",
        },
        "dataset": {
            "id": config.dataset.dataset_id,
            "revision": config.dataset.dataset_revision,
            "rephrasings_id": config.dataset.rephrasings_dataset_id,
            "rephrasings_revision": config.dataset.rephrasings_revision,
            "topic": topic,
            "adapter_version": (audit.get("adapter_version")),
            "split_policy_version": (audit.get("split_policy_version")),
            "gate_g0_passed": audit.get("gate_g0_passed"),
        },
        "model": {
            "id": config.model.name,
            "revision": config.model.revision,
            "tokenizer_id": config.model.resolved_tokenizer_name,
            "tokenizer_revision": config.model.resolved_tokenizer_revision,
            "dtype": config.model.dtype,
            "device": config.model.device,
            "attn_implementation": config.model.attn_implementation,
            "n_layers": (parity or {}).get("model", {}).get("n_layers"),
            "n_heads": (parity or {}).get("model", {}).get("n_heads"),
            "d_model": (parity or {}).get("model", {}).get("d_model"),
            "hook_map_version": config.patching.hook_map_version,
        },
        "prompt_policy": {
            "template": config.prompt.template,
            "template_version": config.prompt.template_version,
            "add_generation_prompt": config.prompt.add_generation_prompt,
            "add_special_tokens": config.prompt.add_special_tokens,
            "system_prompt": config.prompt.system_prompt,
            "chat_template_sha256": (parity or {}).get("model", {}).get("chat_template_sha256"),
            "bos_inserted_by_template": (parity or {}).get("model", {}).get("bos_inserted_by_template"),
        },
        "scoring_policy": {
            "answer_metric": config.scoring.answer_metric,
            "answer_score": config.scoring.answer_score,
            "scoring_version": config.scoring.scoring_version,
            "margin_version": config.scoring.margin_version,
            "correctness_version": config.scoring.correctness_version,
            "distractor_policy": config.scoring.distractor_policy,
            "distractor_count": config.scoring.distractor_count,
            "min_abs_delta": config.scoring.min_abs_delta,
        },
        "corruption_policy": {
            "strategy": config.corruption.strategy,
            "corruption_policy_version": config.corruption.corruption_policy_version,
            "families": list(config.corruption.families),
            "preserve_modality": config.corruption.preserve_modality,
            "preserve_topic_when_possible": config.corruption.preserve_topic_when_possible,
            "preserve_surface_kind": config.corruption.preserve_surface_kind,
            "forbid_same_fact_id": config.corruption.forbid_same_fact_id,
            "clean_surfaces_per_fact": config.corruption.clean_surfaces_per_fact,
            "pair_modality": config.corruption.pair_modality,
            "pair_clean_split": config.corruption.pair_clean_split,
        },
        "pair_thresholds": {
            "min_abs_delta": config.scoring.min_abs_delta,
            "require_clean_correct": True,
            "max_pairs_per_fact_family": config.corruption.max_pairs_per_fact_family,
        },
        "pair_accounting": _pair_accounting(pairs),
        "known_fact_core": {
            "a0_version": core.get("a0_version"),
            "thresholds": core.get("thresholds"),
            "splits_scored": core.get("splits_scored"),
            "n_surfaces_scored": core.get("n_surfaces_scored"),
            "n_facts_screened": core.get("n_facts_screened"),
            "n_eligible": core.get("n_eligible"),
            "eligible_fact_ids": core.get("eligible_fact_ids", []),
            "eligible_detail": [
                {
                    "fact_id": f["fact_id"],
                    "K_f": f["K_f"],
                    "mean_margin": round(f["mean_margin"], 4),
                    "min_margin": round(f["min_margin"], 4),
                    "modalities_passing": f["modalities_passing"],
                }
                for f in sorted(eligible, key=lambda f: (-f["K_f"], -f["mean_margin"]))
            ],
        },
        "exclusions": {
            "excluded_facts": [
                {
                    "fact_id": f["fact_id"],
                    "K_f": f["K_f"],
                    "reasons": f["exclusion_reasons"],
                }
                for f in sorted(excluded, key=lambda f: -f["K_f"])
            ],
            "n_excluded_facts": len(excluded),
            "refused_cells": core.get("refused_cells", []),
            "n_refused_cells": len(core.get("refused_cells", [])),
            "reverse_modality_rule": {
                "rule_id": "reverse_degenerate_v1",
                "statement": (
                    "Reverse-modality cells are excluded from every fact-selective claim. A "
                    "reverse question asks for the fact from the answer side, and within a "
                    "topic the answers collapse to a single entity, so a same-topic distractor "
                    "pool contains the correct answer and the factual margin is undefined."
                ),
                "evidence": {
                    cell: {
                        "n_facts": info["n_facts"],
                        "distinct_answers": info["distinct_answers"],
                        "distinct_ratio": info["distinct_ratio"],
                    }
                    for cell, info in sorted(degeneracy.items())
                },
                "degenerate_cells": degenerate_cells,
                "enforced_by": "scoring.build_distractor_set raises DegenerateDistractorPool",
                "not_rescued": (
                    "No alternate distractor pool is constructed to keep reverse in the primary "
                    "analysis. It is retained only as a labelled topic-level diagnostic."
                ),
            },
        },
        "tolerances": {
            "hook_parity": {
                "capture_only_tolerance": 0.0,
                "capture_only_observed": (capture_check or {}).get("max_abs_per_token_diff"),
                "capture_only_passed": (capture_check or {}).get("passed"),
                "scoring_parity_tolerance": (scoring_check or {}).get("tolerance"),
                "scoring_parity_observed": (scoring_check or {}).get("max_abs_mean_logprob_diff"),
                "resid_post_equals_next_resid_pre_observed": (resid_check or {}).get("max_abs_diff"),
                "rationale": (
                    "Capture clones and returns None, so hook-only execution is gated at exactly "
                    "zero. The scoring comparison contrasts two float32 reduction orders and "
                    "carries a documented tolerance."
                ),
            },
            "patching": {
                "self_patch_in_situ_tolerance": config.patching.self_patch_tolerance,
                "self_patch_in_situ_observed": max(
                    (
                        c.get("max_abs_margin_diff", 0.0)
                        for c in sanity.get("checks", [])
                        if c["check"].endswith("_patch_is_a_no_op")
                    ),
                    default=None,
                ),
                "self_patch_cross_shape_observed": max(
                    (
                        c.get("max_abs_margin_diff", 0.0)
                        for c in sanity.get("checks", [])
                        if c["check"].endswith("cross_shape_capture")
                    ),
                    default=None,
                ),
                "gate_g2_passed": sanity.get("gate_g2_passed"),
                "rationale": (
                    "In-situ capture makes written values bitwise identical, so the patch write "
                    "is gated at zero. Cross-shape capture reflects float32 kernel tiling and "
                    "carries a documented tolerance."
                ),
            },
        },
        "artifact_hashes": {
            name: _hash_if_present(path)
            for name, path in {
                "dataset_audit": root / "dataset_audit" / f"dataset_audit_{topic}.json",
                "instrumentation_parity": parity_path or Path("missing"),
                "known_fact_core": a0_dir / f"known_fact_core_{topic}.json",
                "known_fact_scores": a0_dir / f"known_fact_scores_{topic}.jsonl",
                "distractor_sets": a0_dir / f"distractor_sets_{topic}.jsonl",
                "clean_corrupt_pairs": a0_dir / f"clean_corrupt_pairs_{topic}.jsonl",
                "patch_sanity": a2_dir / "patch_sanity.json",
                "residual_patch_scan": a2_dir / "residual_patch_scan.jsonl",
                "sink_sources_manifest": root / "reference" / "sink_sources_manifest.json",
            }.items()
        },
        "a1_inheritance": {
            "statement": (
                "A1 inherits everything above unchanged. A1 introduces its own discovery "
                "objective and its own discovery-split pairs; those are A1 artifacts and are "
                "not covered by this freeze."
            ),
            "a1_may_not_change": [
                "model/tokenizer revision",
                "dataset revisions",
                "prompt policy",
                "full-sequence factual margin definition",
                "Known-Fact Core membership",
                "reverse-modality exclusion rule",
                "seed",
            ],
            "a1_introduces": [
                "pre-answer discriminative-token objective (discovery only)",
                "discovery-split clean/corrupt pairs",
                "G0 graph with per-head outputs",
                "attribution vectors",
            ],
            "sink_firewall": (
                "No sink repository is read during A1. Only the recorded SHAs are carried in "
                "run manifests."
            ),
        },
    }


def render_markdown(freeze: dict[str, Any]) -> str:
    """Human-readable companion to the JSON freeze."""
    code = freeze["code"]
    model = freeze["model"]
    data = freeze["dataset"]
    core = freeze["known_fact_core"]
    tol = freeze["tolerances"]

    lines = [
        "# P0-P5 freeze",
        "",
        f"Frozen {freeze['created_at']} at commit `{code['commit_sha']}`"
        f"{' (tree dirty)' if code.get('dirty') else ''}.",
        "",
        freeze["purpose"],
        "",
        "**This file is never edited in place.** " + freeze["amendment_policy"].split(". ", 1)[1],
        "",
        "## Pinned revisions",
        "",
        "| Object | Identifier | Revision |",
        "|---|---|---|",
        f"| Dataset | `{data['id']}` | `{data['revision']}` |",
        f"| Paraphrases | `{data['rephrasings_id']}` | `{data['rephrasings_revision']}` |",
        f"| Model | `{model['id']}` | `{model['revision']}` |",
        f"| Tokenizer | `{model['tokenizer_id']}` | `{model['tokenizer_revision']}` |",
        "",
        f"Run settings: {model['dtype']}, {model['attn_implementation']} attention, "
        f"seed {freeze['seeds']['experiment_seed']}, hook map `{model['hook_map_version']}`.",
        "",
        "## Policies",
        "",
        f"- Prompt: `{freeze['prompt_policy']['template_version']}`, "
        f"`add_special_tokens={freeze['prompt_policy']['add_special_tokens']}`, "
        f"BOS from template: {freeze['prompt_policy']['bos_inserted_by_template']}",
        f"- Scoring: `{freeze['scoring_policy']['scoring_version']}`, "
        f"margin `{freeze['scoring_policy']['margin_version']}`, "
        f"correctness `{freeze['scoring_policy']['correctness_version']}`",
        f"- Distractors: `{freeze['scoring_policy']['distractor_policy']}`, "
        f"count {freeze['scoring_policy']['distractor_count']}",
        f"- Corruption: `{freeze['corruption_policy']['corruption_policy_version']}`, "
        f"families {', '.join(freeze['corruption_policy']['families'])}",
        f"- Pair acceptance: `|delta| >= {freeze['pair_thresholds']['min_abs_delta']}` "
        "and the clean surface must be answered correctly",
        "",
        "## Known-Fact Core",
        "",
        f"{core['n_eligible']} of {core['n_facts_screened']} facts eligible over "
        f"{core['n_surfaces_scored']} scored surfaces, splits {core['splits_scored']}.",
        "",
        "| Fact | K_f | mean margin | min margin | modalities |",
        "|---|---|---|---|---|",
    ]
    for f in core["eligible_detail"]:
        lines.append(
            f"| `{f['fact_id'].split(':')[-1]}` | {f['K_f']:.2f} | {f['mean_margin']:+.2f} | "
            f"{f['min_margin']:+.2f} | {', '.join(f['modalities_passing'])} |"
        )

    rev = freeze["exclusions"]["reverse_modality_rule"]
    lines += [
        "",
        f"Excluded: {freeze['exclusions']['n_excluded_facts']} facts, "
        f"{freeze['exclusions']['n_refused_cells']} refused cells.",
        "",
        "## Reverse-modality exclusion (`reverse_degenerate_v1`)",
        "",
        rev["statement"],
        "",
        "| Cell | facts | distinct answers | ratio |",
        "|---|---|---|---|",
    ]
    for cell, info in rev["evidence"].items():
        mark = " **degenerate**" if cell in rev["degenerate_cells"] else ""
        lines.append(
            f"| `{cell}`{mark} | {info['n_facts']} | {info['distinct_answers']} | "
            f"{info['distinct_ratio']:.2f} |"
        )
    lines += ["", rev["not_rescued"], ""]

    lines += [
        "## Pair accounting (attempted vs accepted)",
        "",
        "| Family | attempted | accepted | rate | mean Δ before | mean Δ after |",
        "|---|---|---|---|---|---|",
    ]
    for family, stats in freeze["pair_accounting"].items():
        before = stats["delta_before_filtering"]["mean"]
        after = stats["delta_after_filtering"]["mean"]
        before_s = f"{before:+.2f}" if before is not None else "-"
        after_s = f"{after:+.2f}" if after is not None else "-"
        lines.append(
            f"| `{family}` | {stats['n_attempted']} | {stats['n_accepted']} | "
            f"{stats['acceptance_rate']:.2f} | {before_s} | {after_s} |"
        )

    lines += [
        "",
        "## Tolerances",
        "",
        "| Gate | tolerance | observed |",
        "|---|---|---|",
        f"| capture-only parity | {tol['hook_parity']['capture_only_tolerance']} | "
        f"{tol['hook_parity']['capture_only_observed']} |",
        f"| scoring parity | {tol['hook_parity']['scoring_parity_tolerance']} | "
        f"{tol['hook_parity']['scoring_parity_observed']} |",
        f"| self-patch, in-situ capture | {tol['patching']['self_patch_in_situ_tolerance']} | "
        f"{tol['patching']['self_patch_in_situ_observed']} |",
        f"| self-patch, cross-shape capture | documented | "
        f"{tol['patching']['self_patch_cross_shape_observed']} |",
        "",
        "## What A1 may not change",
        "",
    ]
    for item in freeze["a1_inheritance"]["a1_may_not_change"]:
        lines.append(f"- {item}")
    lines += ["", "## What A1 introduces", ""]
    for item in freeze["a1_inheritance"]["a1_introduces"]:
        lines.append(f"- {item}")
    lines += ["", freeze["a1_inheritance"]["sink_firewall"], ""]
    return "\n".join(lines)
