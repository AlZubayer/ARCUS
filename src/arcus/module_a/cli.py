from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import RunManifest, new_run_id, read_jsonl, run_dir, write_json, write_jsonl
from .audit import build_dataset_audit
from .config import ModuleAConfig, config_sha256, load_config
from .pipeline import Stage, run_stage
from .suite import ADAPTER_VERSION, SPLIT_POLICY_VERSION, build_corpus, example_to_row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arcus-module-a")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config", help="Parse and echo a resolved config")
    validate.add_argument("config")

    audit = sub.add_parser("audit-dataset", help="Gate G0: pin, parse and audit the SUITE source")
    audit.add_argument("--config", required=True)
    audit.add_argument(
        "--all-topics",
        action="store_true",
        help="Audit every topic rather than only the configured pilot topic",
    )
    audit.add_argument(
        "--skip-near-duplicates",
        action="store_true",
        help="Skip the quadratic cross-split near-duplicate scan",
    )

    parity = sub.add_parser(
        "check-parity", help="Gate G1: prove instrumentation does not change the model"
    )
    parity.add_argument("--config", required=True)
    parity.add_argument("--run", default=None, help="Reuse an existing run id")

    a0 = sub.add_parser("run-a0", help="A0: screen base knowledge and freeze the fact core")
    a0.add_argument("--config", required=True)
    a0.add_argument("--run", default=None)
    a0.add_argument(
        "--topics", nargs="*", default=None,
        help="Override the configured topic; pass several to sweep at unchanged thresholds",
    )

    bp = sub.add_parser("build-pairs", help="Build and score matched clean/corrupt pairs")
    bp.add_argument("--config", required=True)
    bp.add_argument("--run", required=True, help="Run id holding the a0 artifacts to consume")

    ps = sub.add_parser("patch-sanity", help="Gate G2: exact-patching sanity gates A-F")
    ps.add_argument("--config", required=True)
    ps.add_argument("--run", required=True)

    sc = sub.add_parser("patch-scan", help="Layer x token restoration/suppression map")
    sc.add_argument("--config", required=True)
    sc.add_argument("--run", required=True)
    sc.add_argument("--family", default="same_topic_fact_swap")

    fz = sub.add_parser("freeze-p5", help="Freeze the validated P0-P5 setup before A1")
    fz.add_argument("--config", required=True)
    fz.add_argument("--a0-run", required=True, help="Run id holding the a0/a2 artifacts")
    fz.add_argument("--parity-run", default=None)

    ao = sub.add_parser(
        "a1-objective", help="Register the A1 discovery objective and check it against M_f"
    )
    ao.add_argument("--config", required=True)
    ao.add_argument("--a0-run", required=True)
    ao.add_argument("--run", default=None, help="A1 run id (defaults to a new one)")
    ao.add_argument("--pair-limit", type=int, default=60)

    ap = sub.add_parser(
        "a1-pairs", help="Build discovery-split clean/corrupt pairs with full accounting"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True, help="A1 run id to write into")
    ap.add_argument(
        "--correctness-run", required=True,
        help="A0 run whose discovery-surface scores supply the clean-anchor filter",
    )
    ap.add_argument(
        "--distractors-run", required=True,
        help="A0 run supplying the frozen distractor pools",
    )
    ap.add_argument(
        "--freeze", default=None,
        help="Freeze artifact supplying fact eligibility (default artifacts/freeze/p0_p5_freeze.json)",
    )
    ap.add_argument("--all-eligible", action="store_true", help="Use all frozen-eligible facts")
    ap.add_argument("--topic", default=None, help="Override the configured topic")
    ap.add_argument(
        "--eligible-from-a0", default=None,
        help="A0 run supplying eligibility for a non-frozen topic (cross-topic controls)",
    )
    ap.add_argument("--name", default=None, help="Output basename (default accepted_pairs)")

    at = sub.add_parser(
        "a1-attribute", help="Compute full signed G0 attribution vectors for targets + controls"
    )
    at.add_argument("--config", required=True)
    at.add_argument("--run", required=True)
    at.add_argument("--distractors-run", required=True)
    at.add_argument("--steps", type=int, default=None, help="Override integration steps")
    at.add_argument("--control-surfaces", type=int, default=3)
    at.add_argument("--control-facts", type=int, default=4)
    at.add_argument("--retain-items", type=int, default=8)
    at.add_argument(
        "--robustness-surfaces", type=int, default=3,
        help="Surfaces per fact for the non-primary corruption families",
    )
    at.add_argument("--name", default="vectors")
    at.add_argument(
        "--pairs-name", default="pairs", help="Basename of the accepted pairs file to read"
    )
    at.add_argument(
        "--cross-topic-pairs", default=None,
        help="Accepted-pairs file for a different topic, for the cross_topic control class",
    )
    at.add_argument("--cross-topic-distractors", default=None, help="Comma-separated topics")
    at.add_argument("--cross-topic-run", default=None, help="A0 run holding those pools")

    sm = sub.add_parser("a1-similarity", help="Within-fact vs control route similarity")
    sm.add_argument("--config", required=True)
    sm.add_argument("--run", required=True)
    sm.add_argument("--name", default="vectors")

    g4 = sub.add_parser("a1-g4", help="Gate G4: validate attribution against exact effects")
    g4.add_argument("--config", required=True)
    g4.add_argument("--run", required=True)
    g4.add_argument("--distractors-run", required=True)
    g4.add_argument("--name", default="vectors")
    g4.add_argument("--n-pairs", type=int, default=3)
    g4.add_argument("--n-top", type=int, default=12)
    g4.add_argument("--n-random", type=int, default=12)

    run = sub.add_parser("run", help="Run a pipeline stage")
    run.add_argument("config")
    run.add_argument("--stage", choices=[s.value for s in Stage], required=True)
    return parser


def _dataset_manifest_block(config: ModuleAConfig) -> dict:
    return {
        "id": config.dataset.dataset_id,
        "revision": config.dataset.dataset_revision,
        "rephrasings_id": config.dataset.rephrasings_dataset_id,
        "rephrasings_revision": config.dataset.rephrasings_revision,
        "topic": config.dataset.topic,
        "project": config.dataset.project,
    }


def cmd_audit_dataset(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    topics = None if args.all_topics else [config.dataset.topic]

    corpus = build_corpus(
        topics=topics,
        suite_dataset_id=config.dataset.dataset_id,
        suite_revision=config.dataset.dataset_revision,
        rephrasings_dataset_id=config.dataset.rephrasings_dataset_id,
        rephrasings_revision=config.dataset.rephrasings_revision,
    )
    audit = build_dataset_audit(
        corpus, topics=topics, run_near_duplicates=not args.skip_near_duplicates
    )

    out = Path(config.experiment.artifact_dir) / "dataset_audit"
    scope = "all_topics" if args.all_topics else config.dataset.topic
    manifest = RunManifest(
        run_id=f"dataset-audit-{scope}",
        stage="a0.dataset_audit",
        config_path=str(args.config),
        config_sha256=config_sha256(args.config),
        seed=config.experiment.seed,
        dataset=_dataset_manifest_block(config),
        policies={
            "adapter_version": ADAPTER_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "audit_scope": scope,
        },
    )
    write_json(out / f"manifest_{scope}.json", manifest.to_dict())
    audit_path = write_json(out / f"dataset_audit_{scope}.json", audit)
    write_jsonl(
        out / f"normalized_examples_{scope}.jsonl",
        (example_to_row(ex) for ex in corpus.examples),
    )

    census = audit["census"]
    print(f"wrote {audit_path}")
    print(f"  examples          : {census['n_examples']}")
    print(f"  dropped           : {census['n_dropped']} {census['drop_reasons']}")
    print(f"  facts per topic   : {audit['fact_identity']['facts_per_topic']}")
    for split, block in census["by_split"].items():
        print(f"  {split:11s}: n={block['n']:5d} modalities={block['by_modality']}")
    print(f"  gate G0 passed    : {audit['gate_g0_passed']}")
    for failure in audit["gate_g0_failures"]:
        print(f"    FAIL {failure['check']}: {failure['detail']}")

    degenerate = [
        cell
        for cell, info in audit["answers"]["degeneracy_by_topic_modality"].items()
        if not info["usable_for_same_topic_distractors"]
    ]
    if degenerate:
        print(f"  degenerate answer cells (D3): {degenerate}")

    return 0 if audit["gate_g0_passed"] else 1


def _model_manifest_block(config: ModuleAConfig, meta) -> dict:
    return {
        "id": config.model.name,
        "revision": config.model.revision,
        "tokenizer_id": meta.tokenizer_id,
        "tokenizer_revision": meta.tokenizer_revision,
        "dtype": meta.dtype,
        "device": meta.device,
        "attention_backend": meta.attn_implementation,
        "architecture": meta.architecture,
        "n_layers": meta.n_layers,
        "hook_map_version": meta.hook_map_version,
    }


def cmd_check_parity(args: argparse.Namespace) -> int:
    from .backend.hf import backend_from_config
    from .stages.parity import run_parity

    config = load_config(args.config)
    backend = backend_from_config(config)

    # A Challenger fact the pilot will meet again in A0, with a matched wrong answer of the
    # same type and a deliberately multi-token answer to exercise answer indexing.
    report = run_parity(
        backend,
        prompt_question="On what date did the Challenger disaster occur?",
        correct_answer="January 28, 1986",
        wrong_answer="July 20, 1969",
        multi_token_answer="Reinforced Carbon-Carbon",
    )

    run_id = args.run or new_run_id("parity", config.experiment.seed)
    out = run_dir(config.experiment.artifact_dir, run_id) / "parity"
    manifest = RunManifest(
        run_id=run_id,
        stage="p2.instrumentation_parity",
        config_path=str(args.config),
        config_sha256=config_sha256(args.config),
        seed=config.experiment.seed,
        model=_model_manifest_block(config, backend.metadata()),
        dataset=_dataset_manifest_block(config),
        prompt=report["prompt"],
        policies={
            "hook_map_version": backend.metadata().hook_map_version,
            "prompt_template_version": backend.metadata().prompt_template_version,
            "scoring_version": config.scoring.scoring_version,
        },
    )
    write_json(out / "manifest.json", manifest.to_dict())
    path = write_json(out / "instrumentation_parity.json", report)

    print(f"wrote {path}")
    meta = backend.metadata()
    print(f"  model     : {meta.model_id}@{meta.model_revision[:12]} {meta.dtype} "
          f"{meta.attn_implementation} eval={meta.eval_mode}")
    print(f"  BOS       : {meta.bos_token!r} id={meta.bos_token_id} "
          f"inserted_by_template={meta.bos_inserted_by_template}")
    print(f"  prompt    : {report['prompt']['n_prompt_tokens']} tokens "
          f"sha256={report['prompt']['sha256'][:12]}")
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {check['check']}")
    print(f"  gate G1 passed: {report['gate_g1_passed']}")
    return 0 if report["gate_g1_passed"] else 1


def cmd_run_a0(args: argparse.Namespace) -> int:
    from .backend.hf import backend_from_config
    from .stages.a0_known_facts import run_a0

    config = load_config(args.config)
    topics = args.topics or [config.dataset.topic]
    backend = backend_from_config(config)
    run_id = args.run or new_run_id("a0", config.experiment.seed)
    out = run_dir(config.experiment.artifact_dir, run_id) / "a0"

    summaries = {}
    for topic in topics:
        print(f"[a0] topic={topic}", flush=True)
        corpus = build_corpus(
            topics=[topic],
            suite_dataset_id=config.dataset.dataset_id,
            suite_revision=config.dataset.dataset_revision,
            rephrasings_dataset_id=config.dataset.rephrasings_dataset_id,
            rephrasings_revision=config.dataset.rephrasings_revision,
        )
        result = run_a0(
            backend,
            corpus,
            minimum_base_accuracy=config.dataset.minimum_base_accuracy,
            minimum_modalities_known=config.dataset.minimum_modalities_known,
            distractor_count=config.scoring.distractor_count,
            seed=config.experiment.seed,
            splits=config.dataset.known_fact_splits,
            allowed_modalities=config.dataset.known_fact_modalities,
            answer_score=config.scoring.answer_score,
        )
        write_json(out / f"known_fact_core_{topic}.json", result["known_fact_core"])
        write_jsonl(out / f"known_fact_scores_{topic}.jsonl", result["known_fact_scores"])
        write_jsonl(out / f"distractor_sets_{topic}.jsonl", result["distractor_sets"])
        write_jsonl(
            out / f"normalized_examples_{topic}.jsonl",
            (example_to_row(ex) for ex in corpus.examples),
        )

        core = result["known_fact_core"]
        summaries[topic] = {
            "n_surfaces_scored": core["n_surfaces_scored"],
            "n_facts_screened": core["n_facts_screened"],
            "n_eligible": core["n_eligible"],
            "n_passing_accuracy_only": core["n_passing_accuracy_only"],
            "eligible_fact_ids": core["eligible_fact_ids"],
            "n_refused_cells": len(core["refused_cells"]),
        }
        print(f"  scored {core['n_surfaces_scored']} surfaces over "
              f"{core['n_facts_screened']} facts")
        print(f"  eligible (accuracy AND modality coverage): {core['n_eligible']}")
        print(f"  passing accuracy only                    : {core['n_passing_accuracy_only']}")

    meta = backend.metadata()
    manifest = RunManifest(
        run_id=run_id,
        stage="a0.known_fact_core",
        config_path=str(args.config),
        config_sha256=config_sha256(args.config),
        seed=config.experiment.seed,
        model=_model_manifest_block(config, meta),
        dataset=_dataset_manifest_block(config),
        prompt={
            "template": config.prompt.template,
            "template_version": config.prompt.template_version,
            "system_prompt_sha256": __import__("hashlib").sha256(
                config.prompt.system_prompt.encode()).hexdigest(),
            "chat_template_sha256": meta.chat_template_sha256,
            "bos_inserted_by_template": meta.bos_inserted_by_template,
        },
        policies={
            "scoring_version": config.scoring.scoring_version,
            "margin_version": config.scoring.margin_version,
            "correctness_version": config.scoring.correctness_version,
            "distractor_policy": config.scoring.distractor_policy,
            "adapter_version": ADAPTER_VERSION,
            "split_policy_version": SPLIT_POLICY_VERSION,
            "splits_scored": config.dataset.known_fact_splits,
            "minimum_base_accuracy": config.dataset.minimum_base_accuracy,
            "minimum_modalities_known": config.dataset.minimum_modalities_known,
        },
        extra={"topics": topics, "summary_by_topic": summaries},
    )
    write_json(out / "manifest.json", manifest.to_dict())
    write_json(out / "summary.json", summaries)
    print(f"wrote {out}")

    best = max(summaries.values(), key=lambda s: s["n_eligible"]) if summaries else None
    if best and best["n_eligible"] < config.dataset.minimum_pilot_facts:
        print(f"  SHORTFALL: no topic reaches {config.dataset.minimum_pilot_facts} eligible facts "
              "at the preregistered thresholds.")
        return 2
    return 0


def _load_a0(config: ModuleAConfig, run_id: str, topic: str):
    """Read the frozen A0 outputs a pairs run depends on."""
    from .scoring import DistractorSet
    from .schema import FactKey, Modality

    base = Path(config.experiment.artifact_dir) / run_id / "a0"
    core = json.loads((base / f"known_fact_core_{topic}.json").read_text(encoding="utf-8"))
    sets: dict = {}
    for row in read_jsonl(base / f"distractor_sets_{topic}.jsonl"):
        key = FactKey(**row["fact_key"])
        modality = Modality(row["modality"])
        sets[(key, modality)] = DistractorSet(
            fact_key=key,
            modality=modality,
            correct_answer=row["correct_answer"],
            distractors=tuple(row["distractors"]),
            policy=row["policy"],
            n_candidates=row["n_candidates"],
            excluded_as_synonymous=tuple(row["excluded_as_synonymous"]),
        )
    return core, sets


def cmd_build_pairs(args: argparse.Namespace) -> int:
    from .schema import Modality, Split
    from .stages.pairs import run_pairs

    config = load_config(args.config)
    topic = config.dataset.topic
    core, distractor_sets = _load_a0(config, args.run, topic)

    # All topics: the cross_topic_matched family needs candidates outside the pilot topic.
    # Only the pilot topic's eligible facts are used as clean surfaces.
    corpus = build_corpus(
        topics=None,
        suite_dataset_id=config.dataset.dataset_id,
        suite_revision=config.dataset.dataset_revision,
        rephrasings_dataset_id=config.dataset.rephrasings_dataset_id,
        rephrasings_revision=config.dataset.rephrasings_revision,
    )
    backend = backend_from_config_cached(config)

    # A clean run must actually exhibit the fact, so clean surfaces are restricted to the
    # ones A0 scored correct.
    scores = read_jsonl(
        Path(config.experiment.artifact_dir) / args.run / "a0" / f"known_fact_scores_{topic}.jsonl"
    )
    correct_ids = {r["surface_form_id"] for r in scores if r["is_correct"]}

    print(f"[pairs] {len(core['eligible_fact_ids'])} eligible facts from a0 run {args.run}; "
          f"{len(correct_ids)} correctly-answered surfaces available as clean anchors")
    result = run_pairs(
        backend,
        corpus,
        eligible_facts=core["eligible_fact_ids"],
        distractor_sets=distractor_sets,
        families=config.corruption.families,
        modality=Modality(config.corruption.pair_modality),
        split=Split(config.corruption.pair_clean_split),
        clean_surfaces_per_fact=config.corruption.clean_surfaces_per_fact,
        max_pairs_per_family=config.corruption.max_pairs_per_fact_family,
        min_abs_delta=config.scoring.min_abs_delta,
        seed=config.experiment.seed,
        answer_score=config.scoring.answer_score,
        correct_surface_ids=correct_ids,
    )

    out = run_dir(config.experiment.artifact_dir, args.run) / "a0"
    write_jsonl(out / f"clean_corrupt_pairs_{topic}.jsonl", result["pairs"])
    write_json(out / f"clean_corrupt_pairs_summary_{topic}.json", result["summary"])

    s = result["summary"]
    print(f"  pairs: {s['n_pairs']} built, {s['n_accepted']} accepted, {s['n_rejected']} rejected")
    print(f"  rejections: {s['rejection_reasons']}")
    for family, stats in s["by_family"].items():
        print(f"    {family:32s} n={stats['n']:4d} accepted={stats['n_accepted']:4d} "
              f"mean_delta={stats['mean_delta']:+.2f}")
    print(f"  facts with accepted pairs: {s['n_facts_with_accepted_pairs']}")
    return 0


def cmd_freeze_p5(args: argparse.Namespace) -> int:
    from .freeze import build_freeze, render_markdown

    config = load_config(args.config)
    freeze = build_freeze(
        config,
        args.config,
        artifact_root=config.experiment.artifact_dir,
        a0_run_id=args.a0_run,
        parity_run_id=args.parity_run,
    )
    out = Path(config.experiment.artifact_dir) / "freeze"
    json_path = write_json(out / "p0_p5_freeze.json", freeze)
    (out / "p0_p5_freeze.md").write_text(render_markdown(freeze), encoding="utf-8")

    missing = [
        name for name, info in freeze["artifact_hashes"].items() if not info["present"]
    ]
    print(f"wrote {json_path}")
    print(f"  commit         : {freeze['code']['commit_sha']} dirty={freeze['code']['dirty']}")
    print(f"  eligible facts : {freeze['known_fact_core']['n_eligible']} "
          f"{[f.split(':')[-1] for f in freeze['known_fact_core']['eligible_fact_ids']]}")
    print(f"  refused cells  : {freeze['exclusions']['n_refused_cells']} "
          f"(reverse_degenerate_v1)")
    total_att = sum(v["n_attempted"] for v in freeze["pair_accounting"].values())
    total_acc = sum(v["n_accepted"] for v in freeze["pair_accounting"].values())
    print(f"  pairs          : {total_acc} accepted of {total_att} attempted")
    print(f"  artifacts      : {len(freeze['artifact_hashes']) - len(missing)} hashed"
          + (f", MISSING {missing}" if missing else ", all present"))
    return 1 if missing else 0


def cmd_a1_objective(args: argparse.Namespace) -> int:
    from .stages.a1_objective import build_objective_artifact

    config = load_config(args.config)
    topic = config.dataset.topic
    _, distractor_sets = _load_a0(config, args.a0_run, topic)

    base = Path(config.experiment.artifact_dir) / args.a0_run / "a0"
    score_rows = read_jsonl(base / f"known_fact_scores_{topic}.jsonl")
    pairs = [
        r
        for r in read_jsonl(base / f"clean_corrupt_pairs_{topic}.jsonl")
        if r["validation_status"] == "accepted" and r["family"] == "same_topic_fact_swap"
    ]

    backend = backend_from_config_cached(config)
    print(f"[a1-objective] {len(score_rows)} surfaces, {len(pairs)} same-topic pairs")
    artifact = build_objective_artifact(
        backend,
        score_rows=score_rows,
        pairs=pairs,
        distractor_sets=distractor_sets,
        pair_limit=args.pair_limit,
    )

    run_id = args.run or new_run_id("a1", config.experiment.seed)
    out = run_dir(config.experiment.artifact_dir, run_id) / "a1"
    path = write_json(out / "discovery_objective.json", artifact)

    meta = backend.metadata()
    write_json(
        out / "manifest.json",
        RunManifest(
            run_id=run_id,
            stage="a1.discovery_objective",
            config_path=str(args.config),
            config_sha256=config_sha256(args.config),
            seed=config.experiment.seed,
            parent_run_ids=[args.a0_run],
            model=_model_manifest_block(config, meta),
            dataset=_dataset_manifest_block(config),
            policies={
                "objective_version": artifact["definition"]["objective_version"],
                "companion_metric": artifact["definition"]["companion_metric"]["name"],
                "scoring_version": config.scoring.scoring_version,
            },
        ).to_dict(),
    )

    sc = artifact["surface_consistency"]
    pc = artifact["pair_direction_consistency"]
    print(f"wrote {path}")
    print(f"  run id           : {run_id}")
    print(f"  surfaces scored  : {sc['n_scored']} ({sc['n_rejected']} rejected)")
    print(f"  common prefix    : mean {sc['common_prefix_tokens']['mean']} tokens, "
          f"zero for {sc['common_prefix_tokens']['n_zero']}/{sc['n_scored']}")
    print(f"  Spearman J vs M  : {sc['spearman_J_vs_M']}  (sign agreement {sc['sign_agreement']})")
    print(f"  clean/corrupt    : Spearman {pc['spearman_deltaJ_vs_deltaM']}, "
          f"sign agreement {pc['sign_agreement']} over {pc['n_pairs']} pairs "
          f"({pc['n_sign_disagreements']} disagreements)")
    for note in artifact["interpretation"]:
        print(f"  note: {note[:150]}")
    return 0


def cmd_a1_pairs(args: argparse.Namespace) -> int:
    from .freeze import _pair_accounting
    from .schema import Modality, Split
    from .stages.pairs import run_pairs

    config = load_config(args.config)
    topic = args.topic or config.dataset.topic
    art = Path(config.experiment.artifact_dir)

    # Eligibility comes from the freeze, never recomputed here. The discovery split has a
    # single modality, so re-running the A0 gate on it would mechanically yield zero
    # eligible facts -- a property of the split, not a finding about the model.
    #
    # Cross-topic CONTROL facts have no freeze of their own, so they may take eligibility
    # from the all-topics A0 run instead; the source is recorded either way.
    if args.eligible_from_a0:
        core_path = (
            art / args.eligible_from_a0 / "a0" / f"known_fact_core_{topic}.json"
        )
        core_json = json.loads(core_path.read_text(encoding="utf-8"))
        eligible = [f.split(":")[-1] for f in core_json["eligible_fact_ids"]]
        freeze_path = core_path
    else:
        freeze_path = Path(args.freeze) if args.freeze else art / "freeze" / "p0_p5_freeze.json"
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        eligible = [f.split(":")[-1] for f in freeze["known_fact_core"]["eligible_fact_ids"]]

    use_all = args.all_eligible or args.topic is not None
    pilot = eligible if use_all else (config.a1.pilot_facts if config.a1 else eligible)
    missing = sorted(set(pilot) - set(eligible))
    if missing:
        print(f"ERROR: requested facts are not in the frozen Known-Fact Core: {missing}")
        return 1
    fact_ids = [f"{topic}:{f}" for f in pilot]

    _, distractor_sets = _load_a0(config, args.distractors_run, topic)

    correctness = read_jsonl(
        art / args.correctness_run / "a0" / f"known_fact_scores_{topic}.jsonl"
    )
    correct_ids = {r["surface_form_id"] for r in correctness if r["is_correct"]}

    corpus = build_corpus(
        topics=None,
        suite_dataset_id=config.dataset.dataset_id,
        suite_revision=config.dataset.dataset_revision,
        rephrasings_dataset_id=config.dataset.rephrasings_dataset_id,
        rephrasings_revision=config.dataset.rephrasings_revision,
    )
    backend = backend_from_config_cached(config)

    print(f"[a1-pairs] facts={pilot}")
    print(f"  eligibility from {freeze_path} ({len(eligible)} frozen-eligible)")
    print(f"  clean anchors restricted to {len(correct_ids)} correct discovery surfaces")

    result = run_pairs(
        backend,
        corpus,
        eligible_facts=fact_ids,
        distractor_sets=distractor_sets,
        families=config.corruption.families,
        modality=Modality(config.corruption.pair_modality),
        split=Split(config.corruption.pair_clean_split),
        clean_surfaces_per_fact=config.corruption.clean_surfaces_per_fact,
        max_pairs_per_family=config.corruption.max_pairs_per_fact_family,
        min_abs_delta=config.scoring.min_abs_delta,
        seed=config.experiment.seed,
        answer_score=config.scoring.answer_score,
        correct_surface_ids=correct_ids,
        prefer_exact_syntax_twin=config.corruption.prefer_exact_syntax_twin,
    )

    out = run_dir(config.experiment.artifact_dir, args.run) / "a1"
    attempted = result["pairs"]
    accepted = [p for p in attempted if p["validation_status"] == "accepted"]

    # Both are persisted: a family's acceptance rate is itself a finding, and it is only
    # recoverable if the rejected pairs survive.
    name = args.name or "pairs"
    write_jsonl(out / f"attempted_{name}.jsonl", attempted)
    write_jsonl(out / f"accepted_{name}.jsonl", accepted)

    accounting = _pair_accounting(attempted)
    controls = {
        "policy_version": config.corruption.corruption_policy_version,
        "families_never_pooled": True,
        "clean_split": config.corruption.pair_clean_split,
        "modality": config.corruption.pair_modality,
        "facts": pilot,
        "eligibility_source": str(freeze_path),
        "correctness_source": args.correctness_run,
        "by_family": accounting,
        "summary": result["summary"],
    }
    write_json(out / f"controls_by_family_{name}.json", controls)

    print(f"  attempted {len(attempted)}, accepted {len(accepted)}")
    print(f"  {'family':32s} {'att':>4} {'acc':>4} {'rate':>5} {'meanD_pre':>10} {'meanD_post':>11}")
    for family, stats in accounting.items():
        pre = stats["delta_before_filtering"]["mean"]
        post = stats["delta_after_filtering"]["mean"]
        print(f"  {family:32s} {stats['n_attempted']:4d} {stats['n_accepted']:4d} "
              f"{stats['acceptance_rate']:5.2f} "
              f"{(f'{pre:+.2f}' if pre is not None else '-'):>10} "
              f"{(f'{post:+.2f}' if post is not None else '-'):>11}")
    quality = result["summary"].get("syntax_match_quality", {})
    if quality:
        print(f"  same_syntax match quality: {quality}")
    return 0


def _build_a1_items(config, args, corpus, distractor_sets, accepted, topic):
    """Assemble target and control attribution items.

    Controls are built the same way targets are, because a cosine between a target vector
    and a control vector only means something if both were produced by the same procedure.
    """
    from .discovery.controls import (
        ControlClass,
        build_fact_control_items,
        build_retain_control_items,
    )
    from .discovery.controls import AttributionItem
    from .schema import ControlType, FactKey, Modality
    from .stages.a1_discovery import TARGET_CLASS

    a1 = config.a1
    pilot = {f"{topic}:{f}" for f in a1.pilot_facts}

    def lookup(row):
        key = FactKey(**row["target_fact_key"])
        return distractor_sets.get((key, Modality(row["modality"])))

    items: list = []

    # Targets: the primary family at every discovery surface, plus the robustness families
    # on a subset (Gate G4E asks whether conclusions survive a change of corruption).
    for family in [a1.primary_family, *a1.robustness_families]:
        per_fact = (
            config.corruption.clean_surfaces_per_fact
            if family == a1.primary_family
            else args.robustness_surfaces
        )
        for row in sorted(accepted, key=lambda r: (r["target_fact_key"]["fact_id"], r["clean_surface_id"])):
            fact_id = f"{topic}:{row['target_fact_key']['fact_id']}"
            if fact_id not in pilot or row["family"] != family:
                continue
            seen = [
                i for i in items
                if i.fact_id == fact_id and i.family == family
            ]
            if len({i.surface_form_id for i in seen}) >= per_fact and row[
                "clean_surface_id"
            ] not in {i.surface_form_id for i in seen}:
                continue
            if any(i.surface_form_id == row["clean_surface_id"] and i.family == family for i in seen):
                continue
            pool = lookup(row)
            if pool is None:
                continue
            items.append(
                AttributionItem(
                    item_id=f"{TARGET_CLASS}|{row['pair_id']}",
                    item_class=TARGET_CLASS,
                    clean_question=row["clean_question"],
                    corrupt_question=row["corrupt_question"],
                    correct_answer=pool.correct_answer,
                    distractors=pool.distractors,
                    fact_id=fact_id,
                    surface_form_id=row["clean_surface_id"],
                    family=family,
                    metadata={"pair_id": row["pair_id"], "delta_M": row.get("delta")},
                )
            )

    # Control class 1: other facts in the same topic, each with its own clean/corrupt pair
    # and its own distractor pool -- their routes, not the pilot facts' routes.
    items += build_fact_control_items(
        accepted, item_class=ControlClass.SAME_TOPIC_DIFFERENT_FACT,
        exclude_fact_ids=pilot, family=a1.primary_family,
        surfaces_per_fact=args.control_surfaces, limit_facts=args.control_facts,
        distractor_lookup=lookup,
    )

    # Control class 5: facts from a different topic entirely. These need their own pairs
    # file. Using a pilot fact under a cross-topic *corruption* would measure the pilot
    # fact's route again, not an unrelated fact's route, so that is not done here.
    if args.cross_topic_pairs:
        cross_rows = read_jsonl(Path(args.cross_topic_pairs))
        items += build_fact_control_items(
            cross_rows, item_class=ControlClass.CROSS_TOPIC,
            exclude_fact_ids=set(), family=a1.primary_family,
            surfaces_per_fact=args.control_surfaces, limit_facts=args.control_facts,
            distractor_lookup=lookup,
        )

    # Control classes 2-4: retain rows, each given its own pool and partner.
    retain = [ex for ex in corpus.examples if not ex.is_forget]
    refusals: list = []
    for item_class, control_type, max_tier in (
        (ControlClass.SEMANTIC_NEIGHBOR, ControlType.SEMANTIC, 1),
        (ControlClass.SAME_SYNTAX, ControlType.SYNTACTIC, None),
        (ControlClass.SAME_LEXICAL, ControlType.LEXICAL, None),
    ):
        built, refused = build_retain_control_items(
            retain, item_class=item_class, control_type=control_type, topic=topic,
            distractor_count=config.scoring.distractor_count,
            seed=config.experiment.seed, limit=args.retain_items,
            semantic_max_tier=max_tier,
        )
        items += built
        refusals += refused
    return items, refusals


def cmd_a1_attribute(args: argparse.Namespace) -> int:
    from .discovery.graph import G0Graph
    from .stages.a1_discovery import (
        discovery_summary,
        run_attribution,
        save_attribution,
    )

    config = load_config(args.config)
    if config.a1 is None:
        print("ERROR: config has no a1 block")
        return 1
    topic = config.dataset.topic
    art = Path(config.experiment.artifact_dir)
    steps = args.steps or config.a1.integration_steps

    _, distractor_sets = _load_a0(config, args.distractors_run, topic)
    if args.cross_topic_distractors:
        for extra_topic in args.cross_topic_distractors.split(","):
            _, extra = _load_a0(config, args.cross_topic_run, extra_topic.strip())
            distractor_sets.update(extra)
    accepted = read_jsonl(art / args.run / "a1" / f"accepted_{args.pairs_name}.jsonl")
    corpus = build_corpus(
        topics=None,
        suite_dataset_id=config.dataset.dataset_id,
        suite_revision=config.dataset.dataset_revision,
        rephrasings_dataset_id=config.dataset.rephrasings_dataset_id,
        rephrasings_revision=config.dataset.rephrasings_revision,
    )

    backend = backend_from_config_cached(config)
    graph = G0Graph.from_backend(backend)
    items, refusals = _build_a1_items(config, args, corpus, distractor_sets, accepted, topic)

    counts: dict = {}
    for it in items:
        counts[it.item_class] = counts.get(it.item_class, 0) + 1
    print(f"[a1-attribute] {len(items)} items over {len(graph)} G0 objects, m={steps}")
    for k, v in sorted(counts.items()):
        print(f"    {k:28s} {v}")
    if refusals:
        print(f"    control refusals: {len(refusals)}")

    matrix, index, skipped = run_attribution(
        backend, graph, items,
        integration_steps=steps, alignment_policy=config.a1.alignment_policy,
    )

    out = run_dir(config.experiment.artifact_dir, args.run) / "a1"
    storage = save_attribution(out / "attribution_full", matrix, index, graph, name=args.name)
    write_jsonl(out / "attribution_full" / f"{args.name}_index.jsonl", index)

    summary = discovery_summary(
        index, skipped, graph,
        integration_steps=steps, alignment_policy=config.a1.alignment_policy,
    )
    summary["storage"] = storage
    summary["control_refusals"] = refusals
    write_json(out / f"attribution_summary_{args.name}.json", summary)

    meta = backend.metadata()
    write_json(
        out / f"manifest_attribute_{args.name}.json",
        RunManifest(
            run_id=args.run, stage="a1.attribution",
            config_path=str(args.config), config_sha256=config_sha256(args.config),
            seed=config.experiment.seed, parent_run_ids=[args.distractors_run],
            model=_model_manifest_block(config, meta),
            dataset=_dataset_manifest_block(config),
            policies={
                "attribution_version": summary["attribution_version"],
                "graph_version": graph.version,
                "objective_version": "discriminative_token_margin_v1",
                "integration_steps": steps,
                "alignment_policy": config.a1.alignment_policy,
                "primary_family": config.a1.primary_family,
                "sink_information_used": False,
            },
        ).to_dict(),
    )

    comp = summary["completeness"]
    health = summary["objective_health"]
    align = summary["alignment"]
    print(f"  wrote {storage['path']} shape={storage['shape']}")
    print(f"  completeness : mean {comp['mean']}  within 5% of 1.0 for "
          f"{comp['fraction_within_5pct_of_1']:.1%}  passed={comp['passed']}")
    print(f"  path effect  : mean {health['path_effect']['mean']}, "
          f"positive for {health['path_effect']['fraction_positive']:.1%}")
    print(f"  J clean > 0  : {health['j_clean']['fraction_positive']:.1%}")
    print(f"  exact-length : {align['n_exact_length']}/{align['n_vectors']} "
          f"(mean |delta| {align['prompt_len_delta']['mean_abs']})")
    if skipped:
        print(f"  skipped      : {len(skipped)}")
    return 0


def _load_vectors(config, run_id: str, name: str):
    """Load the attribution matrix and its index, checking they line up."""
    import numpy as np

    from .analysis.route_similarity import VectorSet

    base = Path(config.experiment.artifact_dir) / run_id / "a1" / "attribution_full"
    payload = np.load(base / f"{name}.npz", allow_pickle=False)
    matrix = payload["scores"].astype(np.float64)
    object_ids = [str(x) for x in payload["object_ids"]]
    rows = read_jsonl(base / f"{name}_index.jsonl")
    if len(rows) != matrix.shape[0]:
        raise SystemExit(
            f"index has {len(rows)} rows but matrix has {matrix.shape[0]}; refusing to guess"
        )
    for row in rows:
        row["_object_ids"] = object_ids
    return VectorSet(matrix=matrix, rows=rows), object_ids


def cmd_a1_similarity(args: argparse.Namespace) -> int:
    from .analysis.route_similarity import (
        Representation,
        analyse,
        backbone_report,
        top_component_stability,
    )
    from .discovery.controls import ControlClass
    from .stages.a1_discovery import TARGET_CLASS

    config = load_config(args.config)
    vectors, object_ids = _load_vectors(config, args.run, args.name)
    seed = config.experiment.seed
    family = config.a1.primary_family
    controls = list(ControlClass.ALL)

    blocks = {
        rep: analyse(
            vectors, target_class=TARGET_CLASS, control_classes=controls,
            family=family, seed=seed, representation=rep,
        )
        for rep in (Representation.RAW, Representation.SUBTRACT, Representation.PROJECT)
    }
    backbone = backbone_report(
        vectors, target_class=TARGET_CLASS, family=family, object_ids=object_ids
    )

    out = run_dir(config.experiment.artifact_dir, args.run) / "a1"
    write_jsonl(out / "route_similarity_raw.jsonl", [blocks[Representation.RAW]])
    write_jsonl(
        out / "route_similarity_residual.jsonl",
        [blocks[Representation.SUBTRACT], blocks[Representation.PROJECT]],
    )
    write_json(out / "generic_fact_backbone.json", backbone)
    write_json(
        out / "top_component_stability.json",
        top_component_stability(
            vectors, target_class=TARGET_CLASS, family=family,
            object_ids=object_ids, top_k=20,
        ),
    )

    def show(block, title):
        w = block["within_fact"]["pooled"]
        ci = block["within_fact"]["bootstrap_ci"]
        print(f"  {title}")
        print(f"    within-fact                mean {w['mean']:+.4f}  "
              f"CI [{ci['lo']:+.4f}, {ci['hi']:+.4f}]  n={w['n']}")
        for control, stats in block["between_by_control_class"].items():
            st = stats["similarity"]
            if st["n"] == 0:
                print(f"    {control:26s} n=0")
                continue
            print(f"    {control:26s} mean {st['mean']:+.4f}  "
                  f"D={stats['distinctness_D']:+.4f}  "
                  f"p={stats['permutation_p_within_gt_between']}  n={st['n']}")

    raw = blocks[Representation.RAW]
    print(f"[a1-similarity] {raw['n_facts']} facts, {raw['n_target_vectors']} target vectors, "
          f"family={family}")
    print(f"  {raw['symmetry']}")
    show(raw, "RAW attribution")
    show(blocks[Representation.SUBTRACT], "RESIDUAL (backbone subtracted, primary)")
    show(blocks[Representation.PROJECT], "RESIDUAL (backbone direction projected out, secondary)")
    c = backbone["cosine_target_vs_pooled_backbone"]
    print(f"  cosine(target, pooled backbone): mean {c['mean']:+.4f} "
          f"[{c['min']:+.4f}, {c['max']:+.4f}]")
    return 0


_BACKEND_CACHE: dict = {}


def backend_from_config_cached(config: ModuleAConfig):
    """One backend per process; loading 3B weights repeatedly is pure waste."""
    from .backend.hf import backend_from_config

    if "backend" not in _BACKEND_CACHE:
        _BACKEND_CACHE["backend"] = backend_from_config(config)
    return _BACKEND_CACHE["backend"]


def _accepted_pairs(config: ModuleAConfig, run_id: str, topic: str, family: str | None) -> list[dict]:
    path = (
        Path(config.experiment.artifact_dir) / run_id / "a0" / f"clean_corrupt_pairs_{topic}.jsonl"
    )
    rows = read_jsonl(path)
    return [
        r
        for r in rows
        if r["validation_status"] == "accepted" and (family is None or r["family"] == family)
    ]


def cmd_patch_sanity(args: argparse.Namespace) -> int:
    from .backend.base import Component
    from .schema import FactKey, Modality
    from .stages.patching import run_sanity_gates

    config = load_config(args.config)
    topic = config.dataset.topic
    _, distractor_sets = _load_a0(config, args.run, topic)
    pairs = _accepted_pairs(config, args.run, topic, "same_topic_fact_swap")
    if not pairs:
        print("no accepted same_topic_fact_swap pairs; run build-pairs first")
        return 1

    backend = backend_from_config_cached(config)
    # Strongest available pair, so the gates are exercised on a real, large-effect case.
    pair = max(pairs, key=lambda r: abs(r["delta"]))
    key = FactKey(**pair["target_fact_key"])
    distractors = distractor_sets[(key, Modality(pair["modality"]))]

    n_layers = backend.metadata().n_layers
    layers = sorted({0, 1, n_layers // 2, n_layers - 2, n_layers - 1})
    report = run_sanity_gates(
        backend,
        clean_question=pair["clean_question"],
        corrupt_question=pair["corrupt_question"],
        distractors=distractors,
        components=[Component(c) for c in config.patching.components],
        layers=layers,
        answer_score=config.scoring.answer_score,
        tolerance=config.patching.self_patch_tolerance,
    )
    report["pair_id"] = pair["pair_id"]
    report["fact_id"] = key.id
    report["layers_probed"] = layers
    report["components_probed"] = config.patching.components

    out = run_dir(config.experiment.artifact_dir, args.run) / "a2"
    path = write_json(out / "patch_sanity.json", report)
    print(f"wrote {path}")
    print(f"  pair {pair['pair_id'][:70]}")
    print(
        f"  clean {report['clean_margin']:+.3f}  corrupt {report['corrupt_margin']:+.3f}  "
        f"delta {report['delta']:+.3f}"
    )
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        extra = ""
        if "max_abs_margin_diff" in check:
            extra = f" (max |diff| {check['max_abs_margin_diff']:.2e})"
        elif "max_abs_score_diff" in check:
            extra = f" (max |diff| {check['max_abs_score_diff']:.2e})"
        print(f"  [{status}] {check['check']}{extra}")
    print(f"  gate G2 passed: {report['gate_g2_passed']}")
    return 0 if report["gate_g2_passed"] else 1


def cmd_patch_scan(args: argparse.Namespace) -> int:
    from .backend.base import Component, PatchDirection
    from .schema import FactKey, Modality
    from .stages.patching import random_location_control, scan_pair

    config = load_config(args.config)
    topic = config.dataset.topic
    _, distractor_sets = _load_a0(config, args.run, topic)
    pairs = _accepted_pairs(config, args.run, topic, args.family)
    if not pairs:
        print(f"no accepted pairs in family {args.family}")
        return 1

    backend = backend_from_config_cached(config)
    n_layers = backend.metadata().n_layers
    layers = list(range(0, n_layers, config.patching.layer_stride))
    offsets = list(range(config.patching.scan_final_k))
    components = [Component(c) for c in config.patching.scan_components]
    directions = [PatchDirection(d) for d in config.patching.directions]

    # One pair per fact, strongest first, so the map spans distinct facts.
    by_fact: dict[str, dict] = {}
    for row in sorted(pairs, key=lambda r: -abs(r["delta"])):
        fact = f"{row['target_fact_key']['topic']}:{row['target_fact_key']['fact_id']}"
        by_fact.setdefault(fact, row)
    selected = list(by_fact.values())[: config.patching.max_pairs_scanned]

    rows: list[dict] = []
    controls: list[dict] = []
    for i, pair in enumerate(selected, start=1):
        key = FactKey(**pair["target_fact_key"])
        distractors = distractor_sets[(key, Modality(pair["modality"]))]
        print(f"  [{i}/{len(selected)}] {key.id} delta={pair['delta']:+.2f}", flush=True)
        rows += scan_pair(
            backend,
            pair_id=pair["pair_id"],
            fact_id=key.id,
            clean_question=pair["clean_question"],
            corrupt_question=pair["corrupt_question"],
            distractors=distractors,
            components=components,
            layers=layers,
            offsets_from_end=offsets,
            directions=directions,
            min_abs_delta=config.scoring.min_abs_delta,
            answer_score=config.scoring.answer_score,
        )
        controls.append(
            {
                "fact_id": key.id,
                "pair_id": pair["pair_id"],
                **random_location_control(
                    backend,
                    clean_question=pair["clean_question"],
                    corrupt_question=pair["corrupt_question"],
                    distractors=distractors,
                    layers=layers,
                    offsets_from_end=offsets,
                    seed=config.experiment.seed,
                    n_samples=12,
                    min_abs_delta=config.scoring.min_abs_delta,
                ),
            }
        )

    out = run_dir(config.experiment.artifact_dir, args.run) / "a2"
    write_jsonl(out / "residual_patch_scan.jsonl", rows)
    summary = {
        "stage": "a2.residual_patch_scan",
        "artifact_kind": "restoration_suppression_map",
        "not_a_claim": (
            "This map shows where replacing a single activation moves the factual margin. "
            "It is not a validated causal circuit and must not be reported as one."
        ),
        "family": args.family,
        "n_pairs_scanned": len(selected),
        "n_interventions": len(rows),
        "layers": layers,
        "offsets_from_end": offsets,
        "components": config.patching.scan_components,
        "directions": config.patching.directions,
        "min_abs_delta": config.scoring.min_abs_delta,
        "random_location_controls": controls,
        "top_sufficiency": sorted(
            (
                r
                for r in rows
                if r["direction"] == "corrupt_to_clean" and r["normalized_effect"] is not None
            ),
            key=lambda r: -r["normalized_effect"],
        )[:15],
        "top_necessity": sorted(
            (
                r
                for r in rows
                if r["direction"] == "clean_to_corrupt" and r["normalized_effect"] is not None
            ),
            key=lambda r: -r["normalized_effect"],
        )[:15],
    }
    write_json(out / "residual_patch_scan_summary.json", summary)

    meta = backend.metadata()
    manifest = RunManifest(
        run_id=args.run,
        stage="a2.residual_patch_scan",
        config_path=str(args.config),
        config_sha256=config_sha256(args.config),
        seed=config.experiment.seed,
        model=_model_manifest_block(config, meta),
        dataset=_dataset_manifest_block(config),
        policies={
            "patching_version": "exact_residual_patch_v1",
            "hook_map_version": meta.hook_map_version,
            "token_alignment_policy": "explicit_indices, aligned from the prompt end",
            "corruption_policy_version": config.corruption.corruption_policy_version,
            "scoring_version": config.scoring.scoring_version,
            "min_abs_delta": config.scoring.min_abs_delta,
        },
        extra={"n_interventions": len(rows), "family": args.family},
    )
    write_json(out / "manifest.json", manifest.to_dict())

    print(f"wrote {out} ({len(rows)} interventions)")
    for row in summary["top_sufficiency"][:5]:
        print(
            f"    sufficiency {row['normalized_effect']:+.3f} at {row['hook_point']} "
            f"offset {row['offset_from_end']} ({row['fact_id']})"
        )
    for control in controls:
        mean = control["mean_sufficiency"]
        top = control["max_sufficiency"]
        print(
            f"    random-site control ({control['fact_id']}): "
            f"mean {mean:+.3f} max {top:+.3f}"
            if mean is not None
            else f"    random-site control ({control['fact_id']}): undefined"
        )
    return 0


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "validate-config":
        print(load_config(args.config).model_dump_json(indent=2))
        return

    if args.command == "audit-dataset":
        raise SystemExit(cmd_audit_dataset(args))

    if args.command == "check-parity":
        raise SystemExit(cmd_check_parity(args))

    if args.command == "freeze-p5":
        raise SystemExit(cmd_freeze_p5(args))

    if args.command == "a1-objective":
        raise SystemExit(cmd_a1_objective(args))

    if args.command == "a1-pairs":
        raise SystemExit(cmd_a1_pairs(args))

    if args.command == "a1-attribute":
        raise SystemExit(cmd_a1_attribute(args))

    if args.command == "a1-similarity":
        raise SystemExit(cmd_a1_similarity(args))

    if args.command == "run-a0":
        raise SystemExit(cmd_run_a0(args))

    if args.command == "build-pairs":
        raise SystemExit(cmd_build_pairs(args))

    if args.command == "patch-sanity":
        raise SystemExit(cmd_patch_sanity(args))

    if args.command == "patch-scan":
        raise SystemExit(cmd_patch_scan(args))

    config = load_config(args.config)
    result = run_stage(config, Stage(args.stage))
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
