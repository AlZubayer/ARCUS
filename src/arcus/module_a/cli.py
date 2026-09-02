from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import RunManifest, new_run_id, run_dir, write_json, write_jsonl
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


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "validate-config":
        print(load_config(args.config).model_dump_json(indent=2))
        return

    if args.command == "audit-dataset":
        raise SystemExit(cmd_audit_dataset(args))

    if args.command == "check-parity":
        raise SystemExit(cmd_check_parity(args))

    config = load_config(args.config)
    result = run_stage(config, Stage(args.stage))
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
