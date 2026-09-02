from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifacts import RunManifest, write_json, write_jsonl
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


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "validate-config":
        print(load_config(args.config).model_dump_json(indent=2))
        return

    if args.command == "audit-dataset":
        raise SystemExit(cmd_audit_dataset(args))

    config = load_config(args.config)
    result = run_stage(config, Stage(args.stage))
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
