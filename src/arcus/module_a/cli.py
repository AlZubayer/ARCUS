from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .pipeline import Stage, run_stage
from .suite import dataset_audit, load_hf_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arcus-module-a")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.add_argument("config")

    audit = sub.add_parser("audit-data")
    audit.add_argument("config")
    audit.add_argument("--split", required=True)

    run = sub.add_parser("run")
    run.add_argument("config")
    run.add_argument("--stage", choices=[s.value for s in Stage], required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    if args.command == "validate-config":
        print(config.model_dump_json(indent=2))
        return

    if args.command == "audit-data":
        if not config.dataset.dataset_name:
            raise SystemExit(
                "dataset.dataset_name is null. Pin the team's verified canonical dataset identifier first."
            )
        rows = load_hf_split(
            config.dataset.dataset_name,
            split=args.split,
            topic=config.dataset.topic,
        )
        report = dataset_audit(rows)
        output = Path(config.experiment.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"dataset_audit_{args.split}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return

    result = run_stage(config, Stage(args.stage))
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
