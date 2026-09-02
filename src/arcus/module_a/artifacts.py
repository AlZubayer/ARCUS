"""Run manifests and artifact writers.

A scientific artifact without a manifest is incomplete and must not be used downstream
(01_SYSTEM_DESIGN.md section 7). Every stage writes one.
"""

from __future__ import annotations

import getpass
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

MANIFEST_VERSION = "run_manifest_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def git_state(repo: str | Path = ".") -> dict[str, Any]:
    """Record the exact code state, including whether the tree was dirty."""

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    status = run("status", "--porcelain")
    return {
        "commit_sha": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
        "dirty_paths": sorted(line[3:] for line in status.splitlines())[:20] if status else [],
    }


def software_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module in ("torch", "transformers", "datasets", "numpy", "pydantic", "huggingface_hub"):
        try:
            import importlib.metadata as md

            versions[module] = md.version(module)
        except Exception:
            versions[module] = None
    return versions


def sink_reference_shas(repo: str | Path = ".") -> dict[str, Any]:
    """Read the committed sink provenance manifest, if present.

    Recorded in every manifest so a later A4/A5 artifact can prove which upstream revision
    defined its intervention semantics. Reading the SHAs is not reading the sink code.
    """
    path = Path(repo) / "artifacts" / "reference" / "sink_sources_manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {s["local_clone"]: s["commit_sha"] for s in payload.get("sources", [])}


@dataclass
class RunManifest:
    """Everything needed to reconstruct one stage of one run."""

    run_id: str
    stage: str
    config_path: str | None = None
    config_sha256: str | None = None
    seed: int | None = None
    parent_run_ids: list[str] = field(default_factory=list)
    model: dict[str, Any] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    prompt: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    manifest_version: str = MANIFEST_VERSION
    created_at: str = field(default_factory=utc_now)
    code: dict[str, Any] = field(default_factory=git_state)
    software: dict[str, Any] = field(default_factory=software_versions)
    sink_references: dict[str, Any] = field(default_factory=sink_reference_shas)
    created_by: str = field(default_factory=lambda: _safe_user())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


def _default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if hasattr(obj, "item"):  # numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy arrays
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=_default) + "\n", encoding="utf-8")
    return p


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=_default) + "\n")
    return p


def read_jsonl(path: str | Path) -> list[Any]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def new_run_id(prefix: str, seed: int | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}" if seed is None else f"{prefix}-{stamp}-s{seed}"


def run_dir(root: str | Path, run_id: str) -> Path:
    p = Path(root) / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p
