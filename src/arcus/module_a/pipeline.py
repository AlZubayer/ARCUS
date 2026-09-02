from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .config import ModuleAConfig


class Stage(StrEnum):
    A0 = "a0"
    A1 = "a1"
    A2 = "a2"
    A3 = "a3"
    A4 = "a4"
    A5 = "a5"
    A6 = "a6"


@dataclass
class StageResult:
    stage: Stage
    status: str
    message: str


def prerequisite(stage: Stage) -> Stage | None:
    order = [Stage.A0, Stage.A1, Stage.A2, Stage.A3, Stage.A4, Stage.A5, Stage.A6]
    i = order.index(stage)
    return None if i == 0 else order[i - 1]


def run_stage(config: ModuleAConfig, stage: Stage) -> StageResult:
    """Stage dispatcher scaffold.

    This function deliberately refuses to fabricate mechanistic outputs. Each stage must be connected
    to a concrete backend and emit auditable artifacts before downstream stages are enabled.
    """
    if stage in {Stage.A4, Stage.A5} and not config.sink.enabled_after_route_validation_only:
        raise RuntimeError("Sink mapping must remain gated behind causal route validation.")

    return StageResult(
        stage=stage,
        status="not_implemented",
        message=(
            f"Stage {stage.value} is scaffolded but has no model backend yet. "
            "Implement exact scoring/patching first; see AGENTS.md and docs/MODULE_A_METHODOLOGY.md."
        ),
    )
