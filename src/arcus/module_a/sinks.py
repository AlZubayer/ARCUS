from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SinkHeadRecord:
    layer: int
    head: int
    anchor: str
    received_attention: float
    projected_value_norm: float | None
    deletion_effect: float | None
    relocation_effect: float | None
    anchor_identity_stable: bool | None = None

    @property
    def head_id(self) -> str:
        return f"L{self.layer}H{self.head}"


def is_sink_candidate(record: SinkHeadRecord, threshold: float) -> bool:
    return record.received_attention >= threshold


def has_causal_anchor_evidence(record: SinkHeadRecord) -> bool:
    """Require more than a high attention column before calling a head functionally sink-related."""
    return (
        record.anchor_identity_stable is True
        and record.deletion_effect is not None
        and record.relocation_effect is not None
    )
