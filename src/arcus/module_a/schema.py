from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re


class Modality(StrEnum):
    DIRECT = "direct"
    REVERSE = "reverse"
    INDIRECT = "indirect"
    UNKNOWN = "unknown"


class ControlType(StrEnum):
    FORGET = "forget"
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    LEXICAL = "lexical"
    GENERAL_KNOWLEDGE = "general_knowledge"
    UNKNOWN = "unknown"


@dataclass(frozen=True, order=True)
class FactKey:
    topic: str
    fact_id: str


@dataclass(frozen=True)
class FactExample:
    topic: str
    question: str
    answer: str
    raw_label: str
    fact_key: FactKey | None
    modality: Modality
    control_type: ControlType
    semantic_tier: int | None = None
    surface_form_id: str | None = None
    source_split: str | None = None


def parse_forget_label(topic: str, label: str) -> tuple[FactKey | None, Modality]:
    """Parse labels such as M3-direct, K1-reverse, or K7-indirect.

    Returns `(None, UNKNOWN)` for unrecognized labels. Never infer a fact ID from text.
    """
    match = re.fullmatch(r"([KM]\d+)-(direct|reverse|indirect)", label)
    if not match:
        return None, Modality.UNKNOWN
    fact_id, modality = match.groups()
    return FactKey(topic=topic, fact_id=fact_id), Modality(modality)


def classify_retain_label(label: str) -> tuple[ControlType, int | None]:
    lower = label.lower()
    if lower.startswith("semantic-"):
        match = re.match(r"semantic-(\d+)-", lower)
        return ControlType.SEMANTIC, int(match.group(1)) if match else None
    if lower.startswith("syntax-"):
        return ControlType.SYNTACTIC, None
    if lower.startswith("lexical-"):
        return ControlType.LEXICAL, None
    if lower.startswith("gk-") or lower.startswith("general-"):
        return ControlType.GENERAL_KNOWLEDGE, None
    return ControlType.UNKNOWN, None
