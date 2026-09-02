"""Typed scientific objects and the fail-closed SUITE label grammar.

Design contract (docs/module_a/01_SYSTEM_DESIGN.md, 02_DATA_AND_SPLITS.md):

* ``fact_id`` never encodes query modality or surface form.
* ``modality`` (direct/reverse/indirect) and ``surface_kind`` (original/paraphrase/
  fill-in) are separate axes and must never be collapsed.
* Fact identity is read from source metadata only. Nothing here infers a fact from
  question text; unparseable labels raise instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256

SCHEMA_VERSION = "suite_schema_v2"
ANSWER_NORMALIZATION_VERSION = "answer_normalization_v1"


class LabelParseError(ValueError):
    """Raised when a SUITE label does not match any registered grammar.

    Deliberately fatal: Gate G0 requires that fact identity is reconstructible from
    source metadata, so an unknown label is a schema change to investigate, never a
    row to silently drop or guess at.
    """


class Modality(StrEnum):
    """Query direction. Distinct from surface form."""

    DIRECT = "direct"
    REVERSE = "reverse"
    INDIRECT = "indirect"


class SurfaceKind(StrEnum):
    """How one realization of a query was produced.

    NOTE (discrepancy D2): docs/module_a/02_DATA_AND_SPLITS.md lists "fill-in" beside
    direct/reverse/indirect as if it were a modality. In the real dataset a
    fill-in-the-blank is an *augmentation* (``blank_*``) of a direct or reverse
    question, so it is recorded here as a surface kind and the modality is preserved.
    """

    ORIGINAL = "original"
    PARAPHRASE = "paraphrase"
    FILL_IN = "fill_in"


class Generator(StrEnum):
    """Which model produced an augmented surface form.

    This is the leakage-grouping axis: ``forget_train`` augmentations are Claude-generated
    and ``SUITE-rephrasings`` augmentations are Gemini-generated, so the two never share a
    generated paraphrase group.
    """

    NONE = "none"
    CLAUDE = "claude"
    GEMINI = "gemini"


class ControlType(StrEnum):
    FORGET = "forget"
    SEMANTIC = "semantic"
    SYNTACTIC = "syntactic"
    LEXICAL = "lexical"
    GENERAL_KNOWLEDGE = "general_knowledge"


class Split(StrEnum):
    """Logical partition (docs/module_a/02_DATA_AND_SPLITS.md section 4)."""

    DISCOVERY = "discovery"
    VALIDATION = "validation"
    STRESS = "stress"


@dataclass(frozen=True, order=True)
class FactKey:
    """Stable identity of one atomic fact. Never encodes modality or surface form."""

    topic: str
    fact_id: str

    @property
    def id(self) -> str:
        return f"{self.topic}:{self.fact_id}"


# --------------------------------------------------------------------------------------
# Label grammars. One regex per family; anything else raises.
# --------------------------------------------------------------------------------------

FORGET_LABEL_RE = re.compile(
    r"^(?P<fact_id>[KM]\d+)-(?P<modality>direct|reverse|indirect)"
    r"(?:@(?P<augmentation>[A-Za-z0-9_]+))?$"
)
SYNTAX_LABEL_RE = re.compile(
    r"^Syntax-(?P<fact_id>[KM]\d+)-(?P<modality>direct|reverse|indirect)"
    r"@(?P<augmentation>[A-Za-z0-9_]+)$"
)
SEMANTIC_LABEL_RE = re.compile(r"^Semantic-(?P<tier>\d+)-(?P<domain>.+)$")
LEXICAL_LABEL_RE = re.compile(r"^Lexical-(?P<domain>.+)$")
GK_LABEL_RE = re.compile(r"^GK-(?P<domain>.+)$")

# original | q_claude7 | blank_gemini3
AUGMENTATION_RE = re.compile(
    r"^(?:(?P<original>original)|(?P<kind>q|blank)_(?P<generator>claude|gemini)(?P<index>\d+))$"
)


@dataclass(frozen=True)
class Augmentation:
    """Parsed augmentation id: which surface form of a question this row is."""

    augmentation_id: str
    surface_kind: SurfaceKind
    generator: Generator
    index: int | None = None


ORIGINAL_AUGMENTATION = Augmentation(
    augmentation_id="original",
    surface_kind=SurfaceKind.ORIGINAL,
    generator=Generator.NONE,
)


def parse_augmentation(augmentation_id: str | None) -> Augmentation:
    """Parse ``original`` / ``q_claude9`` / ``blank_gemini3``. Fail closed."""
    if augmentation_id is None:
        return ORIGINAL_AUGMENTATION
    match = AUGMENTATION_RE.fullmatch(augmentation_id)
    if match is None:
        raise LabelParseError(f"Unrecognized augmentation id: {augmentation_id!r}")
    if match.group("original"):
        return ORIGINAL_AUGMENTATION
    kind = SurfaceKind.PARAPHRASE if match.group("kind") == "q" else SurfaceKind.FILL_IN
    return Augmentation(
        augmentation_id=augmentation_id,
        surface_kind=kind,
        generator=Generator(match.group("generator")),
        index=int(match.group("index")),
    )


@dataclass(frozen=True)
class LabelParse:
    """Everything a SUITE ``label`` string encodes, decomposed."""

    raw_label: str
    control_type: ControlType
    fact_key: FactKey | None = None
    modality: Modality | None = None
    augmentation: Augmentation = ORIGINAL_AUGMENTATION
    semantic_tier: int | None = None
    domain: str | None = None
    # Syntax retain rows point back at the forget row whose template they reuse.
    linked_fact_key: FactKey | None = None
    linked_modality: Modality | None = None


def parse_label(topic: str, label: str) -> LabelParse:
    """Decompose one SUITE label. Raises ``LabelParseError`` on anything unregistered."""
    if not topic:
        raise LabelParseError("topic must be non-empty to build a fact key")

    match = FORGET_LABEL_RE.fullmatch(label)
    if match is not None:
        return LabelParse(
            raw_label=label,
            control_type=ControlType.FORGET,
            fact_key=FactKey(topic=topic, fact_id=match.group("fact_id")),
            modality=Modality(match.group("modality")),
            augmentation=parse_augmentation(match.group("augmentation")),
        )

    match = SYNTAX_LABEL_RE.fullmatch(label)
    if match is not None:
        # Same question template as the linked forget row, different entity. This is the
        # R3 same-syntax control, matched down to the augmentation id (discrepancy D7).
        return LabelParse(
            raw_label=label,
            control_type=ControlType.SYNTACTIC,
            modality=None,
            augmentation=parse_augmentation(match.group("augmentation")),
            linked_fact_key=FactKey(topic=topic, fact_id=match.group("fact_id")),
            linked_modality=Modality(match.group("modality")),
        )

    match = SEMANTIC_LABEL_RE.fullmatch(label)
    if match is not None:
        return LabelParse(
            raw_label=label,
            control_type=ControlType.SEMANTIC,
            semantic_tier=int(match.group("tier")),
            domain=match.group("domain"),
        )

    match = LEXICAL_LABEL_RE.fullmatch(label)
    if match is not None:
        return LabelParse(
            raw_label=label,
            control_type=ControlType.LEXICAL,
            domain=match.group("domain"),
        )

    match = GK_LABEL_RE.fullmatch(label)
    if match is not None:
        return LabelParse(
            raw_label=label,
            control_type=ControlType.GENERAL_KNOWLEDGE,
            domain=match.group("domain"),
        )

    raise LabelParseError(
        f"Unrecognized SUITE label {label!r} for topic {topic!r}. "
        "Update the adapter and the dataset audit rather than inferring fact identity."
    )


# --------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FactExample:
    """One surface realization of one query, with full source provenance."""

    surface_form_id: str
    topic: str
    question: str
    answer: str
    raw_label: str
    control_type: ControlType
    fact_key: FactKey | None = None
    modality: Modality | None = None
    surface_kind: SurfaceKind = SurfaceKind.ORIGINAL
    generator: Generator = Generator.NONE
    augmentation_id: str = "original"
    semantic_tier: int | None = None
    domain: str | None = None
    linked_fact_key: FactKey | None = None
    linked_modality: Modality | None = None
    split: Split | None = None
    source_dataset: str | None = None
    source_revision: str | None = None
    source_split: str | None = None
    source_row_index: int | None = None
    source_column: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def is_forget(self) -> bool:
        return self.control_type is ControlType.FORGET

    @property
    def group_id(self) -> str:
        """Leakage-safe grouping key.

        All generated variants of one (fact, modality) from one generator belong to a
        single group and must not be split across folds.
        """
        fact = self.fact_key.id if self.fact_key else f"{self.topic}:{self.control_type.value}"
        return f"{fact}|{self.modality or 'na'}|{self.generator.value}"


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_answer(text: str) -> str:
    """Benign formatting normalization (``answer_normalization_v1``).

    Used ONLY for duplicate detection, degeneracy checks, and exact-match diagnostics.
    Never applied to the text that is actually scored: teacher-forced scoring always
    consumes the verbatim source answer string.
    """
    cleaned = _WHITESPACE_RE.sub(" ", text.strip()).strip(" .")
    return cleaned.casefold()


def normalize_question(text: str) -> str:
    """Normalization used for cross-split duplicate/leakage detection only."""
    return _WHITESPACE_RE.sub(" ", text.strip()).casefold()


def stable_surface_id(
    *,
    dataset: str,
    source_split: str,
    row_index: int,
    label: str,
    augmentation_id: str,
    question: str,
    answer: str,
) -> str:
    """Deterministic bookkeeping id for one surface form.

    Includes the source row coordinates so two identical strings from different source
    rows stay distinguishable. This is an identifier, never a fact identity.
    """
    payload = "\x1f".join(
        [
            dataset,
            source_split,
            str(row_index),
            label,
            augmentation_id,
            normalize_question(question),
            normalize_answer(answer),
        ]
    ).encode("utf-8")
    return sha256(payload).hexdigest()[:16]
