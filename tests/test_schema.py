"""Label-grammar tests against the real SUITE label forms.

Every literal here was copied from the pinned dataset revisions, not invented.
"""

import pytest

from arcus.module_a.schema import (
    ControlType,
    Generator,
    LabelParseError,
    Modality,
    SurfaceKind,
    normalize_answer,
    normalize_question,
    parse_augmentation,
    parse_label,
    stable_surface_id,
)

TOPIC = "challenger_disaster"


@pytest.mark.parametrize(
    "label,fact_id,modality",
    [
        ("M1-direct", "M1", Modality.DIRECT),
        ("M3-indirect", "M3", Modality.INDIRECT),
        ("K1-reverse", "K1", Modality.REVERSE),
        ("K20-direct", "K20", Modality.DIRECT),
    ],
)
def test_bare_forget_labels(label, fact_id, modality):
    parsed = parse_label(TOPIC, label)
    assert parsed.control_type is ControlType.FORGET
    assert parsed.fact_key.fact_id == fact_id
    assert parsed.fact_key.topic == TOPIC
    assert parsed.modality is modality
    assert parsed.augmentation.surface_kind is SurfaceKind.ORIGINAL


@pytest.mark.parametrize(
    "label,fact_id,modality,aug,kind,generator",
    [
        # These augmented forms are 100% of forget_train and were rejected outright by the
        # previous adapter's fullmatch grammar (discrepancy D1).
        ("M1-direct@original", "M1", Modality.DIRECT, "original", SurfaceKind.ORIGINAL, Generator.NONE),
        ("M1-direct@q_claude9", "M1", Modality.DIRECT, "q_claude9", SurfaceKind.PARAPHRASE, Generator.CLAUDE),
        ("M2-direct@blank_claude3", "M2", Modality.DIRECT, "blank_claude3", SurfaceKind.FILL_IN, Generator.CLAUDE),
        ("K1-reverse@q_gemini10", "K1", Modality.REVERSE, "q_gemini10", SurfaceKind.PARAPHRASE, Generator.GEMINI),
        ("K7-indirect@blank_gemini5", "K7", Modality.INDIRECT, "blank_gemini5", SurfaceKind.FILL_IN, Generator.GEMINI),
    ],
)
def test_augmented_forget_labels(label, fact_id, modality, aug, kind, generator):
    parsed = parse_label(TOPIC, label)
    assert parsed.fact_key.fact_id == fact_id
    assert parsed.modality is modality
    assert parsed.augmentation.augmentation_id == aug
    assert parsed.augmentation.surface_kind is kind
    assert parsed.augmentation.generator is generator


def test_fact_id_never_encodes_modality_or_surface_form():
    direct = parse_label(TOPIC, "M1-direct@q_claude9")
    reverse = parse_label(TOPIC, "M1-reverse@blank_gemini2")
    assert direct.fact_key == reverse.fact_key
    assert direct.modality is not reverse.modality
    assert "direct" not in direct.fact_key.fact_id
    assert "claude" not in direct.fact_key.fact_id


def test_syntax_control_preserves_link_to_forget_row():
    # retain_train Syntax rows reuse a forget row's exact template (discrepancy D7),
    # which makes them an augmentation-matched R3 control.
    parsed = parse_label(TOPIC, "Syntax-M1-direct@q_claude9")
    assert parsed.control_type is ControlType.SYNTACTIC
    assert parsed.linked_fact_key.fact_id == "M1"
    assert parsed.linked_modality is Modality.DIRECT
    assert parsed.augmentation.augmentation_id == "q_claude9"
    # A syntax control is about a different entity, so it carries no fact identity itself.
    assert parsed.fact_key is None
    assert parsed.modality is None


@pytest.mark.parametrize(
    "label,tier,domain",
    [
        ("Semantic-0-retain_challenger_not_disaster", 0, "retain_challenger_not_disaster"),
        ("Semantic-1-The Columbia Space Shuttle Disaster", 1, "The Columbia Space Shuttle Disaster"),
        ("Semantic-15-The Voyager 2 Uranus Flyby", 15, "The Voyager 2 Uranus Flyby"),
    ],
)
def test_semantic_labels(label, tier, domain):
    parsed = parse_label(TOPIC, label)
    assert parsed.control_type is ControlType.SEMANTIC
    assert parsed.semantic_tier == tier
    assert parsed.domain == domain
    assert parsed.fact_key is None


def test_lexical_and_general_knowledge_labels():
    lexical = parse_label(TOPIC, "Lexical-Challenger")
    assert lexical.control_type is ControlType.LEXICAL
    assert lexical.domain == "Challenger"

    # Note the spaces and hyphens inside the real GK domain string.
    gk = parse_label(TOPIC, "GK-2 - Physics - Thermodynamics")
    assert gk.control_type is ControlType.GENERAL_KNOWLEDGE
    assert gk.domain == "2 - Physics - Thermodynamics"


@pytest.mark.parametrize(
    "label",
    ["mystery-label", "M1-sideways", "Semantic-x-Foo", "", "M1", "direct", "Syntax-M1-direct"],
)
def test_unparseable_labels_fail_closed(label):
    # Gate G0: fact identity must be reconstructible from metadata, so an unknown label is
    # a schema change to investigate, never a row to guess at or silently drop.
    with pytest.raises(LabelParseError):
        parse_label(TOPIC, label)


def test_empty_topic_rejected():
    with pytest.raises(LabelParseError):
        parse_label("", "M1-direct")


@pytest.mark.parametrize("aug", ["q_claude0X", "blank_", "gpt_1", "q_claude", "blank_gemini"])
def test_unparseable_augmentation_fails_closed(aug):
    with pytest.raises(LabelParseError):
        parse_augmentation(aug)


def test_normalization_is_conservative():
    assert normalize_answer("  73 seconds. ") == "73 seconds"
    assert normalize_answer("STS-51-L") == "sts-51-l"
    assert normalize_question("  What  was\tthe date? ") == "what was the date?"
    # Normalization is for duplicate/degeneracy checks only; it must not merge distinct answers.
    assert normalize_answer("Michael Smith") != normalize_answer("Michael Smiths")


def test_surface_id_is_deterministic_and_row_scoped():
    kwargs = dict(
        dataset="apeleg/SUITE",
        source_split="forget_train",
        row_index=3,
        label="M1-direct@q_claude9",
        augmentation_id="q_claude9",
        question="What was the mission code?",
        answer="STS-51-L",
    )
    assert stable_surface_id(**kwargs) == stable_surface_id(**kwargs)
    # Identical text from a different source row stays distinguishable.
    assert stable_surface_id(**{**kwargs, "row_index": 4}) != stable_surface_id(**kwargs)
