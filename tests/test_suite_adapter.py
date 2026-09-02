"""Adapter tests using real SUITE row shapes, with no network access.

Rows below are verbatim samples from the pinned revisions.
"""

from dataclasses import replace

import pytest

from arcus.module_a.audit import answer_degeneracy, build_dataset_audit
from arcus.module_a.schema import ControlType, Generator, Modality, Split, SurfaceKind
from arcus.module_a.suite import (
    REPHRASING_AUGMENTATION_COLUMNS,
    DatasetSchemaError,
    SuiteCorpus,
    _drop_cross_split_duplicate_prompts,
    _place,
    assign_split,
    melt_rephrasing_row,
    row_to_example,
    syntax_controls_by_link,
)

SUITE = "apeleg/SUITE"
REV = "3f5f6b0897dac10baacf1aa8b35319a02abccd23"


def make(row, split, index=0, dataset=SUITE, revision=REV):
    return row_to_example(row, row_index=index, dataset=dataset, revision=revision, source_split=split)


FORGET_TRAIN_ROW = {
    "question": "Which official mission code corresponds to the Space Shuttle Challenger's last flight?",
    "answer": "STS-51-L",
    "label": "M1-direct@q_claude9",
    "topic": "challenger_disaster",
}
RETAIN_TRAIN_SYNTAX_ROW = {
    "question": "Which official name corresponds to the Allied invasion plan for the Normandy landings?",
    "answer": "Operation Overlord",
    "label": "Syntax-M1-direct@q_claude9",
    "topic": "challenger_disaster",
}
RETAIN_EVAL_SEMANTIC_ROW = {
    "question": "How many main engines were located on the Challenger orbiter?",
    "answer": "Three",
    "label": "Semantic-0-retain_challenger_not_disaster",
    "topic": "challenger_disaster",
}


def test_forget_train_row_with_augmentation_parses():
    # The previous adapter raised on every row of this shape (discrepancy D1).
    ex = make(FORGET_TRAIN_ROW, "forget_train")
    assert ex.fact_key.fact_id == "M1"
    assert ex.modality is Modality.DIRECT
    assert ex.surface_kind is SurfaceKind.PARAPHRASE
    assert ex.generator is Generator.CLAUDE
    assert ex.augmentation_id == "q_claude9"
    assert ex.source_dataset == SUITE
    assert ex.source_revision == REV
    assert ex.source_row_index == 0


def test_retain_rows_carry_no_fact_identity():
    syntax = make(RETAIN_TRAIN_SYNTAX_ROW, "retain_train")
    assert syntax.control_type is ControlType.SYNTACTIC
    assert syntax.fact_key is None
    assert syntax.linked_fact_key.fact_id == "M1"

    semantic = make(RETAIN_EVAL_SEMANTIC_ROW, "retain_eval")
    assert semantic.control_type is ControlType.SEMANTIC
    assert semantic.semantic_tier == 0
    assert semantic.fact_key is None


def test_missing_required_field_fails_closed():
    with pytest.raises(DatasetSchemaError):
        make({"question": "q", "answer": "a", "topic": "t"}, "forget_train")
    with pytest.raises(DatasetSchemaError):
        make({**FORGET_TRAIN_ROW, "answer": None}, "forget_train")


def test_syntax_controls_indexed_by_linked_surface():
    forget = make(FORGET_TRAIN_ROW, "forget_train")
    syntax = make(RETAIN_TRAIN_SYNTAX_ROW, "retain_train")
    index = syntax_controls_by_link([syntax])
    key = (forget.fact_key, forget.modality, forget.augmentation_id)
    assert index[key].surface_form_id == syntax.surface_form_id


REPHRASING_ROW = {
    "question": "What was the official mission code for the Space Shuttle Challenger's final flight?",
    "answer": "STS-51-L",
    "label": "M1-direct",
    "topic": "challenger_disaster",
    **{c: f"paraphrase text {c}" for c in REPHRASING_AUGMENTATION_COLUMNS},
}


def test_wide_rephrasing_row_melts_to_sixteen_surface_forms():
    out = melt_rephrasing_row(REPHRASING_ROW, row_index=0)
    assert len(out) == 1 + len(REPHRASING_AUGMENTATION_COLUMNS) == 16
    # Fact identity, modality and answer are shared; only the surface varies.
    assert {e.fact_key.fact_id for e in out} == {"M1"}
    assert {e.modality for e in out} == {Modality.DIRECT}
    assert {e.answer for e in out} == {"STS-51-L"}
    assert len({e.surface_form_id for e in out}) == 16
    kinds = {e.surface_kind for e in out}
    assert kinds == {SurfaceKind.ORIGINAL, SurfaceKind.PARAPHRASE, SurfaceKind.FILL_IN}
    assert sum(1 for e in out if e.surface_kind is SurfaceKind.PARAPHRASE) == 10
    assert sum(1 for e in out if e.surface_kind is SurfaceKind.FILL_IN) == 5


def test_melt_skips_blank_augmentation_columns():
    row = {**REPHRASING_ROW, "q_gemini4": "", "blank_gemini2": None}
    assert len(melt_rephrasing_row(row, row_index=0)) == 14


def test_melt_rejects_non_forget_label():
    with pytest.raises(DatasetSchemaError):
        melt_rephrasing_row({**REPHRASING_ROW, "label": "Lexical-Challenger"}, row_index=0)


def test_split_policy_is_provenance_based():
    train = make(FORGET_TRAIN_ROW, "forget_train")
    assert assign_split(train) == (Split.DISCOVERY, None)

    gemini_direct = direct_original = indirect_original = None
    for ex in melt_rephrasing_row(REPHRASING_ROW, row_index=0):
        if ex.generator is Generator.GEMINI:
            gemini_direct = ex
        elif ex.generator is Generator.NONE:
            direct_original = ex
    assert assign_split(gemini_direct) == (Split.VALIDATION, None)
    # The rephrasings "original" duplicates forget_train's "@original" row exactly.
    assert assign_split(direct_original)[1] == "duplicate_of_discovery_original"

    for ex in melt_rephrasing_row({**REPHRASING_ROW, "label": "M1-indirect"}, row_index=1):
        if ex.generator is Generator.NONE:
            indirect_original = ex
    # Indirect has no training counterpart at all (discrepancy D5), so it is stress.
    assert assign_split(indirect_original) == (Split.STRESS, None)


def test_forget_eval_is_dropped_as_reference_only():
    ex = make({**FORGET_TRAIN_ROW, "label": "M1-direct"}, "forget_eval")
    assert assign_split(ex)[1] == "forget_eval_is_reference_only_upstream"


def test_cross_split_identical_prompt_is_dropped_from_the_held_out_side():
    shared = "The calendar date of the Challenger disaster was ____."
    discovery = make(
        {"question": shared, "answer": "January 28, 1986", "label": "M2-direct@blank_claude4",
         "topic": "challenger_disaster"},
        "forget_train",
    )
    validation = melt_rephrasing_row(
        {"question": "unused", "answer": "January 28, 1986", "label": "M2-direct",
         "topic": "challenger_disaster", "blank_gemini1": shared},
        row_index=0,
    )[1]

    corpus = SuiteCorpus()
    _place(corpus, discovery)
    _place(corpus, validation)
    _drop_cross_split_duplicate_prompts(corpus)

    kept = {e.surface_form_id for e in corpus.examples}
    assert discovery.surface_form_id in kept
    assert validation.surface_form_id not in kept
    assert corpus.dropped[-1].reason == "identical_prompt_in_discovery"


def test_answer_degeneracy_detects_reverse_collapse():
    """Reverse questions ask from the answer side, so within a topic they all resolve to
    the same entity. Same-topic distractors therefore cannot separate facts (D3)."""
    rows = [
        {"question": f"q{i}", "answer": "Challenger", "label": f"K{i}-reverse", "topic": "challenger_disaster"}
        for i in range(1, 6)
    ] + [
        {"question": f"d{i}", "answer": f"answer {i}", "label": f"K{i}-direct", "topic": "challenger_disaster"}
        for i in range(1, 6)
    ]
    examples = [make(r, "forget_eval", index=i) for i, r in enumerate(rows)]
    report = answer_degeneracy(examples)

    reverse = report["challenger_disaster|reverse"]
    assert reverse["n_facts"] == 5
    assert reverse["distinct_answers"] == 1
    assert reverse["usable_for_same_topic_distractors"] is False

    assert report["challenger_disaster|direct"]["usable_for_same_topic_distractors"] is True


def test_audit_reports_gate_failure_when_prompts_leak_across_splits():
    corpus = SuiteCorpus()
    shared = "identical prompt"
    corpus.examples = [
        make({"question": shared, "answer": "a", "label": "M1-direct@q_claude1",
              "topic": "challenger_disaster"}, "forget_train"),
        melt_rephrasing_row(
            {"question": "x", "answer": "a", "label": "M1-direct",
             "topic": "challenger_disaster", "q_gemini1": shared},
            row_index=0,
        )[1],
    ]
    corpus.examples[0] = replace(corpus.examples[0], split=Split.DISCOVERY)
    corpus.examples[1] = replace(corpus.examples[1], split=Split.VALIDATION)

    audit = build_dataset_audit(corpus, run_near_duplicates=False)
    assert audit["gate_g0_passed"] is False
    assert any(f["check"] == "no_identical_prompt_across_splits" for f in audit["gate_g0_failures"])
