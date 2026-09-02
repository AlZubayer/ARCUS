from arcus.module_a.schema import ControlType, Modality, classify_retain_label, parse_forget_label


def test_parse_forget_label():
    key, modality = parse_forget_label("challenger_disaster", "M3-direct")
    assert key is not None
    assert key.fact_id == "M3"
    assert modality == Modality.DIRECT


def test_parse_reverse():
    key, modality = parse_forget_label("challenger_disaster", "K1-reverse")
    assert key is not None
    assert key.fact_id == "K1"
    assert modality == Modality.REVERSE


def test_unknown_forget_label_is_not_guessed():
    key, modality = parse_forget_label("challenger_disaster", "mystery-label")
    assert key is None
    assert modality == Modality.UNKNOWN


def test_retain_labels():
    kind, tier = classify_retain_label("Semantic-4-Challenger")
    assert kind == ControlType.SEMANTIC
    assert tier == 4
    kind, tier = classify_retain_label("Lexical-Challenger")
    assert kind == ControlType.LEXICAL
    assert tier is None
