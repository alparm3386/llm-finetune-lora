"""Unit tests for `eval_metrics.py` (dev-plan step 3.5.9) — the metric heart of
step 3.5. Hand-built cases for decision A's TP/FP/FN table (nulls,
hallucinations, wrong values, array set-F1), normalization, unparseable
predictions, and aggregation. Pure functions, no model/torch deps.
"""

import pytest

from eval_metrics import (
    aggregate_field_scores,
    classify_array_field,
    classify_field,
    classify_scalar_field,
    is_array_field,
    is_schema_valid,
    is_valid_json,
    normalize_value,
    parse_prediction,
    precision_recall_f1,
    score_example,
    strip_json_fence,
    validity_rate,
)
from schemas import SCHEMAS

MEDICAL_SCHEMA = SCHEMAS["medical"]


# --- normalize_value -----------------------------------------------------------


def test_normalize_value_strips_and_collapses_whitespace():
    assert normalize_value("  Algopyrin   500mg  ") == "Algopyrin 500mg"


def test_normalize_value_is_case_sensitive():
    assert normalize_value("Algopyrin") != normalize_value("algopyrin")


def test_normalize_value_compares_numbers_as_float():
    assert normalize_value(620000000) == normalize_value(620000000.0)


def test_normalize_value_bool_passes_through_unchanged():
    # bools are excluded from the numeric branch (isinstance(x, int) is True for bool)
    assert normalize_value(True) is True


def test_normalize_value_none_passes_through():
    assert normalize_value(None) is None


# --- classify_scalar_field (decision A's table) ---------------------------------


def test_classify_scalar_both_null_is_ignored():
    assert classify_scalar_field(None, None) == (0, 0, 0)


def test_classify_scalar_hallucination_is_fp():
    assert classify_scalar_field(None, "Algopyrin") == (0, 1, 0)


def test_classify_scalar_miss_is_fn():
    assert classify_scalar_field("Algopyrin", None) == (0, 0, 1)


def test_classify_scalar_correct_is_tp():
    assert classify_scalar_field("Algopyrin", "Algopyrin") == (1, 0, 0)


def test_classify_scalar_wrong_value_is_fp_and_fn():
    assert classify_scalar_field("Algopyrin", "Aspirin") == (0, 1, 1)


def test_classify_scalar_normalizes_before_comparing():
    # whitespace-only difference should still count as a match
    assert classify_scalar_field("Algopyrin 500mg", "  Algopyrin   500mg ") == (1, 0, 0)


def test_classify_scalar_numeric_equality_across_int_float():
    assert classify_scalar_field(620000000, 620000000.0) == (1, 0, 0)


# --- classify_array_field -------------------------------------------------------


def test_classify_array_perfect_match():
    assert classify_array_field(["a", "b"], ["a", "b"]) == (2, 0, 0)


def test_classify_array_partial_overlap():
    # gold={a,b}, pred={b,c} -> TP={b}=1, FP={c}=1, FN={a}=1
    assert classify_array_field(["a", "b"], ["b", "c"]) == (1, 1, 1)


def test_classify_array_both_empty_contributes_nothing():
    assert classify_array_field([], []) == (0, 0, 0)


def test_classify_array_none_treated_as_empty():
    assert classify_array_field(None, None) == (0, 0, 0)
    assert classify_array_field(["a"], None) == (0, 0, 1)
    assert classify_array_field(None, ["a"]) == (0, 1, 0)


def test_classify_array_deduplicates_via_set():
    assert classify_array_field(["a", "a"], ["a"]) == (1, 0, 0)


# --- is_array_field / classify_field dispatch -----------------------------------


def test_is_array_field():
    assert is_array_field(MEDICAL_SCHEMA["properties"]["side_effects"]) is True
    assert is_array_field(MEDICAL_SCHEMA["properties"]["drug_name"]) is False


def test_classify_field_dispatches_array_vs_scalar():
    assert classify_field(MEDICAL_SCHEMA["properties"]["side_effects"], ["a"], ["a"]) == (1, 0, 0)
    assert classify_field(MEDICAL_SCHEMA["properties"]["drug_name"], "x", "x") == (1, 0, 0)


# --- score_example ---------------------------------------------------------------


GOLD_MEDICAL = {
    "drug_name": "Algopyrin",
    "active_ingredient": "metamizol-nátrium",
    "indication": "fájdalomcsillapítás",
    "dosage": None,
    "side_effects": ["hányinger", "szédülés"],
    "contraindications": [],
}


def test_score_example_perfect_prediction_all_tp_or_ignored():
    counts = score_example(MEDICAL_SCHEMA, GOLD_MEDICAL, dict(GOLD_MEDICAL))
    assert counts["drug_name"] == (1, 0, 0)
    assert counts["dosage"] == (0, 0, 0)  # both null, ignored
    assert counts["side_effects"] == (2, 0, 0)


def test_score_example_unparseable_prediction_scores_as_empty():
    # pred=None (decision A): every non-null gold field becomes an FN, no FPs
    counts = score_example(MEDICAL_SCHEMA, GOLD_MEDICAL, None)
    assert counts["drug_name"] == (0, 0, 1)
    assert counts["dosage"] == (0, 0, 0)  # gold null vs. missing -> still ignored
    assert counts["side_effects"] == (0, 0, 2)
    assert all(fp == 0 for (_, fp, _) in counts.values())


def test_score_example_missing_pred_field_is_fn():
    partial_pred = {"drug_name": "Algopyrin"}
    counts = score_example(MEDICAL_SCHEMA, GOLD_MEDICAL, partial_pred)
    assert counts["drug_name"] == (1, 0, 0)
    assert counts["active_ingredient"] == (0, 0, 1)


# --- precision_recall_f1 ----------------------------------------------------------


def test_precision_recall_f1_normal_case():
    result = precision_recall_f1(tp=8, fp=2, fn=2)
    assert result["precision"] == 0.8
    assert result["recall"] == 0.8
    assert result["f1"] == pytest.approx(0.8)


def test_precision_recall_f1_zero_denominators_give_zero():
    result = precision_recall_f1(tp=0, fp=0, fn=0)
    assert result == {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


# --- aggregate_field_scores --------------------------------------------------------


def test_aggregate_field_scores_pools_overall_domain_and_field():
    records = [
        ("medical", {"drug_name": (1, 0, 0), "dosage": (0, 1, 0)}),
        ("medical", {"drug_name": (0, 0, 1), "dosage": (0, 0, 0)}),
        ("business", {"company": (1, 0, 0)}),
    ]
    agg = aggregate_field_scores(records)

    # overall pools every field across every example
    assert agg["overall"]["tp"] == 2
    assert agg["overall"]["fp"] == 1
    assert agg["overall"]["fn"] == 1

    # per-domain pools within a domain
    assert agg["per_domain"]["medical"]["tp"] == 1
    assert agg["per_domain"]["medical"]["fn"] == 1
    assert agg["per_domain"]["business"]["tp"] == 1

    # per-field is keyed "{domain}.{field}" and pools across examples
    assert agg["per_field"]["medical.drug_name"]["tp"] == 1
    assert agg["per_field"]["medical.drug_name"]["fn"] == 1
    assert agg["per_field"]["medical.dosage"]["fp"] == 1
    assert agg["per_field"]["business.company"]["tp"] == 1


def test_aggregate_field_scores_empty_records():
    agg = aggregate_field_scores([])
    assert agg["overall"] == {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert agg["per_domain"] == {}
    assert agg["per_field"] == {}


# --- strip_json_fence / parse_prediction / is_valid_json --------------------------


def test_strip_json_fence_removes_json_language_tag():
    text = '```json\n{"a": 1}\n```'
    assert strip_json_fence(text) == '{"a": 1}'


def test_strip_json_fence_removes_bare_fence():
    text = '```\n{"a": 1}\n```'
    assert strip_json_fence(text) == '{"a": 1}'


def test_strip_json_fence_passthrough_when_no_fence():
    assert strip_json_fence('{"a": 1}') == '{"a": 1}'


def test_parse_prediction_valid_json_object():
    assert parse_prediction('{"a": 1}') == {"a": 1}


def test_parse_prediction_valid_json_inside_fence():
    assert parse_prediction('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_prediction_unparseable_returns_none():
    assert parse_prediction("not json at all") is None


def test_parse_prediction_non_object_json_returns_none():
    # a JSON array or scalar is not a prediction dict
    assert parse_prediction("[1, 2, 3]") is None
    assert parse_prediction("42") is None


def test_is_valid_json_true_and_false():
    assert is_valid_json('{"a": 1}') is True
    assert is_valid_json("garbage") is False


# --- is_schema_valid ---------------------------------------------------------------


def test_is_schema_valid_accepts_conforming_payload():
    assert is_schema_valid(GOLD_MEDICAL, MEDICAL_SCHEMA) is True


def test_is_schema_valid_rejects_missing_required_field():
    bad = dict(GOLD_MEDICAL)
    del bad["drug_name"]
    assert is_schema_valid(bad, MEDICAL_SCHEMA) is False


def test_is_schema_valid_rejects_additional_property():
    bad = dict(GOLD_MEDICAL)
    bad["extra"] = "surprise"
    assert is_schema_valid(bad, MEDICAL_SCHEMA) is False


# --- validity_rate -------------------------------------------------------------------


def test_validity_rate_computes_fraction():
    assert validity_rate([True, True, False, True]) == 0.75


def test_validity_rate_empty_list_is_zero():
    assert validity_rate([]) == 0.0
