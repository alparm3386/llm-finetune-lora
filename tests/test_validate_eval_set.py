"""Unit tests for the eval-set validator and loader (dev-plan steps 3.5.0 / 3.5.2)."""

import json

import pytest

from evaluate import load_eval_examples
from validate_eval_set import (
    EVAL_DOMAINS,
    EvalSetError,
    load_domain_examples,
    parse_args,
    validate_eval_set,
    validate_example,
)

# Minimal schema-valid gold per domain (all required keys, correct typing).
VALID_GOLD = {
    "medical": {
        "drug_name": "Algopyrin",
        "active_ingredient": "metamizol-nátrium",
        "indication": "fájdalomcsillapítás",
        "dosage": None,
        "side_effects": ["hányinger"],
        "contraindications": [],
    },
    "business": {
        "company": "OTP Bank Nyrt.",
        "event_type": "felvásárlás",
        "amount": 620000000,
        "currency": "EUR",
        "date": "2023-02-07",
        "involved_parties": ["OTP Bank Nyrt."],
    },
    "technology": {
        "product": "Latitude 5440",
        "manufacturer": "Dell",
        "version": None,
        "key_specs": ["16 GB RAM"],
        "release_date": None,
        "price": None,
    },
}


def write_eval(eval_dir, per_domain=None):
    """Write one valid example per domain unless `per_domain` overrides the rows."""
    for domain in EVAL_DOMAINS:
        rows = per_domain[domain] if per_domain else [
            {"domain": domain, "document": f"{domain} doc", "gold": VALID_GOLD[domain]}
        ]
        path = eval_dir / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- validate_example ---------------------------------------------------------

def test_validate_example_accepts_valid():
    example = {"domain": "medical", "document": "x", "gold": VALID_GOLD["medical"]}
    assert validate_example(example, "medical") == []


def test_validate_example_flags_wrong_domain():
    example = {"domain": "business", "document": "x", "gold": VALID_GOLD["medical"]}
    problems = validate_example(example, "medical")
    assert any("domain" in p for p in problems)


def test_validate_example_flags_empty_document():
    example = {"domain": "medical", "document": "   ", "gold": VALID_GOLD["medical"]}
    problems = validate_example(example, "medical")
    assert any("document" in p for p in problems)


def test_validate_example_flags_schema_violation():
    bad_gold = dict(VALID_GOLD["medical"])
    del bad_gold["drug_name"]  # required key missing
    example = {"domain": "medical", "document": "x", "gold": bad_gold}
    problems = validate_example(example, "medical")
    assert any("schema" in p for p in problems)


def test_validate_example_flags_additional_property():
    bad_gold = dict(VALID_GOLD["medical"])
    bad_gold["surprise"] = "extra"  # additionalProperties: false
    example = {"domain": "medical", "document": "x", "gold": bad_gold}
    assert validate_example(example, "medical") != []


def test_validate_example_flags_wrong_number_type():
    bad_gold = dict(VALID_GOLD["business"])
    bad_gold["amount"] = "620000000"  # should be number|null, not string
    example = {"domain": "business", "document": "x", "gold": bad_gold}
    assert validate_example(example, "business") != []


# --- load_domain_examples -----------------------------------------------------

def test_load_domain_examples_numbers_lines_and_skips_blanks(tmp_path):
    path = tmp_path / "medical.jsonl"
    row = {"domain": "medical", "document": "x", "gold": VALID_GOLD["medical"]}
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n\n" + json.dumps(row) + "\n")  # blank middle line

    examples = load_domain_examples(tmp_path, "medical")
    assert [ln for ln, _ in examples] == [1, 3]  # blank line 2 skipped, numbering preserved


def test_load_domain_examples_missing_file_raises(tmp_path):
    with pytest.raises(EvalSetError, match="not found"):
        load_domain_examples(tmp_path, "medical")


def test_load_domain_examples_bad_json_raises(tmp_path):
    (tmp_path / "medical.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(EvalSetError, match="invalid JSON"):
        load_domain_examples(tmp_path, "medical")


# --- validate_eval_set --------------------------------------------------------

def test_validate_eval_set_all_clean(tmp_path):
    write_eval(tmp_path)
    report = validate_eval_set(tmp_path)
    assert all(errors == [] for errors in report.values())


def test_validate_eval_set_reports_empty_file(tmp_path):
    write_eval(tmp_path)
    (tmp_path / "medical.jsonl").write_text("", encoding="utf-8")
    report = validate_eval_set(tmp_path)
    assert any("no examples" in e for e in report["medical"])
    assert report["business"] == []  # other domains unaffected


def test_validate_eval_set_reports_missing_file_without_hiding_others(tmp_path):
    write_eval(tmp_path)
    (tmp_path / "medical.jsonl").unlink()
    report = validate_eval_set(tmp_path)
    assert report["medical"] and report["business"] == []


# --- load_eval_examples (evaluate.py) -----------------------------------------

def test_load_eval_examples_flattens_all_domains(tmp_path):
    write_eval(tmp_path)
    examples = load_eval_examples(tmp_path)
    assert len(examples) == len(EVAL_DOMAINS)
    assert {ex["domain"] for ex in examples} == set(EVAL_DOMAINS)


def test_load_eval_examples_raises_on_missing_domain(tmp_path):
    write_eval(tmp_path)
    (tmp_path / "technology.jsonl").unlink()
    with pytest.raises(EvalSetError):
        load_eval_examples(tmp_path)


def test_load_eval_examples_raises_on_empty_domain(tmp_path):
    write_eval(tmp_path)
    (tmp_path / "business.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(EvalSetError, match="empty"):
        load_eval_examples(tmp_path)


def test_load_eval_examples_validates_gold_by_default(tmp_path):
    bad_gold = dict(VALID_GOLD["medical"])
    del bad_gold["indication"]
    write_eval(tmp_path, per_domain={
        "medical": [{"domain": "medical", "document": "x", "gold": bad_gold}],
        "business": [{"domain": "business", "document": "x", "gold": VALID_GOLD["business"]}],
        "technology": [{"domain": "technology", "document": "x", "gold": VALID_GOLD["technology"]}],
    })
    with pytest.raises(EvalSetError, match="invalid eval example"):
        load_eval_examples(tmp_path)


def test_load_eval_examples_skip_validation(tmp_path):
    bad_gold = dict(VALID_GOLD["medical"])
    del bad_gold["indication"]
    write_eval(tmp_path, per_domain={
        "medical": [{"domain": "medical", "document": "x", "gold": bad_gold}],
        "business": [{"domain": "business", "document": "x", "gold": VALID_GOLD["business"]}],
        "technology": [{"domain": "technology", "document": "x", "gold": VALID_GOLD["technology"]}],
    })
    examples = load_eval_examples(tmp_path, validate=False)  # no raise
    assert len(examples) == len(EVAL_DOMAINS)


# --- CLI ----------------------------------------------------------------------

def test_parse_args_defaults():
    args = parse_args([])
    assert args.eval_dir == "data/eval"
    assert args.domain is None


def test_parse_args_domain_choice():
    args = parse_args(["--domain", "medical"])
    assert args.domain == "medical"


def test_parse_args_rejects_unknown_domain():
    with pytest.raises(SystemExit):
        parse_args(["--domain", "finance"])
