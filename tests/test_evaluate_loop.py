"""Unit tests for the eval loop + aggregation (dev-plan step 3.5.5).

Exercises `generate_predictions` / `score_predictions` / `run_eval_cell` /
`run_evaluation` as pure orchestration: `generate_prompt_only` /
`generate_structured` / `build_structured_generators` are monkeypatched so these
tests need neither `torch` nor `outlines`, matching the rest of the local
(GPU-free) test suite.
"""

import pytest

import evaluate as ev

MEDICAL_SCHEMA_GOLD = {
    "drug_name": "Algopyrin",
    "active_ingredient": "metamizol-nátrium",
    "indication": "fájdalomcsillapítás",
    "dosage": None,
    "side_effects": ["hányinger"],
    "contraindications": [],
}

BUSINESS_GOLD = {
    "company": "OTP Bank Nyrt.",
    "event_type": "felvásárlás",
    "amount": 620000000,
    "currency": "EUR",
    "date": "2023-02-07",
    "involved_parties": ["OTP Bank Nyrt."],
}


def make_examples():
    return [
        {"domain": "medical", "document": "doc1", "gold": MEDICAL_SCHEMA_GOLD},
        {"domain": "business", "document": "doc2", "gold": BUSINESS_GOLD},
    ]


# --- apply_limit ----------------------------------------------------------------


def test_apply_limit_none_returns_all():
    examples = make_examples()
    assert ev.apply_limit(examples, None) == examples


def test_apply_limit_truncates():
    examples = make_examples()
    assert ev.apply_limit(examples, 1) == examples[:1]


# --- generate_predictions --------------------------------------------------------


def test_generate_predictions_prompt_only_parses_valid_and_invalid(monkeypatch):
    texts = iter(['{"drug_name": "Algopyrin", "active_ingredient": "x", "indication": "y", "dosage": null, "side_effects": [], "contraindications": []}', "not json"])
    monkeypatch.setattr(ev, "build_inference_prompt", lambda tokenizer, domain, document: document)
    monkeypatch.setattr(ev, "generate_prompt_only", lambda *a, **k: next(texts))

    predictions = ev.generate_predictions(
        model=object(), tokenizer=object(), examples=make_examples(), mode="prompt_only"
    )

    assert predictions[0]["valid"] is True
    assert predictions[0]["pred"]["drug_name"] == "Algopyrin"
    assert predictions[1]["valid"] is False
    assert predictions[1]["pred"] is None
    assert [p["domain"] for p in predictions] == ["medical", "business"]


def test_generate_predictions_structured_uses_per_domain_generator(monkeypatch):
    calls = []

    def fake_generate_structured(generator, prompt, max_new_tokens, seed):
        calls.append(generator)
        return '{"company": "OTP Bank Nyrt.", "event_type": "e", "amount": 1, "currency": null, "date": null, "involved_parties": []}'

    monkeypatch.setattr(ev, "build_inference_prompt", lambda tokenizer, domain, document: document)
    monkeypatch.setattr(ev, "generate_structured", fake_generate_structured)
    generators = {"medical": "medical-gen", "business": "business-gen"}

    predictions = ev.generate_predictions(
        model=object(),
        tokenizer=object(),
        examples=make_examples(),
        mode="structured",
        structured_generators=generators,
    )

    assert calls == ["medical-gen", "business-gen"]
    assert all(p["valid"] for p in predictions)


def test_generate_predictions_unknown_mode_raises(monkeypatch):
    monkeypatch.setattr(ev, "build_inference_prompt", lambda tokenizer, domain, document: document)
    with pytest.raises(ValueError, match="Unknown decode mode"):
        ev.generate_predictions(object(), object(), make_examples(), mode="bogus")


# --- score_predictions ------------------------------------------------------------


def test_score_predictions_perfect_match_gives_f1_one_and_full_validity():
    examples = make_examples()
    predictions = [
        {"domain": ex["domain"], "raw_text": "…", "pred": ex["gold"], "valid": True} for ex in examples
    ]

    result = ev.score_predictions(examples, predictions)

    assert result["f1"]["overall"]["f1"] == 1.0
    assert result["validity_rate"] == 1.0


def test_score_predictions_unparseable_prediction_hurts_recall_not_validity_of_others():
    examples = make_examples()
    predictions = [
        {"domain": "medical", "raw_text": "bad", "pred": None, "valid": False},
        {"domain": "business", "raw_text": "…", "pred": examples[1]["gold"], "valid": True},
    ]

    result = ev.score_predictions(examples, predictions)

    assert result["validity_rate"] == 0.5
    assert result["f1"]["overall"]["fn"] > 0  # missed medical fields
    assert result["f1"]["overall"]["fp"] == 0  # no hallucinated fields


# --- run_eval_cell -----------------------------------------------------------------


def test_run_eval_cell_prompt_only(monkeypatch):
    monkeypatch.setattr(ev, "build_inference_prompt", lambda tokenizer, domain, document: document)
    monkeypatch.setattr(
        ev,
        "generate_prompt_only",
        lambda model, tokenizer, prompt, max_new_tokens, seed: '{"drug_name": null, "active_ingredient": null, "indication": null, "dosage": null, "side_effects": [], "contraindications": []}',
    )
    result = ev.run_eval_cell(object(), object(), [make_examples()[0]], mode="prompt_only")
    assert set(result.keys()) == {"f1", "validity_rate"}


# --- run_evaluation ------------------------------------------------------------------


def test_run_evaluation_assembles_full_2x2(monkeypatch):
    monkeypatch.setattr(ev, "build_inference_prompt", lambda tokenizer, domain, document: document)
    monkeypatch.setattr(
        ev, "build_structured_generators", lambda model, tokenizer: {"business": "gen"}
    )
    monkeypatch.setattr(
        ev, "generate_prompt_only", lambda *a, **k: '{"company": "x", "event_type": "y", "amount": null, "currency": null, "date": null, "involved_parties": []}'
    )
    monkeypatch.setattr(
        ev, "generate_structured", lambda *a, **k: '{"company": "x", "event_type": "y", "amount": null, "currency": null, "date": null, "involved_parties": []}'
    )

    results = ev.run_evaluation(
        base_model=object(),
        base_tokenizer=object(),
        fine_tuned_model=object(),
        fine_tuned_tokenizer=object(),
        examples=[make_examples()[1]],
    )

    assert set(results.keys()) == {
        "base.prompt_only",
        "base.structured",
        "fine_tuned.prompt_only",
        "fine_tuned.structured",
    }


def test_run_evaluation_structured_only_skips_prompt_only_cells(monkeypatch):
    monkeypatch.setattr(ev, "build_inference_prompt", lambda tokenizer, domain, document: document)
    monkeypatch.setattr(
        ev, "build_structured_generators", lambda model, tokenizer: {"business": "gen"}
    )
    monkeypatch.setattr(
        ev, "generate_structured", lambda *a, **k: '{"company": "x", "event_type": "y", "amount": null, "currency": null, "date": null, "involved_parties": []}'
    )

    results = ev.run_evaluation(
        base_model=object(),
        base_tokenizer=object(),
        fine_tuned_model=object(),
        fine_tuned_tokenizer=object(),
        examples=[make_examples()[1]],
        structured_only=True,
    )

    assert set(results.keys()) == {"base.structured", "fine_tuned.structured"}
