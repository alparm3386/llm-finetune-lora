"""Unit tests for `prompt_format.py` (dev-plan step 3.4.7)."""

import json

import pytest

from prompt_format import build_prompt, serialize_gold, to_chat_messages
from schemas import SCHEMAS

DOMAINS = tuple(SCHEMAS)

BUSINESS_GOLD_SHUFFLED_KEYS = {
    "involved_parties": ["Aurora Kft.", "Zenit Zrt."],
    "date": "2025-03-14",
    "amount": 620000000,
    "company": "Aurora Kft.",
    "event_type": "felvásárlás",
    "currency": "EUR",
}


@pytest.mark.parametrize("domain", DOMAINS)
def test_build_prompt_embeds_schema_and_document(domain):
    document = "Ez egy teszt magyar szöveg egy egyedi jelölővel: XYZTOKEN123."
    prompt = build_prompt(domain, document)

    assert "XYZTOKEN123" in prompt
    # The full domain schema (all property names) must be embedded verbatim.
    for field in SCHEMAS[domain]["properties"]:
        assert field in prompt


def test_build_prompt_strips_document_whitespace():
    prompt = build_prompt("technology", "  \n  padded document  \n  ")
    assert "padded document" in prompt
    assert "\n  padded document" not in prompt


def test_serialize_gold_uses_schema_key_order_regardless_of_input_order():
    result = serialize_gold(BUSINESS_GOLD_SHUFFLED_KEYS, "business")
    expected_order = list(SCHEMAS["business"]["properties"])
    assert list(json.loads(result).keys()) == expected_order


def test_serialize_gold_is_ensure_ascii_false():
    gold = {
        "drug_name": "Fájdalomcsillapító Forte",
        "active_ingredient": "ibuprofén",
        "indication": "láz",
        "dosage": None,
        "side_effects": [],
        "contraindications": [],
    }
    result = serialize_gold(gold, "medical")
    assert "Fájdalomcsillapító" in result  # not escaped to á etc.
    assert "\\u" not in result


def test_serialize_gold_roundtrips_values():
    gold = {
        "drug_name": "X",
        "active_ingredient": "Y",
        "indication": "Z",
        "dosage": None,
        "side_effects": ["a", "b"],
        "contraindications": [],
    }
    assert json.loads(serialize_gold(gold, "medical")) == gold


def test_to_chat_messages_without_gold_returns_user_turn_only():
    messages = to_chat_messages("technology", "dokumentum szövege")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_to_chat_messages_with_gold_appends_assistant_turn():
    gold = {
        "product": "Foo",
        "manufacturer": "Bar",
        "version": None,
        "key_specs": [],
        "release_date": None,
        "price": None,
    }
    messages = to_chat_messages("technology", "dokumentum szövege", gold)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert json.loads(messages[1]["content"]) == gold
