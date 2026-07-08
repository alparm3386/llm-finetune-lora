"""Shared prompt construction for training and evaluation (dev-plan step 3.4.1).

Defines the single Hungarian text -> JSON extraction prompt format used by
both `train.py` (fine-tuning targets) and `evaluate.py` (base/fine-tuned
generation), so both stages see an identical prompt. Reuses the domain JSON
Schemas from `schemas.py` as the source of truth for the schema embedded in
the prompt.
"""

from __future__ import annotations

import json
from typing import Any

from schemas import SCHEMAS

INSTRUCTION_TEMPLATE = (
    "Az alábbi magyar nyelvű szövegből nyerd ki a kért adatokat, és add "
    "vissza PONTOSAN egy, a megadott JSON sémának megfelelő JSON objektumot. "
    "Ha egy mező értéke nem szerepel a szövegben, használj null-t (vagy üres "
    "tömböt, ha a mező tömb típusú). Ne adj hozzá magyarázatot vagy egyéb "
    "szöveget a JSON objektumon kívül.\n\n"
    "JSON séma:\n{schema}\n\n"
    "Szöveg:\n{document}"
)


def build_prompt(domain: str, document: str) -> str:
    """Build the user-turn instruction text for one extraction example."""
    schema_text = json.dumps(SCHEMAS[domain], ensure_ascii=False, indent=2)
    return INSTRUCTION_TEMPLATE.format(schema=schema_text, document=document.strip())


def serialize_gold(gold: dict[str, Any], domain: str) -> str:
    """Serialize a gold JSON object deterministically, in schema key order.

    Key order follows `SCHEMAS[domain]["properties"]` rather than `gold`'s own
    (arbitrary) key order, so the training target format is identical across
    examples regardless of how the source JSONL happened to order keys.
    """
    key_order = list(SCHEMAS[domain]["properties"])
    ordered = {key: gold[key] for key in key_order}
    return json.dumps(ordered, ensure_ascii=False)


def to_chat_messages(
    domain: str, document: str, gold: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    """Build Gemma chat-format messages for one example.

    With `gold` given (training), returns a [user, assistant] pair where the
    assistant turn is the serialized gold JSON — the training target. Without
    `gold` (inference/eval), returns just the [user] turn, ready for
    `apply_chat_template(..., add_generation_prompt=True)`.
    """
    messages = [{"role": "user", "content": build_prompt(domain, document)}]
    if gold is not None:
        messages.append({"role": "assistant", "content": serialize_gold(gold, domain)})
    return messages
