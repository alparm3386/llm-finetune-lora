"""Pure metric logic for the before/after evaluation (dev-plan step 3.5.1).

Implements decision A from `DEV_PLAN_3.5.md`: per-field exact-match F1 (TP/FP/FN
classification per field, pooled into micro-F1) plus JSON validity checking.
Zero torch/outlines/model dependencies, so this module is fully unit-testable
without a GPU; `evaluate.py` drives it with real model output.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from jsonschema import ValidationError, validate

# (true_positives, false_positives, false_negatives) for one field on one example.
FieldCounts = tuple[int, int, int]


def normalize_value(value: Any) -> Any:
    """Normalize a scalar value for exact-match comparison (decision A).

    Strings: strip + collapse internal whitespace, case-sensitive (Hungarian
    proper nouns must match exactly). Numbers: compared as float, so
    `620000000 == 620000000.0`. `None` and everything else pass through as-is
    (dates are plain strings in this schema, so no separate date handling).
    """
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return value


def classify_scalar_field(gold: Any, pred: Any) -> FieldCounts:
    """Classify one scalar field's (gold, pred) pair into (tp, fp, fn) per decision A's table."""
    g = normalize_value(gold)
    p = normalize_value(pred)
    if g is None and p is None:
        return (0, 0, 0)
    if g is None:
        return (0, 1, 0)
    if p is None:
        return (0, 0, 1)
    if g == p:
        return (1, 0, 0)
    return (0, 1, 1)


def classify_array_field(gold: list[Any] | None, pred: list[Any] | None) -> FieldCounts:
    """Classify one array field via set overlap: TP = |G∩P|, FP = |P∖G|, FN = |G∖P|."""
    g = {normalize_value(v) for v in (gold or [])}
    p = {normalize_value(v) for v in (pred or [])}
    return (len(g & p), len(p - g), len(g - p))


def is_array_field(field_schema: dict[str, Any]) -> bool:
    return field_schema.get("type") == "array"


def classify_field(field_schema: dict[str, Any], gold: Any, pred: Any) -> FieldCounts:
    """Dispatch to the scalar or array classifier based on the field's JSON Schema type."""
    if is_array_field(field_schema):
        return classify_array_field(gold, pred)
    return classify_scalar_field(gold, pred)


def score_example(
    schema: dict[str, Any], gold: dict[str, Any], pred: dict[str, Any] | None
) -> dict[str, FieldCounts]:
    """Score one example's prediction against gold, per field, per decision A.

    `pred=None` represents an unparseable prediction (prompt-only mode): every
    field is scored as if predicted empty/missing, so non-null gold fields
    become FNs and no FPs are introduced.
    """
    pred = pred or {}
    return {
        field_name: classify_field(field_schema, gold.get(field_name), pred.get(field_name))
        for field_name, field_schema in schema["properties"].items()
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Compute precision/recall/F1 from pooled counts. F1 = 0 when its denominator is 0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _pool(counts: list[FieldCounts]) -> dict[str, float]:
    tp = sum(c[0] for c in counts)
    fp = sum(c[1] for c in counts)
    fn = sum(c[2] for c in counts)
    return precision_recall_f1(tp, fp, fn)


def aggregate_field_scores(
    records: list[tuple[str, dict[str, FieldCounts]]],
) -> dict[str, Any]:
    """Pool per-example field counts into overall / per-domain / per-field micro-F1.

    `records` is a list of `(domain, field_counts)` pairs, one per scored
    example (as returned by `score_example`). Per-field is keyed as
    `"{domain}.{field}"` since field names aren't shared across domains.
    """
    overall: list[FieldCounts] = []
    per_domain: dict[str, list[FieldCounts]] = defaultdict(list)
    per_field: dict[str, list[FieldCounts]] = defaultdict(list)

    for domain, field_counts in records:
        for field_name, counts in field_counts.items():
            overall.append(counts)
            per_domain[domain].append(counts)
            per_field[f"{domain}.{field_name}"].append(counts)

    return {
        "overall": _pool(overall),
        "per_domain": {domain: _pool(counts) for domain, counts in per_domain.items()},
        "per_field": {key: _pool(counts) for key, counts in per_field.items()},
    }


def strip_json_fence(text: str) -> str:
    """Strip a leading/trailing Markdown code fence (```json ... ``` or ``` ... ```).

    Duplicated from `generate_data.py` (rather than imported) to keep this
    module free of `openai`/model-loading imports, per decision C.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


def parse_prediction(text: str) -> dict[str, Any] | None:
    """Parse raw model output as a JSON object. Returns `None` if unparseable (decision A)."""
    try:
        payload = json.loads(strip_json_fence(text))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def is_valid_json(text: str) -> bool:
    """Secondary metric: did the model produce a parseable JSON object at all?"""
    return parse_prediction(text) is not None


def is_schema_valid(payload: dict[str, Any], schema: dict[str, Any]) -> bool:
    """Whether a parsed prediction also conforms to the domain JSON Schema."""
    try:
        validate(instance=payload, schema=schema)
    except ValidationError:
        return False
    return True


def validity_rate(results: list[bool]) -> float:
    """Fraction of `True` in a list of per-example validity checks."""
    return sum(results) / len(results) if results else 0.0
