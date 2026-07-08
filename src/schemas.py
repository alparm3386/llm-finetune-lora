"""Domain JSON Schemas for the Hungarian text-to-JSON extraction task.

Defines the target extraction schema for each of the three project domains
(medical, business, technology), as specified in SCOPE.md. Keys are in
English; the source documents are Hungarian. Fields that may legitimately be
missing from a document are typed as nullable so the model can learn to
emit `null` / an empty array rather than hallucinate a value.

Used by both `generate_data.py` (to validate synthetically generated gold
JSON) and, later, `evaluate.py` (to validate/score model predictions and to
drive structured decoding).
"""

MEDICAL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "drug_name": {"type": "string"},
        "active_ingredient": {"type": "string"},
        "indication": {"type": "string"},
        "dosage": {"type": ["string", "null"]},
        "side_effects": {"type": "array", "items": {"type": "string"}},
        "contraindications": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "drug_name",
        "active_ingredient",
        "indication",
        "dosage",
        "side_effects",
        "contraindications",
    ],
    "additionalProperties": False,
}

BUSINESS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "event_type": {"type": "string"},
        "amount": {"type": ["number", "null"]},
        "currency": {"type": ["string", "null"]},
        "date": {"type": ["string", "null"]},
        "involved_parties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "company",
        "event_type",
        "amount",
        "currency",
        "date",
        "involved_parties",
    ],
    "additionalProperties": False,
}

TECHNOLOGY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "manufacturer": {"type": "string"},
        "version": {"type": ["string", "null"]},
        "key_specs": {"type": "array", "items": {"type": "string"}},
        "release_date": {"type": ["string", "null"]},
        "price": {"type": ["number", "null"]},
    },
    "required": [
        "product",
        "manufacturer",
        "version",
        "key_specs",
        "release_date",
        "price",
    ],
    "additionalProperties": False,
}

SCHEMAS: dict[str, dict] = {
    "medical": MEDICAL_SCHEMA,
    "business": BUSINESS_SCHEMA,
    "technology": TECHNOLOGY_SCHEMA,
}
