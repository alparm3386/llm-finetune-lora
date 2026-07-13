# Evaluation set — labeling guide

This directory holds the **real, hand-labeled** Hungarian evaluation set used by
`src/evaluate.py` (dev-plan step 3.5) for the before/after comparison. Unlike the
LLM-generated training data in `data/synthetic/` (regenerable, gitignored), these
documents are **real text** with **human-checked gold JSON**, and they are
**tracked in git** — they are the ground truth the headline metric is measured
against, so their provenance and correctness matter.

> **Why a separate real set?** The synthetic data trains the model; a metric
> measured on synthetic data would just reward memorizing the generator's quirks.
> A held-out set of genuine Hungarian documents is what makes the reported
> per-field F1 credible (see `SCOPE.md`).

## File format

One file per domain, same `{domain, document, gold}` JSONL shape as the synthetic
data, so the loaders in `train.py` / `evaluate.py` read both identically:

- `medical.jsonl`
- `business.jsonl`
- `technology.jsonl`

Each line is one example:

```json
{"domain": "medical", "document": "…magyar nyelvű forrásszöveg…", "gold": {"drug_name": "…", "…": null}}
```

- **`domain`** — `"medical" | "business" | "technology"` (must match the filename).
- **`document`** — the raw Hungarian source text, verbatim (light cleanup only:
  strip boilerplate headers/footers, page numbers, navigation). Keep it a
  realistic length — a paragraph to a page.
- **`gold`** — the hand-extracted target JSON. It must **validate against the
  domain schema** in `src/schemas.py` (run the validator below). Every schema key
  must be present, in the schema's typing:
  - Missing scalar → `null` (never omit the key, never invent a value).
  - Missing list → `[]`.
  - Numbers as JSON numbers (`amount`, `price`), not strings.
  - Dates as strings, ISO `YYYY-MM-DD` where the source gives a full date;
    otherwise transcribe what the source states.

The schemas are the single source of truth — see the field tables in `SCOPE.md`.

## Sourcing (PII-free, realistic)

Aim for **a few dozen documents per domain**, varied in style and in how many
fields are actually present (some docs should legitimately leave nullable fields
empty — that is what tests whether the model emits `null` instead of
hallucinating).

- **medical** — public drug leaflets / patient information (e.g. OGYÉI /
  EMA-published Hungarian *betegtájékoztató* PDFs). Public regulatory documents,
  no patient data.
- **business** — Hungarian business/financial news (acquisitions, funding,
  results). Public reporting only.
- **technology** — product spec pages, press releases, review intros for consumer
  tech in Hungarian.

**No personal data.** Use only public, published text. Do not include real private
individuals' data; company/product/drug names from public sources are fine (and
are exactly the proper nouns the case-sensitive metric checks).

## Labeling workflow

1. Paste the cleaned source text as `document`.
2. Fill every schema field by hand from the text; use `null` / `[]` for anything
   the text does not state. Do **not** guess.
3. For array fields, list each item once, normalized to how it appears in the text
   (the metric compares them as a case-sensitive set).
4. Validate before committing:

   ```bash
   python src/validate_eval_set.py                 # all domains under data/eval/
   python src/validate_eval_set.py --domain medical # one file
   ```

   The validator checks each `gold` against `src/schemas.py` and reports the file
   and line of any problem. A clean run is required before the eval is meaningful.

## Status

Hand-labeling is tracked as its own checklist item (owner: user) in
`DEV_PLAN_3.5.md` (3.5.0). Until these files are populated, `evaluate.py` can be
unit-tested via fixtures but cannot be *run* end-to-end.
