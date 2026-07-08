# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QLoRA fine-tuning of `google/gemma-4-E2B` (Instruct) for Hungarian structured extraction
(text → JSON), across three domains: **medical**, **business**, **technology**. The full
rationale, dataset design, and success metric are defined in `SCOPE.md` — read it before making
design decisions; don't duplicate its content into code comments.

Status: work in progress. `train.py` and `evaluate.py` are stubs (`NotImplementedError`) awaiting
dev-plan steps 3.4 and 3.5.

## Commands

```bash
pip install -r requirements.txt        # local/reproducible deps; on Colab, `pip install unsloth` instead
python src/generate_data.py --domain all --n 50   # generate synthetic training data via Gemini
python src/generate_data.py --domain medical --n 20 --seed 1  # single domain, custom count/seed
python src/train.py                    # QLoRA fine-tuning (stub — step 3.4)
python src/evaluate.py                 # before/after eval (stub — step 3.5)
```

No test suite, linter, or CI config exists yet.

`generate_data.py` requires `GEMINI_API_KEY` in a local `.env` (copy `.env.example` — never commit
the real `.env`, it's gitignored).

## Architecture

- **`src/schemas.py`** — the single source of truth for the three domain JSON Schemas (medical,
  business, technology), mirroring the tables in `SCOPE.md`. Both `generate_data.py` (validates
  synthetic gold JSON) and the future `evaluate.py`/structured-decoding step consume `SCHEMAS`
  from here. If a domain schema changes, it only needs to change in this file.
- **`src/generate_data.py`** — calls Gemini to synthesize `{document, gold}` pairs per domain,
  validates each `gold` against `schemas.py`, retries on failure, writes JSONL to
  `data/synthetic/{domain}.jsonl`. Style variation comes from rotating `STYLE_HINTS` per domain
  rather than a single fixed prompt, to avoid repetitive generations.
- **`src/train.py`** — will load the base model in 4-bit via Unsloth, attach a LoRA adapter
  (r≈16), and fine-tune on `data/synthetic/`. Target environment: Colab T4.
- **`src/evaluate.py`** — will run the before/after comparison. Key design point from `SCOPE.md`:
  structured decoding (via `outlines`) is applied to **both** the base and fine-tuned model during
  eval, so JSON validity is controlled for and the measured metric (per-field exact-match F1) isolates
  fine-tuning's *content*-accuracy gain, not format compliance. A secondary prompt-only (no
  structured decoding) pass reports raw JSON validity rate to show the 2×2 (base/fine-tuned ×
  prompt-only/structured-decoding) comparison that goes in the README.
- **`data/synthetic/`** — regenerable LLM-generated training data, gitignored by default (except
  files explicitly committed as examples).
- **`data/eval/`** — real, hand-labeled Hungarian documents with gold JSON, used only for the
  before/after eval. Tracked in git (see `.gitignore` exception).

## Conventions

- Input documents are Hungarian; JSON schema keys and code are in English.
- Nullable schema fields (`["string", "null"]`) exist so the model learns to emit `null` /
  `[]` instead of hallucinating values — preserve this typing convention when adding schema fields.
- The README/model card must openly disclose that training data is LLM-generated (see the
  `paraloq/json_data_extraction` precedent cited in `SCOPE.md`).
