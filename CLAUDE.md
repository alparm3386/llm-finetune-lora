# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

QLoRA fine-tuning of `google/gemma-4-E2B` (Instruct) for Hungarian structured extraction
(text → JSON), across three domains: **medical**, **business**, **technology**. The full
rationale, dataset design, and success metric are defined in `SCOPE.md` — read it before making
design decisions; don't duplicate its content into code comments.

Status: work in progress. `train.py` (step 3.4) is implemented but unrun — the actual GPU training
job is step 3.6 (Colab T4). `evaluate.py` is a stub (`NotImplementedError`) awaiting step 3.5.

## Commands

```bash
pip install -r requirements.txt        # local/reproducible deps; on Colab, `pip install unsloth` instead
python src/generate_data.py --domain all --n 50   # generate synthetic training data via the local proxy
python src/generate_data.py --domain medical --n 20 --seed 1  # single domain, custom count/seed
python src/train.py --smoke            # QLoRA fine-tuning; only runs on a CUDA GPU (Colab T4, step 3.6)
python src/evaluate.py                 # before/after eval (stub — step 3.5)
pytest                                 # unit tests for the non-GPU parts (prompt_format.py, train.py data/CLI)
```

No linter or CI config exists yet. `pytest.ini` adds `src/` to the path so tests can import its
modules directly (e.g. `from train import ...`), matching how the scripts import each other.

`generate_data.py` calls an OpenAI-compatible local proxy (serving Claude models) at
`http://127.0.0.1:8000/v1` — no API key needed (a placeholder string is passed to satisfy the SDK).
The proxy must be running locally before invoking the script.

## Architecture

- **`src/schemas.py`** — the single source of truth for the three domain JSON Schemas (medical,
  business, technology), mirroring the tables in `SCOPE.md`. Both `generate_data.py` (validates
  synthetic gold JSON) and the future `evaluate.py`/structured-decoding step consume `SCHEMAS`
  from here. If a domain schema changes, it only needs to change in this file.
- **`src/generate_data.py`** — calls a local OpenAI-compatible proxy (serving Claude models) to
  synthesize `{document, gold}` pairs per domain, validates each `gold` against `schemas.py`,
  retries on failure, writes JSONL to `data/synthetic/{domain}.jsonl`. Style variation comes from
  rotating `STYLE_HINTS`/`NAME_STYLE_HINTS` per domain rather than a single fixed prompt, to avoid
  repetitive generations (entity names, tone, and nullable-field presence are all varied
  deterministically in code, not left purely to the model).
- **`src/train.py`** — loads the base model in 4-bit via Unsloth (`load_base_model`), attaches a
  LoRA adapter (`add_lora_adapter`, r=16, text-only via `finetune_*_layers` flags), builds the
  chat-formatted train/val split from `data/synthetic/` (`build_dataset`), and fine-tunes with a
  response-masked `SFTTrainer` (`build_trainer`). `unsloth`/`trl` are imported lazily inside the
  functions that need them, so the data pipeline and CLI stay importable/testable without a CUDA
  GPU; only actually running training requires one. Target environment: Colab T4
  (`notebooks/train_colab.ipynb`, step 3.6).
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
