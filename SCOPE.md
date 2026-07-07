# SCOPE — `hu-llm-finetune-lora`

> Output of step **3.1** of the dev plan. Defines the scope of the fine-tuning project:
> base model · task · dataset · success metric · environment. The remaining steps (3.2–3.8) follow from this.

## Goal (one sentence)

Adapt a small, open-weight model with LoRA/QLoRA for **structured information extraction in Hungarian**
(text → JSON), and prove the gain with a **before/after eval** — covering the remaining target keywords:
`PyTorch` · `Hugging Face` · `LoRA/QLoRA` · `transformers/PEFT` · `Unsloth`.

## Task

**Structured extraction (text → JSON):** extract a JSON object following a predefined schema from a
free-text Hungarian document. The model input is `{document + target schema}`, the output is the
filled-in JSON.

Why this task: objective, cheap eval (no LLM judge needed), and a small model holds the JSON format
poorly out of the box → **a dramatic before/after jump after LoRA**. The pipeline will be
task-agnostic, so later fine-tuning tasks (instruction following, classification, domain QA) become
just "new dataset + new eval".

### Domains + target schemas (3 domains)

Input documents are in Hungarian; JSON schema keys are in English (consistent with the codebase).

| Domain | Source (real, for eval) | Target schema (draft) |
|---|---|---|
| **medical** | Hungarian medicine **patient information leaflets** (OGYÉI, public, PII-free) | `{drug_name, active_ingredient, indication, dosage, side_effects[], contraindications[]}` |
| **business** | Hungarian business news / press releases | `{company, event_type, amount, currency, date, involved_parties[]}` |
| **technology** | product specs / tech announcements | `{product, manufacturer, version, key_specs[], release_date, price}` |

## Base model

**`google/gemma-4-E2B`** (Gemma 4, "effective-2B"), Instruct variant.

- **Why:** Google brand (strategic), Colab-native, good Hungarian/multilingual coverage, full Unsloth
  support. Unsloth hides the PLE architecture details (QLoRA works out of the box on it).
- **Family ladder for later:** `E2B → E4B` scale-up with the same code (portfolio narrative).
- **License:** Gemma Terms of Use (fine for a public repo + HF model card; noted in the README).

## Dataset

**Own, synthetically generated Hungarian text→JSON dataset** (no ready-made Hungarian equivalent on the
HF Hub; template and precedent: `paraloq/json_data_extraction` — English, Gemini-generated, Apache-2.0).

- **Train (synthetic):** a strong LLM generates `{domain document + schema → gold JSON}` pairs across the
  3 domains. Target: a few hundred examples per domain (with Unsloth, ~100–500 examples already yield a
  striking result on a T4).
- **Eval (real, hand-labeled gold):** a few dozen **real** Hungarian documents per domain, hand-labeled
  with gold JSON → the before/after is measured on **real** data (credibility boost).
- **Authenticity:** the README and HF model card **openly state** that the training data is LLM-generated
  (following the `paraloq` precedent). See the `cv-only-authentic-claims` principle in the dev plan.

## Structured decoding vs. fine-tuning (why both, and what fine-tuning actually proves)

These solve **different layers** and are complementary, not competitors:

| | Structured decoding | Fine-tuning (LoRA/QLoRA) |
|---|---|---|
| Guarantees | **Format** — output is parseable, schema-valid JSON | **Content** — the model extracts the *correct* value |
| How | Masks schema-violating tokens at generation time | Trains the weights on the task |
| Cost | Zero training, zero labeled data | GPU + labeled data |
| Cannot do | Wraps a *wrong* value in valid JSON too | (does not itself guarantee format) |

Structured decoding makes the JSON **valid**, not **correct**. So "fine-tune to get valid JSON" would be a
weak motivation — guided decoding does that for free. This project therefore positions fine-tuning around
the part structured decoding **cannot** fix: **content accuracy**. To isolate that cleanly, the eval
applies **structured decoding to BOTH the base and the fine-tuned model**, so format is controlled for and
the measured delta is purely the extraction-accuracy gain.

**Portfolio narrative (2×2 comparison, goes in the README):**

| | Prompt-only | + Structured decoding |
|---|---|---|
| **Base model** | baseline (format + content both weak) | valid JSON, content still weak |
| **Fine-tuned** | better content, format may still slip | **best** — valid JSON *and* accurate content |

This demonstrates understanding of *where each technique helps* — a stronger signal than a naive
"teach the model JSON" framing.

## Success metric (before/after)

Compare the base (pre-fine-tuning) model vs. the LoRA-adapter model on the **real eval set**, with
**structured decoding enabled for both** (format controlled for):

1. **Per-field exact-match F1** (primary) — the fraction of correctly extracted fields vs. the gold JSON.
   This is the content-accuracy gain fine-tuning is responsible for.
2. **JSON validity rate** (secondary, prompt-only variant) — reported *without* structured decoding, to
   show the fine-tuning also improves raw format adherence (and to motivate the 2×2 above).

Expectation: a measurable, reportable improvement in per-field F1. Results table (the 2×2) goes into the README.

## Environment / stack

- **Training:** Google **Colab** (free **T4 16GB** is enough for E2B; L4/A100 optional for speed).
- **Framework:** **Unsloth** (wraps `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate`,
  `datasets`), **QLoRA** (4-bit NF4), `LoraConfig` around r=16.
- **Structured decoding (eval):** `outlines` (or `transformers` logits processor) to enforce the JSON
  schema at generation time — applied to base *and* fine-tuned during eval to control for format.
- **Adapter hosting:** **Hugging Face Hub** (with a model card — step 3.7).

## Out of scope

- Full fine-tuning (LoRA/QLoRA only).
- Multiple models at once — E2B is the focus, E4B is only an optional scale-up.
- Multimodal (image/audio) — text→text only.
- Production serving / deployment — the goal is a proven before/after, not a product.

---

### Dev-plan steps to update (from this scope)

- **3.1** ✅ (this document)
- **3.2** stack: RunPod → **Colab**, `requirements` → **Unsloth-based**
- **3.6** training: RunPod GPU → **Colab T4**
