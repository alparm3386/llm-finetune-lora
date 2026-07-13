# Dev Plan — Step 3.5: before/after evaluation script

> Detailed breakdown of step **3.5** of [`DEV_PLAN.md`](DEV_PLAN.md) — the
> before/after eval that proves the fine-tuning gain. Compares the **base**
> `google/gemma-4-E2B` against the **fine-tuned** (base + LoRA adapter) model on
> the **real, hand-labeled** Hungarian eval set, producing the 2×2 results table
> for the README. Owner tags: `Sonnet` (mechanical) vs. **`Opus`** (the fiddly
> model-loading / structured-decoding / metric-semantics wiring).
>
> As with 3.4, this step **produces and unit-tests the script**; actually
> *running* it needs a GPU + the trained adapter (after **3.6**) + the real eval
> set (see Dependencies) — that run is folded into **3.6**.

## Context (confirmed from SCOPE.md)

- **Two models:** base (pre-fine-tuning) vs. fine-tuned (base + LoRA adapter
  from 3.6 / HF Hub). Same base load path as `train.py` (Unsloth `FastModel`,
  4-bit), but for inference.
- **Two decode modes:** **prompt-only** (raw `generate`) and **+structured
  decoding** (`outlines`, JSON-schema-constrained). → a **2×2**: {base,
  fine-tuned} × {prompt-only, +structured}.
- **Identical prompt everywhere** (locked decision A from 3.4): reuse
  `prompt_format.to_chat_messages(domain, document)` (no gold) +
  `apply_chat_template(..., add_generation_prompt=True)`. Base, fine-tuned, and
  both decode modes all see the same prompt.
- **Metrics:**
  - **Primary — per-field exact-match F1**, measured in the **+structured**
    row (format controlled for on *both* models), so the delta is purely
    content accuracy.
  - **Secondary — JSON validity rate**, measured in the **prompt-only** row, to
    show fine-tuning also lifts raw format adherence (and to motivate the 2×2).
  - F1 is reported in **all four** cells for the narrative; the *headline*
    number is fine-tuned/+structured vs. base/+structured.
- **Eval set:** real Hungarian docs, hand-labeled, `data/eval/{domain}.jsonl`,
  same `{domain, document, gold}` schema as synthetic. A few dozen per domain.

## Design decisions

### A. Metric semantics — per-field exact-match F1 (**confirmed**)

Treat each field as a retrieval of the correct value, so nullable fields and
hallucinations score correctly (this is *why* the schema has nullable fields —
see `CLAUDE.md`). Per **scalar** field, per example:

| gold | pred | outcome |
|---|---|---|
| null | null | ignored (true negative — not counted) |
| null | non-null | **FP** (hallucination) |
| non-null | null | **FN** (miss) |
| non-null | == gold | **TP** |
| non-null | != gold (both non-null) | **FP + FN** (wrong value: missed truth *and* emitted a wrong one) |

**Array** fields (`side_effects[]`, …): normalize to sets `G`, `P`;
`TP = |G ∩ P|`, `FP = |P∖G|`, `FN = |G∖P|`. Empty/empty contributes nothing.

Aggregate: pool TP/FP/FN → **micro-F1** as the headline, plus **per-domain** and
**per-field** breakdowns. `P = TP/(TP+FP)`, `R = TP/(TP+FN)`, `F1 = 2PR/(P+R)`
(F1 = 0 when the denominator is 0).

**Normalization** (`normalize_value`): strings → strip + collapse internal
whitespace, **case-sensitive** (Hungarian proper nouns: drug/company/product
names). Numbers → compare numerically (`620000000 == 620000000.0`). Dates →
exact string match (schema stores them as strings). `null`/absent are equal.

**Unparseable prediction** (prompt-only mode, model emits non-JSON): validity =
fail; for F1, treat as an empty prediction → every non-null gold field becomes
an FN, no FPs. (In +structured mode this never happens — output is always
schema-valid.)

### B. Structured decoding backend — `outlines` (**`Opus`**, verify API at impl time)

`outlines`, JSON-schema-guided generation fed the domain schema straight from
`schemas.py` (single source of truth). **Risks to verify via context7 during
3.5.4** (outlines' API has churned across 0.1/1.0, and 4-bit + LoRA + Unsloth
integration is non-trivial):
- Does outlines' `transformers` integration wrap an **Unsloth 4-bit** model +
  LoRA adapter cleanly, or do we merge-to-16bit first?
- Does its JSON-schema mode accept our **nullable unions** (`["string","null"]`)
  and `additionalProperties:false` as-is?
- **Fallback** if outlines fights the quantized/PEFT stack: `transformers`
  logits-processor JSON constraint, or merge adapter → 16-bit → outlines.

### C. Module split (mirrors 3.4's `prompt_format.py` extraction)

- **`src/eval_metrics.py`** — pure metric logic (normalization, TP/FP/FN, F1,
  validity). **Zero** torch/outlines deps → fully unit-testable locally. This is
  the heart of the step and gets the most test coverage.
- **`src/evaluate.py`** — model loading, generation (both modes), the eval loop,
  aggregation, reporting, CLI. Heavy deps (`unsloth`/`outlines`/`torch`)
  imported **lazily** inside the functions that need them (the 3.4 lesson), so
  the loop/aggregation/CLI stay importable and testable without a GPU.

### D. Outputs

- `results/eval_results.json` — full structured results (all 4 cells × per-domain
  × per-field), for reproducibility.
- `results/eval_table.md` — the rendered 2×2 markdown table, ready to paste into
  the README (3.8). **`results/` is tracked** (not gitignored — unlike
  `outputs/`), since the numbers are the deliverable.

## Sub-steps

- [x] **3.5.0** `Sonnet` — Eval-set scaffolding (the dedicated eval-data step —
  see Dependencies): write `data/eval/README.md` (a labeling guide: sourcing
  real OGYÉI leaflets / business news / tech specs, PII-free, the
  `{domain, document, gold}` JSONL format, per-domain gold examples) + a small
  `validate_eval_set` helper that checks each hand-labeled `gold` against
  `schemas.py`. The **hand-labeling itself is manual work (owner: user)**, a
  few dozen docs/domain — tracked as its own checklist item, unblocked by this
  scaffolding.
- [x] **3.5.1** **`Opus`** — `src/eval_metrics.py`: `normalize_value`, scalar &
  array TP/FP/FN classification (per decision A), micro-F1 aggregation
  (overall/per-domain/per-field), `is_valid_json` / schema-validity check. Pure
  functions, no model deps.
- [x] **3.5.2** `Sonnet` — Eval-data loading in `evaluate.py`: read
  `data/eval/{domain}.jsonl` (reuse the `{domain, document, gold}` shape; can
  share `load_examples` logic with `train.py`). Clear error if the eval set is
  absent (see Dependencies).
- [x] **3.5.3** **`Opus`** — Model loading for inference: base and base+adapter
  via Unsloth `FastModel` (`for_inference`), 4-bit. Lazy imports. Verify adapter
  attach path (`from_pretrained(adapter)` vs. `PeftModel`).
- [x] **3.5.4** **`Opus`** — Generation, two paths: (a) prompt-only `generate` +
  JSON parse (reuse `strip_json_fence` idea from `generate_data.py`); (b)
  `outlines` schema-constrained. Both reuse `prompt_format`. See decision B
  risks — **fetch current outlines docs via context7 before writing.**
- [x] **3.5.5** `Sonnet` — Eval loop + aggregation: for each {base, ft} ×
  {prompt-only, structured}, run all eval docs, call `eval_metrics` → assemble
  the 2×2 (per-domain + overall). `--limit` for quick shakeouts.
- [x] **3.5.6** `Sonnet` — Reporting: write `results/eval_results.json` +
  `results/eval_table.md` (2×2), and log a readable summary. F1 in all cells,
  validity rate in the prompt-only cells.
- [x] **3.5.7** `Sonnet` — CLI: argparse — `--adapter`, `--base-model`,
  `--eval-dir`, `--out`, `--max-new-tokens`, `--limit`, `--seed`, and a switch to
  skip a mode (e.g. `--structured-only`) for faster iteration.
- [x] **3.5.8** `Sonnet` — `notebooks/eval_colab.ipynb`: thin Colab wrapper —
  load adapter (from 3.6 output or HF Hub) + run `evaluate.py`, mirroring
  `train_colab.ipynb`.
- [x] **3.5.9** `Sonnet` — Local unit tests: `tests/test_eval_metrics.py` (the
  meat — hand-built cases for TP/FP/FN, nulls, hallucinations, wrong values,
  array set-F1, unparseable preds, normalization), plus eval-data-loading and
  CLI-parsing tests. Full GPU eval run is part of **3.6**.

## Dependencies / Constraints

- **No local CUDA GPU** → same as 3.4: build + unit-test the non-GPU parts here
  (metrics, data loading, aggregation shape, CLI); the real end-to-end eval runs
  on Colab in **3.6**, after the adapter exists.
- **Real eval set must be created** — `data/eval/{domain}.jsonl` is currently
  empty (only `.gitkeep`). Now scheduled as **step 3.5.0** (scaffolding +
  labeling guide + validator) plus a tracked **manual hand-labeling** task
  (owner: user), keeping SCOPE's "real eval" credibility. The script and its
  tests (via fixtures) are finishable without it; *running* 3.5 is blocked on
  it.
- **Trained adapter** (3.6 output) is the other run-time input — so the natural
  order is: finish 3.5 script now → 3.6 trains → run 3.5 eval within 3.6.
