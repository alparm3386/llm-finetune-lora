# Dev Plan — Step 3.4: QLoRA training script

> Detailed breakdown of step **3.4** of [`DEV_PLAN.md`](DEV_PLAN.md) — the QLoRA
> fine-tuning script for `google/gemma-4-E2B` on the synthetic Hungarian
> text→JSON extraction data (450 examples, 3 domains). Owner tags: `Sonnet`
> (mechanical) vs. **`Opus`** (the fiddly QLoRA/Trainer wiring). Actual GPU
> training run is step **3.6** (Colab T4) — this step only produces the script.

## Context (confirmed)

- **Data:** `data/synthetic/{medical,business,technology}.jsonl`, 150 each = 450.
  Each record: `{"domain", "document" (Hungarian free text), "gold" (schema JSON)}`.
- **Lengths (doc + gold):** median ~540 tok, p95 ~1180, max ~2400 (rough;
  Hungarian tokenizes heavier). → `max_seq_length = 2048` covers ~95%,
  truncates only the longest handful. Bump to 3072 if we want the long tail.
- **Base model:** `google/gemma-4-E2B` (Instruct), 4-bit NF4 QLoRA via Unsloth.
- **Target env:** Colab T4 16GB (fp16, no bf16). Small-scale `--smoke` config too.

## Locked design decisions

- **A. Shared prompt-format module** (`src/prompt_format.py`) used by *both*
  `train.py` and `evaluate.py` (3.5) → base / fine-tuned / eval all use an
  **identical** prompt. Prompt = instruction + domain JSON schema (from
  `schemas.py`) + Hungarian document; target = gold JSON.
- **B. Chat template:** **Gemma native** template (tokenizer's built-in
  `<start_of_turn>user … <start_of_turn>model …`), most faithful to how the
  Instruct model was trained.
- **C. Train on responses only** — mask prompt tokens, loss only on the gold-JSON
  completion (Unsloth `train_on_responses_only`).
- **D. Small seeded val split** (~5%, ≈22 ex) from synthetic, for `eval_loss`
  monitoring only. The **real** before/after eval is separate (3.5, real set).
- **E. T4 hyperparams:** `r=16, alpha=16, dropout=0`, target modules =
  q/k/v/o/gate/up/down proj, lr `2e-4`, cosine, fp16, batch 2 × grad-accum 4,
  gradient checkpointing on, ~2–3 epochs.
- **F. `--smoke` mode:** same code path, `max_steps≈60` — the "runnable at a few
  hundred steps" config from the plan.
- **G. Script + notebook split:** logic in `src/train.py` (CLI); thin
  `notebooks/train_colab.ipynb` does `pip install unsloth` + calls it.

## Sub-steps

- [x] **3.4.1** `Sonnet` — `src/prompt_format.py`: `build_prompt(domain, document)`,
  `serialize_gold(gold, domain)` (deterministic, schema key order, `ensure_ascii=False`),
  `to_chat_messages(...)`. Reuses `schemas.py`.
- [x] **3.4.2** `Sonnet` — Data loading in `train.py`: read 3 JSONL → HF `Dataset`,
  map to Gemma-chat-formatted text, seeded train/val split.
- [x] **3.4.3** **`Opus`** — Model + LoRA: Unsloth `FastModel` load
  `google/gemma-4-E2B` 4-bit; `get_peft_model` (target modules, r/alpha/dropout,
  gradient checkpointing). Uses `finetune_*_layers` flags (vision frozen,
  text-only) instead of a raw `target_modules` list; `unsloth` imported lazily
  so the module stays importable without CUDA.
- [ ] **3.4.4** **`Opus`** — `SFTTrainer`/`SFTConfig` (trl via Unsloth), T4-tuned
  args, `train_on_responses_only`, checkpointing + save adapter.
- [ ] **3.4.5** `Sonnet` — CLI/config: argparse, `--smoke`, seeds, `--out`,
  epochs/steps, resume-from-checkpoint.
- [ ] **3.4.6** `Sonnet` — `notebooks/train_colab.ipynb`: thin Colab wrapper.
- [ ] **3.4.7** `Sonnet` — Local sanity: unit-test the non-GPU parts (format +
  data loading). Full GPU run is 3.6.

## Constraint

No local CUDA GPU + Unsloth needs one → the training loop **cannot be executed
locally**. Verify non-GPU parts (formatting, data pipeline, argparse) here; the
real end-to-end run is on Colab T4 in step **3.6**.
