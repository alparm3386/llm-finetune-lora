# Dev Plan — `llm-finetune-lora`

> Project 3 of the [portfolio dev plan](../portfolio-dev-plan.md) — ⭐ flagship.
> Covers the remaining keyword gaps: `PyTorch` · `Hugging Face` · `LoRA/QLoRA` · `transformers/PEFT`.
> The only truly from-scratch project — builds on existing GPU/vLLM experience.

- [x] **3.1** `use Opus` — Task and scope decision. **Recorded** in [`SCOPE.md`](SCOPE.md):
  base model **`google/gemma-4-E2B`** (Unsloth, Colab T4; E4B as a scale-up), task **structured
  extraction (text → JSON)** across 3 domains (medical/business/technology, Hungarian input +
  English schema keys), dataset **own synthetic train + real hand-labeled eval** (no ready-made
  Hungarian HF dataset; precedent: `paraloq/json_data_extraction`), success metric **per-field
  exact-match F1 with structured decoding on both models (primary) + JSON validity rate
  (secondary)**.
- [x] **3.2** Repo scaffold: `llm-finetune-lora` created, `requirements`/setup on an
  **Unsloth base** (`unsloth` transitively pulls in `torch`, `transformers`, `peft`, `trl`,
  `bitsandbytes`, `accelerate`, `datasets`), `.gitignore`, MIT license (for the own code), empty
  `README`. Training environment: **Google Colab** (not RunPod).
- [x] **3.3** Synthetic data generator (`src/generate_data.py`, `src/schemas.py`): calls an
  OpenAI-compatible **local proxy serving Claude models** (`claude-haiku-4-5-20251001`, own
  resource — no employer/client keys) to generate `{Hungarian document → gold JSON}` pairs across
  the 3 domains, with a schema-driven prompt + local `jsonschema` validation and retry/backoff.
  Script is done and smoke-tested (2/2 examples OK, distractor/nullable handling correct), plus
  fixes for API resilience (retry on 5xx/429, fail-fast on other 4xx) and data diversity
  (deterministic per-example nullable-field presence plan, name-diversity hints, banned generic
  names). Full set generation: **150/domain** in progress. (Chat-template formatting moves into
  the 3.4 training script.)
- [x] **3.4** QLoRA training script (`src/train.py`, `src/prompt_format.py`,
  `notebooks/train_colab.ipynb`): shared chat-prompt module; data loading + seeded train/val
  split; Unsloth `FastModel` 4-bit load + LoRA adapter (r=alpha=16, text-only via
  `finetune_*_layers`); `SFTTrainer`/`SFTConfig` T4-tuned + `train_on_responses_only`; CLI
  (`--smoke` at 60 steps, seeds, resume-from-checkpoint); thin Colab notebook wrapper; unit tests
  for all non-GPU parts (21 passing). See `DEV_PLAN_3.4.md` for the full breakdown. Training
  itself hasn't run yet — no local CUDA GPU — that's step **3.6** on Colab T4.
- [x] **3.5** Eval script: before/after comparison of the base vs. fine-tuned adapter, with
  structured decoding (`outlines`) applied to both to isolate the content-accuracy gain (see
  `SCOPE.md`'s structured-decoding-vs-fine-tuning section); per-field exact-match F1 as primary
  metric, JSON validity rate (prompt-only) as secondary; results table written out. Detailed
  breakdown in [`DEV_PLAN_3.5.md`](DEV_PLAN_3.5.md) (9/9 sub-steps done, 101 passing tests). Includes
  **3.5.0**: create the **real hand-labeled eval set** (`data/eval/*.jsonl`) — scaffolding +
  validator here, labeling is a manual task. This eval-set + the trained adapter (3.6) are the
  run-time prerequisites, so the full eval run happens within **3.6**.
- [ ] **3.6** Run training on **Colab T4** (Unsloth), save the adapter, record results.
- [ ] **3.7** Upload the adapter to the **Hugging Face Hub** with a model card (task, base model,
  data, hyperparameters, before/after results, limitations).
- [ ] **3.8** `use Opus` — Repo `README.md`: problem → approach → results (with numbers) →
  "how to run" → HF model link. Keyword-rich, but honest. **Pin on the profile.**
