# Dev Plan — `hu-llm-finetune-lora`

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
- [x] **3.2** Repo scaffold: `hu-llm-finetune-lora` created, `requirements`/setup on an
  **Unsloth base** (`unsloth` transitively pulls in `torch`, `transformers`, `peft`, `trl`,
  `bitsandbytes`, `accelerate`, `datasets`), `.gitignore`, MIT license (for the own code), empty
  `README`. Training environment: **Google Colab** (not RunPod).
- [~] **3.3** Synthetic data generator (`src/generate_data.py`, `src/schemas.py`): calls an
  OpenAI-compatible **local proxy serving Claude models** (`claude-haiku-4-5-20251001`, own
  resource — no employer/client keys) to generate `{Hungarian document → gold JSON}` pairs across
  the 3 domains, with a schema-driven prompt + local `jsonschema` validation and retry/backoff.
  Script is done and smoke-tested (2/2 examples OK, distractor/nullable handling correct), plus
  fixes for API resilience (retry on 5xx/429, fail-fast on other 4xx) and data diversity
  (deterministic per-example nullable-field presence plan, name-diversity hints, banned generic
  names). Full set generation: **150/domain** in progress. (Chat-template formatting moves into
  the 3.4 training script.)
- [ ] **3.4** `use Opus` — QLoRA training script: 4-bit quantization (bitsandbytes),
  `LoraConfig` (target modules, r, alpha, dropout), `Trainer`/`SFTTrainer` loop, checkpointing.
  Should also have a config runnable at small scale (a few hundred steps).
- [ ] **3.5** Eval script: before/after comparison of the base vs. fine-tuned adapter, with
  structured decoding (`outlines`) applied to both to isolate the content-accuracy gain (see
  `SCOPE.md`'s structured-decoding-vs-fine-tuning section); per-field exact-match F1 as primary
  metric, JSON validity rate (prompt-only) as secondary; results table written out.
- [ ] **3.6** Run training on **Colab T4** (Unsloth), save the adapter, record results.
- [ ] **3.7** Upload the adapter to the **Hugging Face Hub** with a model card (task, base model,
  data, hyperparameters, before/after results, limitations).
- [ ] **3.8** `use Opus` — Repo `README.md`: problem → approach → results (with numbers) →
  "how to run" → HF model link. Keyword-rich, but honest. **Pin on the profile.**
