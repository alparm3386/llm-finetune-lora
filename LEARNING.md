# Learning Log — Fine-tuning, from the ground up

> A living doc for understanding **what** we're doing and **why**, in this project.
> Not a spec (that's `SCOPE.md`) and not a task tracker (that's `DEV_PLAN*.md`) —
> this is the "explain it to me" companion. We add a new section each time we go
> deeper into a topic. Newest topics go at the bottom; the table of contents
> tracks what's covered so far.

## Table of contents

1. [Big picture: what is fine-tuning, and why LoRA/QLoRA](#1-big-picture-what-is-fine-tuning-and-why-loraqlora)
2. [Does quantization (Q4 vs. full precision) hurt fine-tuned quality?](#2-does-quantization-q4-vs-full-precision-hurt-fine-tuned-quality)
3. [Fixed GPU budget: more data on Q4, or less data on full precision?](#3-fixed-gpu-budget-more-data-on-q4-or-less-data-on-full-precision)

---

## 1. Big picture: what is fine-tuning, and why LoRA/QLoRA

### What is fine-tuning?

A base LLM (here: `google/gemma-4-E2B`) already knows language and reasoning from
pretraining, but wasn't specifically trained to reliably do **our** task: read a
Hungarian medical/business/tech document and output a specific JSON schema.

- **Prompting** (few-shot examples, clear instructions) can get you partway there,
  with zero training.
- **Fine-tuning** goes further: show the model hundreds of `(input → correct
  output)` examples and adjust its internal weights so it gets better and more
  *consistent* at that specific task. Like the difference between instructions vs.
  actual practice on the job.

**Why fine-tune here specifically?** The project's goal (`SCOPE.md`) is to prove
fine-tuning gives a measurable accuracy boost *on top of* prompting + structured
decoding — not just "can the model output valid JSON" (a solved problem via
grammar-constrained decoding), but "does it extract the *right values*." That's
the entire point of the before/after eval in step 3.5.

### Why LoRA / QLoRA instead of full fine-tuning?

Full fine-tuning updates **all** of the model's billions of weights — needs huge
GPU memory, overkill for a narrow task.

- **LoRA** (Low-Rank Adaptation): freeze the whole base model, inject small
  trainable "adapter" matrices alongside specific layers (here: attention
  q/k/v/o and MLP gate/up/down projections). Only ~1% of parameters are trained.
  Result: a small adapter file (MBs) you attach on top of the frozen base model
  at inference time.
- **QLoRA**: additionally load the frozen base model in **4-bit quantized** form
  (compressed weights) so it fits in a small GPU (target: a free Colab T4,
  16GB VRAM), while the LoRA adapters themselves still train in higher precision.
  This is *why* a T4 is enough — a cheap, small GPU can fine-tune a model that
  would otherwise need much more memory.

So: frozen 4-bit base model + small trainable adapter = affordable fine-tuning on
modest hardware.

### Is full fine-tuning just "continuing pretraining"?

Mechanically, yes — same process: forward pass → loss → backprop → update
weights, same next-token-prediction objective, all weights updated (nothing
frozen). What differs is the *phase*:

| | Pretraining | Full fine-tuning |
|---|---|---|
| **Starting point** | Random weights | Already-pretrained weights (not random!) |
| **Data** | Massive, raw, diverse web text (trillions of tokens) | Small, curated, task-specific examples (hundreds–thousands) |
| **Data role** | Model learns language broadly | Model learns to shape existing knowledge toward *your* task |
| **Learning rate** | Relatively higher, more aggressive | Much smaller — nudging, not rebuilding |
| **Duration** | Huge (months on GPU clusters) | Small (minutes–hours) |

**A note on loss masking — it's a separate axis, not a pretrain-vs-finetune split:**
masking depends on *what kind* of fine-tuning you're doing, independent of
full-weights-vs-LoRA:
- **Continued pretraining** (fine-tune on more raw text to build general
  capability, e.g. more Hungarian text): no masking, predict every token —
  literally just "more pretraining" on a narrower corpus. Works with full
  weights *or* LoRA.
- **SFT / instruction tuning** (our case: `prompt → gold JSON` pairs): masking
  is the natural choice, because the *input* (document + schema) is given, not
  something to learn to generate — only the *response* should be learned. Also
  independent of full vs. LoRA; `train_on_responses_only` would apply the same
  way with full fine-tuning too.

So masking tracks whether the data has a clear input/target split (fine-tuning
tasks often do; raw pretraining text doesn't), not whether the training is
"pretraining" or "fine-tuning" per se.

**So could we skip masking and just do continued pretraining on our data
instead of SFT?** Yes, that's a real, named technique — **continued pretraining**
(aka "domain-adaptive pretraining"): keep doing next-token prediction, but on
new domain data (e.g. a pile of Hungarian medical/business/tech text) instead
of `prompt → JSON` pairs. It teaches a *different* thing than what we need,
though:

- **Continued pretraining** (no input/output split): the model gets better at
  Hungarian domain *vocabulary, style, fluency* — "sounds like" a Hungarian
  medical text. No explicit signal for "given this input, produce exactly this
  output" — just statistical patterns of raw text.
- **SFT with masking** (our approach): explicitly teaches the *behavior* —
  "given this input shape, produce this output shape." A much more direct
  signal for a structured task.

Why this matters practically here: continued pretraining needs a *lot* of data
(many MBs–GBs of domain text) to meaningfully shift behavior, since the signal
per token is weak (predict-the-next-word, not do-the-task). We only have ~450
curated examples — plenty for SFT (each example is a strong, direct
demonstration of the exact task), nowhere near enough to move the needle via
continued pretraining.

The other risk introduced by pushing fine-tuning too far: **catastrophic forgetting** — push the learning rate
too hard or train too long on a narrow dataset, and the model can lose some
general pretrained ability while overfitting to the small fine-tuning set. This
is part of *why* LoRA is attractive: since only a tiny adapter is trained and the
original weights stay frozen/untouched, the base model's general knowledge is
structurally protected — you can't forget what you never touched.

### The pipeline, mapped to the dev plan

```
3.1  Decide the task            → "text→JSON extraction, 3 domains, Gemma-4-E2B"
3.3  Generate training data     → synthetic {document, gold JSON} pairs (data/synthetic/*.jsonl)
3.4  Build the training script  → chat-format prompts, attach LoRA, run SFTTrainer
3.5  Build the eval script      → measure base vs fine-tuned accuracy on real hand-labeled docs
3.6  Actually run 3.4 + 3.5     → on Colab T4 GPU (no local GPU in this repo)
3.7  Publish the adapter        → to Hugging Face Hub
3.8  Write it up                → README with the before/after numbers
```

Trickier "why" points:

- **Why synthetic data at all (3.3)?** No ready-made Hungarian dataset exists for
  this exact task, so it's generated with a stronger model (Claude) — a common,
  disclosed pattern (`paraloq/json_data_extraction` precedent). Synthetic data
  can't *prove* the model works on real text, hence the separate small
  hand-labeled real eval set (3.5.0) — that's the credibility check.
- **Why "train on responses only" (3.4 decision C)?** During training the model
  sees `prompt + correct answer` as one sequence, but should only *learn to
  generate the answer*, not memorize/predict the prompt/schema text back. Loss is
  masked to count only the gold-JSON completion tokens.
- **Why structured decoding in eval too, not just training (3.5 decisions A/B)?**
  If we only fine-tune, the model might get better at JSON *formatting* and
  *content* accuracy simultaneously — we wouldn't know which caused an
  improvement. Forcing both base and fine-tuned models through the same
  JSON-schema-constrained decoding (`outlines`) at eval time removes format as a
  variable — any F1 difference measured is purely from fine-tuning's content-accuracy
  gain.

### Open threads (candidates for future sections)

- Data generation (3.3) — how synthetic examples are created and validated
- Prompt/chat formatting (3.4.1) — document+schema → exact training text
- LoRA/QLoRA mechanics (3.4.3) — quantization, target modules, r/alpha/dropout
- The training loop (3.4.4) — SFTTrainer, hyperparameters, steps vs. epochs
- Evaluation & metrics (3.5.1) — per-field F1, why nullable fields matter

---

## 2. Does quantization (Q4 vs. full precision) hurt fine-tuned quality?

**Setup:** imagine we had both a 4-bit (Q4) and a full-precision copy of the
same base model, and fine-tuned both on the same training set. Does the
full-precision one end up with better quality on the *new* skill, or does
fine-tuning erase that gap?

**Short answer: full precision generally still comes out slightly ahead, but
the gap shrinks a lot after fine-tuning — it doesn't fully close, but it's not
"doesn't matter" either.**

Reasoning:

1. **Quantization = lossy compression of the base weights.** 4-bit rounds each
   weight to one of 16 possible values — a small but real information loss vs.
   full precision. The base model ends up slightly "blurrier."

2. **The LoRA adapter compensates only *locally*, for the specific skill being
   trained.** During QLoRA fine-tuning, the adapter's gradients are computed
   against the *quantized* base model's actual behavior — so it learns "given
   how this quantized model behaves, nudge it toward correct outputs for this
   task." It's compensating for that base's actual (slightly noisy)
   representations, not recovering the lost information. Two different frozen
   bases (Q4 vs. FP16) end up with two adapters that each "solved" the task
   starting from a different baseline.

3. **Whether it matters depends on whether the quantization noise interferes
   with the *specific* skill being taught:**
   - A narrow, well-defined skill like structured extraction (map document →
     JSON fields) is fairly "shallow" — mostly format-following +
     copying/locating values, not deep reasoning or rare knowledge. Fine-tuning
     tends to close most of the gap here, thanks to a strong, direct training
     signal per example.
   - Skills relying on subtler capability (nuanced reasoning, rare factual
     recall, a lower-resource language like Hungarian where representations are
     already thinner) are more likely to keep a real, noticeable gap, since the
     quantization noise compounds with an already-weaker signal.

4. **This tracks the actual QLoRA paper's finding** (Dettmers et al., 2023):
   4-bit QLoRA fine-tuning gets *close to* 16-bit full fine-tuning quality on
   many benchmarks — "near-lossless," not identical. That's the empirical case
   for QLoRA's popularity: the trade-off is usually worth it for the memory
   savings, but it's not a free lunch.

**For this project:** QLoRA was picked purely for the **hardware constraint**
(free Colab T4, 16GB) — not because we tested whether Q4 vs. FP16 fine-tuning
quality differs for this specific task. Worth being explicit about that
assumption if it comes up in the README / model card (3.8).

---

## 3. Fixed GPU budget: more data on Q4, or less data on full precision?

**Setup:** limited GPU-time budget for fine-tuning on a specific domain/task.
Better strategy — larger training data on a small quantized (Q4) base model, or
less training data on a full-precision model?

**Short answer: for a narrow, well-defined task (like structured extraction),
going wider — Q4 + more data — usually beats going higher-precision with less
data.**

Reasoning:

1. **Q4 training is cheaper per step** (less memory → bigger batches fit, less
   compute per forward/backward pass). For the same GPU-time budget, you can
   push through *far more* training examples/steps on the quantized model than
   on full precision. The real trade-off: fewer, higher-fidelity updates vs.
   many more, slightly-noisier updates.

2. **For narrow tasks, data coverage tends to dominate the accuracy curve.**
   Most of the quality gain for something like "extract these 8 fields
   reliably" comes from seeing enough varied examples — different phrasings,
   edge cases, nullable-field patterns, all domains well represented. Going
   from 50 → 450 examples typically buys far more accuracy than the small
   ceiling difference between a Q4 and FP16 base (see
   [section 2](#2-does-quantization-q4-vs-full-precision-hurt-fine-tuned-quality)
   — that gap is small for shallow, format-driven tasks).

3. **Quantization noise partially averages out with more gradient updates.**
   More steps means more chances for the LoRA adapter to correct for the base's
   quantization quirks specifically in the weight-space region relevant to the
   task.

4. **Caveat — this flips if extra data is padding, not signal.** Piling on more
   synthetic examples that are repetitive or low-diversity doesn't help — it's
   *diverse, correct* coverage that matters, not raw count. 150 well-varied
   examples/domain can beat 1000 near-duplicate ones.

5. **Caveat #2 — this heuristic weakens for tasks needing deep/rare knowledge
   or subtle reasoning**, where the base model's raw fidelity (not just
   fine-tuning data volume) sets a harder ceiling — extra fine-tuning examples
   can't fully compensate for a blurrier base there. Structured extraction,
   being closer to "format + locate/copy," isn't really in that regime.

**For this project:** "quantized base + as much good data as the budget
allows" is already the direction taken (QLoRA + synthetic data generation) —
just originally for a different reason (free Colab T4 hardware constraint),
not this specific precision/data trade-off.
