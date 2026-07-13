# Walkthrough — Step 3.5: understanding the evaluation script

> A plain-language companion to [`DEV_PLAN_3.5.md`](DEV_PLAN_3.5.md) and
> [`src/evaluate.py`](src/evaluate.py) / [`src/eval_metrics.py`](src/eval_metrics.py)
> / [`src/validate_eval_set.py`](src/validate_eval_set.py). The dev plan says
> *what* to build and tracks status; this doc explains *what the finished
> scripts actually do and why*, for someone new to model evaluation. Big
> picture first, then down into the code. Companion to
> [`DEV_PLAN_3.4_WALKTHROUGH.md`](DEV_PLAN_3.4_WALKTHROUGH.md) (which covers
> training) and [`LEARNING.md`](LEARNING.md) (underlying theory).

## Part 1: The big picture — what is 3.5, really?

Step 3.4 produced a trained LoRA adapter (well — it will, once 3.6 actually
runs it on a GPU). A trained adapter is worthless as a *claim* until you can
answer: **"how much better did fine-tuning actually make the model?"**

Step 3.5 answers that with **one number pair**: F1 score before vs. after
fine-tuning, on real (not synthetic) Hungarian documents the model has never
seen. Everything in this step exists to make that comparison **fair** and
**reproducible**.

"Fair" is the hard part, and it's why this step is bigger than it first looks.
Two models producing different-looking output could differ for two totally
different reasons:

1. The fine-tuned model is *better at extraction* (the thing we actually want
   to measure).
2. The fine-tuned model is *better at following the "output JSON" instruction*
   (a formatting skill, not an extraction skill).

If you only measure "did it produce valid JSON," you conflate these two. A
base model that gets the *format* wrong 40% of the time will score badly even
on documents where it correctly identified every field — you're penalizing it
for punctuation, not comprehension. Step 3.5's entire design is built around
separating these two effects. That's the **2×2** you'll see everywhere:

```
                    prompt-only          +structured decoding
base model          cell 1               cell 2
fine-tuned model    cell 3               cell 4
```

- **Structured decoding** (via the `outlines` library) mechanically *forces*
  every output to be valid JSON matching the schema — the model literally
  cannot emit a stray brace or a missing key. Applying it to **both** models
  means format is no longer a variable: whatever F1 difference remains between
  cell 2 and cell 4 is *purely* about content accuracy — did the model
  correctly find "OTP Bank" vs. hallucinating "OTP Zrt."?
- **Prompt-only** (cells 1 and 3) is the model with no such crutch. Comparing
  cell 1 → cell 3's validity rate shows whether fine-tuning *also* taught the
  model to follow instructions better — a real, separate benefit, but not the
  headline number.

**The headline metric is cell 4 vs. cell 2** (fine-tuned/structured vs.
base/structured F1). Everything else in this step is scaffolding to produce
those four cells correctly and reproducibly.

Three scripts do the work:

| File | Job |
|---|---|
| [`src/eval_metrics.py`](src/eval_metrics.py) | Pure scoring logic — no model, no GPU. Given a gold answer and a prediction, compute right/wrong. |
| [`src/validate_eval_set.py`](src/validate_eval_set.py) | Sanity-checks the hand-labeled real documents before they're trusted as ground truth. |
| [`src/evaluate.py`](src/evaluate.py) | The orchestrator: loads both models, generates both ways, calls `eval_metrics` to score, writes the report. |

## Part 2: Why these particular pieces? (the "why", before the "how")

**① Metrics must be model-agnostic and GPU-free, so they can be trusted and
tested in isolation.** [`eval_metrics.py`](src/eval_metrics.py) imports
nothing but `json` and `jsonschema`. This isn't a style choice — it means the
scoring math (the part most likely to have an off-by-one or a subtle bug) can
be hammered with unit tests on a laptop, with zero risk that a flaky GPU run
or a model quirk is masking a bug in the metric itself. `tests/test_eval_metrics.py`
is 39 tests, more than any other test file in the project — this module is
the one place a silent bug would invisibly corrupt the entire project's
headline result.

**② A field is scored as "retrieval," not "string equality," because
`null` is a valid, correct answer.** This project's schemas deliberately make
fields nullable (`"dosage": ["string", "null"]`) so the model can *truthfully*
say "the document doesn't mention this" instead of inventing a plausible-
sounding value. A naive metric ("did predicted `dosage` equal gold `dosage`?")
treats "both said `null`" as unremarkable — but a metric that instead treats
each field as **something to retrieve** correctly rewards this: predicting
`null` when gold is `null` is a *true negative*, ignored (correctly — there
was nothing to retrieve); predicting a *value* when gold is `null` is a
*hallucination* and punished as a false positive. See Part 3 for the full
table.

**③ The real eval set must be independent of the training data, or the
number is meaningless.** `data/synthetic/` (used for training, step 3.4) is
LLM-generated — testing on it would just measure whether the model memorized
the *generator's* patterns, not whether it can read real Hungarian text. This
is why `data/eval/` is a **separate, hand-labeled, git-tracked** directory
(see [`data/eval/README.md`](data/eval/README.md)) — real documents, real
human-verified answers, checked against the schema by
[`validate_eval_set.py`](src/validate_eval_set.py) before they're trusted.

**④ Every heavy model dependency is a lazy import — same discipline as 3.4.**
`unsloth`, `outlines`, `torch`, `transformers.set_seed` are all imported
*inside* the functions that need them, not at the top of `evaluate.py`. That's
why the whole eval loop, aggregation, and CLI were buildable and testable on
this machine (no GPU) *before* the adapter even exists — the same lesson
carried over from `train.py`.

## Part 3: `eval_metrics.py` — the scoring rules, in detail

This is the part worth understanding precisely, because it's the actual
definition of "correct."

### The five outcomes for one scalar field

For each field (e.g. `drug_name`), per document, compare gold vs. prediction:

| gold | prediction | outcome | why |
|---|---|---|---|
| `null` | `null` | ignored (not counted at all) | nothing to retrieve, nothing missed |
| `null` | a value | **FP** (false positive) | the model hallucinated something that isn't there |
| a value | `null` | **FN** (false negative) | the model missed something that *is* there |
| a value | same value | **TP** (true positive) | correct |
| a value | a *different* value | **FP + FN**, both | the model missed the truth **and** stated something wrong — double-counted, because two separate failures happened |

This is implemented almost verbatim in
[`classify_scalar_field()`](src/eval_metrics.py#L36):

```python
def classify_scalar_field(gold, pred):
    g, p = normalize_value(gold), normalize_value(pred)
    if g is None and p is None: return (0, 0, 0)   # both null -> ignored
    if g is None: return (0, 1, 0)                  # hallucination -> FP
    if p is None: return (0, 0, 1)                  # miss -> FN
    if g == p: return (1, 0, 0)                     # correct -> TP
    return (0, 1, 1)                                # wrong value -> FP + FN
```

### Array fields: set overlap, not order

Fields like `side_effects` are lists. Order in the source text is
incidental — the model shouldn't be punished for listing side effects in a
different order than the gold answer did. So arrays are compared as **sets**:
`TP = |gold ∩ pred|`, `FP = |pred − gold|` (extra items), `FN = |gold − pred|`
(missing items) — [`classify_array_field()`](src/eval_metrics.py#L51).

### Normalization — what counts as "the same value"

Before any comparison, [`normalize_value()`](src/eval_metrics.py#L21):
- Strings: strip leading/trailing whitespace, collapse internal runs of
  whitespace to one space. **Case-sensitive** — Hungarian proper nouns (drug
  names, company names) must match exactly; "OTP Bank" ≠ "otp bank" is a real
  miss, not noise.
- Numbers: compared as `float`, so `620000000 == 620000000.0` — the model
  isn't punished for JSON's int/float ambiguity.
- `None`: passes through unchanged (handled explicitly upstream anyway).

### Aggregating many fields into one number: micro-F1

Once every field of every document has a `(TP, FP, FN)` triple, they're all
pooled together (not averaged per-field first) — this is called **micro-F1**,
and it's the right choice here because it weights every individual field
prediction equally, rather than letting a rare field (say, `currency`, mostly
null) count as much as a common one. [`aggregate_field_scores()`](src/eval_metrics.py#L100)
pools three ways simultaneously: **overall** (the headline number),
**per-domain** (medical vs. business vs. technology), and **per-field**
(e.g. `medical.drug_name` — useful for spotting which specific field the
model struggles with). Standard formulas from there:
`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`,
`F1 = 2·P·R/(P+R)` (defined as 0 if the denominator is 0) —
[`precision_recall_f1()`](src/eval_metrics.py#L85).

### The unparseable-prediction case

In prompt-only mode, the model might not even produce valid JSON. That's
handled by treating an unparseable prediction as `pred=None`, i.e. an *empty*
prediction dict — [`score_example()`](src/eval_metrics.py#L69). Every non-null
gold field becomes an FN ("missed"); no FPs are introduced (you can't
hallucinate a field that isn't there in a payload that doesn't parse). This
can never happen in structured-decoding mode, since `outlines` guarantees
valid JSON by construction.

## Part 4: `validate_eval_set.py` — trusting the ground truth

Before any of the scoring math above means anything, the *gold* answers
themselves have to be right. This script is a standalone, dependency-light
checker (`python src/validate_eval_set.py`) you run *while hand-labeling*
`data/eval/*.jsonl` — well before `evaluate.py`'s heavy model deps even need to
exist. It checks, per example:
- The envelope (`domain`, `document`, `gold`) is well-formed and `domain`
  matches the filename.
- `gold` validates against the domain's JSON Schema from
  [`schemas.py`](src/schemas.py) — every required key present, correct types,
  no stray extra keys (`additionalProperties: false`).

`evaluate.py`'s own loader (`load_eval_examples`,
[evaluate.py:63](src/evaluate.py#L63)) reuses this same validation by default,
so a malformed label can't silently corrupt the metric — it raises loudly
instead.

## Part 5: `evaluate.py`, station by station

Same "assembly line" shape as `train.py`, but for inference + comparison
instead of training:

```
1. Load eval set     2. Load both models      3. Generate, 2 modes x 2 models
  (real, labeled)  →   (base, fine-tuned)   →    (prompt-only, structured)
                                                          ↓
                    5. Write report          ←     4. Score against gold
                 (eval_results.json/.md)          (via eval_metrics.py)
```

### Station 1 — loading the eval set

[`load_eval_examples()`](src/evaluate.py#L63) reads all three domain JSONL
files, validates each `gold` (by default), and raises a clear `EvalSetError`
if a domain file is missing or empty — the eval set is a genuinely *required*
input (there's no synthetic fallback for it), so failing loudly here beats a
confusing crash three stations later.

### Station 2 — `load_inference_model()` ([evaluate.py:96](src/evaluate.py#L96))

Loads a model for inference via Unsloth, same 4-bit-load + float32-AltUp-patch
recipe as `train.py`'s `load_base_model` (that patch is now a shared helper,
`upcast_gemma_per_layer_modules`, imported from `train.py` — no need to
duplicate it). Called twice:

- **Base model:** `load_inference_model(model_name)`, no adapter.
- **Fine-tuned model:** `load_inference_model(model_name, adapter_dir=...)` —
  passing the *adapter's own directory* as the model to load. This works
  because `train.py`'s `save_adapter` wrote an `adapter_config.json` inside
  that directory whose `base_model_name_or_path` field points back at the
  original base model; Unsloth follows that pointer, reloads the 4-bit base,
  and re-attaches the LoRA weights on top — one call does both steps.

`FastModel.for_inference(model)` at the end switches Unsloth into its faster
inference-only mode (as opposed to the training-ready mode used in 3.4).

### Station 3 — generation, two paths ([evaluate.py:133](src/evaluate.py#L133))

Both paths start from the exact same prompt string —
[`build_inference_prompt()`](src/evaluate.py#L136) reuses
`to_chat_messages(domain, document)` (the same function `train.py` uses to
build training targets, just without the `gold` argument) and
`apply_chat_template(..., add_generation_prompt=True)`. This is the concrete
mechanism behind "format is controlled for": literally the same characters go
into the model in all four cells.

**Prompt-only** ([`generate_prompt_only()`](src/evaluate.py#L155)) — plain
`model.generate()`, greedy (`do_sample=False`, so results are reproducible
run to run), decode just the newly generated tokens (slice off the prompt
length first). Whatever comes out, comes out — no safety net.

**Structured** ([`build_structured_generators()`](src/evaluate.py#L183) +
[`generate_structured()`](src/evaluate.py#L209)) — uses the `outlines`
library. `outlines.Generator(model, JsonSchema(schema))` compiles the domain's
JSON Schema into a constraint that's enforced *during* decoding: at every
generation step, only tokens that keep the output on a valid path through the
schema are allowed. The three domain generators are built **once per model**
(schema compilation is the expensive part) and reused across every document of
that domain. Output is guaranteed schema-valid — there's no such thing as an
unparseable structured prediction.

### Station 4 — the eval loop ([evaluate.py:225](src/evaluate.py#L225))

Three layers, each a thin wrapper on the one below:

- [`generate_predictions()`](src/evaluate.py#L237) — runs *one* decode mode
  over *all* documents, parsing each raw output into a dict (or `None` if
  unparseable) via `eval_metrics.parse_prediction`.
- [`score_predictions()`](src/evaluate.py#L273) — feeds every
  `(gold, prediction)` pair through `eval_metrics.score_example`, then
  `aggregate_field_scores` — this is where `eval_metrics.py`'s pure math
  actually gets called with real model output.
- [`run_eval_cell()`](src/evaluate.py#L294) — generate then score, for one
  cell.
- [`run_evaluation()`](src/evaluate.py#L310) — calls `run_eval_cell` four
  times (or two, with `--structured-only`), once per `{base, fine_tuned} ×
  {prompt_only, structured}` combination, and returns the whole 2×2 as one
  dict keyed `"base.structured"`, `"fine_tuned.prompt_only"`, etc.

`apply_limit()` ([evaluate.py:228](src/evaluate.py#L228)) is the plumbing
behind `--limit N` — truncate to the first N eval documents, for a fast
shakeout of the whole pipeline before committing to a full (slow, GPU-time-
consuming) run over the entire real eval set.

### Station 5 — reporting ([evaluate.py:348](src/evaluate.py#L348))

Two output files, both written by [`write_eval_results()`](src/evaluate.py#L441):

- **`results/eval_results.json`** — the full 2×2, including every
  per-domain/per-field breakdown `aggregate_field_scores` computed, plus run
  metadata (which models, seed, how many examples, timestamp). This is the
  "reproducibility" artifact — enough detail to answer "why did this number
  change" later.
- **`results/eval_table.md`** — a rendered, paste-ready Markdown table:
  a headline sentence (base structured F1 → fine-tuned structured F1), the
  2×2 table itself (F1 in every cell; validity rate only in the prompt-only
  row, since it's not meaningful for structured output), and a per-domain
  breakdown. Built for pasting straight into the README (step 3.8).

### The CLI ([evaluate.py:472](src/evaluate.py#L472))

`python src/evaluate.py --adapter outputs/adapter [options]` ties station 1-5
together in `main()`. `--adapter` is the only required flag (there's no
default — evaluating *requires* a trained adapter to exist). Everything else
has a sensible default: `--eval-dir data/eval`, `--out results`,
`--max-new-tokens 512`, `--seed 0`. `--limit N` and `--structured-only` are
the two "make this faster while I'm iterating" flags.

## Part 6: Where 3.5 stops, and why

Every function above was written and unit-tested on this machine — 101 tests
pass with **zero GPU dependency** touched (verified by importing `evaluate.py`
without `torch`/`outlines`/`unsloth` installed and running the full suite).
But *running* the eval for real needs three things this machine doesn't have:

1. **A GPU** — `unsloth`/`outlines`/`torch` model loading and generation only
   work on CUDA.
2. **A trained adapter** — the actual output of step 3.6 (not built yet).
3. **A populated real eval set** — `data/eval/*.jsonl` are still empty; the
   hand-labeling (per `data/eval/README.md`) is manual work, tracked
   separately.

So step 3.5's job was specifically to make sure that when all three of those
finally exist (on Colab, step 3.6), running
`python src/evaluate.py --adapter outputs/adapter` "just works" and produces a
trustworthy number — with all the actual thinking (what counts as correct,
how to keep the comparison fair, how to report it) already done and tested.

## The one-sentence summary

**3.5 builds a fair-by-construction before/after comparison — a pure,
heavily-tested scoring module that treats every field as "retrieve correctly
or don't hallucinate," fed by a 2×2 of {base, fine-tuned} × {prompt-only,
schema-constrained decoding} so that the headline number isolates fine-tuning's
actual content-accuracy gain from mere formatting compliance — all wired,
tested, and waiting for a GPU, an adapter, and a labeled eval set to produce
a real result in step 3.6.**
