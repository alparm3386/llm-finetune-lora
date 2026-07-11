# Walkthrough — Step 3.4: understanding the training script

> A plain-language companion to [`DEV_PLAN_3.4.md`](DEV_PLAN_3.4.md) and
> [`src/train.py`](src/train.py). The dev plan says *what* to build and tracks
> status; this doc explains *what the finished script actually does and why*,
> for someone new to fine-tuning / LoRA. Big picture first, then down into the
> code, station by station. Companion to [`LEARNING.md`](LEARNING.md) (which
> covers the underlying theory).

## Part 1: The big picture — what is 3.4, really?

Step 3.4 builds **one script** ([`src/train.py`](src/train.py)) whose entire job is:

> Take the base Gemma model + our 450 example pairs, and produce a small trained
> **LoRA adapter** file.

That's it. It doesn't *run* the training locally (no GPU on this machine) — it
just assembles all the machinery correctly so that when you hit "run" on Colab
(step 3.6), a fine-tuned adapter pops out the other end.

Think of it as an assembly line with **5 stations**. The script wires them
together in this order:

```
1. Load the base model  →  2. Attach the LoRA adapter  →  3. Prepare the data
        (4-bit Gemma)          (the ~1% we'll train)         (450 → chat text → train/val)
                                                                      ↓
                    5. Save the adapter  ←  4. Train (the actual learning loop)
```

Each station is one function in `train.py`:

| Station | Function | One-line job |
|---|---|---|
| 1 | `load_base_model()` | Download Gemma, load it compressed to 4-bit |
| 2 | `add_lora_adapter()` | Bolt the tiny trainable adapter onto it |
| 3 | `build_dataset()` | Turn our JSONL into model-ready chat text, split train/val |
| 4 | `build_trainer()` | Configure *how* to train (the learning loop + settings) |
| 5 | `save_adapter()` | Write out just the adapter file |

`main()` ([train.py:335](src/train.py#L335)) is literally just those 5 called in
order. If you read only one thing, read `main()` — it's the whole story in ~30
lines.

## Part 2: Why these particular pieces? (the "why", before the "how")

Three big design ideas drive everything:

**① We only train a tiny adapter, not the whole model (LoRA/QLoRA).**
The base Gemma is loaded **frozen and 4-bit compressed** (station 1). We add
small trainable matrices (station 2). Only those get updated. This is what makes
it fit on a free T4 — the whole reason the project is doable at all.

**② The model must see the *exact same prompt* during training and later during
evaluation.**
That's why [`prompt_format.py`](src/prompt_format.py) exists as a **shared**
module. Both `train.py` and the future `evaluate.py` call the same
`to_chat_messages()`. If the prompts drifted even slightly, you'd be training on
one task and testing on another. (Locked decision A.)

**③ We only want the model to learn to *write the JSON*, not to memorize the
question.**
Each training example is `[question] + [answer]` glued together. But we mask the
question so the model is only graded on producing the answer. That's
`train_on_responses_only` (station 4). (Locked decision C.)

## Part 3: The technical details, station by station

Tracing one example through the pipeline.

### Station 3 first (the data) — the most tangible

Raw data on disk (`data/synthetic/medical.jsonl`, one JSON per line):

```json
{"domain": "medical", "document": "A beteg, Kovács János...", "gold": {"patient_name": "Kovács János", ...}}
```

`build_dataset()` ([train.py:110](src/train.py#L110)) does three things:
1. **`load_examples()`** reads all 3 domain files into one list.
2. **`format_example()`** turns each `{domain, document, gold}` into a single
   string of chat text.
3. **`train_test_split`** carves off 5% (~22 examples) as a validation set, just
   to watch `eval_loss` during training. (The *real* before/after eval is
   separate — step 3.5.)

The magic is in step 2, driven by [`prompt_format.py`](src/prompt_format.py):

- **`build_prompt()`** ([prompt_format.py:28](src/prompt_format.py#L28)) —
  assembles the **user turn**: a fixed Hungarian instruction ("extract the data,
  return exactly one JSON object matching this schema, use `null` for missing
  fields, no extra text") + the JSON schema + the Hungarian document.
- **`serialize_gold()`** ([prompt_format.py:34](src/prompt_format.py#L34)) —
  turns the gold answer dict into the **target JSON string**, forcing keys into
  schema order so every example is formatted identically.
- **`to_chat_messages()`** wraps them as `[{user…}, {assistant…}]`.

Then `tokenizer.apply_chat_template()` ([train.py:106](src/train.py#L106))
inserts Gemma's special turn markers. The final training string looks like:

```
<start_of_turn>user
Az alábbi magyar nyelvű szövegből nyerd ki...
JSON séma: {...}
Szöveg: A beteg, Kovács János...<end_of_turn>
<start_of_turn>model
{"patient_name": "Kovács János", ...}<end_of_turn>
```

**That whole string is one training example** — prompt and answer, concatenated.
Note the `<start_of_turn>user` / `<start_of_turn>model` markers; they come back
at station 4.

### Station 1 — `load_base_model()` ([train.py:128](src/train.py#L128))

```python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E2B-it",
    load_in_4bit=True,          # ← the "Q" in QLoRA: compress weights to 4-bit
    full_finetuning=False,      # ← we're NOT touching base weights
    ...
)
```

- **`load_in_4bit=True`** is the QLoRA memory trick — the frozen base is
  compressed so it fits the T4's 16GB.
- **The lazy `from unsloth import ...` inside the function** (not at module top)
  is deliberate: Unsloth needs CUDA, which the local machine lacks. Importing it
  only *inside* this function keeps the rest of `train.py` importable so the
  local unit tests can run. Same pattern in stations 2 and 4.

### Station 2 — `add_lora_adapter()` ([train.py:154](src/train.py#L154))

```python
FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # Gemma-4 is multimodal; freeze the vision tower
    finetune_language_layers=True,     # train the text tower
    finetune_attention_modules=True,   # q/k/v/o projections
    finetune_mlp_modules=True,         # gate/up/down projections
    r=16, lora_alpha=16, lora_dropout=0.0,
    use_gradient_checkpointing="unsloth",  # another VRAM saver
)
```

This is [`LEARNING.md` section 7](LEARNING.md) made real: LoRA wraps the
attention (`q/k/v/o`) and MLP (`gate/up/down`) projections. We use Unsloth's
`finetune_*_layers` **flags** instead of a hand-written `target_modules` list —
because Gemma-4 is multimodal and we must explicitly freeze the vision half
(`finetune_vision_layers=False`). After this call, ~99% of the model is frozen;
only the adapter matrices are trainable.

### Station 4 — `build_trainer()` ([train.py:182](src/train.py#L182))

The fiddliest station (marked "Opus" in the plan). Two parts:

**(a) `SFTConfig`** — all the training knobs (locked decision E):
- `per_device_train_batch_size=2` × `gradient_accumulation_steps=4` →
  **effective batch size 8** (T4 can't hold 8 at once, so we accumulate 4
  mini-batches before each weight update).
- `learning_rate=2e-4`, `cosine` schedule, `adamw_8bit` optimizer, 3 epochs.
- `fp16=not is_bfloat16_supported()` — the T4 has no bf16, so this auto-picks
  fp16 while still working on a fancier L4/A100.

**(b) `train_on_responses_only`** ([train.py:245](src/train.py#L245)) — decision
C, where those turn markers return:

```python
train_on_responses_only(
    trainer,
    instruction_part="<start_of_turn>user\n",   # ← everything after this = masked
    response_part="<start_of_turn>model\n",      # ← everything after this = learned
)
```

It tells the trainer: *"find the `<start_of_turn>model` marker in each example;
compute loss only on the tokens after it."* So the model is graded purely on
producing the gold JSON, not on echoing back the question. ⚠️ The comment at
[train.py:82](src/train.py#L82) warns these exact strings matter — get them wrong
and masking silently zeroes *everything*, and training does nothing.

### Station 5 — `save_adapter()` ([train.py:252](src/train.py#L252))

```python
model.save_pretrained(output_dir)      # saves ONLY the adapter, not the 4-bit base
tokenizer.save_pretrained(output_dir)
```

We save just the adapter (a few MB), because that's how it'll be consumed:
`evaluate.py` and the future HF Hub upload both load the base model separately
and attach this adapter on top.

## Part 4: The `--smoke` flag and the CLI

The remaining code (`parse_args`, `resolve_max_steps`,
[train.py:263-332](src/train.py#L263-L332)) is plumbing so it can run two ways:
- `python src/train.py` → full 3-epoch run
- `python src/train.py --smoke` → caps at 60 steps for a quick "does the
  pipeline even work end-to-end" shakeout, *before* burning an hour on the real
  run.

`resolve_max_steps()` encodes the priority: explicit `--max-steps` wins, else
`--smoke` uses 60, else train full epochs.

## The one-sentence summary

**3.4 wires up a 5-stage pipeline — load frozen 4-bit Gemma → attach a tiny LoRA
adapter → format our 450 pairs as masked chat examples → run the SFT training
loop → save just the adapter — all built so it's testable locally but actually
*runs* on a Colab T4 in step 3.6.**
