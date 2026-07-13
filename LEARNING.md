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
4. [Training memory vs. inference memory](#4-training-memory-vs-inference-memory)
5. [A model = architecture + weights: where each lives, and how deployment works](#5-a-model--architecture--weights-where-each-lives-and-how-deployment-works)
6. [Why PyTorch/HF instead of the "original" framework (JAX) a model was trained in?](#6-why-pytorchhf-instead-of-the-original-framework-jax-a-model-was-trained-in)
7. [Where exactly does LoRA attach in the architecture?](#7-where-exactly-does-lora-attach-in-the-architecture)
8. [Open weight vs. open source, and reading a real config.json](#8-open-weight-vs-open-source-and-reading-a-real-configjson)
9. [3.4.1 — what one training example actually looks like (`prompt_format.py`)](#9-341--what-one-training-example-actually-looks-like-prompt_formatpy)
10. [A checklist for reading any new model's config.json](#10-a-checklist-for-reading-any-new-models-configjson)
11. [`architectures` names a real class — config numbers plug into hand-written code, not the other way around](#11-architectures-names-a-real-class--config-numbers-plug-into-hand-written-code-not-the-other-way-around)
12. [Peeking at `modeling_gemma4.py`: how complex is it, really?](#12-peeking-at-modeling_gemma4py-how-complex-is-it-really)
13. [Is HF `transformers` the only widely-used port?](#13-is-hf-transformers-the-only-widely-used-port)
14. [What host runtime does each port need? HF `transformers` vs. vLLM, with examples](#14-what-host-runtime-does-each-port-need-hf-transformers-vs-vllm-with-examples)
15. [Is HF `transformers` just for experiments, and vLLM for prod?](#15-is-hf-transformers-just-for-experiments-and-vllm-for-prod)
16. [Deploying a LoRA adapter via vLLM](#16-deploying-a-lora-adapter-via-vllm)
17. [What's in a checkpoint zip, PEFT vs. Unsloth, and what deployment actually needs](#17-whats-in-a-checkpoint-zip-peft-vs-unsloth-and-what-deployment-actually-needs)

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

---

## 4. Training memory vs. inference memory

**Setup:** full fine-tuning needs "huge GPU memory" — but doesn't *inference*
need huge memory too, since it's the same big model?

**Short answer: no — training and inference have very different memory
profiles, and that asymmetry is exactly why QLoRA is cheap to train but not
noticeably cheaper to *run* than a fully fine-tuned model.**

In units of "weight memory" (X = size of the weights alone):

- **Inference** only needs the weights loaded (no gradients, no optimizer
  state) — roughly **1x**, regardless of *how* the model was trained. A fully
  fine-tuned model and a LoRA-adapted model are basically the same size at
  inference (LoRA just adds a tiny adapter on top of the same frozen base).
- **Training** needs much more, because of what has to be kept around for
  backprop:
  - Weights: 1x
  - Gradients (one per trainable weight): 1x
  - Adam optimizer state (momentum + variance per weight): 2x
  - **Subtotal: ~4x**, before even counting activations (intermediate values
    kept for backprop), which add more on top and scale with batch
    size/sequence length — so real-world estimates are often **~4-6x**
    inference memory for full fine-tuning.

This is why QLoRA is cheap on *both* axes: the frozen base needs no
gradients/optimizer state at all (only the tiny adapter does), so training
memory drops close to inference memory. Full fine-tuning, by contrast, pays
that ~4-6x multiplier on the *entire* model, every one of its billions of
weights.

---

## 5. A model = architecture + weights: where each lives, and how deployment works

A model is really two separate things bundled together:

- **Architecture**: the code defining the computation graph — how many
  layers, attention mechanism, activation functions, normalization scheme,
  position embeddings. Fixed by design, versioned, lives in a *library*
  (e.g. Hugging Face's `transformers`, in a file like `modeling_gemma3.py`).
- **Weights**: the actual learned numbers sitting inside that architecture's
  matrices. Meaningless without knowing which architecture to plug them into
  — and the architecture with random weights just produces garbage. You need
  both.

**Where the architecture is actually defined:** a downloaded HF model folder
looks like `config.json` + `model.safetensors` + tokenizer files. The weights
file has no code in it — just tensors keyed by name. `config.json` has an
`"architectures"` field (e.g. `"Gemma3ForCausalLM"`) that's a pointer to a
Python class **already living in the `transformers` library** (installed via
`pip install transformers`) — not something downloaded per-model.

**How loading connects the two** — `AutoModelForCausalLM.from_pretrained(...)`:
1. Downloads `config.json`, reads the `"architectures"` field
2. Looks up the matching class already installed in the library
3. Instantiates it with the config's hyperparameters (empty/random weights)
4. Downloads `model.safetensors`, loads tensors into the matching layers
5. Returns a ready-to-run model object

**The dependency chain, end to end:**
```
JAX/PyTorch/TensorFlow   → low-level tensor math + autograd (the "engines")
        ↓
Model developer (e.g. Google) trains internally — architecture design + weights
        ↓
A PyTorch port is written + weights converted → published to the HF Hub
        ↓
`transformers` library — hosts/standardizes the port, easy loading API
        ↓
You: AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B")
```
Note the direction: the model developer doesn't build on top of HF — they
train independently (often in a different framework, see
[section 6](#6-why-pytorchhf-instead-of-the-original-framework-jax-a-model-was-trained-in)),
and a PyTorch-compatible port + weights get published *to* HF afterward. HF is
a distribution/compatibility hub, not a foundation the model was built on.

**Deploying a downloaded model, in increasing order of "real":**
- **Script-level**: `from_pretrained(...)` + `model.generate(...)` in Python —
  already "deployed" in the loosest sense.
- **Serve it as an API**: vLLM (`vllm serve ...`, optimized for throughput —
  continuous batching, paged attention), TGI (Hugging Face's own serving
  stack), llama.cpp/Ollama (for quantized GGUF models, great for local/CPU or
  small-GPU use), or a DIY FastAPI wrapper around `generate()`.
- **With a LoRA adapter specifically** (this project's eventual output): load
  the base model, then `PeftModel.from_pretrained(base_model, "path/to/adapter")`
  to attach it — either kept as a separate small file layered on at load time,
  or merged once (`model.merge_and_unload()`) into a single standalone
  checkpoint.

**For this project:** step 3.7 (publish adapter to HF Hub) means the eventual
deploy story is just "load the base model + attach the adapter" — no custom
architecture work needed, since Gemma's architecture already ships in
`transformers`/Unsloth.

---

## 6. Why PyTorch/HF instead of the "original" framework (JAX) a model was trained in?

**Setup:** Google trains Gemma internally, and it likely isn't PyTorch they
use — so wouldn't it be simpler in production to stick with the original
framework, whatever that was, rather than going through a PyTorch/HF port?

**Short answer: it depends entirely on who you are — for Google, staying in
their native stack is simpler; for practically everyone else (this project
included), the PyTorch/HF path is simpler, precisely because of the "extra"
conversion step, not despite it.**

The framework landscape has **three** major players, not two:
- **PyTorch** (Meta-originated, dominant in the open community)
- **TensorFlow** (Google, older, declining in research use)
- **JAX** (Google's own framework) — and this is the one Google's own model
  teams, including Gemma, mostly train with internally (often via Flax on
  top of it), not PyTorch or TensorFlow.

**JAX is fully open source** (Apache 2.0, `google/jax` on GitHub) — anyone
can `pip install jax` and use it, on GPU or TPU, no special access needed.
The gap isn't legal/access-based, it's ecosystem maturity for a specific
workflow.

**Why JAX is genuinely the better tool — for Google's job:**
1. **Functional composability** — `grad`/`vmap`/`pmap`/`jit` let you
   transform plain math functions (auto-diff, auto-vectorize,
   auto-parallelize) with clean composition, which suits writing *massively
   distributed* training code (thousands of TPU chips) more naturally than
   PyTorch's more imperative `nn.Module` style.
2. **XLA compilation, TPU-first** — JAX traces the whole computation and
   compiles it ahead of time via XLA, tuned hard for TPU pods specifically
   (PyTorch has been catching up with `torch.compile`, but JAX was XLA-first
   from day one).
3. **Built for frontier-scale pretraining** — thousands of TPUs, weeks of
   compute, Google's own hardware. JAX was purpose-built for exactly this.

**Why PyTorch/HF is genuinely the better tool — for almost everyone else's
job (fine-tune/deploy an existing checkpoint on commodity GPUs):**
1. **Ecosystem gravity** — nearly all the practical tooling (vLLM, TGI,
   Ollama, llama.cpp for serving; PEFT/LoRA, Unsloth, `bitsandbytes` for
   fine-tuning/quantization; thousands of community checkpoints/tutorials)
   was built around PyTorch/HF. JAX's equivalents (Flax, Optax, some LoRA
   implementations) exist but are far less battle-tested for this exact
   "download a checkpoint → fine-tune cheaply → serve it" workflow.
2. **Hardware reality** — JAX shines on TPUs; most practitioners (this
   project included, targeting a free Colab T4) have **GPUs**, where
   PyTorch's CUDA tooling is the mature, default choice.
3. **The "extra step" is invisible to you as a consumer** — the PyTorch port
   already exists and is maintained by others; you just `pip install
   transformers` and load it. You're not paying a conversion cost, you're
   benefiting from one someone else already paid.

Neither framework is "worse" in general — each is the right tool for a
different nail. Google uses the tool built for training frontier models at
massive scale on their own hardware; this project uses the tool built for
fine-tuning/deploying an existing checkpoint on a commodity GPU.

---

## 7. Where exactly does LoRA attach in the architecture?

**LoRA is not a new layer inserted into the graph — it's a wrapper around an
existing linear layer's forward pass.** A linear layer computes `y = Wx`;
LoRA changes this to `y = Wx + (BA)x`, where `W` (the original, large weight
matrix) stays frozen, and `B`/`A` are two small new matrices whose product
approximates a low-rank *update* to `W`. Same position in the network, same
layer slot — just modified math, with a small addition on top.

**Which layers, concretely** — within one Transformer block:
```
Input
  ↓
[Self-Attention]
  ├─ q_proj (query)   ← LoRA target
  ├─ k_proj (key)     ← LoRA target
  ├─ v_proj (value)   ← LoRA target
  └─ o_proj (output)  ← LoRA target
  ↓
[Add + Norm]  (residual connection — no weights, nothing to adapt)
  ↓
[MLP / Feed-Forward]        ("Multi-Layer Perceptron" — a small standalone
  ├─ gate_proj  ← LoRA target   feed-forward network: expand the vector to a
  ├─ up_proj    ← LoRA target   wider size, apply a non-linearity, then
  └─ down_proj  ← LoRA target   compress back down. Typically holds the
  ↓                             majority of a Transformer's total parameters.
[Add + Norm]  (again, skipped)
  ↓
Output → feeds into next block
```
LoRA does **not** touch normalization layers (no large weight matrix worth
adapting) or residual connections (not even a learned layer). It *can*
optionally target the embedding layer and final LM head too, but this
project's `train.py` keeps it attention+MLP only via the `finetune_*_layers`
flags (text-only, per the Architecture section above).

**Critically, this whole pattern repeats identically at every block in the
stack** — not just once at the top or bottom. A model with, say, 26
Transformer blocks gets 26 × 7 = 182 separately LoRA-wrapped layers, each with
its own small adapter matrices. Total adapter size stays tiny (~1% of model
parameters) because each individual pair is small, but it's applied
*throughout the whole depth* of the network — letting the adapter shift
behavior at every stage of the model's processing, not just at the input or
output.

**Is this architecture-specific, or a general library?** General library —
that's the whole appeal. Hugging Face's **PEFT** (and Unsloth's faster
implementation) takes a loaded model plus a list of **target module names**
(exactly the `q_proj`/`k_proj`/.../`down_proj` names above), walks the model,
and swaps each matching named submodule for a wrapped version adding the `BA`
term. Nothing about this requires the architecture's original developer
(Google, for Gemma) to build in special support — it works on any model built
from standard `nn.Linear` layers, which is effectively all Transformer
implementations. The only architecture-*awareness* needed is knowing which
module names exist in a given family — which Unsloth ships sensible presets
for per model family, so `train.py` doesn't have to guess.

---

## 8. Open weight vs. open source, and reading a real config.json

**Setup:** if `config.json` fully discloses the architecture (needed just to
run the model at all), what does "open weight, not open source" actually
withhold?

**Short answer: the architecture is essentially always public for open-weight
models — what's withheld is the training data, the training code/recipe, and
often the license's freedom to use/modify/redistribute.**

- **Open weight** = you can download and run the finished checkpoint.
  Necessarily includes a public architecture (`config.json` +
  `modeling_*.py` in `transformers`) — otherwise the weights would be
  unusable, defeating the point of releasing them.
- **Open source** (stricter, per the OSI's Open Source AI definition) =
  additionally, training data + training code are disclosed *and* the
  license imposes no usage restrictions. Llama, Gemma, Mistral, Qwen are all
  "open weight" but **not** "open source" by this bar — their licenses carry
  usage restrictions (e.g. Llama's >700M-MAU clause), and none disclose their
  full training data/recipe. A few research efforts (OLMo, Pythia) go further
  and qualify as genuinely open source.

### Reading a real `config.json` — `google/gemma-4-E2B`'s `text_config`

Fetched the live file from the HF Hub; the `text_config` block reads like
this (trimmed to the interesting fields):

```json
{
  "num_hidden_layers": 35,
  "hidden_size": 1536,
  "num_attention_heads": 8,
  "num_key_value_heads": 1,
  "head_dim": 256,
  "sliding_window": 512,
  "max_position_embeddings": 131072,
  "vocab_size": 262144,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                  "sliding_attention", "full_attention", "..."],
  "rope_parameters": {
    "full_attention":    {"rope_theta": 1000000.0, "rope_type": "proportional", "partial_rotary_factor": 0.25},
    "sliding_attention": {"rope_theta": 10000.0,    "rope_type": "default"}
  },
  "num_kv_shared_layers": 20,
  "final_logit_softcapping": 30.0,
  "tie_word_embeddings": true,
  "enable_moe_block": false
}
```

What each interesting bit means:

- **`num_key_value_heads: 1` vs. `num_attention_heads: 8`** — Grouped-Query
  Attention pushed to the extreme (effectively Multi-Query Attention): all 8
  query heads share a single K/V head. Ties directly back to
  [section 7](#7-where-exactly-does-lora-attach-in-the-architecture)'s
  `q_proj`/`k_proj`/`v_proj` — here `k_proj`/`v_proj` are tiny relative to
  `q_proj`, which shrinks the KV-cache a lot (cheaper long-context inference).
- **`layer_types` alternating pattern** — 4 `sliding_attention` layers then 1
  `full_attention`, repeating. Sliding layers only look at a local window
  (`sliding_window: 512` tokens, cheap); full-attention layers look at the
  entire sequence (expensive, but needed for long-range dependencies). Mixing
  the two is what makes a 131k-token context (`max_position_embeddings`)
  affordable.
- **Per-layer-type RoPE** (`rope_parameters`) — position encoding is tuned
  differently depending on the layer's attention range: sliding layers use a
  small `rope_theta` (short-range), full-attention layers use a much larger
  one with a `partial_rotary_factor` (tuned for long-range positions).
  Splitting this by layer type is a direct consequence of the local/global
  attention split above.
- **`num_kv_shared_layers: 20`** — 20 of the 35 layers share a KV cache
  instead of each keeping their own, another memory-saving trick, consistent
  with this being an "effective size" (E2B) model aimed at lean deployment.
- **`final_logit_softcapping: 30.0`** — clips final logits into a bounded
  range before softmax; a Gemma-specific stability trick.
- **`enable_moe_block: false`** — the code path for mixture-of-experts
  exists (shared with a bigger sibling in the same family) but is switched
  off for this checkpoint.

**Ties back to earlier sections:** this is `config.json` acting exactly as
described in [section 5](#5-a-model--architecture--weights-where-each-lives-and-how-deployment-works)
— a complete, inspectable spec sheet, nothing hidden. And it's *why* a LoRA
adapter is tied to one exact model (section 7): the adapter's matrix shapes
are derived straight from these numbers (`hidden_size`, `head_dim`, etc.) —
change the config, and the adapter no longer fits.

---

## 9. 3.4.1 — what one training example actually looks like (`prompt_format.py`)

`src/prompt_format.py` is the module that decides exactly what text the model
sees and what it's supposed to produce — shared by `train.py` and (later)
`evaluate.py` so both stages use a byte-identical prompt. If training and eval
prompts drifted even slightly, you'd effectively be evaluating a different
task than the one trained.

Three small functions do all the work:

1. **`build_prompt(domain, document)`** — assembles the "user" turn: a fixed
   Hungarian instruction ("extract the requested data, return exactly one
   JSON object matching the schema, use `null`/`[]` for missing fields, no
   extra text") + the domain's JSON Schema (from `schemas.py`, pretty-printed)
   + the raw Hungarian document.

2. **`serialize_gold(gold, domain)`** — turns the gold-answer dict into the
   target JSON string, but **forces key order to match the schema's property
   order**, not the source JSONL's arbitrary order. Reasoning: if training
   targets had inconsistent key ordering across examples, the model would
   waste learning capacity on "what order do keys come in" instead of the
   actual field values — a small format-consistency decision that keeps the
   training signal focused on content.

3. **`to_chat_messages(domain, document, gold=None)`** — wraps both into
   `[{"role": "user", ...}, {"role": "assistant", ...}]`, the format Gemma's
   chat template expects. With `gold` omitted (inference/eval time), it
   returns just the user turn, ready for
   `apply_chat_template(..., add_generation_prompt=True)` — this is the hook
   `evaluate.py` (3.5) will reuse.

Concretely, one training example renders to something like:
```
<start_of_turn>user
Az alábbi magyar nyelvű szövegből nyerd ki...
JSON séma: {...}
Szöveg: [the Hungarian document]
<start_of_turn>model
{"patient_name": "...", "diagnosis": "...", ...}
```
That whole string — prompt + target concatenated — is what gets tokenized and
fed to the model. `train_on_responses_only` (decision C, section 1) then masks
everything except the `<start_of_turn>model` completion, so loss is only
computed on the gold-JSON part, not the prompt/schema text.

---

## 10. A checklist for reading any new model's config.json

**Setup:** when sizing up an unfamiliar model's `config.json`, what actually
matters, in what order?

1. **`architectures` + `model_type`.** Tells you which class loads the model
   (e.g. `Gemma4ForConditionalGeneration`) — check it against your installed
   `transformers_version`; a mismatch means `from_pretrained()` will fail or
   need a newer/dev install.

2. **Single-tower or composite?** Check first, before anything else. Some
   configs are one flat block; others (like `google/gemma-4-E2B`, fetched live
   from the Hub) nest separate `text_config` / `vision_config` / `audio_config`
   blocks — three distinct towers glued together, not one Transformer. This is
   exactly what decided `train.py`'s `finetune_vision_layers=False` call
   (section 7) — you can't know that flag is needed without seeing this
   nesting.

3. **Size knobs, for compute/VRAM budgeting:** `num_hidden_layers` × `hidden_size`
   gives rough depth/width; `intermediate_size` shows the MLP expansion ratio
   (here 6144 = 4× the 1536 `hidden_size`, see section 7's MLP note);
   `vocab_size` (262144 here) sizes the embedding matrix, which adds real
   memory even though it's usually not a LoRA target.

4. **Attention shape — the easy-to-miss gotcha:** `num_attention_heads` vs.
   `num_key_value_heads` reveals GQA/MQA (here: 8 query heads share **1** KV
   head — extreme grouped-query attention, shrinking the KV cache a lot). Also
   don't assume `num_attention_heads × head_dim == hidden_size` — for this
   model it's `8 × 256 = 2048 ≠ 1536`; some architectures just don't tie those
   together.

5. **Context length & attention pattern:** `max_position_embeddings` for the
   trained ceiling; `layer_types` + `sliding_window` for whether attention
   alternates between cheap local windows and full global attention (this
   model: sliding every layer except every 5th, which is what makes a
   131k-token context affordable — see section 8's worked example).

6. **Unfamiliar fields → don't guess, look them up.** Fields like
   `use_double_wide_mlp`, `num_kv_shared_layers`, `hidden_size_per_layer_input`,
   `enable_moe_block` are architecture-specific quirks, not generic Transformer
   vocabulary. `enable_moe_block: false` + `num_experts: null` here means
   mixture-of-experts scaffolding exists in the code path but is switched off
   for this particular checkpoint — a sibling model in the family likely uses it.

7. **Tokenization/special tokens:** `bos_token_id`, `eos_token_id`,
   `pad_token_id`, plus any modality tokens (`image_token_id`, `audio_token_id`,
   etc. — this model has four, one per modality/boundary). Getting pad/eos
   wrong is a classic silent bug: loss computed over padding, or generation
   that never stops.

**For QLoRA + Unsloth specifically:** in practice, only #2 (composite towers →
which `finetune_*_layers` flags apply) and #3/#4 (does it fit a T4 at 4-bit)
require manual judgment — Unsloth already knows Gemma's internals, so LoRA
target shapes don't need to be hand-derived. This whole checklist matters most
for a genuinely new/obscure architecture Unsloth doesn't have a preset for,
where you'd write a `target_modules` list by hand instead.

---

## 11. `architectures` names a real class — config numbers plug into hand-written code, not the other way around

**Setup:** so is `"Gemma4ForConditionalGeneration"` just an arbitrary label in
`config.json`, or does the architecture somehow get built up automatically
from the config's numbers?

**Short answer: neither extreme — it's a binding pointer to a real,
hand-written Python class that must already exist in `transformers`, and that
class defines *how* computation happens; the config only supplies the
*numbers* that get plugged into it.**

**The rough end-to-end procedure, extending [section 6](#6-why-pytorchhf-instead-of-the-original-framework-jax-a-model-was-trained-in)'s dependency chain:**
1. Google designs the architecture and trains the weights internally, in JAX
   (often via Flax).
2. Someone — usually Google's own team working directly with Hugging Face
   around release day (not a random outside guess, though for smaller/community
   models it genuinely can be an external contributor) — **hand-ports** that
   architecture's math into a new PyTorch class, e.g. `Gemma4ForConditionalGeneration`,
   added to the `transformers` library.
3. That class ships in a `transformers` release; `config.json`'s
   `"architectures"` field names it exactly, as a contract: "the code that
   runs this checkpoint is the class of this exact name, in this library."

**Why the class — not the config — is where the architecture actually lives:**
`config.json` is pure data (`hidden_size: 1536`, `num_attention_heads: 8`, ...)
— numbers with no information about *how* they're used. The class is
hand-written code that says things like:

```python
class Gemma4Attention(nn.Module):
    def __init__(self, config):
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim)
        ...
    def forward(self, x):
        q = self.q_proj(x)
        # rotary embeddings, sliding-window mask, attention math...
```

The config's numbers get **plugged into** an already-designed structure
(matrix sizes, how many blocks to stack) — but *that there's a `q_proj` at
all, that attention alternates sliding/full every 5th layer, the exact RoPE
formula per layer type* (all from [section 8](#8-open-weight-vs-open-source-and-reading-a-real-configjson)'s
worked example) is logic someone wrote by hand, matching Google's original JAX
implementation's math. None of that is derivable from the config alone.

**The stakes of getting the port wrong:** if the hand-written class's math
doesn't exactly match the original JAX implementation (e.g. a slightly
different RoPE formula, or attention masking off-by-one), the ported model can
silently produce different — often subtly worse — outputs than the original,
*even with the exact original weights loaded in*. Loading succeeds, no error
is raised, the model just quietly behaves a bit wrong. This is why widely-used
ports (major model families) get heavy scrutiny and cross-checking against
reference outputs before release — the port is the fragile, easy-to-get-wrong
step, not the config.

**One-line summary:** config = "what sizes to plug in," class = "the actual
computation graph." Two genuinely separate artifacts; the second one is the
hard, hand-written part, not something auto-derived from the first.

---

## 12. Peeking at `modeling_gemma4.py`: how complex is it, really?

**Setup:** having established the class (not the config) defines the
architecture (section 11), what does that class actually look like — and is
it too complex to realistically understand?

**Scale:** `modeling_gemma4.py` on the `transformers` GitHub repo is roughly
**2,650 lines, 34 classes**. That's a lot for "one model" — but it's not one
Transformer, it's **three cohabiting in one file**: text, vision, and audio,
each with their own attention/MLP/embedding classes, plus glue code merging
them into one sequence.

**Rough breakdown by tower:**

| Tower | Key classes | What's distinctive |
|---|---|---|
| **Text** | `Gemma4TextAttention`, `Gemma4TextMLP`, `Gemma4TextDecoderLayer`, `Gemma4TextRotaryEmbedding`, plus `Gemma4TextExperts`/`Gemma4TextRouter` | The MoE routing classes exist in code but are dead for this checkpoint (`enable_moe_block: false`, section 10) — alive for a bigger sibling model in the family. Handles the sliding/full attention split + per-layer RoPE seen in the config (section 8). |
| **Vision** | `Gemma4VisionAttention`, `Gemma4VisionMLP`, `Gemma4VisionPatchEmbedder`, `Gemma4VisionRotaryEmbedding` | Uses **2D** rotary embeddings (images need x/y position, not just sequence position) — genuinely different math from text RoPE, not just a relabeled copy. |
| **Audio** | `Gemma4AudioAttention`, `Gemma4AudioFeedForward`, `Gemma4AudioCausalConv1d`, `Gemma4AudioSubSampleConvProjection` | Convolution layers *before* attention, subsampling raw audio to a manageable sequence length — a different architectural family (CNN-then-Transformer) bolted on. |
| **Glue** | `Gemma4MultimodalEmbedder`, various `*Output` dataclasses | Merges token/image/audio embeddings into one sequence before the text decoder runs. |

**A concrete illustration of section 11's point** — this attention snippet
has zero corresponding config.json field:
```python
query_states = self.q_proj(hidden_states).view(hidden_shape)
query_states = self.q_norm(query_states)
query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
```
That `q_norm` (normalizing queries before applying rotary embeddings) is a
Gemma-family stability trick — hand-written logic, not derivable from any
config number.

**Is this too complex to understand?** No — the *scale* is mostly repetition,
not novel ideas:
- Once you understand one attention block (q/k/v/o proj → RoPE → softmax →
  weighted sum), you've understood most of `Gemma4TextAttention`,
  `Gemma4VisionAttention`, and `Gemma4AudioAttention` — variations on one
  theme, not three unrelated concepts. The `*Output` dataclasses are just
  "named bags of tensors," no logic at all.
- The genuinely novel ~20% is small and individually well-documented:
  `q_norm`, 2D vs. 1D RoPE, causal 1D convolutions before audio attention,
  MoE routing. Each is a nameable, googleable concept with a paper behind it
  — you don't hold all 2,650 lines in your head, you pattern-match "oh, this
  is the RoPE part."
- Practically, none of this needs reading to *use* the model — that's the
  whole point of `AutoModelForCausalLM.from_pretrained()` + Unsloth (section
  5): someone already verified this code matches Google's original math, so
  it's a black box (text/audio/image in, text out) for everyday use.
  Cracking it open here was pure curiosity, not something `train.py` requires.

**Caveat on this section:** fetched via a summarizing web-fetch tool rather
than a raw file read, so treat exact line counts/snippets as "probably right"
rather than verified — worth a direct read of the source if this ever matters
for real debugging.

---

## 13. Is HF `transformers` the only widely-used port?

**Setup:** HF `transformers` is "a port" for e.g. Gemma (section 11) — is it
the only big community port, or are there others?

**Short answer: HF is the dominant one, but far from the only one — several
independent reimplementations exist, each targeting a different deployment
niche, and each is its own separate hand-written translation of the original
JAX math (same fragility as section 11's porting risk, just repeated across
ecosystems).**

**Some of the widely-used ports/runtimes:**

- **`llama.cpp` / GGUF** — a from-scratch C/C++ reimplementation, no PyTorch
  dependency at all. Optimized for CPU and consumer GPUs, quantized GGUF
  formats. Powers Ollama, LM Studio. A genuinely independent codebase from
  HF's — attention/RoPE/etc. re-derived in C++ by different people.
- **vLLM** — loads HF-format weights, but runs its **own internal model
  implementation** (not just calling `transformers`), rewritten for
  high-throughput serving (paged attention, continuous batching). Using HF
  checkpoints doesn't mean using HF's forward-pass code.
- **MLX** — Apple's framework for Apple Silicon (M-series chips), with its
  own model ports for local on-Mac inference.
- **TensorRT-LLM** — NVIDIA's own port, compiled for maximum throughput on
  NVIDIA GPUs, common in enterprise production serving.
- **Google's own native formats** — since Google trains in JAX/Flax (section
  6), they often also publish **native JAX/Flax weights** directly (Kaggle,
  Google's own `gemma` JAX library) — not even a "port" in the same sense,
  closer to the original training-time representation.
- **ONNX Runtime** — a hardware-agnostic export format, used to deploy models
  onto all kinds of specialized/edge/mobile hardware.

**Why HF's port is the default for fine-tuning specifically (this project):**
not because it's uniquely "more correct," but because the *tooling
ecosystem* — PEFT, Unsloth, `bitsandbytes`, `trl`/`SFTTrainer` — is built
directly against `transformers`' class interfaces. The other ports mostly
target **inference/serving**, not training; you wouldn't typically fine-tune
against a GGUF or TensorRT-LLM build.

**The interesting implication, extending section 11:** every one of these
ports is an *independent* hand-written translation of the same original math
— so each could, in principle, carry subtly different bugs or numerical
precision quirks vs. the JAX original, and vs. each other. Popular models get
cross-validated fairly heavily (output comparisons against reference), but
it's a real, occasionally-relevant reason "which port/runtime did you run
this on" can affect reproducing exact numbers.

---

## 14. What host runtime does each port need? HF `transformers` vs. vLLM, with examples

**Setup:** llama.cpp and vLLM each need their own port of a model (section
13) — so what "host runtime" does *HF's* port actually need? And does vLLM
need its own separate port even for a model already in `transformers`?

**HF `transformers` is self-contained: Python + PyTorch (+ CUDA/cuDNN if
using GPU), nothing else.** No separate server process, no special runtime.
Every layer (`Gemma4TextAttention`, `Gemma4TextMLP`, ...) is a plain
`torch.nn.Module`; `model.generate(...)` just runs a Python script, eagerly,
calling into PyTorch's C++ core (`libtorch`) for the actual matrix math. This
is the same stack this project already uses locally — no extra dependency
beyond what `load_base_model()` in
[`train.py`](src/train.py#L129-L151) already pulls in via Unsloth's
`FastModel` wrapper around this same idea.

**vLLM needs its own separate port too, even for a model HF already
supports.** Confirmed by checking vLLM's model registry directly
(`vllm/model_executor/models/registry.py`) — it's a dict mapping
`config.json`'s `"architectures"` string to vLLM's *own* module/class:
```python
"Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration"),
```
So `"Gemma4ForConditionalGeneration"` is a binding contract (section 11)
consumed by **two independent implementations** — HF's `modeling_gemma4.py`
and vLLM's own `gemma4_mm` module — both keyed off the exact same string, but
neither depending on the other's code.

**How to check whether a given model has vLLM support, in general:**
1. Easiest: vLLM's [Supported Models docs](https://docs.vllm.ai/en/latest/models/supported_models.html) — searchable, no code needed.
2. Source-level: grep `registry.py` for the model's `"architectures"` string
   from its `config.json`. If it's missing, `vllm serve` fails outright with
   an unsupported-architecture error at load time — no ambiguity.

**Why vLLM bothers re-porting instead of just calling `transformers`:**
`transformers`' eager Python execution is optimized for *correctness and
flexibility* (easy to fine-tune/debug/patch — exactly why Unsloth's
`finetune_*_layers` trick works, section 7), not for serving throughput.
vLLM's whole reason to exist is rewriting the model's execution into a form
that adds continuous batching, PagedAttention (efficient KV-cache memory
layout), and kernel-level optimizations that plain eager PyTorch doesn't do —
serving many concurrent requests efficiently, not just running one script.

**Example — HF `transformers`, one Python process, no server:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E2B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")

messages = [{"role": "user", "content": "Mondj egy viccet magyarul."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

output = model.generate(inputs, max_new_tokens=200)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Example — vLLM, offline batch generation:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="google/gemma-4-E2B")
sampling_params = SamplingParams(max_tokens=200)

outputs = llm.generate(["Mondj egy viccet magyarul."], sampling_params)
print(outputs[0].outputs[0].text)
```

**Example — vLLM as an actual server** (OpenAI-compatible HTTP API, the
typical company-deployment shape):
```bash
vllm serve google/gemma-4-E2B --port 8000
```
Any OpenAI-client-compatible code then just points `base_url` at
`http://localhost:8000/v1` — the same pattern this project's own
`generate_data.py` already uses against its local proxy.

**The practical difference:** the HF snippet is "a script that runs a model
once." The vLLM snippet is "a serving engine" — batches concurrent requests
together on the GPU and is what actually sits behind a production API. That
throughput/flexibility trade-off is the whole reason the second, independent
port exists at all.

**For this project:** both the T4 training run (3.6) and the "load base +
attach adapter" deploy story (3.7, section 5) stay in the `transformers`/
PyTorch runtime — the right call here, since this is a fine-tuning + modest
demo deploy, not a high-throughput production server where vLLM's trade-offs
would pay off.

---

## 15. Is HF `transformers` just for experiments, and vLLM for prod?

**Setup:** rough mental model check — vLLM is for prod deployment (nice
OpenAI-style endpoint, real traffic), HF `transformers` is for trying things
once / experiments. Is that right?

**Mostly right, with one important refinement: the cleaner split isn't
"experiment vs. prod," it's "can train/modify the model" vs. "can serve the
model at scale."**

**vLLM for prod** — right, and it's not really about the OpenAI-style API
being "nice" (that's just the interface). The actual value is underneath:
PagedAttention + continuous batching (section 14) let one GPU serve **many
concurrent requests** far more cheaply than looping `model.generate()`
per-request. If the deploy shape is "one endpoint, real traffic, cost-per-request
matters" — that's exactly the trade-off vLLM was built for.

**HF `transformers` is not *just* for one-off experiments — it's the only one
of the two that can train at all.** vLLM is an inference-only serving engine
(no backward pass); this project's entire training stack — PEFT, Unsloth,
`trl`/`SFTTrainer`, `train.py` itself — is built directly on `transformers`'
Python object model, because that's the only layer where gradients/backprop
exist. So `transformers` isn't "the toy version you graduate from," it's
"the layer where modification happens at all." Its other legitimate uses
beyond training:
- One-off experiments (what prompted this question)
- Low-traffic/single-user apps, where vLLM's batching setup isn't worth it
- Debugging/inspecting internals (hooking intermediate activations, etc.) —
  much easier against eager Python objects than through vLLM's
  optimized-away internals

**The cleaner mental model:** `transformers` owns "can modify/train the
model" entirely (nothing else in this landscape can); vLLM/TensorRT-LLM/etc.
own "can serve the model at scale." **This project actually straddles both**:
`train.py` uses `transformers`/Unsloth because it *must* (training only
exists there), while the eventual demo deploy (3.7-3.8) will likely *also*
stay on `transformers` — not because it's "just for experiments," but because
"one demo, low traffic" genuinely doesn't need vLLM's scale-serving
trade-offs yet.

---

## 16. Deploying a LoRA adapter via vLLM

**Setup:** train with `transformers`/Unsloth (section 15 — the only layer
that can train), then deploy the result via vLLM (the scale-serving layer,
section 14). How does the LoRA adapter specifically get into vLLM?

**vLLM has native LoRA support — no merging required.** It can load the base
model once into GPU memory and swap adapters per-request, which is the actual
point of adapters over full fine-tunes: one base model, many cheap
task-specific variants layered on top.

**As a server:**
```bash
vllm serve google/gemma-4-E2B \
  --enable-lora \
  --lora-modules hu-extract=/path/to/models/lora_adapter \
  --port 8000
```
`/path/to/models/lora_adapter` is exactly what `save_adapter()` writes in
[`train.py`](src/train.py#L252-L260) — the `args.out` directory `main()`
passes at the end of training (adapter weights + tokenizer, not merged base
weights — see the function's own docstring). `hu-extract` is an arbitrary
name, used as the `"model"` field in requests.

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
resp = client.chat.completions.create(
    model="hu-extract",   # selects the LoRA adapter, not the base model
    messages=[{"role": "user", "content": "..."}],
)
```
Multiple `--lora-modules` can be registered at once, all sharing the same
loaded base model underneath.

**Offline, no server:**
```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="google/gemma-4-E2B", enable_lora=True)
sampling_params = SamplingParams(max_tokens=200)

lora_request = LoRARequest("hu-extract", 1, "/path/to/models/lora_adapter")
outputs = llm.generate(["Szöveg: ..."], sampling_params, lora_request=lora_request)
print(outputs[0].outputs[0].text)
```
The `1` is an arbitrary numeric adapter ID vLLM tracks internally — must be
unique per adapter if several are loaded at once.

**Worth flagging for this project specifically:** vLLM's LoRA hot-swap
support requires target modules it knows how to handle — standard
`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj` (exactly
decision E's target modules, section 7) are well-supported, so no conflict
with the current setup. If the target modules ever changed to something
unusual, that would be the first thing to check against vLLM's LoRA docs.

---

## 17. What's in a checkpoint zip, PEFT vs. Unsloth, and what deployment actually needs

**Setup:** a first real adapter zip out of a Colab run turned out to be 380 MB
uncompressed to 500 MB — surprisingly large for something described as a
"tiny" LoRA adapter (section 1). Unpacking why, and what a checkpoint even
is, led into PEFT and Unsloth's actual relationship.

### What is a "checkpoint," generally?

A saved snapshot of training state, not just the model weights. Three things
bundled together:
- **Weights** — the actual learned parameters (here: just the LoRA adapter's
  `B`/`A` matrices, section 7 — the frozen base isn't re-saved).
- **Optimizer state** (`optimizer.pt`) — AdamW's per-weight momentum/variance
  buffers. Needed only to *resume* training with correct momentum; irrelevant
  for inference.
- **Metadata** (`trainer_state.json`, `rng_state.pth`, `training_args.bin`) —
  step/epoch counters, RNG state, the config used. Also resume-only.

Checkpoints exist so a training run can survive interruption (Colab
disconnects, T4 session timeouts) without losing hours of progress, and so
you can pick the best-performing checkpoint rather than assuming the last one
is best.

### Why the zip was 380 MB, not ~100 MB

Inspecting it (`zipfile` + `trainer_state.json`) showed **three full copies**
of the adapter bundled together: the final `adapter_model.safetensors`
(101 MB) plus two rolling training checkpoints (`checkpoint-140`,
`checkpoint-159`), each *also* containing their own 101 MB adapter weights,
50 MB `optimizer.pt`, and 32 MB `tokenizer.json`. None of that duplication was
a bug — `SAVE_STEPS = 20` / `SAVE_TOTAL_LIMIT = 2` at
[`train.py:72-74`](src/train.py#L72-L74) intentionally keeps the 2 most
recent periodic checkpoints as a resume safety net — it's just that the
*deployment* zip shouldn't also carry that resume scaffolding.

**Why `checkpoint-159` isn't a round multiple of `SAVE_STEPS=20`:** the HF
`Trainer` always does one extra save at the true end of training, regardless
of `save_steps` alignment. `trainer_state.json` confirmed
`global_step == max_steps == 159`, `epoch: 3.0` — training simply finished at
step 159 (`≈ ceil(train_examples / effective_batch_size) × epochs`, with
`effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE × GRAD_ACCUM_STEPS = 2×4 = 8`),
so `checkpoint-140` (last periodic multiple of 20) and `checkpoint-159`
(final, off-grid) are both legitimate, not a sign anything is missing.

**Is 101 MB itself reasonable for an r=16 adapter?** Yes, given
`finetune_mlp_modules=True` at [`train.py:192`](src/train.py#L192) — LoRA is
applied to MLP projections (`gate`/`up`/`down`, the widest matrices in a
Transformer block, section 7's MLP note) in addition to attention, which is
"decision E," a deliberate accuracy/size tradeoff, not an oversight.

### What deployment actually needs

Minimal set to `PeftModel.from_pretrained(base, adapter_dir)`:
- `adapter_model.safetensors` — the weights.
- `adapter_config.json` — tells PEFT which base model to attach to, and the
  `r`/`alpha`/target-modules shape; without it the safetensors file is just
  unlabeled numbers.

Worth keeping alongside for exact reproducibility (guards against tokenizer/
chat-template drift vs. the base model on the Hub), but not strictly required
to load the adapter itself: `tokenizer.json`, `tokenizer_config.json`,
`chat_template.jinja`.

Skip entirely for a deploy bundle: `processor_config.json` (vision-processor
config — irrelevant, `finetune_vision_layers=False`), `README.md`
(auto-generated model card, no functional role), and everything under
`checkpoints/*` (`optimizer.pt`, `rng_state.pth`, `trainer_state.json`,
`training_args.bin` — resume-only state). That trims the ~380 MB zip down to
roughly the ~101 MB adapter + ~32 MB tokenizer ≈ **~134 MB** for something
actually deployable.

### PEFT vs. Unsloth — what each one is actually doing

**PEFT** = **P**arameter-**E**fficient **F**ine-**T**uning — Hugging Face's
library (and the umbrella term) for the whole family of "train a small
subset of parameters, freeze the rest" techniques. LoRA (section 1, section
7) is one method in that family; others include prefix tuning, prompt
tuning, (IA)³, and classic bottleneck adapters. PEFT defines *what* to
train: the `LoraConfig`, which named modules get wrapped, the resulting
`adapter_config.json`/`adapter_model.safetensors` artifact shape.

**Unsloth doesn't replace PEFT — it's an optimization layer wrapping it.**
`FastModel.get_peft_model(...)` at [`train.py:187`](src/train.py#L187) still
produces a standard PEFT `LoraConfig`/adapter under the hood; Unsloth just
constructs it through hand-written Triton/CUDA kernels and manually-derived
backward passes instead of generic PyTorch autograd, claiming ~2x faster
training and much lower VRAM for the LoRA+quantized-model combination
specifically. That's *why* a T4 (16 GB) is enough here — plain
`transformers`+`peft`+`bitsandbytes` would likely be slower and might not
even fit `gemma-4-E2B` (multimodal, plus MLP-included LoRA, section 10's
composite-tower point) in that budget, not just "would run a bit slower."

Concretely, everything Unsloth touches in `train.py` is a drop-in
accelerator over a standard piece:

| Unsloth call | Standard equivalent it wraps |
|---|---|
| `FastModel.get_peft_model(...)` ([`train.py:187`](src/train.py#L187)) | `peft.get_peft_model(model, LoraConfig(...))` |
| `load_base_model` (Unsloth's `FastModel.from_pretrained`) | `transformers.AutoModelForCausalLM.from_pretrained(..., quantization_config=BitsAndBytesConfig(load_in_4bit=True))` |
| `use_gradient_checkpointing="unsloth"` ([`train.py:197`](src/train.py#L197)) | `gradient_checkpointing=True` on `TrainingArguments`/`SFTConfig` (Unsloth's variant claims extra VRAM savings) |
| `train_on_responses_only` | hand-rolled loss-masking of the prompt tokens (same effect, more boilerplate) |

**The output stays portable either way.** Because Unsloth produces a
standard PEFT adapter, the saved `adapter_model.safetensors` +
`adapter_config.json` load with plain `PeftModel.from_pretrained` later —
Unsloth is a *training-time* accelerator, not a runtime dependency for
inference/deployment (section 5, section 16's vLLM deploy path needs no
Unsloth at all).
