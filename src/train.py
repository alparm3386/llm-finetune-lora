"""QLoRA fine-tuning of Gemma 4 E2B with Unsloth.

This module runs the actual fine-tuning job: load the base model in 4-bit
(QLoRA) via Unsloth, attach a LoRA adapter, and train it on the synthetic
text-to-JSON dataset produced by `generate_data.py`. Intended to run on a
Google Colab T4 instance (see `notebooks/train_colab.ipynb`, step 3.4.6).

Pushing the trained adapter to the Hugging Face Hub is a separate step (3.7);
this script only saves it locally.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from prompt_format import to_chat_messages
from schemas import SCHEMAS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOMAINS = tuple(SCHEMAS)

# Default RNG seed for the data split, LoRA init, and trainer — matches
# Unsloth's conventional `random_state = 3407`.
DEFAULT_SEED = 3407

# Fraction of the synthetic data held out for `eval_loss` monitoring during
# training. The *real* before/after eval (3.5) uses a separate, hand-labeled
# set — this split has no bearing on the reported success metric.
VAL_FRACTION = 0.05

# Base model. `unsloth/gemma-4-E2B-it` is Unsloth's dynamic-4bit build of the
# `google/gemma-4-E2B` Instruct model named in SCOPE.md — same weights, but
# pre-quantized for a faster download and lower-VRAM QLoRA load on a T4. Pass a
# different `--model` (e.g. the vanilla `google/gemma-4-E2B`) to override.
DEFAULT_MODEL_NAME = "unsloth/gemma-4-E2B-it"

# max_seq_length = 2048 covers ~95% of doc+gold lengths (see DEV_PLAN_3.4.md);
# bump to 3072 for the long tail at the cost of VRAM.
MAX_SEQ_LENGTH = 2048

# LoRA hyperparameters (locked decision E): r=alpha=16, no dropout, adapters on
# both attention (q/k/v/o) and MLP (gate/up/down) projections of the language
# tower only — vision layers are left frozen since the task is text-only.
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# T4-tuned training hyperparameters (locked decision E). Effective batch size
# is 2 x 4 = 8. `adamw_8bit` + gradient checkpointing keep the E2B QLoRA run
# inside the T4's 16 GB. LR 2e-4 with a cosine schedule is the Unsloth default
# for LoRA SFT.
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
OPTIMIZER = "adamw_8bit"
LOGGING_STEPS = 1
# eval/checkpoint cadence; the synthetic train set is small (~430 ex -> ~160
# steps over 3 epochs at effective batch 8), so every 20 steps is a few times
# per epoch. `save_total_limit` caps Colab disk use.
EVAL_STEPS = 20
SAVE_STEPS = 20
SAVE_TOTAL_LIMIT = 2

# `--smoke` step budget: the "runnable at a few hundred steps" config from the
# plan, for a quick end-to-end shakeout rather than a full training run.
SMOKE_MAX_STEPS = 60

# Gemma chat-template turn markers for `train_on_responses_only`: loss is
# computed only on the model's completion (the gold JSON), with the user turn
# (instruction + schema + document) masked to -100. Gemma 2/3/3n use
# "<start_of_turn>user\n" / "<start_of_turn>model\n", but Gemma 4's template
# renders turns as "<|turn>user\n" / "<|turn>model\n" (confirmed via
# `tokenizer.apply_chat_template` on unsloth/gemma-4-E2B-it) — using the wrong
# markers masks every label to -100 and raises in `train_on_responses_only`.
GEMMA_INSTRUCTION_PART = "<|turn>user\n"
GEMMA_RESPONSE_PART = "<|turn>model\n"


def load_examples(data_dir: Path) -> list[dict[str, Any]]:
    """Read all per-domain synthetic JSONL files into a single list of examples."""
    examples: list[dict[str, Any]] = []
    for domain in DOMAINS:
        path = data_dir / f"{domain}.jsonl"
        with path.open(encoding="utf-8") as f:
            examples.extend(json.loads(line) for line in f if line.strip())
    return examples


def format_example(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Render one {domain, document, gold} example as Gemma-chat-formatted text.

    Uses the tokenizer's native chat template (locked design decision B) so
    training text matches exactly what `evaluate.py` will feed the model at
    inference time via the same `to_chat_messages` helper.
    """
    messages = to_chat_messages(example["domain"], example["document"], example["gold"])
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


def build_dataset(
    data_dir: Path, tokenizer: Any, seed: int, val_fraction: float = VAL_FRACTION
) -> tuple[Dataset, Dataset]:
    """Load, chat-format, and split the synthetic dataset into train/val.

    `tokenizer` is passed in (rather than loaded here) since it's produced by
    the model-loading step (3.4.3) and this function only needs its chat
    template, not the model itself.
    """
    examples = load_examples(data_dir)
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda ex: format_example(ex, tokenizer), remove_columns=dataset.column_names
    )
    split = dataset.train_test_split(test_size=val_fraction, seed=seed)
    return split["train"], split["test"]


def load_base_model(
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load the base Gemma model in 4-bit (QLoRA) and its tokenizer via Unsloth.

    `unsloth` is imported lazily inside this function (not at module top) so the
    rest of `train.py` — the data pipeline in particular — stays importable on a
    machine without CUDA/Unsloth, which is what the local 3.4.7 sanity tests rely
    on. The actual GPU load only happens when this is called on Colab (step 3.6).

    `dtype=None` lets Unsloth auto-detect (fp16 on a T4, which has no bf16).
    """
    import torch
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto: fp16 on T4 (no bf16 support)
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )

    # Gemma 4's per-layer AltUp mechanism (small `per_layer_input_gate` /
    # `per_layer_projection` / `per_layer_model_projection` modules) computes its
    # activations in float32 even though Unsloth loads the rest of the model in
    # float16 on a T4 (no bf16 support). Left as-is, this raises
    # `RuntimeError: expected mat1 and mat2 to have the same dtype` on the very
    # first training step. Upcasting just these small modules (not the ~4.7GB
    # `embed_tokens_per_layer` table, which would OOM a T4) fixes the mismatch
    # at negligible memory cost (confirmed via forward+backward pass on Colab).
    for name, module in model.named_modules():
        if (
            "per_layer" in name
            and "embed_tokens_per_layer" not in name
            and getattr(module, "weight", None) is not None
            and module.weight.dtype == torch.float16
        ):
            module.float()

    return model, tokenizer


def add_lora_adapter(model: Any, seed: int) -> Any:
    """Attach a LoRA adapter to the 4-bit base model (locked decision E).

    Gemma 4 E2B is multimodal, so we use Unsloth's `finetune_*_layers` flags
    rather than a raw `target_modules` list: `finetune_vision_layers=False`
    freezes the vision tower (text-only task), while the attention + MLP flags
    select exactly the q/k/v/o and gate/up/down projections of the language
    tower. `use_gradient_checkpointing="unsloth"` trims VRAM to fit the T4.
    """
    from unsloth import FastModel

    return FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # text-only task -> freeze vision tower
        finetune_language_layers=True,
        finetune_attention_modules=True,  # q/k/v/o proj
        finetune_mlp_modules=True,  # gate/up/down proj
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
        loftq_config=None,
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    val_dataset: Dataset,
    output_dir: str,
    seed: int,
    max_seq_length: int = MAX_SEQ_LENGTH,
    num_train_epochs: int = NUM_TRAIN_EPOCHS,
    max_steps: int | None = None,
) -> Any:
    """Build the trl `SFTTrainer`, tuned for a Colab T4 and masked to responses.

    `max_steps`, when given (e.g. `--smoke`), overrides `num_train_epochs`. The
    trainer is wrapped with `train_on_responses_only` so loss is computed only
    on the gold-JSON completion, not the prompt (locked decision C).

    `unsloth` / `trl` are imported lazily so the module stays importable without
    CUDA (see `load_base_model`).
    """
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    # -1 tells trl to ignore max_steps and train for the full epoch count.
    effective_max_steps = max_steps if max_steps is not None else -1

    config = SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,  # incompatible with response-only masking
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=num_train_epochs,
        max_steps=effective_max_steps,
        learning_rate=LEARNING_RATE,
        # T4 has no bf16; `is_bfloat16_supported()` keeps this correct on the
        # optional L4/A100 too (SCOPE.md) instead of hardcoding fp16.
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim=OPTIMIZER,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        seed=seed,
        output_dir=output_dir,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=config,
    )

    return train_on_responses_only(
        trainer,
        instruction_part=GEMMA_INSTRUCTION_PART,
        response_part=GEMMA_RESPONSE_PART,
    )


def save_adapter(model: Any, tokenizer: Any, output_dir: str) -> None:
    """Save the trained LoRA adapter + tokenizer (not the merged base weights).

    Only the adapter is saved — `evaluate.py` (3.5) loads the base model and
    attaches this adapter, and the HF Hub upload (3.7) publishes just the
    adapter, mirroring how it will be consumed.
    """
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QLoRA fine-tune Gemma 4 E2B on the synthetic Hungarian text-to-JSON "
            "extraction data. Designed for a Colab T4."
        )
    )
    parser.add_argument(
        "--data-dir",
        default="data/synthetic",
        help="Directory of per-domain synthetic JSONL files (default: data/synthetic).",
    )
    parser.add_argument(
        "--out",
        default="outputs",
        help=(
            "Output directory: the final LoRA adapter is saved here, and "
            "training checkpoints go under <out>/checkpoints (default: outputs)."
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Base model to load (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=MAX_SEQ_LENGTH,
        help=f"Max sequence length (default: {MAX_SEQ_LENGTH}).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_TRAIN_EPOCHS,
        help=f"Number of training epochs (default: {NUM_TRAIN_EPOCHS}); ignored if --max-steps/--smoke set.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Cap training at this many steps, overriding --epochs (default: none).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Quick end-to-end shakeout: caps training at {SMOKE_MAX_STEPS} steps (unless --max-steps is given).",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})."
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Resume training. Bare flag resumes from the latest checkpoint in "
            "<out>/checkpoints; pass a path to resume from a specific one."
        ),
    )
    return parser.parse_args(argv)


def resolve_max_steps(max_steps: int | None, smoke: bool) -> int | None:
    """Resolve the effective `--max-steps`: explicit value wins, else `--smoke`
    uses the smoke budget, else `None` (train the full `--epochs`)."""
    if max_steps is not None:
        return max_steps
    return SMOKE_MAX_STEPS if smoke else None


def main() -> None:
    args = parse_args()
    max_steps = resolve_max_steps(args.max_steps, args.smoke)

    logger.info("Loading base model %s (4-bit QLoRA)...", args.model)
    model, tokenizer = load_base_model(args.model, args.max_seq_length)
    model = add_lora_adapter(model, args.seed)

    logger.info("Building dataset from %s...", args.data_dir)
    train_ds, val_ds = build_dataset(Path(args.data_dir), tokenizer, args.seed)
    logger.info("Train: %d examples, val: %d examples", len(train_ds), len(val_ds))

    checkpoint_dir = str(Path(args.out) / "checkpoints")
    trainer = build_trainer(
        model,
        tokenizer,
        train_ds,
        val_ds,
        output_dir=checkpoint_dir,
        seed=args.seed,
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        max_steps=max_steps,
    )

    logger.info("Starting training%s...", " (smoke)" if max_steps == SMOKE_MAX_STEPS else "")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    logger.info("Saving LoRA adapter to %s...", args.out)
    save_adapter(model, tokenizer, args.out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
