"""QLoRA fine-tuning of Gemma 4 E2B with Unsloth.

This module will run the actual fine-tuning job: load the base model in
4-bit (QLoRA) via Unsloth, attach a LoRA adapter, and train it on the
synthetic text-to-JSON dataset produced by `generate_data.py`. Intended to
run on a Google Colab T4 instance.

# TODO (3.4):
#   - Train with an SFTTrainer (trl), tuned for a single Colab T4 GPU.
#   - Save the resulting LoRA adapter (and optionally push it to the Hugging Face Hub).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from prompt_format import to_chat_messages
from schemas import SCHEMAS

DOMAINS = tuple(SCHEMAS)

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
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto: fp16 on T4 (no bf16 support)
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
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


def main() -> None:
    raise NotImplementedError("Step 3.4: QLoRA training not yet implemented")


if __name__ == "__main__":
    main()
