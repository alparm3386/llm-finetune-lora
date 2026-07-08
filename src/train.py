"""QLoRA fine-tuning of Gemma 4 E2B with Unsloth.

This module will run the actual fine-tuning job: load the base model in
4-bit (QLoRA) via Unsloth, attach a LoRA adapter, and train it on the
synthetic text-to-JSON dataset produced by `generate_data.py`. Intended to
run on a Google Colab T4 instance.

# TODO (3.4):
#   - Load `google/gemma-4-E2B` (Instruct) in 4-bit via Unsloth's FastLanguageModel.
#   - Configure LoraConfig (r~16) and attach the adapter to the base model.
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


def main() -> None:
    raise NotImplementedError("Step 3.4: QLoRA training not yet implemented")


if __name__ == "__main__":
    main()
