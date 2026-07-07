"""QLoRA fine-tuning of Gemma 4 E2B with Unsloth.

This module will run the actual fine-tuning job: load the base model in
4-bit (QLoRA) via Unsloth, attach a LoRA adapter, and train it on the
synthetic text-to-JSON dataset produced by `generate_data.py`. Intended to
run on a Google Colab T4 instance.

# TODO (3.4):
#   - Load `google/gemma-4-E2B` (Instruct) in 4-bit via Unsloth's FastLanguageModel.
#   - Configure LoraConfig (r~16) and attach the adapter to the base model.
#   - Load the synthetic JSONL dataset from `data/synthetic/` and format it into
#     the model's chat/instruction template.
#   - Train with an SFTTrainer (trl), tuned for a single Colab T4 GPU.
#   - Save the resulting LoRA adapter (and optionally push it to the Hugging Face Hub).
"""


def main() -> None:
    raise NotImplementedError("Step 3.4: QLoRA training not yet implemented")


if __name__ == "__main__":
    main()
