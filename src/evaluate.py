"""Before/after evaluation of the fine-tuned model.

This module will compare the base (pre-fine-tuning) model against the
LoRA-adapted model on the real, hand-labeled Hungarian evaluation set, and
report the improvement in structured extraction quality.

Framing (see SCOPE.md): structured decoding guarantees *format*, fine-tuning
improves *content*. To isolate the content-accuracy gain, structured decoding
is applied to BOTH the base and the fine-tuned model, so format is controlled
for and the measured delta is purely the extraction-accuracy improvement.

# TODO (3.5):
#   - Load the base model and the fine-tuned (base + LoRA adapter) model.
#   - Run both models on the real eval documents in `data/eval/`, per domain.
#   - Primary metric: per-field exact-match F1 vs. the gold JSON, WITH structured
#     decoding (outlines) enabled for both models (format controlled for).
#   - Secondary metric: JSON validity rate WITHOUT structured decoding (prompt-only),
#     to show fine-tuning also improves raw format adherence.
#   - Print the 2x2 results table (prompt-only vs. +structured decoding) x (base vs.
#     fine-tuned), per domain and overall.
"""


def main() -> None:
    raise NotImplementedError("Step 3.5: before/after evaluation not yet implemented")


if __name__ == "__main__":
    main()
