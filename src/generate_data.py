"""Synthetic training data generation.

This module will generate the synthetic Hungarian text-to-JSON training set
used to fine-tune the model. For each of the three target domains (medical,
business, technology) it will prompt a strong LLM to produce realistic
Hungarian documents paired with their gold JSON extraction, following the
domain schemas defined in SCOPE.md. The resulting examples are written out
as JSONL files under `data/synthetic/`.

# TODO (3.3):
#   - Load/define the per-domain target JSON schemas (medical, business, technology).
#   - Build prompts that ask a strong LLM to generate {Hungarian document, gold JSON} pairs
#     for a given domain and schema.
#   - Generate a target number of examples per domain (aim: a few hundred each).
#   - Validate each generated example (JSON parses, matches schema keys).
#   - Write the resulting examples to `data/synthetic/<domain>.jsonl`.
"""


def main() -> None:
    raise NotImplementedError("Step 3.3: synthetic data generation not yet implemented")


if __name__ == "__main__":
    main()
