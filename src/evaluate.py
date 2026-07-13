"""Before/after evaluation of the fine-tuned model (dev-plan step 3.5).

Compares the base (pre-fine-tuning) `google/gemma-4-E2B` against the fine-tuned
(base + LoRA adapter) model on the real, hand-labeled Hungarian eval set, and
reports the improvement in structured extraction quality.

Framing (see SCOPE.md): structured decoding guarantees *format*, fine-tuning
improves *content*. To isolate the content-accuracy gain, structured decoding is
applied to BOTH models, so format is controlled for and the measured delta is
purely the extraction-accuracy improvement — hence the 2x2:
{base, fine-tuned} x {prompt-only, +structured}.

Metric logic lives in `eval_metrics.py` (pure, GPU-free). Model loading /
generation deps (`unsloth` / `outlines` / `torch`) are imported lazily inside the
functions that need them, so this module's data loading, eval loop, aggregation,
and CLI stay importable and unit-testable without a GPU (the 3.4 lesson).

Sub-step status within this file:
  3.5.2 eval-data loading      — done
  3.5.3 inference model loading — TODO
  3.5.4 generation (2 paths)    — TODO
  3.5.5 eval loop + aggregation — TODO
  3.5.6 reporting               — TODO
  3.5.7 CLI                      — TODO
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from validate_eval_set import (
    DEFAULT_EVAL_DIR,
    EVAL_DOMAINS,
    EvalSetError,
    load_domain_examples,
    validate_example,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_eval_examples(
    eval_dir: Path, domains: tuple[str, ...] = EVAL_DOMAINS, validate: bool = True
) -> list[dict[str, Any]]:
    """Load the hand-labeled eval set as a flat list of `{domain, document, gold}`.

    Reuses `validate_eval_set.load_domain_examples` so the eval loop reads the set
    exactly as the standalone validator does. Raises `EvalSetError` if a domain
    file is missing or empty (the eval set is a required run-time input — see
    `DEV_PLAN_3.5.md` Dependencies), or, when `validate=True`, if any `gold` fails
    its domain schema (a bad label would silently corrupt the metric).
    """
    examples: list[dict[str, Any]] = []
    for domain in domains:
        domain_examples = load_domain_examples(eval_dir, domain)
        if not domain_examples:
            raise EvalSetError(
                f"Eval set for domain {domain!r} is empty ({eval_dir / f'{domain}.jsonl'}). "
                "Populate it per data/eval/README.md before running the eval."
            )
        for line_no, example in domain_examples:
            if validate:
                problems = validate_example(example, domain)
                if problems:
                    raise EvalSetError(
                        f"{domain}.jsonl:{line_no}: invalid eval example: {'; '.join(problems)}"
                    )
            examples.append(example)
    return examples


def main() -> None:
    raise NotImplementedError(
        "Step 3.5: eval loop not yet wired (sub-steps 3.5.3–3.5.7 in progress)"
    )


if __name__ == "__main__":
    main()
