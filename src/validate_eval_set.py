"""Validate the hand-labeled evaluation set against the domain schemas (step 3.5.0).

Standalone, dependency-light (only `json` / `jsonschema` / `schemas`) so it can be
run during manual labeling — before any of `evaluate.py`'s heavy model deps
exist — to catch malformed `gold` JSON early:

    python src/validate_eval_set.py                  # all domains under data/eval/
    python src/validate_eval_set.py --domain medical # a single domain file

`evaluate.py` (step 3.5.2) reuses `load_domain_examples` / `validate_example` from
here so the eval loop reads and sanity-checks the set the same way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

from schemas import SCHEMAS

DEFAULT_EVAL_DIR = "data/eval"
EVAL_DOMAINS = tuple(SCHEMAS)


class EvalSetError(Exception):
    """Raised when the eval set is missing or malformed (surfaced clearly to the user)."""


def load_domain_examples(eval_dir: Path, domain: str) -> list[tuple[int, dict[str, Any]]]:
    """Read one domain's JSONL as `(line_number, example)` pairs.

    Line numbers are 1-based and skip blank lines, so validation errors can point
    at the offending source line. Raises `EvalSetError` if the file is absent or a
    line is not valid JSON.
    """
    path = eval_dir / f"{domain}.jsonl"
    if not path.exists():
        raise EvalSetError(f"Eval file not found: {path}")

    examples: list[tuple[int, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                examples.append((line_no, json.loads(line)))
            except json.JSONDecodeError as exc:
                raise EvalSetError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
    return examples


def validate_example(example: dict[str, Any], domain: str) -> list[str]:
    """Return a list of human-readable problems with one example (empty == valid).

    Checks the `{domain, document, gold}` envelope and validates `gold` against
    `SCHEMAS[domain]` (which enforces required keys, types, and
    `additionalProperties:false`).
    """
    problems: list[str] = []

    example_domain = example.get("domain")
    if example_domain != domain:
        problems.append(f"'domain' is {example_domain!r}, expected {domain!r}")

    document = example.get("document")
    if not isinstance(document, str) or not document.strip():
        problems.append("'document' is missing or empty")

    gold = example.get("gold")
    if not isinstance(gold, dict):
        problems.append("'gold' is missing or not a JSON object")
    else:
        try:
            jsonschema_validate(instance=gold, schema=SCHEMAS[domain])
        except ValidationError as exc:
            location = "/".join(str(p) for p in exc.absolute_path) or "<root>"
            problems.append(f"gold does not match schema at {location}: {exc.message}")

    return problems


def validate_eval_set(
    eval_dir: Path, domains: tuple[str, ...] = EVAL_DOMAINS
) -> dict[str, list[str]]:
    """Validate every example in each domain file; return {domain: [error lines]}.

    An empty list for a domain means all its examples are valid. A missing file is
    reported as a single error line for that domain rather than raising, so one
    absent domain doesn't hide problems in the others.
    """
    report: dict[str, list[str]] = {}
    for domain in domains:
        errors: list[str] = []
        try:
            examples = load_domain_examples(eval_dir, domain)
        except EvalSetError as exc:
            report[domain] = [str(exc)]
            continue
        if not examples:
            errors.append(f"{eval_dir / f'{domain}.jsonl'}: no examples (file is empty)")
        for line_no, example in examples:
            for problem in validate_example(example, domain):
                errors.append(f"{domain}.jsonl:{line_no}: {problem}")
        report[domain] = errors
    return report


def _iter_report_lines(report: dict[str, list[str]]) -> Iterator[str]:
    for domain, errors in report.items():
        if errors:
            yield f"[{domain}] {len(errors)} problem(s):"
            yield from (f"  {e}" for e in errors)
        else:
            yield f"[{domain}] OK"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the hand-labeled eval set against the domain schemas."
    )
    parser.add_argument(
        "--eval-dir",
        default=DEFAULT_EVAL_DIR,
        help=f"Directory of per-domain eval JSONL files (default: {DEFAULT_EVAL_DIR}).",
    )
    parser.add_argument(
        "--domain",
        choices=EVAL_DOMAINS,
        help="Validate only this domain (default: all).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    domains = (args.domain,) if args.domain else EVAL_DOMAINS
    report = validate_eval_set(Path(args.eval_dir), domains)

    for line in _iter_report_lines(report):
        print(line)

    total_problems = sum(len(errors) for errors in report.values())
    if total_problems:
        print(f"\n{total_problems} problem(s) found.")
        sys.exit(1)
    print("\nAll eval examples valid.")


if __name__ == "__main__":
    main()
