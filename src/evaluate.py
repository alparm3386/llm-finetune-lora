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
  3.5.3 inference model loading — done
  3.5.4 generation (2 paths)    — done
  3.5.5 eval loop + aggregation — done
  3.5.6 reporting               — done
  3.5.7 CLI                      — done
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_metrics import aggregate_field_scores, parse_prediction, score_example, validity_rate
from prompt_format import to_chat_messages
from schemas import SCHEMAS
from train import DEFAULT_MODEL_NAME, MAX_SEQ_LENGTH, upcast_gemma_per_layer_modules
from validate_eval_set import (
    DEFAULT_EVAL_DIR,
    EVAL_DOMAINS,
    EvalSetError,
    load_domain_examples,
    validate_example,
)

# The four cells of the 2x2 (locked naming used throughout the eval loop,
# reporting, and CLI).
MODEL_VARIANTS = ("base", "fine_tuned")
DECODE_MODES = ("prompt_only", "structured")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Reuse train.py's base-model default and seq length so eval loads the same base
# the adapter was trained on. Eval is inference-only, so a shorter completion
# budget than train's max_seq_length is enough for one JSON object.
DEFAULT_BASE_MODEL = DEFAULT_MODEL_NAME
DEFAULT_MAX_NEW_TOKENS = 512


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


# --- 3.5.3: inference model loading -------------------------------------------


def load_inference_model(
    model_name: str = DEFAULT_BASE_MODEL,
    adapter_dir: str | None = None,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load a model for inference via Unsloth `FastModel`, 4-bit, `for_inference`.

    Two callers: the **base** eval (`adapter_dir=None` → load `model_name`
    straight) and the **fine-tuned** eval (`adapter_dir` set). For the latter we
    pass the saved adapter directory as `model_name`: `save_adapter` (train.py)
    wrote the adapter + its `adapter_config.json`, whose `base_model_name_or_path`
    points Unsloth back at the same base to reload in 4-bit and re-attach the
    LoRA — the Unsloth-native reload path. (Fallback if that ever fights the
    quantized stack: load the base via this function with `adapter_dir=None`, then
    `PeftModel.from_pretrained(model, adapter_dir)`.)

    Mirrors `train.load_base_model`: lazy `unsloth` import (keeps this module
    GPU-free to import), `dtype=None` auto-detect, and the shared Gemma-4
    per-layer float32 upcast — inference does a forward pass too, so it hits the
    same fp16 AltUp dtype mismatch on a T4 if left unpatched.
    """
    from unsloth import FastModel

    load_name = adapter_dir if adapter_dir is not None else model_name
    model, tokenizer = FastModel.from_pretrained(
        model_name=load_name,
        max_seq_length=max_seq_length,
        dtype=None,  # auto: fp16 on T4 (no bf16 support)
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    upcast_gemma_per_layer_modules(model)
    FastModel.for_inference(model)  # Unsloth's 2x-faster inference mode
    return model, tokenizer


# --- 3.5.4: generation, two paths ---------------------------------------------


def build_inference_prompt(tokenizer: Any, domain: str, document: str) -> str:
    """Render the single eval prompt string for one document (no gold).

    Reuses `to_chat_messages(domain, document)` (the user turn only) +
    `apply_chat_template(..., add_generation_prompt=True)`, the identical prompt
    used for training targets and fed to BOTH decode modes (locked decision A) —
    so base/fine-tuned × prompt-only/structured all see exactly the same text.
    """
    messages = to_chat_messages(domain, document)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _set_seed(seed: int) -> None:
    """Best-effort determinism for a generate call (greedy is deterministic anyway)."""
    from transformers import set_seed

    set_seed(seed)


def generate_prompt_only(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
) -> str:
    """Prompt-only path: raw greedy `generate`, decode just the completion.

    No format constraint — this is the mode whose JSON-validity rate is the
    secondary metric. `parse_prediction` (eval_metrics) tolerates non-JSON output
    here. Greedy (`do_sample=False`) for a reproducible headline number.
    """
    import torch

    _set_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    completion_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def build_structured_generators(
    model: Any, tokenizer: Any, domains: tuple[str, ...] = EVAL_DOMAINS
) -> dict[str, Any]:
    """Build one schema-constrained `outlines` generator per domain (built once).

    outlines' index/FSM compilation is per-schema, so we build the three
    generators once per model and reuse them across all docs of that domain. The
    domain JSON Schema is fed straight from `schemas.py` (single source of truth),
    including its nullable unions (`["string","null"]`) and
    `additionalProperties:false`.

    `outlines`/`torch` are imported lazily. **Decision B risk (verify on Colab in
    3.6):** if outlines fights the Unsloth 4-bit + LoRA stack, the fallback is to
    merge the adapter to 16-bit first (`model.save_pretrained_merged`) and wrap
    that, or use a transformers logits-processor JSON constraint.
    """
    import outlines
    from outlines.types import JsonSchema

    outlines_model = outlines.from_transformers(model, tokenizer)
    return {
        domain: outlines.Generator(outlines_model, JsonSchema(json.dumps(SCHEMAS[domain])))
        for domain in domains
    }


def generate_structured(
    generator: Any,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
) -> str:
    """Structured path: schema-constrained decode via a prebuilt outlines generator.

    Output is always schema-valid JSON by construction, so this is the mode the
    primary per-field F1 is measured in (format controlled for on both models).
    Same `prompt` string as `generate_prompt_only` (decision A).
    """
    _set_seed(seed)
    return generator(prompt, max_new_tokens=max_new_tokens, do_sample=False)


# --- 3.5.5: eval loop + aggregation --------------------------------------------


def apply_limit(examples: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """Truncate to the first `limit` examples, or return all of them if `limit` is None.

    Used by the CLI's `--limit` for a fast shakeout of the loop/plumbing without
    waiting on a full-set GPU run.
    """
    return examples if limit is None else examples[:limit]


def generate_predictions(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    mode: str,
    structured_generators: dict[str, Any] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Run one decode mode over every example, returning raw + parsed predictions.

    `mode` is `"prompt_only"` (needs just `model`/`tokenizer`) or `"structured"`
    (needs `structured_generators`, one per domain, from
    `build_structured_generators`). Every example gets the identical prompt from
    `build_inference_prompt` (decision A). Parsing (`eval_metrics.parse_prediction`)
    happens here for both modes: prompt-only output may be unparseable (`pred`
    becomes `None`, scored as decision A's empty-prediction case); structured
    output is schema-valid JSON by construction, so parsing only recovers the dict.
    """
    predictions: list[dict[str, Any]] = []
    for example in examples:
        prompt = build_inference_prompt(tokenizer, example["domain"], example["document"])
        if mode == "prompt_only":
            raw_text = generate_prompt_only(model, tokenizer, prompt, max_new_tokens, seed)
        elif mode == "structured":
            generator = structured_generators[example["domain"]]
            raw_text = generate_structured(generator, prompt, max_new_tokens, seed)
        else:
            raise ValueError(f"Unknown decode mode: {mode!r}")
        pred = parse_prediction(raw_text)
        predictions.append(
            {"domain": example["domain"], "raw_text": raw_text, "pred": pred, "valid": pred is not None}
        )
    return predictions


def score_predictions(
    examples: list[dict[str, Any]], predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Score one cell's predictions against gold: per-field F1 (all aggregation
    levels, via `eval_metrics.aggregate_field_scores`) plus JSON validity rate.

    F1 is meaningful in every cell (decision A: "F1 is reported in all four
    cells"); validity rate is only informative for `prompt_only` cells (structured
    output is always valid by construction), but is included everywhere for
    uniformity — the reporting step (3.5.6) picks which numbers to surface.
    """
    records = [
        (ex["domain"], score_example(SCHEMAS[ex["domain"]], ex["gold"], pred["pred"]))
        for ex, pred in zip(examples, predictions)
    ]
    return {
        "f1": aggregate_field_scores(records),
        "validity_rate": validity_rate([p["valid"] for p in predictions]),
    }


def run_eval_cell(
    model: Any,
    tokenizer: Any,
    examples: list[dict[str, Any]],
    mode: str,
    structured_generators: dict[str, Any] | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate + score one cell of the 2x2 (one model variant x one decode mode)."""
    predictions = generate_predictions(
        model, tokenizer, examples, mode, structured_generators, max_new_tokens, seed
    )
    return score_predictions(examples, predictions)


def run_evaluation(
    base_model: Any,
    base_tokenizer: Any,
    fine_tuned_model: Any,
    fine_tuned_tokenizer: Any,
    examples: list[dict[str, Any]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
    structured_only: bool = False,
) -> dict[str, dict[str, Any]]:
    """Assemble the full 2x2: {base, fine_tuned} x {prompt_only, structured}.

    Keyed `"{variant}.{mode}"` (e.g. `"fine_tuned.structured"`), each value the
    `score_predictions` result for that cell. `structured_only=True` skips the
    `prompt_only` cells (CLI `--structured-only`, for faster iteration — the
    +structured cells carry the headline F1 metric).

    Structured generators are built once per model variant (schema-index
    compilation is the expensive part — see `build_structured_generators`), then
    reused across all examples in that variant's structured cell.
    """
    results: dict[str, dict[str, Any]] = {}
    variants = [
        ("base", base_model, base_tokenizer),
        ("fine_tuned", fine_tuned_model, fine_tuned_tokenizer),
    ]
    for variant_name, model, tokenizer in variants:
        structured_generators = build_structured_generators(model, tokenizer)
        results[f"{variant_name}.structured"] = run_eval_cell(
            model, tokenizer, examples, "structured", structured_generators, max_new_tokens, seed
        )
        if not structured_only:
            results[f"{variant_name}.prompt_only"] = run_eval_cell(
                model, tokenizer, examples, "prompt_only", None, max_new_tokens, seed
            )
    return results


# --- 3.5.6: reporting -----------------------------------------------------------


def _fmt_f1(value: float) -> str:
    return f"{value:.3f}"


def _fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_results_payload(
    results: dict[str, dict[str, Any]], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Wrap the raw 2x2 `run_evaluation` output with run metadata, for `eval_results.json`.

    `results` already carries the full per-cell / per-domain / per-field
    breakdown (nested inside each cell's `f1`, via
    `eval_metrics.aggregate_field_scores`) — decision D's "reproducibility" ask —
    so this just adds the run parameters (models, seed, sizes, timestamp) needed
    to interpret a saved run later.
    """
    return {"metadata": metadata, "results": results}


def render_2x2_table(results: dict[str, dict[str, Any]]) -> str:
    """Render the headline 2x2 (decode mode x model variant) as a markdown table.

    Overall F1 appears in every cell (decision D: "F1 is reported in all four
    cells"); validity rate is added only to the prompt-only row (decision D's
    secondary metric — structured output is always valid by construction).
    """
    lines = ["| Decode mode | Base | Fine-tuned |", "|---|---|---|"]
    for mode_label, mode_key in [("Prompt-only", "prompt_only"), ("+Structured decoding", "structured")]:
        cells = []
        for variant in MODEL_VARIANTS:
            cell = results.get(f"{variant}.{mode_key}")
            if cell is None:
                cells.append("—")
                continue
            f1 = _fmt_f1(cell["f1"]["overall"]["f1"])
            if mode_key == "prompt_only":
                cells.append(f"F1 {f1}, valid {_fmt_percent(cell['validity_rate'])}")
            else:
                cells.append(f"F1 {f1}")
        lines.append(f"| {mode_label} | {cells[0]} | {cells[1]} |")
    return "\n".join(lines)


def render_per_domain_table(
    results: dict[str, dict[str, Any]], mode_key: str = "structured"
) -> str:
    """Render per-domain F1 (base vs. fine-tuned) for one decode mode, default +structured."""
    domains = sorted(
        {
            domain
            for variant in MODEL_VARIANTS
            for domain in results.get(f"{variant}.{mode_key}", {}).get("f1", {}).get("per_domain", {})
        }
    )
    lines = ["| Domain | Base F1 | Fine-tuned F1 |", "|---|---|---|"]
    for domain in domains:
        row = [domain]
        for variant in MODEL_VARIANTS:
            cell = results.get(f"{variant}.{mode_key}")
            per_domain = cell["f1"]["per_domain"] if cell else {}
            row.append(_fmt_f1(per_domain[domain]["f1"]) if domain in per_domain else "—")
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    return "\n".join(lines)


def build_eval_table_markdown(results: dict[str, dict[str, Any]]) -> str:
    """Build the full `eval_table.md` content: headline sentence + 2x2 + per-domain breakdown.

    The headline compares fine-tuned/+structured vs. base/+structured F1 (decision
    D: "the *headline* number is fine-tuned/+structured vs. base/+structured"),
    since structured decoding controls for format on both sides.
    """
    lines = ["# Evaluation results", ""]
    base_structured = results.get("base.structured")
    ft_structured = results.get("fine_tuned.structured")
    if base_structured and ft_structured:
        base_f1 = _fmt_f1(base_structured["f1"]["overall"]["f1"])
        ft_f1 = _fmt_f1(ft_structured["f1"]["overall"]["f1"])
        lines.append(f"**Headline (+structured decoding): base F1 {base_f1} → fine-tuned F1 {ft_f1}.**")
        lines.append("")
    lines.append(render_2x2_table(results))
    if "base.structured" in results or "fine_tuned.structured" in results:
        lines.extend(["", "## Per-domain F1 (+structured decoding)", "", render_per_domain_table(results)])
    lines.append("")
    return "\n".join(lines)


def write_eval_results(
    results: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
    results_json_path: Path,
    results_table_path: Path,
) -> None:
    """Write both reporting outputs (decision D), creating parent directories as needed."""
    results_json_path.parent.mkdir(parents=True, exist_ok=True)
    results_json_path.write_text(
        json.dumps(build_results_payload(results, metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    results_table_path.parent.mkdir(parents=True, exist_ok=True)
    results_table_path.write_text(build_eval_table_markdown(results), encoding="utf-8")


def log_summary(results: dict[str, dict[str, Any]]) -> None:
    """Log one readable line per cell: overall F1 and JSON validity rate."""
    for key in (f"{variant}.{mode}" for variant in MODEL_VARIANTS for mode in DECODE_MODES):
        cell = results.get(key)
        if cell is None:
            continue
        logger.info(
            "%-24s F1=%s  validity=%s",
            key,
            _fmt_f1(cell["f1"]["overall"]["f1"]),
            _fmt_percent(cell["validity_rate"]),
        )


# --- 3.5.7: CLI -------------------------------------------------------------------

DEFAULT_RESULTS_DIR = "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Before/after evaluation: base vs. fine-tuned (base + LoRA adapter) on "
            "the real, hand-labeled Hungarian eval set, at {prompt-only, +structured "
            "decoding}."
        )
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="Path to the trained LoRA adapter directory (train.py's --out).",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help=f"Base model to load (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--eval-dir",
        default=DEFAULT_EVAL_DIR,
        help=f"Directory of per-domain hand-labeled eval JSONL files (default: {DEFAULT_EVAL_DIR}).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_RESULTS_DIR,
        help=f"Output directory for eval_results.json / eval_table.md (default: {DEFAULT_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Max new tokens per generation (default: {DEFAULT_MAX_NEW_TOKENS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N eval examples (default: all — quick shakeout with a small N).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--structured-only",
        action="store_true",
        help="Skip the prompt-only cells and run only +structured decoding (faster iteration).",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    examples = apply_limit(load_eval_examples(Path(args.eval_dir)), args.limit)
    logger.info("Loaded %d eval examples from %s", len(examples), args.eval_dir)

    logger.info("Loading base model %s...", args.base_model)
    base_model, base_tokenizer = load_inference_model(args.base_model)

    logger.info("Loading fine-tuned model (adapter %s)...", args.adapter)
    fine_tuned_model, fine_tuned_tokenizer = load_inference_model(
        args.base_model, adapter_dir=args.adapter
    )

    results = run_evaluation(
        base_model,
        base_tokenizer,
        fine_tuned_model,
        fine_tuned_tokenizer,
        examples,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        structured_only=args.structured_only,
    )

    log_summary(results)

    metadata = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "eval_dir": args.eval_dir,
        "num_examples": len(examples),
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "structured_only": args.structured_only,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = Path(args.out)
    results_json_path = out_dir / "eval_results.json"
    results_table_path = out_dir / "eval_table.md"
    write_eval_results(results, metadata, results_json_path, results_table_path)
    logger.info("Wrote %s and %s", results_json_path, results_table_path)


if __name__ == "__main__":
    main()
