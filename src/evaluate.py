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
  3.5.5 eval loop + aggregation — TODO
  3.5.6 reporting               — TODO
  3.5.7 CLI                      — TODO
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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


def main() -> None:
    raise NotImplementedError(
        "Step 3.5: eval loop not yet wired (sub-steps 3.5.5–3.5.7 in progress)"
    )


if __name__ == "__main__":
    main()
