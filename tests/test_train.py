"""Unit tests for the non-GPU parts of `train.py` (dev-plan step 3.4.7):
data loading/formatting, the train/val split, and CLI arg resolution. The
GPU-dependent parts (`load_base_model`, `add_lora_adapter`, `build_trainer`)
are exercised on Colab in step 3.6, not here.
"""

import json

import pytest

from train import (
    DOMAINS,
    SMOKE_MAX_STEPS,
    build_dataset,
    format_example,
    load_examples,
    parse_args,
    resolve_max_steps,
)


class FakeTokenizer:
    """Minimal stand-in for a Gemma tokenizer's chat template."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        turns = [f"<start_of_turn>{m['role']}\n{m['content']}<end_of_turn>" for m in messages]
        return "\n".join(turns)


GOLD_BY_DOMAIN = {
    "medical": {
        "drug_name": "X",
        "active_ingredient": "Y",
        "indication": "Z",
        "dosage": None,
        "side_effects": [],
        "contraindications": [],
    },
    "business": {
        "company": "X Kft.",
        "event_type": "felvásárlás",
        "amount": 100,
        "currency": "EUR",
        "date": "2025-01-01",
        "involved_parties": ["X Kft."],
    },
    "technology": {
        "product": "X",
        "manufacturer": "Y",
        "version": None,
        "key_specs": [],
        "release_date": None,
        "price": None,
    },
}


def write_synthetic_data(data_dir, n_per_domain=10):
    """Write `n_per_domain` synthetic examples per domain into `data_dir`."""
    for domain in DOMAINS:
        path = data_dir / f"{domain}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i in range(n_per_domain):
                example = {
                    "domain": domain,
                    "document": f"{domain} dokumentum #{i}",
                    "gold": GOLD_BY_DOMAIN[domain],
                }
                f.write(json.dumps(example, ensure_ascii=False) + "\n")


def test_load_examples_reads_all_domains(tmp_path):
    write_synthetic_data(tmp_path, n_per_domain=5)
    examples = load_examples(tmp_path)

    assert len(examples) == 5 * len(DOMAINS)
    assert {ex["domain"] for ex in examples} == set(DOMAINS)


def test_load_examples_skips_blank_lines(tmp_path):
    write_synthetic_data(tmp_path, n_per_domain=1)
    # Inject a trailing blank line into one file, as some editors/tools do.
    path = tmp_path / "medical.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")

    examples = load_examples(tmp_path)
    assert len(examples) == len(DOMAINS)  # blank line did not become an example


def test_format_example_renders_gemma_chat_markers():
    example = {"domain": "technology", "document": "doc", "gold": GOLD_BY_DOMAIN["technology"]}
    result = format_example(example, FakeTokenizer())

    assert set(result.keys()) == {"text"}
    assert "<start_of_turn>user" in result["text"]
    assert "<start_of_turn>assistant" in result["text"]


def test_build_dataset_splits_and_formats(tmp_path):
    write_synthetic_data(tmp_path, n_per_domain=20)  # 60 total
    train_ds, val_ds = build_dataset(tmp_path, FakeTokenizer(), seed=0, val_fraction=0.1)

    assert len(train_ds) + len(val_ds) == 60
    assert len(val_ds) == 6
    assert train_ds.column_names == ["text"]
    assert val_ds.column_names == ["text"]


def test_build_dataset_is_deterministic_for_a_given_seed(tmp_path):
    write_synthetic_data(tmp_path, n_per_domain=20)
    train_a, val_a = build_dataset(tmp_path, FakeTokenizer(), seed=0, val_fraction=0.1)
    train_b, val_b = build_dataset(tmp_path, FakeTokenizer(), seed=0, val_fraction=0.1)

    assert train_a["text"] == train_b["text"]
    assert val_a["text"] == val_b["text"]


@pytest.mark.parametrize(
    ("max_steps", "smoke", "expected"),
    [
        (None, False, None),
        (None, True, SMOKE_MAX_STEPS),
        (10, False, 10),
        (10, True, 10),  # explicit --max-steps wins over --smoke
    ],
)
def test_resolve_max_steps(max_steps, smoke, expected):
    assert resolve_max_steps(max_steps, smoke) == expected


def test_parse_args_defaults():
    args = parse_args([])
    assert args.data_dir == "data/synthetic"
    assert args.out == "outputs"
    assert args.max_steps is None
    assert args.smoke is False
    assert args.resume_from_checkpoint is None


def test_parse_args_resume_bare_flag_means_latest_checkpoint():
    args = parse_args(["--resume-from-checkpoint"])
    assert args.resume_from_checkpoint is True


def test_parse_args_resume_with_explicit_path():
    args = parse_args(["--resume-from-checkpoint", "outputs/checkpoints/checkpoint-40"])
    assert args.resume_from_checkpoint == "outputs/checkpoints/checkpoint-40"
