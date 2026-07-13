"""Unit tests for evaluate.py's CLI arg parsing (dev-plan step 3.5.7)."""

import pytest

import evaluate as ev


def test_parse_args_requires_adapter():
    with pytest.raises(SystemExit):
        ev.parse_args([])


def test_parse_args_defaults():
    args = ev.parse_args(["--adapter", "outputs/adapter"])

    assert args.adapter == "outputs/adapter"
    assert args.base_model == ev.DEFAULT_BASE_MODEL
    assert args.eval_dir == ev.DEFAULT_EVAL_DIR
    assert args.out == ev.DEFAULT_RESULTS_DIR
    assert args.max_new_tokens == ev.DEFAULT_MAX_NEW_TOKENS
    assert args.limit is None
    assert args.seed == 0
    assert args.structured_only is False


def test_parse_args_overrides():
    args = ev.parse_args(
        [
            "--adapter",
            "outputs/adapter",
            "--base-model",
            "unsloth/gemma-4-E2B-it",
            "--eval-dir",
            "data/eval",
            "--out",
            "custom-results",
            "--max-new-tokens",
            "256",
            "--limit",
            "5",
            "--seed",
            "42",
            "--structured-only",
        ]
    )

    assert args.out == "custom-results"
    assert args.max_new_tokens == 256
    assert args.limit == 5
    assert args.seed == 42
    assert args.structured_only is True
