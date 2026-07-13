"""Unit tests for the reporting layer (dev-plan step 3.5.6)."""

import json

import evaluate as ev


def make_cell(f1: float, validity: float, per_domain: dict[str, float]) -> dict:
    return {
        "f1": {
            "overall": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": f1},
            "per_domain": {
                domain: {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": v}
                for domain, v in per_domain.items()
            },
            "per_field": {},
        },
        "validity_rate": validity,
    }


def make_full_results() -> dict:
    return {
        "base.prompt_only": make_cell(0.40, 0.60, {"medical": 0.3, "business": 0.5}),
        "base.structured": make_cell(0.45, 1.0, {"medical": 0.4, "business": 0.5}),
        "fine_tuned.prompt_only": make_cell(0.70, 0.90, {"medical": 0.65, "business": 0.75}),
        "fine_tuned.structured": make_cell(0.80, 1.0, {"medical": 0.78, "business": 0.82}),
    }


# --- build_results_payload -------------------------------------------------------


def test_build_results_payload_wraps_metadata_and_results():
    results = make_full_results()
    metadata = {"seed": 0, "base_model": "unsloth/gemma-4-E2B-it"}

    payload = ev.build_results_payload(results, metadata)

    assert payload["metadata"] == metadata
    assert payload["results"] == results


# --- render_2x2_table --------------------------------------------------------------


def test_render_2x2_table_includes_all_four_cells_and_validity_only_prompt_only():
    table = ev.render_2x2_table(make_full_results())

    assert "F1 0.400" in table and "valid 60.0%" in table  # base prompt-only
    assert "F1 0.450" in table  # base structured
    assert "F1 0.450, valid" not in table  # structured row has no "valid" text
    assert "F1 0.800" in table  # fine-tuned structured


def test_render_2x2_table_handles_missing_cell():
    results = {"base.structured": make_cell(0.45, 1.0, {})}
    table = ev.render_2x2_table(results)
    assert "—" in table  # missing cells render as em-dash, not a KeyError


# --- render_per_domain_table --------------------------------------------------------


def test_render_per_domain_table_lists_domains_sorted_with_both_variants():
    table = ev.render_per_domain_table(make_full_results())
    lines = table.splitlines()

    assert any(line.startswith("| business") for line in lines)
    assert any(line.startswith("| medical") for line in lines)
    # business appears after medical (sorted)
    domain_lines = [l for l in lines if l.startswith("| business") or l.startswith("| medical")]
    assert domain_lines[0].startswith("| business")


# --- build_eval_table_markdown -------------------------------------------------------


def test_build_eval_table_markdown_includes_headline_and_tables():
    markdown = ev.build_eval_table_markdown(make_full_results())

    assert "Headline" in markdown
    assert "base F1 0.450" in markdown
    assert "fine-tuned F1 0.800" in markdown
    assert "Per-domain F1" in markdown


def test_build_eval_table_markdown_no_headline_when_structured_cells_missing():
    results = {"base.prompt_only": make_cell(0.4, 0.6, {})}
    markdown = ev.build_eval_table_markdown(results)
    assert "Headline" not in markdown


# --- write_eval_results ---------------------------------------------------------------


def test_write_eval_results_creates_both_files(tmp_path):
    results = make_full_results()
    metadata = {"seed": 0}
    json_path = tmp_path / "out" / "eval_results.json"
    table_path = tmp_path / "out" / "eval_table.md"

    ev.write_eval_results(results, metadata, json_path, table_path)

    assert json_path.exists()
    assert table_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metadata"] == metadata
    assert payload["results"]["base.structured"]["f1"]["overall"]["f1"] == 0.45

    assert "Headline" in table_path.read_text(encoding="utf-8")


# --- log_summary ---------------------------------------------------------------------


def test_log_summary_does_not_raise_on_partial_results(caplog):
    results = {"base.structured": make_cell(0.45, 1.0, {})}
    ev.log_summary(results)  # only asserts no exception on missing cells
