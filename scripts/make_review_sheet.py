"""Render the hand-labeled eval set into a human-readable review sheet.

Turns the per-domain `data/eval/*.jsonl` files (one compact JSON object per line)
into a single `data/eval/REVIEW.md`: for each example, the source link, the
Hungarian document (line-wrapped), and the gold JSON as a field/value table --
so a human can proofread the set without squinting at raw JSONL.

This sheet is a *review aid only*: the eval pipeline (`src/evaluate.py`) always
reads the `.jsonl` files, never this sheet. Edit the `.jsonl`, re-run
`python src/validate_eval_set.py`, then regenerate:

    python scripts/make_review_sheet.py
    python scripts/make_review_sheet.py --eval-dir data/eval --out data/eval/REVIEW.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Domain render order + the "headline" field used as each entry's title. Kept in
# sync with src/schemas.py (medical/business/technology).
DOMAIN_ORDER = ("medical", "business", "technology")
NAME_FIELD = {"medical": "drug_name", "business": "company", "technology": "product"}


def _fmt_val(v: Any) -> str:
    """Render one gold value for a Markdown table cell."""
    if v is None:
        return "_null_"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"`{v}`"
    if isinstance(v, list):
        if not v:
            return "_[]_"
        return "<br>".join("• " + str(x) for x in v)
    return str(v)


def _load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_sheet(eval_dir: Path) -> str:
    lines: list[str] = []
    lines.append("# Eval set — review sheet")
    lines.append("")
    lines.append(
        "> Human-readable rendering of `data/eval/*.jsonl` for verification. "
        "**Not** read by the eval itself — the pipeline always reads the `.jsonl` "
        "files. Edit the `.jsonl`, not this sheet, then re-run "
        "`python src/validate_eval_set.py` and regenerate with "
        "`python scripts/make_review_sheet.py`."
    )
    lines.append("")

    rows_by_domain: dict[str, list[dict[str, Any]]] = {}
    for dom in DOMAIN_ORDER:
        path = eval_dir / f"{dom}.jsonl"
        rows_by_domain[dom] = _load(path) if path.exists() else []

    total = sum(len(r) for r in rows_by_domain.values())
    lines.append("| domain | count |")
    lines.append("|---|---|")
    for dom in DOMAIN_ORDER:
        lines.append(f"| {dom} | {len(rows_by_domain[dom])} |")
    lines.append(f"| **total** | **{total}** |")
    lines.append("")

    for dom in DOMAIN_ORDER:
        rows = rows_by_domain[dom]
        lines.append(f"## {dom} ({len(rows)})")
        lines.append("")
        name_field = NAME_FIELD[dom]
        for i, row in enumerate(rows, 1):
            gold = row.get("gold", {})
            name = gold.get(name_field, "?")
            lines.append(f"### {dom} #{i} — {name}")
            lines.append(f"🔗 [source]({row.get('source', '')})")
            lines.append("")
            lines.append("**Document:**")
            lines.append("")
            for doc_line in row.get("document", "").split("\n"):
                lines.append("> " + doc_line if doc_line.strip() else ">")
            lines.append("")
            lines.append("**Gold:**")
            lines.append("")
            lines.append("| field | value |")
            lines.append("|---|---|")
            for key, value in gold.items():
                lines.append(f"| {key} | {_fmt_val(value)} |")
            lines.append("")
            lines.append("---")
            lines.append("")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-dir", default="data/eval", help="Directory of per-domain eval JSONL files.")
    parser.add_argument("--out", default="data/eval/REVIEW.md", help="Output Markdown path.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    sheet = build_sheet(Path(args.eval_dir))
    out_path = Path(args.out)
    out_path.write_text(sheet, encoding="utf-8")
    total = sheet.count("\n### ")
    print(f"wrote {out_path} — {total} entries, {out_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
