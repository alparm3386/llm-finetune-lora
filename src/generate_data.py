"""Synthetic training data generation (dev-plan step 3.3).

Generates the synthetic Hungarian text-to-JSON training set used to
fine-tune the model. For each of the three target domains (medical,
business, technology) this script prompts Google's Gemini API to produce a
realistic Hungarian document paired with its gold JSON extraction,
following the domain schemas defined in `schemas.py` (which mirror
SCOPE.md's table). Each generated example is validated against the
domain's JSON Schema and written out as JSONL under `data/synthetic/`.

Requires a `GEMINI_API_KEY` in a local `.env` file (see `.env.example`).
This script only performs API calls when actually run; it is not executed
as part of writing/reviewing this code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from jsonschema import ValidationError, validate

from schemas import SCHEMAS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ("medical", "business", "technology")

# Short style/sub-type hints rotated into the prompt per example, so a fixed
# prompt template doesn't yield repetitive documents. One list per domain.
STYLE_HINTS: dict[str, list[str]] = {
    "medical": [
        "hivatalos betegtájékoztató, orvosi szaknyelvvel",
        "rövid, közérthető betegtájékoztató-kivonat",
        "gyógyszertári tájékoztató szórólap, barátságos hangnemben",
        "receptre felírható gyógyszer alkalmazási előírása, tömör stílusban",
        "vényköteles gyógyszer betegtájékoztatója, figyelmeztetésekkel",
    ],
    "business": [
        "rövid hír egy tőzsdei felvásárlásról",
        "sajtóközlemény céges összeolvadásról",
        "gazdasági napilap cikke egy befektetési körről",
        "vállalati közlemény negyedéves eredményekről",
        "hír egy startup finanszírozási körének lezárásáról",
    ],
    "technology": [
        "termékbejelentés egy új okostelefonról",
        "műszaki blogbejegyzés egy laptop specifikációiról",
        "sajtóközlemény egy szoftvertermék új verziójáról",
        "rövid híradás egy elektronikai eszköz piaci bevezetéséről",
        "termékoldal-szerű leírás egy háztartási gépről",
    ],
}

GENERATION_TEMPERATURE_RANGE = (0.7, 1.0)

# Backoff (seconds) between retries. Validation/parse retries can be quick;
# transient API errors (503/429) need more breathing room, capped at 60s.
VALIDATION_BACKOFF_CAP = 30.0
API_BACKOFF_CAP = 60.0


def _describe_field(name: str, spec: dict[str, Any]) -> str:
    """Render one schema field as a human-readable Hungarian type description.

    Used to embed the expected `gold` structure directly in the prompt text,
    since the google-genai SDK's native `response_schema` cannot represent the
    JSON-Schema union types (e.g. ["string", "null"]) used in `schemas.py`.
    """
    field_type = spec.get("type")
    if field_type == "array":
        item_type = spec.get("items", {}).get("type", "string")
        return f"- {name}: {item_type} elemek tömbje (ha nincs adat: üres tömb [])"
    if isinstance(field_type, list):
        # Union type, e.g. ["string", "null"] -> nullable scalar.
        base = next((t for t in field_type if t != "null"), "string")
        nullable = "null" in field_type
        suffix = " (nullable: ha nincs adat, legyen null)" if nullable else ""
        return f"- {name}: {base}{suffix}"
    return f"- {name}: {field_type}"


def build_field_spec(domain: str) -> str:
    """Render the domain's gold fields as a Hungarian bullet list for the prompt."""
    props = SCHEMAS[domain]["properties"]
    return "\n".join(_describe_field(name, spec) for name, spec in props.items())


def build_example_envelope(domain: str) -> str:
    """Render one concrete {document, gold} example (as JSON text) for the prompt."""
    example_golds: dict[str, dict[str, Any]] = {
        "medical": {
            "drug_name": "Algopyrin",
            "active_ingredient": "metamizol-nátrium",
            "indication": "erős fájdalom és magas láz csillapítása",
            "dosage": "napi 1-2 tabletta, étkezés után",
            "side_effects": ["allergiás bőrreakció", "vérnyomásesés"],
            "contraindications": [],
        },
        "business": {
            "company": "MOL Nyrt.",
            "event_type": "felvásárlás",
            "amount": 620000000,
            "currency": "EUR",
            "date": "2025-03-14",
            "involved_parties": ["MOL Nyrt.", "Aurora Energy Kft."],
        },
        "technology": {
            "product": "NovaBook Pro 14",
            "manufacturer": "Lumitech",
            "version": None,
            "key_specs": ["14 hüvelykes OLED kijelző", "32 GB RAM", "1 TB SSD"],
            "release_date": "2025-09-01",
            "price": 749900,
        },
    }
    example = {
        "document": "<ide kerül a generált, természetes magyar nyelvű szöveg>",
        "gold": example_golds[domain],
    }
    return json.dumps(example, ensure_ascii=False, indent=2)


def build_prompt(domain: str, style_hint: str) -> str:
    """Build the Hungarian-document generation prompt for one example."""
    return (
        f"Generálj egy szintetikus, de realisztikus MAGYAR nyelvű szöveget a(z) "
        f"'{domain}' témakörben, a következő stílusban: {style_hint}.\n\n"
        "A szöveg legyen természetes, folyó szöveg (NE a JSON mezők egyszerű "
        "felsorolása), változó hosszúságú és stílusú. Illessz bele 1-2 olyan "
        "hihető, de a sémához nem tartozó részletet is (pl. egy mellékes "
        "szereplő neve, egy dátum, egy szám), amelyek elterelik a figyelmet, "
        "de nem részei a kinyerendő adatoknak.\n\n"
        "Add vissza PONTOSAN egy JSON objektumot, két kulccsal:\n"
        "- 'document': a generált magyar nyelvű szöveg (string)\n"
        "- 'gold': a szövegből kinyerhető adatok az alábbi mezőkkel.\n\n"
        "A 'gold' objektum mezői és típusai:\n"
        f"{build_field_spec(domain)}\n\n"
        "Fontos szabályok:\n"
        "- Néhány opcionális (nullable) mezőt SZÁNDÉKOSAN hagyj ki a "
        "szövegből — ilyenkor a 'gold'-ban az értékük legyen null, tömb "
        "típusnál pedig üres tömb [].\n"
        "- Ne találj ki adatot: a 'gold' kizárólag a szövegben ténylegesen "
        "szereplő információt tartalmazza.\n"
        "- Minden felsorolt mező szerepeljen a 'gold' objektumban.\n\n"
        "Példa a válasz formátumára (a tartalmat te generáld, ez csak a "
        "szerkezetet mutatja):\n"
        f"{build_example_envelope(domain)}"
    )


def generate_one_example(
    client: genai.Client,
    domain: str,
    model: str,
    rng: random.Random,
    max_retries: int,
) -> dict[str, Any] | None:
    """Generate and validate a single {document, gold} example for a domain.

    Retries up to `max_retries` times (with exponential backoff) on:
      - parse/validation failures (bad JSON or schema-invalid gold), and
      - transient API errors: any 5xx ``ServerError``, and 429 rate-limit
        ``ClientError``.
    Non-transient client errors (other 4xx, e.g. 400/401/403 — bad request,
    bad key, forbidden) are NOT transient and abort immediately with a clear
    message. Returns None if all retries are exhausted.
    """
    domain_schema = SCHEMAS[domain]
    style_hint = rng.choice(STYLE_HINTS[domain])
    prompt = build_prompt(domain, style_hint)

    for attempt in range(1, max_retries + 1):
        temperature = rng.uniform(*GENERATION_TEMPERATURE_RANGE)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            payload = json.loads(response.text)
            document = payload["document"]
            gold = payload["gold"]
            if not isinstance(document, str) or not document.strip():
                raise ValueError("empty or non-string 'document' field")
            validate(instance=gold, schema=domain_schema)
            return {"domain": domain, "document": document, "gold": gold}
        except genai_errors.ClientError as exc:
            # 429 = rate limit -> transient, retry. Any other 4xx is our bug
            # (bad request / bad key / forbidden) -> fail fast.
            if exc.code != 429:
                raise SystemExit(
                    f"Non-retryable API error (HTTP {exc.code}): {exc}. "
                    "This is likely a bad request or an invalid/unauthorized "
                    "GEMINI_API_KEY, not a transient outage."
                ) from exc
            logger.warning(
                "[%s] attempt %d/%d rate-limited (429): %s",
                domain, attempt, max_retries, exc,
            )
            if attempt < max_retries:
                time.sleep(min(5 * 2 ** (attempt - 1), API_BACKOFF_CAP))
        except genai_errors.ServerError as exc:
            # 5xx (e.g. 503 UNAVAILABLE, 500) -> transient, always retry.
            logger.warning(
                "[%s] attempt %d/%d server error (%s): %s",
                domain, attempt, max_retries, exc.code, exc,
            )
            if attempt < max_retries:
                time.sleep(min(5 * 2 ** (attempt - 1), API_BACKOFF_CAP))
        except (
            json.JSONDecodeError,
            KeyError,
            ValueError,
            ValidationError,
        ) as exc:
            logger.warning(
                "[%s] attempt %d/%d failed: %s", domain, attempt, max_retries, exc
            )
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), VALIDATION_BACKOFF_CAP))
    return None


def generate_domain(
    client: genai.Client,
    domain: str,
    n: int,
    model: str,
    rng: random.Random,
    max_retries: int,
    out_dir: Path,
    delay: float,
) -> int:
    """Generate `n` examples for one domain, writing valid ones to JSONL.

    Returns the number of examples skipped (failed after all retries).
    """
    out_path = out_dir / f"{domain}.jsonl"
    skipped = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            example = generate_one_example(client, domain, model, rng, max_retries)
            if example is None:
                skipped += 1
                logger.warning("[%s] skipping example %d/%d (all retries failed)", domain, i, n)
                continue
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info("[%s] %d/%d", domain, i, n)
            # Polite delay between successful requests to avoid tripping
            # free-tier rate limits.
            if delay > 0:
                time.sleep(delay)
    return skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Hungarian text-to-JSON training data via Gemini."
    )
    parser.add_argument(
        "--domain",
        choices=(*DOMAINS, "all"),
        default="all",
        help="Domain to generate data for (default: all).",
    )
    parser.add_argument(
        "--n", type=int, default=50, help="Number of examples per domain (default: 50)."
    )
    parser.add_argument(
        "--model", default="gemini-2.5-flash", help="Gemini model name (default: gemini-2.5-flash)."
    )
    parser.add_argument(
        "--out",
        default="data/synthetic",
        help="Output directory for per-domain JSONL files (default: data/synthetic).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max generation attempts per example before skipping it (default: 5).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay (seconds) after each successful request, to avoid rate limits (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill in your key."
        )

    client = genai.Client(api_key=api_key)
    rng = random.Random(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = DOMAINS if args.domain == "all" else (args.domain,)

    total_skipped = 0
    for domain in domains:
        skipped = generate_domain(
            client, domain, args.n, args.model, rng, args.max_retries, out_dir, args.delay
        )
        total_skipped += skipped
        logger.info("[%s] done: %d generated, %d skipped", domain, args.n - skipped, skipped)

    logger.info("All domains done. Total skipped: %d", total_skipped)


if __name__ == "__main__":
    main()
