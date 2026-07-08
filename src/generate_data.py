"""Synthetic training data generation (dev-plan step 3.3).

Generates the synthetic Hungarian text-to-JSON training set used to
fine-tune the model. For each of the three target domains (medical,
business, technology) this script prompts an LLM to produce a realistic
Hungarian document paired with its gold JSON extraction, following the
domain schemas defined in `schemas.py` (which mirror SCOPE.md's table).
Each generated example is validated against the domain's JSON Schema and
written out as JSONL under `data/synthetic/`.

The LLM backend is an OpenAI-compatible local proxy (serving Claude models)
at `http://127.0.0.1:8000/v1` — no API key is required since it's a
personally-owned, localhost-only resource. This script only performs API
calls when actually run; it is not executed as part of writing/reviewing
this code.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import openai
from jsonschema import ValidationError, validate
from openai import OpenAI

from schemas import SCHEMAS

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DOMAINS = ("medical", "business", "technology")

# Local OpenAI-compatible proxy (claude-fast-proxy, personally owned/authored
# by the user — no API key required). Keeps one persistent Claude Agent SDK
# connection open across requests instead of spawning a fresh CLI session per
# call, which is dramatically faster for a batch of independent completions.
LOCAL_PROXY_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_PROXY_PLACEHOLDER_API_KEY = "not-needed"

# claude-fast-proxy disables extended thinking outright, so max_tokens isn't
# mapped to a thinking budget the way the old wrapper did; this value is
# accepted for OpenAI-client compatibility but otherwise unused.
MAX_RESPONSE_TOKENS = 700

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

# Naming style hints rotated per call to push the model toward inventing a
# fresh, unique entity name each time instead of defaulting to a handful of
# "safe" generic names (observed: e.g. "Innovatech Zrt." in 36% of business
# examples). One list per domain, analogous to STYLE_HINTS.
NAME_STYLE_HINTS: dict[str, list[str]] = {
    "medical": [
        "latin/görög gyógyszerészeti tő alapú név (pl. -in, -ol, -ex végződés)",
        "márkanév, amely a hatóanyagra utal rövidítve",
        "generikus gyártói név + hatáserősség a névben",
        "klasszikus, régóta bejegyzett hangzású magyar gyógyszernév",
        "modern, nemzetközi hangzású gyógyszermárka-név",
    ],
    "business": [
        "magyar-angol összetételű, modern cégnév (pl. -Tech, -Solutions utótaggal, de NE 'Innovatech')",
        "családi/alapítói névre épülő hagyományos magyar cégnév",
        "rövidítésekből álló, tőzsdei hangzású cégnév",
        "playful, startup-stílusú, egyszavas márkanév",
        "iparági utótagú (Kft., Zrt., Nyrt.) formális cégnév, egyedi törzsnévvel",
    ],
    "technology": [
        "termékcsalád-szerű névelnevezés (pl. sorozatszám vagy generációjelzés a névben)",
        "egyszavas, modern, nemzetközi hangzású terméknév",
        "gyártó + termékvonal + verziószám kombinációja",
        "playful, fogyasztói márkanév, könnyen megjegyezhető",
        "szakmai/prémium hangzású terméknév, betűszóval vagy kóddal",
    ],
}

# Generic entity names observed to repeat across generations — explicitly
# banned in the prompt so the model can't fall back on them.
BANNED_GENERIC_NAMES: dict[str, list[str]] = {
    "medical": ["Algopyrin", "Aspirin Plus", "MediCare"],
    "business": ["Innovatech Zrt.", "TechCorp", "GlobalCorp Kft."],
    "technology": ["TechPro", "NovaTech", "SmartDevice X"],
}

GENERATION_TEMPERATURE_RANGE = (0.7, 1.0)

# Target probability that a nullable (optional) field is actually present
# and filled in with a concrete value in a given example, vs. omitted from
# the document (and thus null/empty in gold). Decided deterministically per
# example in code (not left to the model's own judgment), since a prompt
# instruction alone ("sometimes omit") was observed to collapse to "almost
# always omit" (e.g. medical `dosage` null in 95% of examples).
NULLABLE_FIELD_PRESENCE_RATE = 0.75

# Backoff (seconds) between retries. Validation/parse retries can be quick;
# transient API errors (503/429) need more breathing room, capped at 60s.
VALIDATION_BACKOFF_CAP = 30.0
API_BACKOFF_CAP = 60.0


def is_nullable_scalar(spec: dict[str, Any]) -> bool:
    """True if a field's JSON Schema type is a nullable scalar union, e.g. ["string", "null"]."""
    field_type = spec.get("type")
    return isinstance(field_type, list) and "null" in field_type


def build_presence_plan(domain: str, rng: random.Random) -> dict[str, bool]:
    """Decide, per nullable field, whether it is present (True) or omitted (False).

    Each nullable scalar field independently gets `True` (present, filled in
    with a concrete value) with probability `NULLABLE_FIELD_PRESENCE_RATE`,
    else `False` (omitted from the document -> null in gold). This is decided
    in code rather than left to the model, so the actual output ratio matches
    the intended ~75/25 split instead of drifting to "almost always null".
    """
    props = SCHEMAS[domain]["properties"]
    return {
        name: rng.random() < NULLABLE_FIELD_PRESENCE_RATE
        for name, spec in props.items()
        if is_nullable_scalar(spec)
    }


def _describe_field(name: str, spec: dict[str, Any], presence: dict[str, bool]) -> str:
    """Render one schema field as a human-readable Hungarian type description.

    Used to embed the expected `gold` structure directly in the prompt text,
    since this backend has no native JSON-schema-constrained response mode
    (unlike e.g. Gemini's `response_schema`) — the model is guided purely by
    prompt instructions. For nullable scalar fields, also states the per-example presence decision
    from `presence` (see `build_presence_plan`), so the prompt gives an
    explicit, deterministic instruction instead of a vague "sometimes".
    """
    field_type = spec.get("type")
    if field_type == "array":
        item_type = spec.get("items", {}).get("type", "string")
        return f"- {name}: {item_type} elemek tömbje (ha nincs adat: üres tömb [])"
    if isinstance(field_type, list):
        # Union type, e.g. ["string", "null"] -> nullable scalar.
        base = next((t for t in field_type if t != "null"), "string")
        if presence.get(name, True):
            directive = " -> EBBEN A PÉLDÁBAN legyen konkrét érték a szövegben (NE null)."
        else:
            directive = " -> EBBEN A PÉLDÁBAN szándékosan hiányozzon a szövegből (legyen null)."
        return f"- {name}: {base} (nullable){directive}"
    return f"- {name}: {field_type}"


def build_field_spec(domain: str, presence: dict[str, bool]) -> str:
    """Render the domain's gold fields as a Hungarian bullet list for the prompt."""
    props = SCHEMAS[domain]["properties"]
    return "\n".join(_describe_field(name, spec, presence) for name, spec in props.items())


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


def build_prompt(
    domain: str,
    style_hint: str,
    name_hint: str,
    presence: dict[str, bool],
) -> str:
    """Build the Hungarian-document generation prompt for one example."""
    banned = ", ".join(f"'{n}'" for n in BANNED_GENERIC_NAMES[domain])
    return (
        f"Generálj egy szintetikus, de realisztikus MAGYAR nyelvű szöveget a(z) "
        f"'{domain}' témakörben, a következő stílusban: {style_hint}.\n\n"
        "A szöveg legyen természetes, folyó szöveg (NE a JSON mezők egyszerű "
        "felsorolása), változó hosszúságú és stílusú. Illessz bele 1-2 olyan "
        "hihető, de a sémához nem tartozó részletet is (pl. egy mellékes "
        "szereplő neve, egy dátum, egy szám), amelyek elterelik a figyelmet, "
        "de nem részei a kinyerendő adatoknak.\n\n"
        "Egyedi elnevezés: találj ki egy VADONATÚJ, EGYEDI, a magyar piacon "
        f"hihető nevet a szöveg főszereplőjének (cég/gyógyszer/termék), "
        f"ebben a névalkotási stílusban: {name_hint}. Minden egyes generálás "
        "alkalmával más és más nevet találj ki — SOHA ne használd újra a "
        "korábban már látott vagy alábbi, túlságosan elcsépelt, generikus "
        f"neveket: {banned}. Legyél kreatív és változatos.\n\n"
        "Add vissza PONTOSAN egy JSON objektumot, két kulccsal:\n"
        "- 'document': a generált magyar nyelvű szöveg (string)\n"
        "- 'gold': a szövegből kinyerhető adatok az alábbi mezőkkel.\n\n"
        "A 'gold' objektum mezői és típusai:\n"
        f"{build_field_spec(domain, presence)}\n\n"
        "Fontos szabályok:\n"
        "- A fenti mezőlistában minden nullable mezőnél EXPLICIT utasítás "
        "szerepel arra, hogy ebben a konkrét példában legyen-e benne "
        "konkrét érték a szövegben, vagy szándékosan hiányozzon — ezt "
        "PONTOSAN kövesd (a legtöbb dokumentumban a legtöbb adat ténylegesen "
        "szerepelni szokott, csak elvétve hiányzik egy-egy részlet).\n"
        "- Ne találj ki adatot: a 'gold' kizárólag a szövegben ténylegesen "
        "szereplő információt tartalmazza.\n"
        "- Minden felsorolt mező szerepeljen a 'gold' objektumban.\n\n"
        "Példa a válasz formátumára (a tartalmat te generáld, ez csak a "
        "szerkezetet mutatja):\n"
        f"{build_example_envelope(domain)}\n\n"
        "A válaszod KIZÁRÓLAG a nyers JSON objektum legyen — NE tegyél elé "
        "vagy mögé semmilyen magyarázatot, és NE csomagold Markdown "
        "kódblokkba (NE használj ```json``` vagy ``` jelölést)."
    )


def strip_json_fence(text: str) -> str:
    """Strip a leading/trailing Markdown code fence (```json ... ``` or ``` ... ```).

    Chat models often wrap JSON in a fenced code block even when explicitly
    told not to; this normalizes the response before `json.loads`.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the closing ```.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
    return stripped.strip()


def generate_one_example(
    client: OpenAI,
    domain: str,
    model: str,
    rng: random.Random,
    max_retries: int,
) -> dict[str, Any] | None:
    """Generate and validate a single {document, gold} example for a domain.

    Retries up to `max_retries` times (with exponential backoff) on:
      - parse/validation failures (bad JSON or schema-invalid gold), and
      - transient API errors: any 5xx, and 429 rate-limit responses.
    Non-transient client errors (other 4xx, e.g. 400/401/403 — bad request,
    bad key, forbidden) are NOT transient and abort immediately with a clear
    message. Returns None if all retries are exhausted.
    """
    domain_schema = SCHEMAS[domain]
    style_hint = rng.choice(STYLE_HINTS[domain])
    name_hint = rng.choice(NAME_STYLE_HINTS[domain])
    presence = build_presence_plan(domain, rng)
    prompt = build_prompt(domain, style_hint, name_hint, presence)

    for attempt in range(1, max_retries + 1):
        temperature = rng.uniform(*GENERATION_TEMPERATURE_RANGE)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_RESPONSE_TOKENS,
            )
            raw_text = response.choices[0].message.content or ""
            payload = json.loads(strip_json_fence(raw_text))
            document = payload["document"]
            gold = payload["gold"]
            if not isinstance(document, str) or not document.strip():
                raise ValueError("empty or non-string 'document' field")
            validate(instance=gold, schema=domain_schema)
            return {"domain": domain, "document": document, "gold": gold}
        except openai.APIStatusError as exc:
            status = exc.status_code
            is_retryable = status == 429 or 500 <= status < 600
            if not is_retryable:
                raise SystemExit(
                    f"Non-retryable API error (HTTP {status}): {exc}. "
                    "This is likely a bad request or a misconfigured proxy, "
                    "not a transient outage."
                ) from exc
            logger.warning(
                "[%s] attempt %d/%d API error (HTTP %s): %s",
                domain, attempt, max_retries, status, exc,
            )
            if attempt < max_retries:
                time.sleep(min(5 * 2 ** (attempt - 1), API_BACKOFF_CAP))
        except openai.APIConnectionError as exc:
            # Local proxy unreachable (e.g. not started yet) -> transient, retry.
            logger.warning(
                "[%s] attempt %d/%d connection error: %s", domain, attempt, max_retries, exc
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
    client: OpenAI,
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
            # Polite delay between successful requests, to avoid hammering
            # the local proxy back-to-back.
            if delay > 0:
                time.sleep(delay)
    return skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic Hungarian text-to-JSON training data via a local "
            "OpenAI-compatible proxy serving Claude models."
        )
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
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Model name served by the local proxy (default: claude-haiku-4-5-20251001).",
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
        # Less critical than with a shared cloud API quota (this is a local
        # proxy), but still polite to avoid hammering it back-to-back.
        help="Delay (seconds) after each successful request, to avoid hammering the local proxy (default: 0.5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Local, personally-owned proxy — no real API key needed, just a
    # non-empty placeholder (the SDK requires one).
    client = OpenAI(base_url=LOCAL_PROXY_BASE_URL, api_key=LOCAL_PROXY_PLACEHOLDER_API_KEY)
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
