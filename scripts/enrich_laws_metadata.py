#!/usr/bin/env python3
"""Enrich laws_de.csv with extra metadata columns.

By default only adds a ``citation_field`` column (cheap, no LLM calls).
Pass ``--summary`` to also generate LLM summaries (slow, requires OpenAI key).

Usage:
  python scripts/enrich_laws_metadata.py                       # citation_field only
  python scripts/enrich_laws_metadata.py --summary             # citation_field + summary
  python scripts/enrich_laws_metadata.py --summary --model gpt-4o-mini --rate-limit-delay 0.5
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Citation-field extraction (no LLM needed)
# ---------------------------------------------------------------------------

# Patterns to extract referenced citations from law text
_CITATION_PATTERNS = [
    r"BGE\s+\d+\s+(?:I{1,3}|IV|V|VI|Ia|IIa|IIIa|[IVX]+)\s+\d+(?:\s+E\.\s+[\d.a-z und]+)?(?:\s+S\.\s+\d+(?:\s+f\.)?)?",
    r"Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?(?:\s+(?:lit\.\s+[a-z](?:\s+und\s+[a-z])?\s+)?[A-Z][A-Za-z/]{1,}(?:\s+\[[^\]]+\])?)?",
    r"§§?\s*\d+(?:\s*(?:und\s+\d+|f{1,2}\.)|Abs\.\s+\d+)?(?:\s+[A-Za-z/]+)?",
    r"\d+[A-Z]_?\d+/\d{4}",
]
_CITATION_RE = re.compile("|".join(f"({p})" for p in _CITATION_PATTERNS))


def extract_citation_field(text: str) -> str:
    """Return semicolon-separated list of legal citations found in *text*."""
    if not text or not text.strip():
        return ""
    seen = set()
    results = []
    for m in _CITATION_RE.finditer(text):
        cite = re.sub(r"\s+", " ", m.group(0)).strip()
        if cite and cite not in seen:
            seen.add(cite)
            results.append(cite)
    return "; ".join(results)


# ---------------------------------------------------------------------------
# LLM summary (opt-in)
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """\
Du bist ein Schweizer Rechtsexperte. Fasse den folgenden Gesetzesartikel in 1-2 \
kurzen Sätzen auf Deutsch zusammen. Konzentriere dich auf den Regelungsgegenstand \
und die wichtigsten Rechtsfolgen. Antworte NUR mit der Zusammenfassung, ohne \
Einleitung oder Erklärung.

Gesetzesartikel:
---
{text}
---

Zusammenfassung:"""


def generate_summary(text: str, client, model: str) -> str:
    if not text or not text.strip():
        return ""
    truncated = text[:3000]
    prompt = SUMMARY_PROMPT.replace("{text}", truncated)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Schweizer Rechtsassistent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"  WARNING: summary failed: {exc}")
        return ""


def load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached summaries from {path}")
        return cache
    return {}


def save_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Enrich laws CSV with metadata")
    parser.add_argument("--input", default=str(DATA_DIR / "laws_de.csv"))
    parser.add_argument("--output", default=str(DATA_DIR / "laws_de_enriched.csv"))
    parser.add_argument("--summary", action="store_true",
                        help="Also generate LLM summaries (slow, requires OpenAI key)")
    parser.add_argument("--cache", default=str(CACHE_DIR / "laws_summary_cache.json"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--rate-limit-delay", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Save cache every N new summaries")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # --- Always add citation_field (fast, no LLM) ---
    print("Extracting citation_field ...")
    df["citation_field"] = df["text"].fillna("").apply(extract_citation_field)
    n_with_cites = (df["citation_field"] != "").sum()
    print(f"  {n_with_cites}/{len(df)} rows have at least one extracted citation")

    # --- Optionally add summary (slow, LLM) ---
    if args.summary:
        from openai import OpenAI

        cache_path = Path(args.cache)
        print(f"Model:  {args.model}")

        client = OpenAI()
        cache = load_cache(cache_path)

        summaries = []
        new_count = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
            citation = str(row.get("citation", "")).strip()
            text = str(row.get("text", "")).strip()

            if citation and citation in cache:
                summaries.append(cache[citation])
                continue

            summary = generate_summary(text, client, args.model)
            summaries.append(summary)

            if citation:
                cache[citation] = summary
            new_count += 1

            if args.rate_limit_delay > 0:
                time.sleep(args.rate_limit_delay)

            if new_count % args.save_every == 0:
                save_cache(cache_path, cache)

        df["summary"] = summaries
        save_cache(cache_path, cache)
        print(f"{new_count} new summaries generated, {len(cache)} total cached")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    main()
