#!/usr/bin/env python3
"""
Parse court_considerations CSV: extract every citation and its case/summary from each row.
Each (source row, cited reference) becomes one output row.
"""

import csv
import re
from pathlib import Path

# Citation patterns (Swiss/German legal) – order matters (more specific first)
PATTERNS = [
    # BGE 139 I 2 S. 7; BGE 138 I 189 E. 2.1 S. 190; BGE 115 Ia 148 E. 1a und b S. 152 f.
    (r"BGE\s+\d+\s+(?:I{1,3}|IV|V|VI|Ia|IIa|IIIa|[IVX]+)\s+\d+(?:\s+E\.\s+[\d.a-z und]+)?(?:\s+S\.\s+\d+(?:\s+f\.)?)?", "BGE"),
    # Art. 34 Abs. 1 BV; Art. 95 lit. c und d BGG; Art. 34 BV
    (r"Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?(?:\s+lit\.\s+[a-z]\s+und\s+[a-z])?\s+[A-Za-z/]{2,}(?:\s+\[[^\]]+\])?", "Art."),
    # §§ 12 und 26 GOG; § 25 Abs. 3 PBG/SZ; §§ 25 ff. PBG/SZ; § 27 Abs. 2 PBG/SZ (include law code)
    (r"§§?\s*\d+(?:\s*(?:und\s+\d+|\s*f\.?|ff\.)|Abs\.\s+\d+)?(?:\s+[A-Za-z/]+)?", "§"),
    # 1C_403/2011; 4A_355/2021; 1P.150/2003; 4P.226/2004
    (r"\d+[A-Z]_?\d+/\d{4}(?:\s+\d{2}\.\d{2}\.\d{4})?(?:\s+E\.\s+[\d.]+)?", "Docket"),
    # VGE III 2009 101; VGE 895/05
    (r"VGE(?:\s+[IVX]+)?\s*\d+\s*/\s*\d+|\bVGE\s+\d+\s*/\s*\d+", "VGE"),
]


def expand_citation_to_full_names(cite: str) -> list[str]:
    """Expand abbreviated citations to full names (one per item).

    E.g. 'Art. 95 lit. c und d BGG' -> ['Art. 95 lit. c BGG', 'Art. 95 lit. d BGG']
    """
    cite = re.sub(r"\s+", " ", cite).strip()
    # Art. N [Abs. M] lit. X und Y LAW or lit. a, b und c LAW
    m = re.match(
        r"Art\.\s+(\d+[a-z]?)(?:\s+Abs\.\s+\d+)?\s+lit\.\s+([a-z](?:\s*,\s*[a-z]|\s+und\s+[a-z])*)\s+([A-Za-z/]{2,}(?:\s+\[[^\]]+\])?)",
        cite,
        re.IGNORECASE,
    )
    if not m:
        return [cite]
    art_num, letters_part, law_code = m.group(1), m.group(2), m.group(3)
    # Split letters: "c und d" -> [c, d]; "a, b und c" -> [a, b, c]
    letters = re.split(r"\s*,\s*|\s+und\s+", letters_part)
    letters = [x.strip() for x in letters if x.strip()]
    if len(letters) <= 1:
        return [cite]
    return [f"Art. {art_num} lit. {letter} {law_code}" for letter in letters]


def find_citations(text: str) -> list[tuple[str, int]]:
    """Return list of (citation_string, start_position)."""
    found = []
    for pattern, _ in PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            cite = m.group(0).strip()
            cite = re.sub(r"\s+", " ", cite)
            if not cite:
                continue
            # Expand to full names (e.g. Art. 95 lit. c und d BGG -> Art. 95 lit. c BGG, Art. 95 lit. d BGG)
            expanded = expand_citation_to_full_names(cite)
            for c in expanded:
                if (c, m.start()) not in [(x, p) for x, p in found]:
                    found.append((c, m.start()))
    found.sort(key=lambda x: x[1])
    return found


def extract_summary_around(text: str, position: int, cite: str, max_chars: int = 400) -> str:
    """Extract a summary: sentence or context around the citation."""
    # Extend to sentence boundaries if possible
    start = max(0, position - 80)
    end = min(len(text), position + len(cite) + 200)
    chunk = text[start:end]
    # Try to start at last sentence start before citation
    for sep in ". ", ". ", "). ":
        idx = chunk.find(sep, 0, min(len(chunk), position - start + 50))
        if idx != -1:
            chunk = chunk[idx + len(sep) :].lstrip()
            break
    # Truncate to max_chars and clean
    if len(chunk) > max_chars:
        chunk = chunk[: max_chars].rsplit(". ", 1)[0] + "." if ". " in chunk[: max_chars] else chunk[: max_chars] + "..."
    return chunk.replace("\n", " ").strip()


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    in_path = data_dir / "court_considerations_first_10.csv"
    out_path = data_dir / "court_considerations_first_10_parsed.csv"

    rows_out = []
    with open(in_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source_citation = row.get("citation", "").strip()
            text = (row.get("text") or "").strip()
            if not text:
                continue
            citations = find_citations(text)
            if not citations:
                # No secondary citation found; treat whole row as one "case" with the source citation as the only citation
                rows_out.append(
                    {
                        "source_citation": source_citation,
                        "cited_citation": source_citation,
                        "summary": text[:500].replace("\n", " ") + ("..." if len(text) > 500 else ""),
                    }
                )
                continue
            for cite, pos in citations:
                summary = extract_summary_around(text, pos, cite)
                rows_out.append(
                    {
                        "source_citation": source_citation,
                        "cited_citation": cite,
                        "summary": summary,
                    }
                )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_citation", "cited_citation", "summary"])
        w.writeheader()
        w.writerows(rows_out)

    print(f"Read {in_path}")
    print(f"Wrote {len(rows_out)} rows to {out_path}")


if __name__ == "__main__":
    main()
