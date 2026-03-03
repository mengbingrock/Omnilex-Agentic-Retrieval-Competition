"""Extract Swiss legal citations from court consideration texts.

Supports BGE (Federal Court decisions), docket-style references,
VGE (cantonal court), and law article citations.
"""

import re
from dataclasses import dataclass, field


@dataclass
class ExtractedCitation:
    """A citation extracted from text with its source context."""

    raw_text: str
    citation_type: str  # "BGE", "Docket", "VGE", "Art.", "§"
    canonical_id: str
    position: int = 0


# ---------------------------------------------------------------------------
# Swiss legal citation patterns (ordered most-specific first)
# ---------------------------------------------------------------------------

PATTERNS: list[tuple[str, str]] = [
    # BGE 139 I 2 S. 7; BGE 138 I 189 E. 2.1 S. 190
    (
        r"BGE\s+\d+\s+(?:I{1,3}|IV|V|VI|Ia|IIa|IIIa|[IVX]+[a-z]?)\s+\d+"
        r"(?:\s+E\.\s+[\d.a-z]+(?:\s+und\s+[a-z])?)?"
        r"(?:\s+S\.\s+\d+(?:\s+f\.)?)?",
        "BGE",
    ),
    # Art. 34 Abs. 1 BV; Art. 95 lit. c und d BGG
    (
        r"Art\.\s+\d+[a-z]?"
        r"(?:\s+Abs\.\s+\d+)?"
        r"(?:\s+lit\.\s+[a-z](?:\s+und\s+[a-z])?)?"
        r"\s+[A-Za-z/]{2,}"
        r"(?:\s+\[[^\]]+\])?",
        "Art.",
    ),
    # §§ 12 und 26 GOG; § 25 Abs. 3 PBG/SZ
    (
        r"§§?\s*\d+"
        r"(?:\s*(?:und\s+\d+|\s*f{1,2}\.)|Abs\.\s+\d+)?"
        r"(?:\s+[A-Za-z/]+)?",
        "§",
    ),
    # 1C_403/2011; 4A_355/2021; 1P.150/2003
    (
        r"\d+[A-Z]_?\d+/\d{4}"
        r"(?:\s+\d{2}\.\d{2}\.\d{4})?"
        r"(?:\s+E\.\s+[\d.]+)?",
        "Docket",
    ),
    # VGE III 2009 101; VGE 895/05
    (
        r"VGE(?:\s+[IVX]+)?\s*\d+\s*/\s*\d+|\bVGE\s+\d+\s*/\s*\d+",
        "VGE",
    ),
]

# Pre-compiled regexes
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), t) for p, t in PATTERNS]

# Regex to extract the BGE case-level ID (without consideration / page suffix)
_BGE_CASE_RE = re.compile(
    r"BGE\s+(\d+)\s+([IVX]+[a-z]?)\s+(\d+)", re.IGNORECASE
)


def _canonicalize_bge(raw: str) -> str:
    """Normalize a BGE citation to case-level canonical form: 'BGE {vol} {sec} {page}'."""
    m = _BGE_CASE_RE.search(raw)
    if m:
        return f"BGE {m.group(1)} {m.group(2)} {m.group(3)}"
    return raw.strip()


def _canonicalize(raw: str, ctype: str) -> str:
    """Return a canonical string for deduplication and linking."""
    raw = re.sub(r"\s+", " ", raw).strip()
    if ctype == "BGE":
        return _canonicalize_bge(raw)
    if ctype == "Art.":
        return re.sub(r"\s+", " ", raw).strip()
    return raw


def extract_citations(text: str) -> list[ExtractedCitation]:
    """Extract all citations from *text*, deduplicated by canonical ID.

    Returns a list of :class:`ExtractedCitation` sorted by position in the text.
    """
    if not text:
        return []

    seen_spans: list[tuple[int, int]] = []
    results: dict[str, ExtractedCitation] = {}

    for regex, ctype in _COMPILED_PATTERNS:
        for m in regex.finditer(text):
            start, end = m.start(), m.end()

            # skip if this span overlaps a previously captured match
            if any(s <= start < e or s < end <= e for s, e in seen_spans):
                continue
            seen_spans.append((start, end))

            raw = re.sub(r"\s+", " ", m.group(0)).strip()
            canonical = _canonicalize(raw, ctype)

            if canonical not in results:
                results[canonical] = ExtractedCitation(
                    raw_text=raw,
                    citation_type=ctype,
                    canonical_id=canonical,
                    position=start,
                )

    return sorted(results.values(), key=lambda c: c.position)


def extract_bge_citations(text: str) -> list[str]:
    """Return deduplicated list of canonical BGE case IDs found in *text*.

    Only returns BGE-type citations (court decisions), not law articles.
    This is the main entry point for building case-to-case edges.
    """
    cites = extract_citations(text)
    seen: set[str] = set()
    out: list[str] = []
    for c in cites:
        if c.citation_type == "BGE" and c.canonical_id not in seen:
            seen.add(c.canonical_id)
            out.append(c.canonical_id)
    return out


def extract_all_case_citations(text: str) -> list[str]:
    """Return deduplicated canonical IDs for all case-type citations (BGE + Docket + VGE)."""
    cites = extract_citations(text)
    seen: set[str] = set()
    out: list[str] = []
    for c in cites:
        if c.citation_type in ("BGE", "Docket", "VGE") and c.canonical_id not in seen:
            seen.add(c.canonical_id)
            out.append(c.canonical_id)
    return out


def count_case_citations(text: str) -> dict[str, int]:
    """Count how many times each case-type citation appears in *text*.

    Unlike :func:`extract_all_case_citations` which deduplicates, this counts
    every occurrence so we can compute TF (term-frequency) for edge weighting.

    Returns:
        ``{canonical_id: count}``
    """
    if not text:
        return {}

    counts: dict[str, int] = {}
    case_types = ("BGE", "Docket", "VGE")

    for regex, ctype in _COMPILED_PATTERNS:
        if ctype not in case_types:
            continue
        for m in regex.finditer(text):
            raw = re.sub(r"\s+", " ", m.group(0)).strip()
            canonical = _canonicalize(raw, ctype)
            counts[canonical] = counts.get(canonical, 0) + 1

    return counts


def parse_case_id_from_csv_citation(citation_field: str) -> str | None:
    """Extract the case-level BGE ID from a ``court_considerations.csv`` citation field.

    The CSV ``citation`` column looks like ``BGE 139 I 2 E. 5.1`` or
    ``BGE 139 I 2 E. 1.12.2011``.  We strip everything after the page number.

    Returns:
        Canonical case ID (e.g. ``"BGE 139 I 2"``) or *None* if unparseable.
    """
    m = _BGE_CASE_RE.search(citation_field)
    if m:
        return f"BGE {m.group(1)} {m.group(2)} {m.group(3)}"
    return None
