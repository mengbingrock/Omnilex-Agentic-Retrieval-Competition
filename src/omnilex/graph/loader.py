"""Parse court_considerations.csv and laws_de.csv, extract citations, and bulk-load the citation graph into Neo4j.

Pipeline
--------
1.  Read court_considerations.csv (``citation``, ``text`` columns).
2.  Group rows by case-level BGE ID (e.g. "BGE 139 I 2").
3.  Concatenate all consideration texts per case → the node's ``text``.
4.  For each case, extract outgoing BGE citations from the text → edges.
5.  Read laws_de.csv (``citation``, ``text``, ``title`` columns) → :Law nodes.
6.  Extract citations from law texts → :CITES_LAW / :CITES edges from laws.
7.  Bulk-insert nodes and edges into Neo4j.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, BATCH_SIZE
from .extractor import (
    count_art_citations,
    count_case_citations,
    extract_all_case_citations,
    extract_citations,
    parse_case_id_from_csv_citation,
)
from .schema import create_schema


# ---------------------------------------------------------------------------
# 1. Parse CSV into per-case records
# ---------------------------------------------------------------------------

def _iter_csv_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def parse_cases_from_csv(
    csv_path: Path,
    *,
    max_rows: int | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """Group CSV rows into per-case records.

    Returns:
        ``{case_id: {"id": ..., "volume": ..., "section": ..., "page": ..., "text": ...}}``
    """
    cases: dict[str, dict] = {}
    skipped = 0

    for i, row in enumerate(_iter_csv_rows(csv_path)):
        if max_rows and i >= max_rows:
            break
        citation_field = (row.get("citation") or "").strip()
        text = (row.get("text") or "").strip()
        case_id = parse_case_id_from_csv_citation(citation_field)
        if case_id is None:
            skipped += 1
            continue

        if case_id not in cases:
            m = re.match(r"BGE\s+(\d+)\s+([IVX]+[a-z]?)\s+(\d+)", case_id)
            cases[case_id] = {
                "id": case_id,
                "volume": int(m.group(1)) if m else None,
                "section": m.group(2) if m else None,
                "page": int(m.group(3)) if m else None,
                "texts": [],
            }
        cases[case_id]["texts"].append(text)

    # join consideration texts into a single string per case
    for case in cases.values():
        case["text"] = "\n\n".join(case.pop("texts"))

    if verbose:
        print(f"Parsed {len(cases)} unique cases from CSV ({skipped} rows skipped)")
    return cases


def parse_laws_from_csv(
    csv_path: Path,
    *,
    max_rows: int | None = None,
    verbose: bool = False,
) -> dict[str, dict]:
    """Parse ``laws_de.csv`` into per-provision law records.

    The CSV has columns ``citation``, ``text``, ``title``.
    The ``citation`` column uses SR-number format, e.g. ``"Art. 3 Abs. 1 112"``.

    Returns:
        ``{law_id: {"id": ..., "text": ..., "title": ..., "sr_number": ..., "book": ...}}``
    """
    _sr_re = re.compile(r"(\d[\d.]*)\s*$")
    laws: dict[str, dict] = {}
    skipped = 0

    for i, row in enumerate(_iter_csv_rows(csv_path)):
        if max_rows and i >= max_rows:
            break
        citation = (row.get("citation") or "").strip()
        text = (row.get("text") or "").strip()
        title = (row.get("title") or "").strip()

        if not citation:
            skipped += 1
            continue

        law_id = re.sub(r"\s+", " ", citation).strip()

        # Extract the SR number (last numeric token, e.g. "112", "131.211")
        sr_match = _sr_re.search(law_id)
        sr_number = sr_match.group(1) if sr_match else None

        if law_id not in laws:
            laws[law_id] = {
                "id": law_id,
                "text": text,
                "title": title,
                "sr_number": sr_number,
            }
        else:
            # Same citation may appear multiple times (shouldn't, but handle it)
            laws[law_id]["text"] += "\n\n" + text

    if verbose:
        print(f"Parsed {len(laws)} law provisions from CSV ({skipped} rows skipped)")
    return laws


# ---------------------------------------------------------------------------
# 2. Build edge lists
# ---------------------------------------------------------------------------

def build_case_edges(
    cases: dict[str, dict],
    *,
    verbose: bool = False,
) -> list[dict]:
    """Extract CITES edges with TF-IDF weights from concatenated texts.

    Weight formula per edge (source → target)::

        TF  = 1 + log(count)          # how often source cites target
        IDF = log(N / df)              # rarity: N cases / df cases that cite target
        weight = TF * IDF

    Only creates edges where *both* source and target exist as nodes.
    """
    case_ids = set(cases)
    N = len(cases)

    # Pass 1: count per-source citation frequencies & collect document-frequency
    source_counts: dict[str, dict[str, int]] = {}
    df_counter: dict[str, int] = defaultdict(int)  # target → number of sources citing it

    for case_id, case in tqdm(cases.items(), desc="Counting citations", disable=not verbose):
        counts = count_case_citations(case["text"])
        filtered: dict[str, int] = {}
        for target_id, cnt in counts.items():
            if target_id == case_id or target_id not in case_ids:
                continue
            filtered[target_id] = cnt
            df_counter[target_id] += 1
        source_counts[case_id] = filtered

    # Pass 2: compute TF-IDF and build edges
    edges: list[dict] = []
    for source_id, targets in source_counts.items():
        for target_id, raw_count in targets.items():
            tf = 1.0 + math.log(raw_count) if raw_count > 0 else 0.0
            idf = math.log(N / df_counter[target_id]) if df_counter[target_id] > 0 else 0.0
            weight = round(tf * idf, 6)
            edges.append({
                "source": source_id,
                "target": target_id,
                "weight": weight,
                "tf": raw_count,
            })

    if verbose:
        print(f"Built {len(edges)} CITES edges with TF-IDF weights")
    return edges


def build_law_nodes_and_edges(
    cases: dict[str, dict],
    *,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Extract :Law nodes and TF-IDF weighted :CITES_LAW edges from case texts.

    Weight formula (same as :func:`build_case_edges`)::

        TF  = 1 + log(count)
        IDF = log(N / df)
        weight = TF * IDF

    Returns:
        ``(law_nodes, law_edges)`` where law_nodes is ``[{"id": ..., "book": ...}]``
        and law_edges is ``[{"source": ..., "target": ..., "weight": ..., "tf": ...}]``.
    """
    book_re = re.compile(r"Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+([A-Za-z/]{2,})")
    law_map: dict[str, dict] = {}
    N = len(cases)

    # Pass 1: count per-case Art. citation frequencies & document frequency
    source_counts: dict[str, dict[str, int]] = {}
    df_counter: dict[str, int] = defaultdict(int)

    for case_id, case in tqdm(cases.items(), desc="Counting law citations", disable=not verbose):
        counts = count_art_citations(case["text"])
        for target_id in counts:
            if target_id not in law_map:
                bm = book_re.search(target_id)
                law_map[target_id] = {
                    "id": target_id,
                    "book": bm.group(1) if bm else None,
                }
            df_counter[target_id] += 1
        source_counts[case_id] = counts

    # Pass 2: compute TF-IDF and build edges
    edges: list[dict] = []
    for source_id, targets in source_counts.items():
        for target_id, raw_count in targets.items():
            tf = 1.0 + math.log(raw_count) if raw_count > 0 else 0.0
            idf = math.log(N / df_counter[target_id]) if df_counter[target_id] > 0 else 0.0
            weight = round(tf * idf, 6)
            edges.append({
                "source": source_id,
                "target": target_id,
                "weight": weight,
                "tf": raw_count,
            })

    law_nodes = list(law_map.values())
    if verbose:
        print(f"Found {len(law_nodes)} unique law provisions, {len(edges)} CITES_LAW edges (TF-IDF weighted)")
    return law_nodes, edges


def build_law_citation_edges(
    laws: dict[str, dict],
    case_ids: set[str],
    law_ids: set[str],
    *,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Extract TF-IDF weighted citations from law provision texts.

    Scans each law provision's text for:
    - BGE / Docket / VGE citations → ``law_to_case`` edges
    - Art. citations → ``law_to_law`` edges (when target exists in *law_ids*)

    Weight formula (same as :func:`build_case_edges`)::

        TF  = 1 + log(count)
        IDF = log(N / df)
        weight = TF * IDF

    Args:
        laws: Output of :func:`parse_laws_from_csv`.
        case_ids: Set of known :Case node IDs.
        law_ids: Set of known :Law node IDs (both SR-number and abbreviation-based).

    Returns:
        ``(law_to_case_edges, law_to_law_edges)`` — each edge carries
        ``weight`` and ``tf`` keys.
    """
    N = len(laws)

    # Pass 1: count per-law citation frequencies & document frequency
    l2c_source_counts: dict[str, dict[str, int]] = {}
    l2c_df: dict[str, int] = defaultdict(int)

    l2l_source_counts: dict[str, dict[str, int]] = {}
    l2l_df: dict[str, int] = defaultdict(int)

    for law_id, law in tqdm(laws.items(), desc="Counting law text citations", disable=not verbose):
        text = law.get("text", "")
        if not text:
            continue

        # Case citations from law text
        case_counts = count_case_citations(text)
        filtered_case: dict[str, int] = {}
        for target_id, cnt in case_counts.items():
            if target_id in case_ids:
                filtered_case[target_id] = cnt
                l2c_df[target_id] += 1
        l2c_source_counts[law_id] = filtered_case

        # Art. citations from law text
        art_counts = count_art_citations(text)
        filtered_art: dict[str, int] = {}
        for target_id, cnt in art_counts.items():
            if target_id != law_id and target_id in law_ids:
                filtered_art[target_id] = cnt
                l2l_df[target_id] += 1
        l2l_source_counts[law_id] = filtered_art

    # Pass 2: compute TF-IDF
    def _build_weighted(
        source_counts: dict[str, dict[str, int]],
        df_counter: dict[str, int],
    ) -> list[dict]:
        edges: list[dict] = []
        for source_id, targets in source_counts.items():
            for target_id, raw_count in targets.items():
                tf = 1.0 + math.log(raw_count) if raw_count > 0 else 0.0
                idf = math.log(N / df_counter[target_id]) if df_counter[target_id] > 0 else 0.0
                weight = round(tf * idf, 6)
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "weight": weight,
                    "tf": raw_count,
                })
        return edges

    law_to_case = _build_weighted(l2c_source_counts, l2c_df)
    law_to_law = _build_weighted(l2l_source_counts, l2l_df)

    if verbose:
        print(
            f"Built {len(law_to_case)} Law→Case edges, "
            f"{len(law_to_law)} Law→Law edges (TF-IDF weighted)"
        )
    return law_to_case, law_to_law


# ---------------------------------------------------------------------------
# 3. Bulk-load into Neo4j
# ---------------------------------------------------------------------------

def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_case_nodes(cases: list[dict], driver, batch_size: int = BATCH_SIZE) -> int:
    """MERGE :Case nodes."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(cases, batch_size)), desc="Loading Case nodes"):
            result = session.run(
                """
                UNWIND $batch AS c
                MERGE (n:Case {id: c.id})
                SET n.volume   = c.volume,
                    n.section  = c.section,
                    n.page     = c.page,
                    n.text     = c.text
                RETURN count(n) AS cnt
                """,
                batch=batch,
            )
            total += result.single()["cnt"]
    return total


def load_law_nodes(laws: list[dict], driver, batch_size: int = BATCH_SIZE) -> int:
    """MERGE :Law nodes.  Accepts both minimal (``id``, ``book``) and enriched
    (``text``, ``title``, ``sr_number``) records from ``laws_de.csv``."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(laws, batch_size)), desc="Loading Law nodes"):
            result = session.run(
                """
                UNWIND $batch AS l
                MERGE (n:Law {id: l.id})
                SET n.book      = COALESCE(l.book, n.book),
                    n.text      = COALESCE(l.text, n.text),
                    n.title     = COALESCE(l.title, n.title),
                    n.sr_number = COALESCE(l.sr_number, n.sr_number)
                RETURN count(n) AS cnt
                """,
                batch=batch,
            )
            total += result.single()["cnt"]
    return total


def load_cites_edges(edges: list[dict], driver, batch_size: int = BATCH_SIZE * 2) -> int:
    """MERGE (:Case)-[:CITES]->(:Case) edges with TF-IDF weight."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(edges, batch_size)), desc="Loading CITES edges"):
            result = session.run(
                """
                UNWIND $batch AS e
                MATCH (s:Case {id: e.source})
                MATCH (t:Case {id: e.target})
                MERGE (s)-[r:CITES]->(t)
                SET r.weight = e.weight,
                    r.tf     = e.tf
                RETURN count(r) AS cnt
                """,
                batch=batch,
            )
            total += result.single()["cnt"]
    return total


def load_cites_law_edges(edges: list[dict], driver, batch_size: int = BATCH_SIZE * 2) -> int:
    """MERGE (:Case)-[:CITES_LAW]->(:Law) edges with TF-IDF weight."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(edges, batch_size)), desc="Loading CITES_LAW edges"):
            result = session.run(
                """
                UNWIND $batch AS e
                MATCH (s:Case {id: e.source})
                MATCH (t:Law {id: e.target})
                MERGE (s)-[r:CITES_LAW]->(t)
                SET r.weight = e.weight,
                    r.tf     = e.tf
                RETURN count(r) AS cnt
                """,
                batch=batch,
            )
            total += result.single()["cnt"]
    return total


def load_law_cites_case_edges(edges: list[dict], driver, batch_size: int = BATCH_SIZE * 2) -> int:
    """MERGE (:Law)-[:CITES]->(:Case) edges with TF-IDF weight."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(edges, batch_size)), desc="Loading Law→Case edges"):
            result = session.run(
                """
                UNWIND $batch AS e
                MATCH (s:Law {id: e.source})
                MATCH (t:Case {id: e.target})
                MERGE (s)-[r:CITES]->(t)
                SET r.weight = e.weight,
                    r.tf     = e.tf
                RETURN count(r) AS cnt
                """,
                batch=batch,
            )
            total += result.single()["cnt"]
    return total


def load_law_cites_law_edges(edges: list[dict], driver, batch_size: int = BATCH_SIZE * 2) -> int:
    """MERGE (:Law)-[:CITES_LAW]->(:Law) edges with TF-IDF weight."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(edges, batch_size)), desc="Loading Law→Law edges"):
            result = session.run(
                """
                UNWIND $batch AS e
                MATCH (s:Law {id: e.source})
                MATCH (t:Law {id: e.target})
                MERGE (s)-[r:CITES_LAW]->(t)
                SET r.weight = e.weight,
                    r.tf     = e.tf
                RETURN count(r) AS cnt
                """,
                batch=batch,
            )
            total += result.single()["cnt"]
    return total


# ---------------------------------------------------------------------------
# 4. High-level pipeline
# ---------------------------------------------------------------------------

def build_and_load(
    csv_path: Path,
    *,
    laws_csv_path: Path | None = None,
    max_rows: int | None = None,
    clear_first: bool = False,
    verbose: bool = True,
) -> dict:
    """End-to-end: parse CSVs → extract citations → load Neo4j.

    Args:
        csv_path: Path to ``court_considerations.csv``.
        laws_csv_path: Path to ``laws_de.csv``.  When provided, law provisions
            are loaded as ``:Law`` nodes and their internal citations are parsed
            to build additional edges.
        max_rows: Limit rows read from each CSV (for testing).
        clear_first: Wipe existing graph data before loading.
        verbose: Print progress.

    Returns:
        Dict of loading statistics.
    """
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        # Optionally clear existing data
        if clear_first:
            from .schema import clear_all_data
            print("Clearing existing data...")
            deleted = clear_all_data(driver)
            print(f"  Deleted {deleted} nodes.")

        # Create schema
        print("Creating schema...")
        create_schema(driver)

        # ---- Cases ----
        print("Parsing court_considerations CSV...")
        cases = parse_cases_from_csv(csv_path, max_rows=max_rows, verbose=verbose)

        # Build case-to-case edges
        case_edges = build_case_edges(cases, verbose=verbose)

        # Load Case nodes
        case_list = list(cases.values())
        for c in case_list:
            if len(c.get("text", "")) > 100_000:
                c["text"] = c["text"][:100_000] + "…"

        cases_loaded = load_case_nodes(case_list, driver)
        cites_loaded = load_cites_edges(case_edges, driver)

        stats: dict = {
            "cases_parsed": len(cases),
            "cases_loaded": cases_loaded,
            "cites_edges": cites_loaded,
        }

        # ---- Laws (optional) ----
        if laws_csv_path is not None:
            print(f"\nParsing laws CSV ({laws_csv_path.name})...")
            laws = parse_laws_from_csv(laws_csv_path, max_rows=max_rows, verbose=verbose)

            # Build law-node list (with text, title, sr_number)
            law_list = list(laws.values())
            for l_node in law_list:
                text = l_node.get("text", "")
                if len(text) > 100_000:
                    l_node["text"] = text[:100_000] + "…"

            laws_loaded = load_law_nodes(law_list, driver)

            # Also extract abbreviation-based Law nodes from case texts
            abbr_law_nodes, case_law_edges = build_law_nodes_and_edges(cases, verbose=verbose)
            abbr_laws_loaded = load_law_nodes(abbr_law_nodes, driver)
            case_cites_law = load_cites_law_edges(case_law_edges, driver)

            # Collect all known IDs for edge matching
            case_id_set = set(cases.keys())
            law_id_set = set(laws.keys()) | {n["id"] for n in abbr_law_nodes}

            # Extract citations from law texts
            law_to_case, law_to_law = build_law_citation_edges(
                laws, case_id_set, law_id_set, verbose=verbose,
            )
            l2c_loaded = load_law_cites_case_edges(law_to_case, driver)
            l2l_loaded = load_law_cites_law_edges(law_to_law, driver)

            stats.update({
                "laws_parsed": len(laws),
                "laws_loaded": laws_loaded,
                "abbr_law_nodes": abbr_laws_loaded,
                "case_cites_law_edges": case_cites_law,
                "law_to_case_edges": l2c_loaded,
                "law_to_law_edges": l2l_loaded,
            })

        if verbose:
            print("\n=== Load complete ===")
            for k, v in stats.items():
                print(f"  {k}: {v}")
        return stats

    finally:
        driver.close()


def export_edges_csv(
    csv_path: Path,
    out_path: Path,
    *,
    laws_csv_path: Path | None = None,
    max_rows: int | None = None,
    verbose: bool = True,
) -> Path:
    """Parse CSVs and write all edge lists to CSV files (no Neo4j required).

    Writes the case-to-case CITES edges to *out_path*.  When *laws_csv_path*
    is given, also writes ``law_to_case_edges.csv`` and ``law_to_law_edges.csv``
    next to *out_path*.
    """
    cases = parse_cases_from_csv(csv_path, max_rows=max_rows, verbose=verbose)
    case_edges = build_case_edges(cases, verbose=verbose)

    df = pd.DataFrame(case_edges)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote {len(df)} case-to-case edges to {out_path}")

    if laws_csv_path is not None:
        laws = parse_laws_from_csv(laws_csv_path, max_rows=max_rows, verbose=verbose)

        # Case→Law edges (TF-IDF weighted)
        abbr_law_nodes, case_law_edges = build_law_nodes_and_edges(cases, verbose=verbose)

        # Collect all known IDs for matching
        case_id_set = set(cases.keys())
        law_id_set = set(laws.keys()) | {n["id"] for n in abbr_law_nodes}

        # Law→Case and Law→Law edges (TF-IDF weighted)
        law_to_case, law_to_law = build_law_citation_edges(
            laws, case_id_set, law_id_set, verbose=verbose,
        )

        out_dir = out_path.parent
        if case_law_edges:
            cl_path = out_dir / "case_to_law_edges.csv"
            pd.DataFrame(case_law_edges).to_csv(cl_path, index=False)
            if verbose:
                print(f"Wrote {len(case_law_edges)} case→law edges to {cl_path}")
        if law_to_case:
            l2c_path = out_dir / "law_to_case_edges.csv"
            pd.DataFrame(law_to_case).to_csv(l2c_path, index=False)
            if verbose:
                print(f"Wrote {len(law_to_case)} law→case edges to {l2c_path}")
        if law_to_law:
            l2l_path = out_dir / "law_to_law_edges.csv"
            pd.DataFrame(law_to_law).to_csv(l2l_path, index=False)
            if verbose:
                print(f"Wrote {len(law_to_law)} law→law edges to {l2l_path}")

    return out_path
