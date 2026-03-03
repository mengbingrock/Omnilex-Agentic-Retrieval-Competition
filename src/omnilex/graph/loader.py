"""Parse court_considerations.csv, extract citations, and bulk-load the citation graph into Neo4j.

Pipeline
--------
1.  Read the CSV (``citation``, ``text`` columns).
2.  Group rows by case-level BGE ID (e.g. "BGE 139 I 2").
3.  Concatenate all consideration texts per case → the node's ``text``.
4.  For each case, extract outgoing BGE citations from the text → edges.
5.  Optionally extract law article citations → :Law nodes + :CITES_LAW edges.
6.  Bulk-insert nodes and edges into Neo4j.
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
    """Extract :Law nodes and :CITES_LAW edges from case texts.

    Returns:
        (law_nodes, law_edges) where law_nodes is ``[{"id": ..., "book": ...}]``
        and law_edges is ``[{"source": case_id, "target": law_id}]``.
    """
    law_map: dict[str, dict] = {}
    edges: list[dict] = []

    book_re = re.compile(r"Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+([A-Za-z/]{2,})")

    for case_id, case in tqdm(cases.items(), desc="Extracting law citations", disable=not verbose):
        cites = extract_citations(case["text"])
        for c in cites:
            if c.citation_type != "Art.":
                continue
            law_id = c.canonical_id
            if law_id not in law_map:
                bm = book_re.search(law_id)
                law_map[law_id] = {
                    "id": law_id,
                    "book": bm.group(1) if bm else None,
                }
            edges.append({"source": case_id, "target": law_id})

    law_nodes = list(law_map.values())
    if verbose:
        print(f"Found {len(law_nodes)} unique law provisions, {len(edges)} CITES_LAW edges")
    return law_nodes, edges


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
    """MERGE :Law nodes."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(laws, batch_size)), desc="Loading Law nodes"):
            result = session.run(
                """
                UNWIND $batch AS l
                MERGE (n:Law {id: l.id})
                SET n.book = l.book
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
    """MERGE (:Case)-[:CITES_LAW]->(:Law) edges."""
    total = 0
    with driver.session() as session:
        for batch in tqdm(list(_batched(edges, batch_size)), desc="Loading CITES_LAW edges"):
            result = session.run(
                """
                UNWIND $batch AS e
                MATCH (s:Case {id: e.source})
                MATCH (t:Law {id: e.target})
                MERGE (s)-[r:CITES_LAW]->(t)
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
    max_rows: int | None = None,
    clear_first: bool = False,
    verbose: bool = True,
) -> dict:
    """End-to-end: parse CSV → extract case-to-case citations → load Neo4j.

    Args:
        csv_path: Path to ``court_considerations.csv``.
        max_rows: Limit rows read from CSV (for testing).
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

        # Parse
        print("Parsing CSV...")
        cases = parse_cases_from_csv(csv_path, max_rows=max_rows, verbose=verbose)

        # Build case-to-case edges
        case_edges = build_case_edges(cases, verbose=verbose)

        # Load nodes
        case_list = list(cases.values())
        for c in case_list:
            if len(c.get("text", "")) > 100_000:
                c["text"] = c["text"][:100_000] + "…"

        cases_loaded = load_case_nodes(case_list, driver)

        # Load edges
        cites_loaded = load_cites_edges(case_edges, driver)

        stats = {
            "cases_parsed": len(cases),
            "cases_loaded": cases_loaded,
            "cites_edges": cites_loaded,
        }
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
    max_rows: int | None = None,
    verbose: bool = True,
) -> Path:
    """Parse CSV and write the CITES edge list to a CSV file (no Neo4j required).

    Useful for offline analysis or import via ``LOAD CSV``.
    """
    cases = parse_cases_from_csv(csv_path, max_rows=max_rows, verbose=verbose)
    case_edges = build_case_edges(cases, verbose=verbose)

    df = pd.DataFrame(case_edges)
    df.to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote {len(df)} edges to {out_path}")
    return out_path
