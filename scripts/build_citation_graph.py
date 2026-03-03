#!/usr/bin/env python3
"""Build the Swiss legal citation graph in Neo4j.

Usage
-----
    # Full pipeline: start Neo4j, parse CSV, load graph
    python scripts/build_citation_graph.py

    # Quick test with first 100 CSV rows
    python scripts/build_citation_graph.py --max-rows 100

    # Export edges to CSV without Neo4j
    python scripts/build_citation_graph.py --export-only

    # Skip auto-start (Neo4j already running)
    python scripts/build_citation_graph.py --no-start
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as ``python scripts/build_citation_graph.py`` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from omnilex.graph import (
    build_and_load,
    check_health,
    export_edges_csv,
    get_stats,
    start_neo4j,
)


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "court_considerations.csv"
DEFAULT_EDGES_OUT = DATA_DIR / "processed" / "citation_edges.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Swiss legal citation graph in Neo4j.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to court_considerations.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit CSV rows to read (for testing).",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe existing graph before loading.",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Skip starting Neo4j (assume it is already running).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export edge CSV; do not load into Neo4j.",
    )
    parser.add_argument(
        "--edges-out",
        type=Path,
        default=DEFAULT_EDGES_OUT,
        help=f"Output path for edge CSV (default: {DEFAULT_EDGES_OUT})",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"ERROR: CSV not found at {args.csv}")
        raise SystemExit(1)

    # ---- Export-only mode ----
    if args.export_only:
        args.edges_out.parent.mkdir(parents=True, exist_ok=True)
        export_edges_csv(args.csv, args.edges_out, max_rows=args.max_rows)
        return

    # ---- Ensure Neo4j is running ----
    if not args.no_start:
        print("=== Starting Neo4j (local) ===")
        if not start_neo4j():
            print("ERROR: Could not start Neo4j. Make sure it is installed (brew install neo4j).")
            raise SystemExit(1)
    else:
        if not check_health():
            print("ERROR: Neo4j is not reachable. Start it with 'neo4j start' or remove --no-start.")
            raise SystemExit(1)

    # ---- Build & load ----
    print(f"\n=== Building citation graph from {args.csv} ===")
    stats = build_and_load(
        args.csv,
        max_rows=args.max_rows,
        clear_first=args.clear,
        verbose=True,
    )

    # ---- Print summary ----
    print("\n=== Neo4j graph summary ===")
    try:
        for k, v in get_stats().items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"  (could not fetch stats: {exc})")


if __name__ == "__main__":
    main()
