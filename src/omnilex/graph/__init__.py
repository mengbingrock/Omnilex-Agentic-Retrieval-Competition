"""Citation graph module – Neo4j-backed graph of Swiss BGE court decisions.

Quick start::

    from omnilex.graph import CitationGraphRetriever, build_and_load

    # Build the graph from the CSV (one-time)
    build_and_load(Path("data/court_considerations.csv"))

    # Query it
    with CitationGraphRetriever() as r:
        neighbours = r.get_neighbors("BGE 139 I 2", k=10)
"""

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from .docker_setup import (
    check_health,
    get_connection,
    get_stats,
    is_container_running,
    start_neo4j,
    stop_neo4j,
)
from .extractor import (
    ExtractedCitation,
    count_case_citations,
    extract_all_case_citations,
    extract_bge_citations,
    extract_citations,
    parse_case_id_from_csv_citation,
)
from .loader import (
    build_and_load,
    build_case_edges,
    export_edges_csv,
    parse_cases_from_csv,
)
from .retriever import CitationGraphRetriever, RetrievedCase
from .schema import clear_all_data, create_schema, drop_schema, get_schema_info

__all__ = [
    # config
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    # docker
    "start_neo4j",
    "stop_neo4j",
    "is_container_running",
    "check_health",
    "get_connection",
    "get_stats",
    # extractor
    "ExtractedCitation",
    "count_case_citations",
    "extract_citations",
    "extract_bge_citations",
    "extract_all_case_citations",
    "parse_case_id_from_csv_citation",
    # loader
    "parse_cases_from_csv",
    "build_case_edges",
    "build_and_load",
    "export_edges_csv",
    # retriever
    "CitationGraphRetriever",
    "RetrievedCase",
    # schema
    "create_schema",
    "drop_schema",
    "clear_all_data",
    "get_schema_info",
]
