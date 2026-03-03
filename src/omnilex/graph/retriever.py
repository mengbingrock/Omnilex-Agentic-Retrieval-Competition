"""Query the Neo4j citation graph for related cases.

Provides citation-based retrieval (1-hop, 2-hop neighbours) that can be
combined with BM25 or embedding search for hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from neo4j import GraphDatabase

from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


@dataclass
class RetrievedCase:
    """A case retrieved from the citation graph."""

    case_id: str
    citation_distance: int  # hops in the graph (0 = self, -1 = not connected)
    score: float = 0.0
    volume: int | None = None
    section: str | None = None
    page: int | None = None
    text: str = ""
    retrieval_method: str = "citation_graph"


class CitationGraphRetriever:
    """Retrieve related cases by traversing the citation graph in Neo4j."""

    def __init__(self, driver=None):
        if driver is None:
            self._driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self._own_driver = True
        else:
            self._driver = driver
            self._own_driver = False

    def close(self):
        if self._own_driver:
            self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Core retrieval
    # ------------------------------------------------------------------

    def get_neighbors(
        self,
        case_id: str,
        k: int = 20,
        max_hops: int = 2,
        include_text: bool = False,
    ) -> list[RetrievedCase]:
        """Return up to *k* cases within *max_hops*, ranked by TF-IDF edge weight.

        For 1-hop neighbours the score is the edge's TF-IDF weight.
        For 2-hop neighbours the score is the product of the two edge weights
        (attenuated by distance).
        """
        text_return = "n.text AS text," if include_text else "'' AS text,"

        two_hop_clause = ""
        if max_hops >= 2:
            two_hop_clause = """
                    UNION
                    WITH q
                    MATCH (q)-[r1:CITES]->()-[r2:CITES]->(c:Case)
                    WHERE c.id <> q.id
                    RETURN c, 2 AS dist,
                           coalesce(r1.weight, 1.0) * coalesce(r2.weight, 1.0) AS w
                    UNION
                    WITH q
                    MATCH (q)<-[r1:CITES]-()<-[r2:CITES]-(c:Case)
                    WHERE c.id <> q.id
                    RETURN c, 2 AS dist,
                           coalesce(r1.weight, 1.0) * coalesce(r2.weight, 1.0) AS w
            """

        with self._driver.session() as s:
            result = s.run(
                f"""
                MATCH (q:Case {{id: $qid}})
                CALL {{
                    WITH q
                    MATCH (q)-[r:CITES]->(c:Case)
                    RETURN c, 1 AS dist, coalesce(r.weight, 1.0) AS w
                    UNION
                    WITH q
                    MATCH (q)<-[r:CITES]-(c:Case)
                    RETURN c, 1 AS dist, coalesce(r.weight, 1.0) AS w
                    {two_hop_clause}
                }}
                WITH c, min(dist) AS d, max(w) AS best_w
                ORDER BY d ASC, best_w DESC
                LIMIT $k
                RETURN c.id AS id,
                       c.volume AS volume,
                       c.section AS section,
                       c.page AS page,
                       {text_return}
                       d AS distance,
                       best_w / d AS score
                """,
                qid=case_id,
                k=k,
            )
            return [
                RetrievedCase(
                    case_id=r["id"],
                    citation_distance=r["distance"],
                    score=r["score"],
                    volume=r["volume"],
                    section=r["section"],
                    page=r["page"],
                    text=r["text"] or "",
                )
                for r in result
            ]

    def get_citing_cases(self, case_id: str, limit: int = 100) -> list[str]:
        """Cases that cite *case_id* (direct inbound edges)."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (c:Case)-[:CITES]->(target:Case {id: $qid})
                RETURN c.id AS id
                ORDER BY c.volume DESC
                LIMIT $limit
                """,
                qid=case_id,
                limit=limit,
            )
            return [r["id"] for r in result]

    def get_cited_cases(self, case_id: str, limit: int = 100) -> list[str]:
        """Cases cited by *case_id* (direct outbound edges)."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (source:Case {id: $qid})-[:CITES]->(c:Case)
                RETURN c.id AS id
                ORDER BY c.volume DESC
                LIMIT $limit
                """,
                qid=case_id,
                limit=limit,
            )
            return [r["id"] for r in result]

    def get_cited_laws(self, case_id: str) -> list[str]:
        """Law provisions cited by *case_id*."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (source:Case {id: $qid})-[:CITES_LAW]->(l:Law)
                RETURN l.id AS id
                ORDER BY l.id
                """,
                qid=case_id,
            )
            return [r["id"] for r in result]

    def get_cases_citing_law(self, law_id: str, limit: int = 100) -> list[str]:
        """Cases that cite a given law provision."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (c:Case)-[:CITES_LAW]->(l:Law {id: $lid})
                RETURN c.id AS id
                ORDER BY c.volume DESC
                LIMIT $limit
                """,
                lid=law_id,
                limit=limit,
            )
            return [r["id"] for r in result]

    # ------------------------------------------------------------------
    # Co-citation analysis
    # ------------------------------------------------------------------

    def co_cited_cases(self, case_id: str, k: int = 20) -> list[tuple[str, int]]:
        """Find cases frequently co-cited with *case_id*.

        Returns (case_id, co_citation_count) pairs sorted by count descending.
        Two cases are co-cited when a third case cites both of them.
        """
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (target:Case {id: $qid})<-[:CITES]-(citing:Case)-[:CITES]->(other:Case)
                WHERE other.id <> $qid
                WITH other.id AS id, count(DISTINCT citing) AS cnt
                ORDER BY cnt DESC
                LIMIT $k
                RETURN id, cnt
                """,
                qid=case_id,
                k=k,
            )
            return [(r["id"], r["cnt"]) for r in result]

    # ------------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------------

    def case_exists(self, case_id: str) -> bool:
        with self._driver.session() as s:
            result = s.run("MATCH (c:Case {id: $qid}) RETURN count(c) AS cnt", qid=case_id)
            return result.single()["cnt"] > 0

    def graph_stats(self) -> dict:
        """Summary statistics of the loaded graph."""
        with self._driver.session() as s:
            cases = s.run("MATCH (n:Case) RETURN count(n) AS c").single()["c"]
            laws = s.run("MATCH (n:Law) RETURN count(n) AS c").single()["c"]
            cites = s.run("MATCH ()-[r:CITES]->() RETURN count(r) AS c").single()["c"]
            cites_law = s.run("MATCH ()-[r:CITES_LAW]->() RETURN count(r) AS c").single()["c"]

            # Degree distribution (top-cited cases)
            top_cited = s.run(
                """
                MATCH (c:Case)<-[r:CITES]-()
                WITH c.id AS id, count(r) AS in_degree
                ORDER BY in_degree DESC
                LIMIT 10
                RETURN id, in_degree
                """
            )
            top_cited_list = [(r["id"], r["in_degree"]) for r in top_cited]

        return {
            "cases": cases,
            "laws": laws,
            "cites_edges": cites,
            "cites_law_edges": cites_law,
            "top_cited_cases": top_cited_list,
        }
