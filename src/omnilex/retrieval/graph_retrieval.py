"""Graph-based retrieval index using the Neo4j citation graph.

Finds related cases by traversing CITES edges in the citation graph.
Compatible with the SearchIndex protocol (same API as BM25Index / EmbeddingIndex).

Search queries can be:
- A case ID (e.g. ``"BGE 139 I 2"``) → direct graph traversal.
- Free text containing BGE citations → extracts citation IDs as seed nodes.
- Any text → falls back to an optional BM25/embedding index to identify seed cases.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from ..graph.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
from ..graph.extractor import extract_bge_citations, parse_case_id_from_csv_citation
from ..graph.retriever import CitationGraphRetriever, RetrievedCase

if TYPE_CHECKING:
    from neo4j import Driver


class GraphRetrievalIndex:
    """Citation-graph retrieval with the same interface as BM25Index / EmbeddingIndex.

    Given a query (case ID or text), retrieves the most related cases
    by traversing the Neo4j citation graph (1-hop and 2-hop neighbours,
    ranked by TF-IDF edge weight).

    Typical usage::

        idx = GraphRetrievalIndex()
        idx.build(court_docs)          # registers docs so results carry full metadata

        results = idx.search("BGE 139 I 2", top_k=10)
        for doc in results:
            print(doc["citation"], doc["_score"])
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        *,
        driver: "Driver | None" = None,
        max_hops: int = 2,
        include_text_from_graph: bool = True,
        fallback_index=None,
    ):
        """
        Args:
            documents: Optional list of document dicts to register immediately.
            text_field: Key for the document text in each dict.
            citation_field: Key for the citation string in each dict.
            driver: An existing Neo4j driver; one is created automatically if *None*.
            max_hops: Maximum graph traversal depth (1 or 2).
            include_text_from_graph: Fetch case text from Neo4j when the doc
                is not in the local document store.
            fallback_index: Optional BM25Index or EmbeddingIndex used to map
                free-text queries to seed case IDs when no BGE citation is found
                in the query string.
        """
        self.text_field = text_field
        self.citation_field = citation_field
        self.max_hops = max_hops
        self.include_text_from_graph = include_text_from_graph
        self.fallback_index = fallback_index

        self.documents: list[dict] = []
        self._case_id_to_idx: dict[str, int] = {}

        self._driver = driver
        self._own_driver = driver is None

        if documents:
            self.build(documents)

    def _get_retriever(self) -> CitationGraphRetriever:
        if self._driver is not None:
            return CitationGraphRetriever(driver=self._driver)
        return CitationGraphRetriever()

    def build(
        self,
        documents: list[dict],
        *,
        show_progress: bool = False,
        **_kwargs,
    ) -> None:
        """Register documents and build the case-ID → document mapping.

        Unlike BM25 / embedding indexes this does not compute a local index
        structure — the graph already lives in Neo4j.  ``build()`` only stores
        the document metadata so that ``search()`` can return full result dicts.
        """
        self.documents = documents
        self._case_id_to_idx = {}

        for i, doc in enumerate(documents):
            citation = doc.get(self.citation_field, "")
            case_id = parse_case_id_from_csv_citation(citation)
            if case_id:
                self._case_id_to_idx[case_id] = i

    def _extract_seed_case_ids(self, query: str) -> list[str]:
        """Derive one or more seed case IDs from a query string.

        Strategy (ordered by priority):
        1. If the query itself is a canonical case ID (e.g. "BGE 139 I 2") use it.
        2. Extract all BGE citations from the query text.
        3. If a fallback index is provided, search it and take citations from the
           top result(s).
        """
        case_id = parse_case_id_from_csv_citation(query)
        if case_id:
            return [case_id]

        bge_ids = extract_bge_citations(query)
        if bge_ids:
            return bge_ids

        if self.fallback_index is not None:
            try:
                fb_results = self.fallback_index.search(query, top_k=3)
                seeds: list[str] = []
                for doc in fb_results:
                    cit = doc.get(self.citation_field, "")
                    cid = parse_case_id_from_csv_citation(cit)
                    if cid and cid not in seeds:
                        seeds.append(cid)
                if seeds:
                    return seeds
            except Exception:
                pass

        return []

    def _retrieved_case_to_doc(
        self, rc: RetrievedCase, return_scores: bool
    ) -> dict:
        """Convert a ``RetrievedCase`` into the standard document dict format."""
        idx = self._case_id_to_idx.get(rc.case_id)
        if idx is not None:
            doc = self.documents[idx].copy()
        else:
            doc = {
                self.citation_field: rc.case_id,
                self.text_field: rc.text,
            }
            if rc.volume is not None:
                doc["volume"] = rc.volume
            if rc.section is not None:
                doc["section"] = rc.section
            if rc.page is not None:
                doc["page"] = rc.page

        doc["_citation_distance"] = rc.citation_distance
        doc["_retrieval_method"] = rc.retrieval_method

        if return_scores:
            doc["_score"] = rc.score

        return doc

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the citation graph for cases related to *query*.

        Args:
            query: A case ID (e.g. ``"BGE 139 I 2"``), a text containing
                BGE citations, or free text (requires ``fallback_index``).
            top_k: Maximum number of results.
            return_scores: Attach ``_score`` to each result dict.

        Returns:
            List of document dicts, ordered by graph relevance (TF-IDF weight
            attenuated by hop distance).
        """
        if not query or not query.strip():
            return []

        seed_ids = self._extract_seed_case_ids(query.strip())
        if not seed_ids:
            return []

        retriever = self._get_retriever()
        try:
            seen: set[str] = set(seed_ids)
            all_results: list[RetrievedCase] = []

            for seed_id in seed_ids:
                neighbours = retriever.get_neighbors(
                    seed_id,
                    k=top_k,
                    max_hops=self.max_hops,
                    include_text=self.include_text_from_graph,
                )
                for rc in neighbours:
                    if rc.case_id not in seen:
                        seen.add(rc.case_id)
                        all_results.append(rc)

            all_results.sort(key=lambda r: r.score, reverse=True)
            all_results = all_results[:top_k]

            return [
                self._retrieved_case_to_doc(rc, return_scores)
                for rc in all_results
            ]
        finally:
            if self._own_driver:
                retriever.close()

    def save(self, path: Path | str) -> None:
        """Save the document index to disk.

        Only persists documents and the case-ID mapping — the graph itself
        lives in Neo4j.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "documents": self.documents,
            "case_id_to_idx": self._case_id_to_idx,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "max_hops": self.max_hops,
            "include_text_from_graph": self.include_text_from_graph,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        driver: "Driver | None" = None,
        fallback_index=None,
    ) -> "GraphRetrievalIndex":
        """Load a saved document index from disk.

        Args:
            path: Path to the saved ``.pkl`` file.
            driver: Optional Neo4j driver.
            fallback_index: Optional fallback search index for free-text queries.
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
            driver=driver,
            max_hops=data.get("max_hops", 2),
            include_text_from_graph=data.get("include_text_from_graph", True),
            fallback_index=fallback_index,
        )
        instance.documents = data["documents"]
        instance._case_id_to_idx = data["case_id_to_idx"]
        return instance
