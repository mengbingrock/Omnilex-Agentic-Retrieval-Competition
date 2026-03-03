"""LLM-compatible search tools for agentic retrieval.

These tools are designed to be used with ReAct-style agents or
LangChain-compatible tool interfaces.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .bm25_index import BM25Index

if TYPE_CHECKING:
    from omnilex.graph.retriever import CitationGraphRetriever

logger = logging.getLogger(__name__)


@runtime_checkable
class SearchIndex(Protocol):
    """Protocol for search backends (BM25Index, EmbeddingIndex, etc.)."""

    documents: list

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]: ...
    def save(self, path) -> None: ...


class LawSearchTool:
    """Tool for searching Swiss federal laws corpus.

    Searches the SR (Systematische Rechtssammlung) collection
    using BM25 keyword matching.
    """

    name: str = "search_laws"
    description: str = """Search Swiss federal laws (SR/Systematische Rechtssammlung) by keywords.
Input: Search query string (can be in German, French, Italian, or English)
Output: List of relevant law citations with text excerpts

Use this tool to find relevant federal law provisions for a legal question.
Example queries: "contract formation requirements", "Vertragsabschluss", "divorce grounds"
"""

    def __init__(
        self,
        index: SearchIndex | BM25Index,
        index: SearchIndex | BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize law search tool.

        Args:
            index: Search index for federal laws (BM25Index or EmbeddingIndex)
            index: Search index for federal laws (BM25Index or EmbeddingIndex)
            top_k: Number of results to return
            max_excerpt_length: Maximum characters for text excerpts
        """
        self.index = index
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        if not query or not query.strip():
            self._last_results = []
            return "Error: Empty query. Please provide search terms."

        results = self.index.search(query, top_k=self.top_k)
        self._last_results = results

        if not results:
            return f"No relevant federal laws found for: '{query}'"

        formatted = []
        for doc in results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

            # Truncate text for readability
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."

            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search.

        Returns:
            List of citation strings from the most recent search
        """
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

    def search_with_metadata(self, query: str) -> list[dict]:
        """Execute search and return full result objects.

        Args:
            query: Search query string

        Returns:
            List of result dictionaries with full metadata
        """
        return self.index.search(query, top_k=self.top_k, return_scores=True)


class CourtSearchTool:
    """Tool for searching Swiss Federal Court decisions corpus.

    Searches court decisions (BGE and docket-style citations)
    using BM25 keyword matching.
    """

    name: str = "search_courts"
    description: str = """Search Swiss Federal Court decisions by keywords.
Input: Search query string (German, French, Italian, or English)
Output: List of relevant court decision citations with excerpts

Use this tool to find relevant case law and judicial interpretations.
Example queries: "negligence standard of care", "Sorgfaltspflicht", "contract interpretation"
"""

    def __init__(
        self,
        index: SearchIndex | BM25Index,
        index: SearchIndex | BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize court search tool.

        Args:
            index: Search index for court decisions (BM25Index or EmbeddingIndex)
            index: Search index for court decisions (BM25Index or EmbeddingIndex)
            top_k: Number of results to return
            max_excerpt_length: Maximum characters for text excerpts
        """
        self.index = index
        self.top_k = top_k
        self.max_excerpt_length = max_excerpt_length
        self._last_results: list[dict] = []

    def __call__(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        if not query or not query.strip():
            self._last_results = []
            return "Error: Empty query. Please provide search terms."

        results = self.index.search(query, top_k=self.top_k)
        self._last_results = results

        if not results:
            return f"No relevant court decisions found for: '{query}'"

        formatted = []
        for doc in results:
            citation = doc.get("citation", "Unknown")
            text = doc.get("text", "")

            # Truncate text for readability
            if len(text) > self.max_excerpt_length:
                text = text[: self.max_excerpt_length] + "..."

            formatted.append(f"- {citation}: {text}")

        return "\n".join(formatted)

    def get_last_citations(self) -> list[str]:
        """Return citations from the last search.

        Returns:
            List of citation strings from the most recent search
        """
        return [doc.get("citation", "") for doc in self._last_results if doc.get("citation")]

    def search_with_metadata(self, query: str) -> list[dict]:
        """Execute search and return full result objects.

        Args:
            query: Search query string

        Returns:
            List of result dictionaries with full metadata
        """
        return self.index.search(query, top_k=self.top_k, return_scores=True)


class CombinedSearchTool:
    """Tool that searches both laws and court decisions.

    Useful for comprehensive legal research across all sources.
    """

    name: str = "search_all"
    description: str = """Search both Swiss federal laws (SR) and Federal Court decisions (BGE).
Input: Search query string
Output: Combined list of relevant citations from both corpora

Use this for comprehensive research when you need both statutory law and case law.
"""

    def __init__(
        self,
        law_index: SearchIndex | BM25Index,
        court_index: SearchIndex | BM25Index,
        law_index: SearchIndex | BM25Index,
        court_index: SearchIndex | BM25Index,
        top_k_each: int = 3,
        max_excerpt_length: int = 250,
    ):
        """Initialize combined search tool.

        Args:
            law_index: Search index for federal laws (BM25Index or EmbeddingIndex)
            court_index: Search index for court decisions (BM25Index or EmbeddingIndex)
            law_index: Search index for federal laws (BM25Index or EmbeddingIndex)
            court_index: Search index for court decisions (BM25Index or EmbeddingIndex)
            top_k_each: Number of results from each corpus
            max_excerpt_length: Maximum characters for excerpts
        """
        self.law_tool = LawSearchTool(
            law_index,
            top_k=top_k_each,
            max_excerpt_length=max_excerpt_length,
        )
        self.court_tool = CourtSearchTool(
            court_index,
            top_k=top_k_each,
            max_excerpt_length=max_excerpt_length,
        )

    def __call__(self, query: str) -> str:
        """Execute search on both corpora.

        Args:
            query: Search query string

        Returns:
            Combined formatted results
        """
        return self.run(query)

    def run(self, query: str) -> str:
        """Execute search on both corpora.

        Args:
            query: Search query string

        Returns:
            Combined formatted results
        """
        law_results = self.law_tool.run(query)
        court_results = self.court_tool.run(query)

        output = [
            "=== Federal Laws (SR) ===",
            law_results,
            "",
            "=== Court Decisions ===",
            court_results,
        ]

        return "\n".join(output)


class CitationExplorerTool:
    """Tool for exploring citation relationships of a specific law or case.

    Given a citation ID (law provision or court decision), queries the Neo4j
    citation graph to find:
    1. What cites this entity most (top inbound citations by TF-IDF weight)
    2. What this entity cites most (top outbound citations by TF-IDF weight)
    """

    name: str = "explore_citations"
    description: str = """Explore citation relationships of a specific law or court decision.
Input: A citation ID (e.g. "Art. 34 BV", "BGE 139 I 2")
Output: Two lists — what cites this entity most, and what this entity cites most.

Use this tool to understand the citation network around a known law or case.
Example inputs: "Art. 41 Abs. 1 OR", "BGE 145 II 32", "Art. 1 ZGB"
"""

    def __init__(
        self,
        graph_retriever: CitationGraphRetriever,
        top_k: int = 5,
    ):
        """Initialize citation explorer tool.

        Args:
            graph_retriever: Neo4j graph retriever for citation queries
            top_k: Number of results per direction (inbound + outbound)
        """
        self.graph_retriever = graph_retriever
        self.top_k = top_k

    def __call__(self, citation_id: str) -> str:
        return self.run(citation_id)

    def run(self, citation_id: str) -> str:
        """Explore citation relationships for a given citation ID.

        Args:
            citation_id: Law provision or court decision ID

        Returns:
            Formatted string with inbound and outbound citation info
        """
        if not citation_id or not citation_id.strip():
            return "Error: Empty citation ID. Please provide a law or case citation."

        citation_id = citation_id.strip()

        try:
            inbound = self.graph_retriever.get_top_inbound(citation_id, k=self.top_k)
            outbound = self.graph_retriever.get_top_outbound(citation_id, k=self.top_k)
        except Exception as exc:
            logger.debug("Citation explorer failed for %s: %s", citation_id, exc)
            return f"Error querying graph for '{citation_id}': {exc}"

        sections = [f"=== Citation Explorer: {citation_id} ==="]

        # Inbound: what cites this
        sections.append(f"\n--- Cited BY (top {self.top_k}) ---")
        if inbound:
            for r in inbound:
                sections.append(
                    f"  {r['id']} [{r['type']}] (weight={r['weight']:.2f}, tf={r['tf']})"
                )
        else:
            sections.append("  (no inbound citations found)")

        # Outbound: what this cites
        sections.append(f"\n--- CITES (top {self.top_k}) ---")
        if outbound:
            for r in outbound:
                sections.append(
                    f"  {r['id']} [{r['type']}] (weight={r['weight']:.2f}, tf={r['tf']})"
                )
        else:
            sections.append("  (no outbound citations found)")

        return "\n".join(sections)

    def search_with_metadata(self, citation_id: str) -> dict:
        """Return structured inbound/outbound citation data.

        Args:
            citation_id: Law provision or court decision ID

        Returns:
            Dict with ``"citation_id"``, ``"inbound"``, ``"outbound"`` keys
        """
        citation_id = (citation_id or "").strip()
        try:
            inbound = self.graph_retriever.get_top_inbound(citation_id, k=self.top_k)
            outbound = self.graph_retriever.get_top_outbound(citation_id, k=self.top_k)
        except Exception as exc:
            logger.debug("Citation explorer failed for %s: %s", citation_id, exc)
            inbound, outbound = [], []

        return {
            "citation_id": citation_id,
            "inbound": inbound,
            "outbound": outbound,
        }


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools.

    Returns:
        String with tool names and descriptions for use in prompts
    """
    tools = [
        ("search_laws", LawSearchTool.description),
        ("search_courts", CourtSearchTool.description),
        ("explore_citations", CitationExplorerTool.description),
    ]

    lines = []
    for i, (name, desc) in enumerate(tools, 1):
        lines.append(f"{i}. {name}: {desc.strip()}")

    return "\n\n".join(lines)
