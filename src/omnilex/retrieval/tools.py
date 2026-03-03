"""LLM-compatible search tools for agentic retrieval.

These tools are designed to be used with ReAct-style agents or
LangChain-compatible tool interfaces.
"""

from typing import Protocol, runtime_checkable

from .bm25_index import BM25Index


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
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize law search tool.

        Args:
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
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize court search tool.

        Args:
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
        top_k_each: int = 3,
        max_excerpt_length: int = 250,
    ):
        """Initialize combined search tool.

        Args:
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


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools.

    Returns:
        String with tool names and descriptions for use in prompts
    """
    tools = [
        ("search_laws", LawSearchTool.description),
        ("search_courts", CourtSearchTool.description),
    ]

    lines = []
    for i, (name, desc) in enumerate(tools, 1):
        lines.append(f"{i}. {name}: {desc.strip()}")

    return "\n\n".join(lines)
