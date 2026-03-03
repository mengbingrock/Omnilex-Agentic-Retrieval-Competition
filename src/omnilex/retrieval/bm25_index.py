"""BM25 indexing and search for legal document corpora."""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 index for keyword search over legal documents.

    Supports Swiss federal laws (SR) and court decisions (BGE).
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
    ):
        """Initialize BM25 index.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text in dict
            citation_field: Key for citation string in dict
        """
        self.text_field = text_field
        self.citation_field = citation_field

        self.documents: list[dict] = []
        self.index: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

        if documents:
            self.build(documents)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing.

        Simple whitespace + lowercase tokenization.
        Can be overridden for language-specific tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Lowercase and split on non-alphanumeric characters
        text = text.lower()
        tokens = re.split(r"\W+", text)
        # Filter empty tokens
        return [t for t in tokens if t]

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
        """
        self.documents = documents

        # Tokenize all documents
        self._tokenized_corpus = []
        for doc in documents:
            text = doc.get(self.text_field, "")
            tokens = self.tokenize(text)
            self._tokenized_corpus.append(tokens)

        # Build BM25 index
        self.index = BM25Okapi(self._tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the index with a query.

        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: Whether to include BM25 scores in results

        Returns:
            List of matching documents (with optional scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Tokenize query
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue

            doc = self.documents[idx].copy()
            if return_scores:
                doc["_score"] = float(scores[idx])
            results.append(doc)

        return results

    def save(self, path: Path | str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index (creates .pkl file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self.documents,
            "tokenized_corpus": self._tokenized_corpus,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        """Load index from disk.

        Args:
            path: Path to saved index

        Returns:
            Loaded BM25Index instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
        )
        instance.documents = data["documents"]
        instance._tokenized_corpus = data["tokenized_corpus"]
        instance.index = BM25Okapi(instance._tokenized_corpus)

        return instance


def build_index(
    documents: list[dict],
    text_field: str = "text",
    citation_field: str = "citation",
) -> BM25Index:
    """Build a BM25 index from documents.

    Convenience function for quick index creation.

    Args:
        documents: List of document dictionaries
        text_field: Key for document text
        citation_field: Key for citation string

    Returns:
        Built BM25Index
    """
    return BM25Index(
        documents=documents,
        text_field=text_field,
        citation_field=citation_field,
    )


def search(
    index: BM25Index,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Search an index with a query.

    Convenience function for quick search.

    Args:
        index: BM25Index to search
        query: Search query string
        top_k: Number of results

    Returns:
        List of matching documents
    """
    return index.search(query, top_k=top_k)


def load_jsonl_corpus(path: Path | str) -> list[dict]:
    """Load a corpus from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of document dictionaries
    """
    path = Path(path)
    documents = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    return documents


def save_jsonl_corpus(documents: list[dict], path: Path | str) -> None:
    """Save a corpus to a JSONL file.

    Args:
        documents: List of document dictionaries
        path: Path to save JSONL file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")