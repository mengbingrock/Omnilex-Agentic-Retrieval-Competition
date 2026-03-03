"""Semantic reranker for second-pass relevance refinement.

Uses a cross-encoder model (e.g. BAAI/bge-reranker-v2-m3) to re-score
search results based on deep query–document interaction, then re-orders
them by relevance.

Typical workflow:
    1. Initial retrieval (BM25 or embedding) returns a broad candidate set.
    2. SemanticReranker re-scores that candidate set with a cross-encoder.
    3. Top results are returned in refined order.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class SemanticReranker:
    """Cross-encoder reranker for search result refinement.

    Wraps a ``sentence_transformers.CrossEncoder`` so callers only
    need to pass ``(query, documents)`` and get back reranked results.

    Args:
        model_name: HuggingFace model id (default: ``BAAI/bge-reranker-v2-m3``).
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``, …).
            ``None`` = auto-detect.
        batch_size: Texts scored per forward pass.
        max_length: Maximum token length for the cross-encoder input.
        normalize: If ``True``, apply sigmoid to raw logits so scores ∈ [0, 1].
    """

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or self.DEFAULT_MODEL
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(
            self.model_name,
            device=device,
            max_length=self.max_length,
        )

    def score(
        self,
        query: str,
        texts: Sequence[str],
    ) -> np.ndarray:
        """Score each text against the query.

        Args:
            query: The search query.
            texts: Candidate texts to score.

        Returns:
            1-D float array of scores, one per text.
        """
        if not texts:
            return np.array([], dtype=np.float32)

        pairs = [[query, t] for t in texts]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(raw_scores, dtype=np.float32)
        if self.normalize:
            scores = _sigmoid(scores)
        return scores

    def rerank(
        self,
        query: str,
        documents: list[dict],
        text_field: str = "text",
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank a list of document dicts by cross-encoder relevance.

        Args:
            query: The search query.
            documents: Candidate documents (each must have *text_field*).
            text_field: Key used to extract text from each document dict.
            top_k: Keep only the top-k results after reranking.
                ``None`` keeps all, reordered.

        Returns:
            List of document dicts in descending relevance order.
            Each dict gets an extra ``_rerank_score`` key.
        """
        if not documents:
            return []

        texts = [doc.get(text_field, "") for doc in documents]
        scores = self.score(query, texts)
        ranked_indices = np.argsort(scores)[::-1]

        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        results = []
        for idx in ranked_indices:
            doc = dict(documents[idx])
            doc["_rerank_score"] = float(scores[idx])
            results.append(doc)
        return results


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
