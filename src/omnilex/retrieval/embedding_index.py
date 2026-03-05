"""OpenAI embedding-based index for semantic search over legal document corpora.

Memory-efficient implementation:
- Pre-allocates a numpy matrix and fills it batch-by-batch (no Python list accumulator).
- Streams chunk expansion so documents + chunk_docs are never both fully in memory.
- Supports incremental checkpointing so a crash doesn't lose hours of API work.
- Retries transient network / API errors with exponential back-off.
"""

import logging
import os
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tiktoken
from tqdm import tqdm

from ..config import EMBEDDING_BACKEND, EMBEDDING_MODEL

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from huggingface_hub import InferenceClient
    from openai import OpenAI

logger = logging.getLogger(__name__)

# Max input tokens per OpenAI embedding model
_MODEL_MAX_TOKENS: dict[str, int] = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191,
}

# Embedding dimensions per model
_MODEL_EMBEDDING_DIM: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# Retry configuration for transient errors
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each retry

_DEFAULT_RECURSIVE_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")


def _get_client() -> "OpenAI":
    """Lazy import to avoid requiring openai at import time when using BM25 only."""
    from openai import OpenAI
    return OpenAI()


def _get_huggingface_client() -> "InferenceClient":
    """Lazy import to avoid requiring huggingface_hub unless needed."""
    from huggingface_hub import InferenceClient

    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    return InferenceClient(token=token)


def _get_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _get_local_transformer_client(model_name: str) -> Any:
    """Load a local sentence-transformers model for embeddings (GPU when available)."""
    try:
        from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Local transformer backend requires sentence-transformers. "
            "Install it or set OMNILEX_EMBEDDING_BACKEND=auto."
        ) from exc
    device = _get_device()
    logger.info("Using device %s for embedding model %s", device, model_name)
    return SentenceTransformer(model_name, device=device)


class EmbeddingIndex:
    """Semantic search index using OpenAI or Hugging Face embeddings.

    Compatible with the same interface as BM25Index: build(), search(), save(), load().
    Supports Swiss federal laws (SR) and court decisions (BGE).
    """

    DEFAULT_MODEL = EMBEDDING_MODEL
    BATCH_SIZE = 100  # API accepts multiple inputs per request

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        *,
        model: str | None = None,
        openai_client: "OpenAI | None" = None,
        batch_size: int = BATCH_SIZE,
        rate_limit_delay: float = 0.0,
        show_progress: bool = True,
        progress_desc: str = "Building embedding index",
    ):
        """Initialize embedding index.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text in dict
            citation_field: Key for citation string in dict
            model: OpenAI embedding model (default: text-embedding-3-small)
            openai_client: Optional OpenAI client (uses default from env if not set)
            batch_size: Number of texts to embed per API call
            rate_limit_delay: Seconds to wait between batch API calls
            show_progress: Whether to show progress bar when building from documents
            progress_desc: Label for the progress bar (e.g. "Laws", "Courts")
        """
        self.text_field = text_field
        self.citation_field = citation_field
        self.model = model or self.DEFAULT_MODEL
        self._backend = self._resolve_backend(self.model)
        # Only eagerly set the client when one is explicitly provided.
        # All backends lazily create a client via the `client` property on
        # first use (e.g. search or build), avoiding heavy model loads at
        # import / load time.
        if self._backend == "openai" and openai_client is not None:
            self._client = openai_client
        else:
            self._client = None
        self.batch_size = batch_size
        # No rate limit for local model (hf_local); only throttle API backends.
        self.rate_limit_delay = 0.0 if self._backend == "hf_local" else rate_limit_delay

        self.documents: list[dict] = []
        self._embeddings: np.ndarray | None = None  # shape (n_docs, dim); used when faiss unavailable
        self._faiss_index: Any = None  # faiss.IndexFlatIP when faiss is used
        self._resolved_embedding_dim: int | None = None

        if documents:
            self.build(documents, show_progress=show_progress, progress_desc=progress_desc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            if self._backend == "hf_inference":
                self._client = _get_huggingface_client()
            elif self._backend == "hf_local":
                self._client = _get_local_transformer_client(self.model)
            else:
                self._client = _get_client()
        return self._client

    @property
    def embedding_dim(self) -> int:
        if self._resolved_embedding_dim is not None:
            return self._resolved_embedding_dim
        if self.model in _MODEL_EMBEDDING_DIM:
            self._resolved_embedding_dim = _MODEL_EMBEDDING_DIM[self.model]
            return self._resolved_embedding_dim

        # For unknown models (typically HF ids), infer the vector size once.
        sample_embedding = self._embed_batch_with_retry(["dimension probe"])[0]
        self._resolved_embedding_dim = len(sample_embedding)
        return self._resolved_embedding_dim

    @staticmethod
    def _is_huggingface_model(model_name: str) -> bool:
        """Heuristic: HF models usually look like `org/model-name`."""
        return "/" in model_name and not model_name.startswith("text-embedding-")

    def _resolve_backend(self, model_name: str) -> str:
        """Resolve backend: openai, hf_inference, or hf_local."""
        if not self._is_huggingface_model(model_name):
            return "openai"
        # For HF models: auto and local_transformer both use local sentence-transformers.
        if EMBEDDING_BACKEND.strip().lower() in ("auto", "local_transformer"):
            return "hf_local"
        return "hf_inference"

    def _get_encoder(self):
        """Get tiktoken encoding for the current model."""
        try:
            return tiktoken.encoding_for_model(self.model)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _chunk_text(self, text: str, overlap_tokens: int = 0) -> list[str]:
        """Split text with recursive chunking adapted from third-party baseline.

        Args:
            text: Text to chunk
            overlap_tokens: Number of tokens to overlap between consecutive chunks

        Returns:
            List of text chunks, each within the token limit
        """
        if not text or not text.strip():
            return [""]
        max_tokens = _MODEL_MAX_TOKENS.get(self.model, 8191)
        enc = self._get_encoder()
        n_tokens = len(enc.encode(text))
        if n_tokens <= max_tokens:
            return [text]

        segments = self._recursive_split(
            text=text,
            enc=enc,
            max_tokens=max_tokens,
            separators=list(_DEFAULT_RECURSIVE_SEPARATORS),
            sep_idx=0,
            forced_overlap=overlap_tokens,
        )
        return self._merge_recursive_segments(
            segments=segments,
            max_tokens=max_tokens,
            chunk_overlap=overlap_tokens,
        )

    def _force_token_based_splits(
        self,
        text: str,
        enc,
        max_tokens: int,
        chunk_overlap: int,
    ) -> list[dict[str, str | int]]:
        """Fallback: strict token-window splits when separators are exhausted."""
        tokens = enc.encode(text)
        n_tokens = len(tokens)
        if n_tokens == 0:
            return [{"text": "", "n_tokens": 0}]

        segments: list[dict[str, str | int]] = []
        start_tok = 0
        step = max(1, max_tokens - chunk_overlap)

        while start_tok < n_tokens:
            end_tok = min(start_tok + max_tokens, n_tokens)
            seg_tokens = tokens[start_tok:end_tok]
            seg_text = enc.decode(seg_tokens)
            segments.append({"text": seg_text, "n_tokens": end_tok - start_tok})
            if end_tok == n_tokens:
                break
            start_tok += step

        return segments

    def _recursive_split(
        self,
        text: str,
        enc,
        max_tokens: int,
        separators: list[str],
        sep_idx: int,
        forced_overlap: int,
    ) -> list[dict[str, str | int]]:
        """Recursively split text into token-bounded segments."""
        n_tokens = len(enc.encode(text))
        if n_tokens <= max_tokens:
            return [{"text": text, "n_tokens": n_tokens}]

        if sep_idx >= len(separators):
            return self._force_token_based_splits(
                text=text,
                enc=enc,
                max_tokens=max_tokens,
                chunk_overlap=forced_overlap,
            )

        sep = separators[sep_idx]
        if sep not in text:
            return self._recursive_split(
                text=text,
                enc=enc,
                max_tokens=max_tokens,
                separators=separators,
                sep_idx=sep_idx + 1,
                forced_overlap=forced_overlap,
            )

        pieces: list[str] = []
        cursor = 0
        sep_len = len(sep)
        text_len = len(text)
        while cursor < text_len:
            idx = text.find(sep, cursor)
            if idx == -1:
                if cursor < text_len:
                    pieces.append(text[cursor:text_len])
                break
            end_idx = idx + sep_len
            pieces.append(text[cursor:end_idx])
            cursor = end_idx

        segments: list[dict[str, str | int]] = []
        for piece in pieces:
            segments.extend(
                self._recursive_split(
                    text=piece,
                    enc=enc,
                    max_tokens=max_tokens,
                    separators=separators,
                    sep_idx=sep_idx + 1,
                    forced_overlap=forced_overlap,
                )
            )
        return segments

    def _merge_recursive_segments(
        self,
        segments: list[dict[str, str | int]],
        max_tokens: int,
        chunk_overlap: int,
    ) -> list[str]:
        """Merge recursive segments into final overlapping chunks."""
        if not segments:
            return [""]

        chunks: list[str] = []
        current: list[dict[str, str | int]] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = int(seg["n_tokens"])
            if current and current_tokens + seg_tokens > max_tokens:
                chunks.append("".join(str(s["text"]) for s in current))

                overlap_tail: list[dict[str, str | int]] = []
                overlap_count = 0
                for s in reversed(current):
                    s_tokens = int(s["n_tokens"])
                    if overlap_count + s_tokens > chunk_overlap:
                        break
                    overlap_tail.insert(0, s)
                    overlap_count += s_tokens

                current = overlap_tail
                current_tokens = overlap_count

            current.append(seg)
            current_tokens += seg_tokens

        if current:
            chunks.append("".join(str(s["text"]) for s in current))

        return chunks or [""]

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        """Call embedding API for a single batch, with retries."""
        delay = _RETRY_BASE_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if self._backend == "hf_local":
                    text_batch = [text if text.strip() else " " for text in batch]
                    vectors_np = self.client.encode(
                        text_batch,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        normalize_embeddings=False,
                        convert_to_numpy=True,
                    )
                    vectors_np = np.asarray(vectors_np, dtype=np.float32)
                    if vectors_np.ndim == 1:
                        vectors_np = vectors_np.reshape(1, -1)
                    return vectors_np.tolist()

                if self._backend == "hf_inference":
                    vectors: list[list[float]] = []
                    for text in batch:
                        text_input = text if text.strip() else " "
                        raw = self.client.feature_extraction(
                            text_input,
                            model=self.model,
                        )
                        arr = np.asarray(raw, dtype=np.float32)
                        if arr.ndim == 1:
                            vec = arr
                        elif arr.ndim >= 2:
                            vec = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
                        else:
                            vec = np.zeros((self.embedding_dim,), dtype=np.float32)
                        vectors.append(vec.astype(np.float32).tolist())
                    return vectors

                response = self.client.embeddings.create(input=batch, model=self.model)
                return [item.embedding for item in response.data]
            except Exception as exc:
                if attempt == _MAX_RETRIES:
                    raise
                logger.warning(
                    "Embedding API error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt, _MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")  # pragma: no cover

    # ------------------------------------------------------------------
    # Core: embed a list of texts (memory-efficient)
    # ------------------------------------------------------------------

    def _embed_texts(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
        progress_desc: str = "Embedding",
    ) -> np.ndarray:
        """Embed a list of texts via API.

        When a text exceeds the model token limit it is chunked and each
        chunk is embedded separately.
        """
        max_tokens = _MODEL_MAX_TOKENS.get(self.model, 8191)
        enc = self._get_encoder()

        flat_texts: list[str] = []
        for t in texts:
            n_tok = len(enc.encode(t))
            if n_tok <= max_tokens:
                flat_texts.append(t)
            else:
                flat_texts.extend(self._chunk_text(t))

        total = len(flat_texts)
        if total == 0:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        batch_embeddings = self._embed_batch_with_retry(flat_texts)
        embeddings = np.array(batch_embeddings, dtype=np.float32)
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        return embeddings

    # ------------------------------------------------------------------
    # Build (streaming, checkpoint-friendly)
    # ------------------------------------------------------------------

    def build(
        self,
        documents: list[dict],
        *,
        show_progress: bool = True,
        progress_desc: str = "Building embedding index",
        chunk_overlap_tokens: int = 0,
        preprocess_fn=None,
    ) -> None:
        """Build embedding index from documents.

        Single-pass streaming: each document is chunked, embedded, and added to
        the index immediately.  Only the final chunk metadata is kept in
        ``self.documents``; no intermediate lists are accumulated.

        Args:
            documents: List of document dictionaries
            show_progress: Show a progress bar while embedding (default True)
            progress_desc: Label for the progress bar
            chunk_overlap_tokens: Token overlap between consecutive chunks
            preprocess_fn: Optional callable(doc) -> doc applied to each document
                before embedding (e.g. metadata enrichment).
        """
        self.documents = []
        n_docs = len(documents)
        chunk_count = 0

        if faiss is not None:
            dim = self.embedding_dim
            index = faiss.IndexFlatIP(dim)
            doc_iter: Any = enumerate(documents)
            if show_progress:
                doc_iter = tqdm(doc_iter, total=n_docs, unit="doc", desc=progress_desc)

            for _di, doc in doc_iter:
                if preprocess_fn is not None:
                    doc = preprocess_fn(doc)
                text = doc.get(self.text_field, "")
                chunks = self._chunk_text(text, overlap_tokens=chunk_overlap_tokens)
                n_chunks = len(chunks)
                meta = {k: v for k, v in doc.items() if k != self.text_field}

                for ci, chunk in enumerate(chunks):
                    emb = self._embed_batch_with_retry([chunk])[0]
                    emb = np.array(emb, dtype=np.float32).reshape(1, -1)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    index.add(emb)
                    del emb

                    self.documents.append({
                        **meta,
                        self.text_field: chunk,
                        "_chunk_index": ci,
                        "_n_chunks": n_chunks,
                    })
                    chunk_count += 1

                    if self.rate_limit_delay > 0:
                        time.sleep(self.rate_limit_delay)

            self._faiss_index = index
            self._embeddings = None
        else:
            emb_list: list[list[float]] = []
            doc_iter = enumerate(documents)
            if show_progress:
                doc_iter = tqdm(doc_iter, total=n_docs, unit="doc", desc=progress_desc)

            for _di, doc in doc_iter:
                if preprocess_fn is not None:
                    doc = preprocess_fn(doc)
                text = doc.get(self.text_field, "")
                chunks = self._chunk_text(text, overlap_tokens=chunk_overlap_tokens)
                n_chunks = len(chunks)
                meta = {k: v for k, v in doc.items() if k != self.text_field}

                for ci, chunk in enumerate(chunks):
                    emb = self._embed_batch_with_retry([chunk])[0]
                    emb_list.append(emb)

                    self.documents.append({
                        **meta,
                        self.text_field: chunk,
                        "_chunk_index": ci,
                        "_n_chunks": n_chunks,
                    })
                    chunk_count += 1

                    if self.rate_limit_delay > 0:
                        time.sleep(self.rate_limit_delay)

            embeddings = np.array(emb_list, dtype=np.float32)
            del emb_list
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms
            self._faiss_index = None
            self._embeddings = embeddings

        logger.info(
            "Built index: %d documents → %d chunks", n_docs, chunk_count,
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the index with a query (cosine similarity).

        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: Whether to include similarity scores in results

        Returns:
            List of matching documents (with optional _score)
        """
        if self._faiss_index is None and self._embeddings is None:
            raise ValueError("Index not built. Call build() first.")

        if not query or not query.strip():
            return []

        q_emb = self._embed_texts([query.strip()])
        if q_emb.shape[0] > 1:
            q_emb = q_emb.mean(axis=0, keepdims=True)
        q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) or 1.0)
        q_emb = np.ascontiguousarray(q_emb.astype(np.float32))

        if self._faiss_index is not None:
            scores, top_indices = self._faiss_index.search(q_emb, top_k)
            scores = scores.flatten()
            top_indices = top_indices.flatten()
        else:
            scores = np.dot(self._embeddings, q_emb.T).flatten()
            top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for j, idx in enumerate(top_indices):
            if idx < 0:
                continue  # FAISS returns -1 for missing neighbors when k > n
            doc = self.documents[int(idx)].copy()
            if return_scores:
                doc["_score"] = float(scores[j])
            results.append(doc)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Save index to disk.

        When FAISS is available, saves a .faiss index plus a .pkl metadata file.
        Otherwise saves a single .pkl with embeddings.

        Args:
            path: Base path (e.g. index.pkl). FAISS index is saved as index.faiss.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        use_faiss = self._faiss_index is not None
        data = {
            "documents": self.documents,
            "embeddings": None if use_faiss else self._embeddings,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "model": self.model,
            "use_faiss": use_faiss,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        if use_faiss:
            faiss_path = path.with_suffix(".faiss")
            faiss.write_index(self._faiss_index, str(faiss_path))

    @classmethod
    def load(
        cls,
        path: Path | str,
        openai_client: "OpenAI | None" = None,
    ) -> "EmbeddingIndex":
        """Load index from disk.

        Args:
            path: Path to saved index
            openai_client: Optional OpenAI client for future queries (not stored)

        Returns:
            Loaded EmbeddingIndex instance
        """
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
            model=data.get("model", cls.DEFAULT_MODEL),
            openai_client=openai_client,
        )
        instance.documents = data["documents"]
        if data.get("use_faiss") and faiss is not None:
            faiss_path = path.with_suffix(".faiss")
            if faiss_path.exists():
                instance._faiss_index = faiss.read_index(str(faiss_path))
                instance._embeddings = None
            else:
                instance._faiss_index = None
                instance._embeddings = data.get("embeddings")
        else:
            instance._faiss_index = None
            instance._embeddings = data["embeddings"]
        return instance
