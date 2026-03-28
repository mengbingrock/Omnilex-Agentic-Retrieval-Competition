#!/usr/bin/env python3
"""
Agentic Precedent Retrieval **with Semantic Embedding**

Two-stage pipeline:
  1. Initial retrieval (BM25, embedding, or hybrid dense+sparse) returns a candidate set.
  2. The ReAct agent reasons over results + extracts citations from
     court decision texts.

Hybrid retrieval (index_type="hybrid") combines dense (embedding) and sparse (BM25)
via Reciprocal Rank Fusion (RRF), improving recall on keyword and semantic queries.

Usage:
  python 04_semantic_embedding_pipeline.py

Requirements:
  - OpenAI: export OPENAI_API_KEY=sk-...
  - Local HuggingFace: set llm_backend="local_transformer" and llm_model (e.g. Qwen/Qwen2.5-7B-Instruct)
  - Data files in ../data/ directory

Based on: 03_agentic_precedent_retrieval.py
"""

# ======================================================================
# IMPORTS
# ======================================================================

import json
import logging as _logging
import pickle
import re
import time
import warnings
from datetime import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

from omnilex.config import OPENAI_API_KEY, EMBEDDING_MODEL, get_semantic_embedding_config
from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.embedding_index import EmbeddingIndex
from omnilex.retrieval.tools import LawSearchTool, CourtSearchTool, SearchIndex, CitationExplorerTool
from omnilex.graph.retriever import CitationGraphRetriever
from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.scorer import evaluate_submission


# ---------------------------------------------------------------------------
# LLM client: OpenAI or local HuggingFace (AutoModelForCausalLM + AutoTokenizer)
# ---------------------------------------------------------------------------

def _create_local_hf_client(model_name: str, max_new_tokens: int = 512, adapter_path: str = ""):
    """Create a local HuggingFace chat client using transformers.

    Supports PEFT/LoRA adapters in two ways:
    1. Explicit: pass adapter_path pointing to a directory with adapter_config.json
    2. Auto-detect: if model_name itself points to a PEFT adapter directory

    When a PEFT adapter is detected, the base model is loaded and the adapter
    is merged via merge_and_unload() for faster inference.

    Requires transformers >= 4.37 for Qwen2/Qwen2.5.
    """
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Qwen2/Qwen2.5 need transformers >= 4.37
    min_version = (4, 37)
    try:
        from packaging import version as pkg_version
        if pkg_version.parse(getattr(transformers, "__version__", "0")) < pkg_version.parse(f"{min_version[0]}.{min_version[1]}"):
            raise RuntimeError(
                f"Local Qwen2/Qwen2.5 models require transformers>={min_version[0]}.{min_version[1]}. "
                f"Current: {getattr(transformers, '__version__', 'unknown')}. "
                "Run: pip install --upgrade 'transformers>=4.37'"
            )
    except ImportError:
        pass  # no packaging, skip version check

    # --- PEFT adapter resolution ---
    resolved_base_model = model_name
    resolved_adapter = adapter_path

    # Auto-detect: if model_name itself is a PEFT adapter directory
    adapter_config_at_model = Path(model_name) / "adapter_config.json"
    if adapter_config_at_model.exists() and not resolved_adapter:
        with open(adapter_config_at_model) as f:
            adapter_cfg = json.load(f)
        resolved_base_model = adapter_cfg["base_model_name_or_path"]
        resolved_adapter = model_name
        print(f"Auto-detected PEFT adapter at {model_name}")
        print(f"  Base model: {resolved_base_model}")

    # Validate explicit adapter_path and read base model from its config
    if resolved_adapter and resolved_adapter != model_name:
        adapter_cfg_path = Path(resolved_adapter) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            raise FileNotFoundError(
                f"adapter_path '{resolved_adapter}' does not contain adapter_config.json"
            )
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        resolved_base_model = adapter_cfg["base_model_name_or_path"]
        print(f"Using PEFT adapter at {resolved_adapter}")
        print(f"  Base model (from adapter config): {resolved_base_model}")

    # --- Load base model ---
    # Transformers 5.x auto-detects adapter_config.json in the model dir and
    # tries to load the adapter via load_adapter(), which can crash in
    # caching_allocator_warmup.  We handle adapter loading manually below, so
    # temporarily hide the file to prevent auto-detection.
    _base_adapter_cfg = Path(resolved_base_model) / "adapter_config.json"
    _hidden_cfg = _base_adapter_cfg.with_suffix(".json._skip")
    _did_hide = _base_adapter_cfg.exists() and resolved_adapter is not None
    if _did_hide:
        _base_adapter_cfg.rename(_hidden_cfg)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            torch_dtype="auto",
            device_map="auto",
        )
    except ModuleNotFoundError as e:
        if "Qwen2" in str(e) or "Qwen3" in str(e):
            raise ModuleNotFoundError(
                f"{e}\n\nQwen2/Qwen2.5 require a newer transformers. "
                "Run: pip install --upgrade 'transformers>=4.37'"
            ) from e
        raise
    finally:
        if _did_hide and _hidden_cfg.exists():
            _hidden_cfg.rename(_base_adapter_cfg)

    # --- Apply and merge PEFT adapter ---
    if resolved_adapter:
        from peft import PeftModel
        print(f"Loading PEFT adapter from {resolved_adapter} ...")
        model = PeftModel.from_pretrained(model, resolved_adapter)
        model = model.merge_and_unload()
        print("  Adapter merged into base model.")

    tokenizer_path = resolved_adapter if resolved_adapter else resolved_base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return _LocalHFChatClient(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)


class _LocalHFCompletions:
    def __init__(self, client):
        self._client = client

    def create(self, *, model: str, messages: list, temperature: float = 0.1, stop: list[str] | None = None):
        return self._client._create(model=model, messages=messages, temperature=temperature, stop=stop)


class _LocalHFChatNamespace:
    def __init__(self, client):
        self._client = client

    @property
    def completions(self):
        return _LocalHFCompletions(self._client)


class _LocalHFChatClient:
    """Thin wrapper so local HF can be used like OpenAI for chat completions."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    @property
    def chat(self):
        return _LocalHFChatNamespace(self)

    def _create(self, *, model: str, messages: list, temperature: float = 0.1, stop: list[str] | None = None):
        """OpenAI-style chat completion using apply_chat_template + generate."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens}
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        generated_ids = self.model.generate(**model_inputs, **gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if stop:
            for s in stop:
                if s in response:
                    response = response.split(s)[0]
        return _SimpleResponse(choices=[_SimpleChoice(message=_SimpleMessage(content=response.strip()))])


class _SimpleMessage:
    def __init__(self, content: str):
        self.content = content


class _SimpleChoice:
    def __init__(self, message):
        self.message = message


class _SimpleResponse:
    def __init__(self, choices: list):
        self.choices = choices


def get_llm_client(config: dict):
    """Return OpenAI client or local HuggingFace client based on config."""
    backend = (config.get("llm_backend") or "openai").strip().lower()
    if backend == "local_transformer":
        model_name = config.get("llm_model") or "Qwen/Qwen2.5-7B-Instruct"
        max_new_tokens = config.get("llm_max_new_tokens", 512)
        adapter_path = config.get("llm_adapter_path", "")
        return _create_local_hf_client(model_name, max_new_tokens=max_new_tokens, adapter_path=adapter_path)
    return OpenAI(api_key=OPENAI_API_KEY)




# ======================================================================
# CONFIGURATION
# ======================================================================

CONFIG = get_semantic_embedding_config(create_dirs=True)

# LLM client: OpenAI or local HuggingFace (transformers)
client = get_llm_client(CONFIG)

# For EmbeddingIndex we only pass OpenAI client when using OpenAI embeddings
openai_client_for_embeddings = client if isinstance(client, OpenAI) else None

print("Configuration:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")


# ======================================================================
# HYBRID RETRIEVAL (Dense + Sparse)
# ======================================================================
# Combines BM25 (sparse) and embedding (dense) retrieval via Reciprocal
# Rank Fusion (RRF), following the pattern of dense retrieval from
# thirdparty/meta-enriched-rag-for-legal-llms and extending with sparse.

RRF_K = 60  # RRF constant: score(d) = sum 1/(k + rank); k=60 is standard


class HybridIndex:
    """Combined dense (embedding) + sparse (BM25) retrieval using RRF.

    Implements the SearchIndex protocol so it can be used with LawSearchTool.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        embedding_index: EmbeddingIndex,
        rrf_k: int = RRF_K,
    ):
        self.bm25_index = bm25_index
        self.embedding_index = embedding_index
        self.rrf_k = rrf_k
        # Use embedding index's documents as canonical (same corpus as BM25)
        self.documents = embedding_index.documents

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
        candidate_multiplier: int = 2,
    ) -> list[dict]:
        """Search using both indices and merge results with RRF.

        Fetches more candidates from each retriever (top_k * candidate_multiplier)
        then fuses by Reciprocal Rank Fusion and returns top_k by fused score.
        """
        n = max(top_k, 10) * candidate_multiplier
        bm25_results = self.bm25_index.search(query, top_k=n, return_scores=True)
        dense_results = self.embedding_index.search(query, top_k=n, return_scores=True)

        # RRF: citation -> sum of 1/(k + rank) over both lists
        rrf_scores: dict[str, float] = {}
        doc_by_citation: dict[str, dict] = {}

        for rank, doc in enumerate(bm25_results, start=1):
            cit = doc.get("citation", "")
            if not cit:
                continue
            rrf_scores[cit] = rrf_scores.get(cit, 0.0) + 1.0 / (self.rrf_k + rank)
            if cit not in doc_by_citation:
                doc_by_citation[cit] = doc.copy()

        for rank, doc in enumerate(dense_results, start=1):
            cit = doc.get("citation", "")
            if not cit:
                continue
            rrf_scores[cit] = rrf_scores.get(cit, 0.0) + 1.0 / (self.rrf_k + rank)
            if cit not in doc_by_citation:
                doc_by_citation[cit] = doc.copy()

        # Sort by RRF score descending and take top_k
        sorted_citations = sorted(
            rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True
        )[:top_k]

        results = []
        for cit in sorted_citations:
            doc = doc_by_citation[cit].copy()
            if return_scores:
                doc["_score"] = float(rrf_scores[cit])
            results.append(doc)

        return results


# ======================================================================
# METADATA ENRICHMENT (inspired by thirdparty meta-enriched-rag-for-legal-llms)
# ======================================================================

SUMMARY_PROMPT = """\
Du bist ein Schweizer Rechtsexperte. Fasse den folgenden Gesetzesartikel in 1-2 \
kurzen Sätzen auf Deutsch zusammen. Konzentriere dich auf den Regelungsgegenstand \
und die wichtigsten Rechtsfolgen. Antworte NUR mit der Zusammenfassung, ohne \
Einleitung oder Erklärung.

Gesetzesartikel:
---
{text}
---

Zusammenfassung:"""


def _generate_summary(text: str, llm_client, model: str, temperature: float = 0.0) -> str:
    """Generate a short LLM summary for a single document text.

    Works with both OpenAI-style clients and the local HF wrapper defined above.
    """
    if not text or not text.strip():
        return ""
    # Truncate very long texts to keep the summary call fast / within context
    truncated = text[:3000]
    prompt = SUMMARY_PROMPT.replace("{text}", truncated)
    try:
        msgs = [
            {"role": "system", "content": "Du bist ein hilfreicher Schweizer Rechtsassistent."},
            {"role": "user", "content": prompt},
        ]
        print(f"[LLM CALL] _generate_summary | model={model} | input_len={len(prompt)}")
        response = llm_client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
        )
        result = (response.choices[0].message.content or "").strip()
        print(f"[LLM RESP] _generate_summary | output_len={len(result)} | preview={result[:120]!r}")
        return result
    except Exception as exc:
        print(f"[LLM ERR]  _generate_summary | {exc}")
        logger_meta.warning("Summary generation failed: %s", exc)
        return ""


logger_meta = _logging.getLogger(__name__ + ".meta_enrichment")


def enrich_documents_with_metadata(
    documents: list[dict],
    llm_client,
    model: str,
    *,
    text_field: str = "text",
    citation_field: str = "citation",
    cache_path: Path | str | None = None,
    rate_limit_delay: float = 0.0,
    show_progress: bool = True,
    add_summary: bool = False,
) -> list[dict]:
    """Prepend citation, title, and an LLM-generated summary to each document's text.

    This enriches the text that will later be embedded so the embedding captures
    metadata semantics (following the meta-enriched RAG pattern from
    thirdparty/meta-enriched-rag-for-legal-llms).

    Already-generated summaries are cached to ``cache_path`` so re-runs skip
    expensive LLM calls.

    Returns a **new** list of document dicts (originals are not mutated).
    """
    # Load or initialise summary cache  {citation: summary}
    cache: dict[str, str] = {}
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached summaries from {cache_path}")

    enriched: list[dict] = []
    new_summaries = 0
    iterator = enumerate(documents)
    if show_progress:
        iterator = tqdm(iterator, total=len(documents), desc="Enriching docs with metadata")

    for _i, doc in iterator:
        citation = str(doc.get(citation_field, "")).strip()
        text = str(doc.get(text_field, "")).strip()

        # If the citation was manually concatenated at the start, strip it
        # so it only appears in the metadata block.
        if citation and text.startswith(citation):
            after = text[len(citation):]
            if after and after[0] in (":", "-", "\n", " "):
                text = after.lstrip(":- \n").strip()

        # Look up or generate summary
        summary = ""
        if add_summary:
            if citation and citation in cache:
                summary = cache[citation]
            else:
                summary = _generate_summary(text, llm_client, model)
                if citation:
                    cache[citation] = summary
                new_summaries += 1
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)

        # Build enriched text: citation ONLY in <metadata> block
        metadata_lines: list[str] = []
        if citation:
            metadata_lines.append(f"Zitat: {citation}")
        if summary:
            metadata_lines.append(f"Zusammenfassung: {summary}")

        if metadata_lines:
            metadata_block = "<metadata>\n" + "\n".join(metadata_lines) + "\n</metadata>"
            enriched_text = text + "\n\n" + metadata_block
        else:
            enriched_text = text

        new_doc = {**doc, text_field: enriched_text}
        enriched.append(new_doc)

    # Persist summary cache
    if cache_path is not None and new_summaries > 0:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(cache)} summaries to {cache_path} ({new_summaries} new)")

    print(f"Metadata enrichment complete: {len(enriched)} documents enriched, {new_summaries} new summaries generated")
    return enriched


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------

def deduplicate_laws(
    documents: list[dict],
    text_field: str = "text",
    citation_field: str = "citation",
) -> list[dict]:
    """Deduplicate documents with identical text, merging their citations.

    When multiple law articles share the exact same text content, they are
    collapsed into a single document.  The citation field becomes a
    semicolon-separated list of all original citations so the metadata
    captures every reference.

    Returns a new list (originals are not mutated).
    """
    from collections import OrderedDict

    grouped: OrderedDict[str, dict] = OrderedDict()
    for doc in documents:
        text = str(doc.get(text_field, "")).strip()
        if text in grouped:
            existing_citation = str(grouped[text].get(citation_field, ""))
            new_citation = str(doc.get(citation_field, "")).strip()
            if new_citation and new_citation not in existing_citation:
                grouped[text][citation_field] = existing_citation + "; " + new_citation
        else:
            grouped[text] = {**doc}

    deduped = list(grouped.values())
    n_removed = len(documents) - len(deduped)
    if n_removed > 0:
        print(f"Deduplication: {len(documents)} → {len(deduped)} documents ({n_removed} duplicates merged)")
    return deduped


# ---------------------------------------------------------------------------
# Single-document enrichment helpers (for interleaved enrich + embed)
# ---------------------------------------------------------------------------

def _load_summary_cache(cache_path: Path | str | None) -> dict[str, str]:
    """Load summary cache from JSON file, or return empty dict."""
    if cache_path is None:
        return {}
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached summaries from {cache_path}")
        return cache
    return {}


def _save_summary_cache(cache_path: Path | str | None, cache: dict[str, str]) -> None:
    """Persist summary cache to JSON file."""
    if cache_path is None or not cache:
        return
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(cache)} summaries to {cache_path}")


def _make_single_doc_enricher(
    llm_client,
    model: str,
    text_field: str = "text",
    citation_field: str = "citation",
    cache: dict[str, str] | None = None,
    rate_limit_delay: float = 0.0,
    add_summary: bool = False,
):
    """Return a function that enriches a single document with metadata.

    The returned callable can be passed as ``preprocess_fn`` to
    ``EmbeddingIndex.build()`` so enrichment and embedding happen in a
    single per-document pass instead of two separate passes.
    """
    if cache is None:
        cache = {}

    def _enrich(doc: dict) -> dict:
        citation = str(doc.get(citation_field, "")).strip()
        text = str(doc.get(text_field, "")).strip()

        # If the citation was manually concatenated at the start (e.g. "Art. 22 ...: <content>"),
        # strip it so it only appears in the metadata block.
        # If the citation appears naturally within the text body, keep it.
        if citation and text.startswith(citation):
            after = text[len(citation):]
            # Only strip if followed by separator (colon, dash, newline) — indicates concatenation
            if after and after[0] in (":", "-", "\n", " "):
                text = after.lstrip(":- \n").strip()

        # Look up or generate summary
        summary = ""
        if add_summary:
            if citation and citation in cache:
                summary = cache[citation]
            else:
                summary = _generate_summary(text, llm_client, model)
                if citation:
                    cache[citation] = summary
                if rate_limit_delay > 0:
                    time.sleep(rate_limit_delay)

        # Build enriched text: citation ONLY in <metadata> block, not in text body
        metadata_lines: list[str] = []
        if citation:
            metadata_lines.append(f"Zitat: {citation}")
        if summary:
            metadata_lines.append(f"Zusammenfassung: {summary}")

        if metadata_lines:
            metadata_block = "<metadata>\n" + "\n".join(metadata_lines) + "\n</metadata>"
            enriched_text = text + "\n\n" + metadata_block
        else:
            enriched_text = text

        return {**doc, text_field: enriched_text}

    return _enrich


# ======================================================================
# HELPER FUNCTIONS
# ======================================================================

def extract_citations_from_text(text: str, max_length: int = 2000) -> list[str]:
    """
    Extract law citations from court decision text.
    Finds patterns like: Art. 221 Abs. 1 StPO, Art. 1 ZGB, etc.

    Args:
        text: Court decision text
        max_length: Maximum text length to process (for efficiency)

    Returns:
        List of extracted law citations
    """
    if not text:
        return []

    text = text[:max_length]

    pattern = r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+[A-Za-z]{2,}'
    citations = re.findall(pattern, text)

    return list(set(citations))


# Test the extraction function
test_text = """
Das Bundesgericht hat entschieden, dass Art. 221 Abs. 1 StPO die Grundlage 
für Untersuchungshaft bildet. Gemäss Art. 1 ZGB ist das Gesetz anzuwenden.
Siehe auch Art. 41 OR zur Haftung und Art. 117 StGB zur fahrlässigen Tötung.
"""

print("Testing citation extraction:")
extracted = extract_citations_from_text(test_text)
print(f"Found {len(extracted)} citations: {sorted(extracted)}")

# ======================================================================
# TOOL DEFINITIONS (standard OpenAI tool-calling format)
# ======================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_laws",
            "description": (
                "Durchsuche Schweizer Bundesgesetze (SR/Systematische Rechtssammlung). "
                "Gibt relevante Gesetzesbestimmungen mit Zitaten und Textauszügen zurück. "
                "Verwende für Gesetzesrecht: Kodizes, Gesetze, Verordnungen. "
                "Suche IMMER auf Deutsch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Deutsche Suchanfrage für Gesetzesartikel",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_courts",
            "description": (
                "Durchsuche Schweizer Bundesgerichtsentscheide (BGE). "
                "Gibt relevante Rechtsprechung mit Zitaten und Auszügen zurück. "
                "Verwende für Gerichtsentscheide und Präzedenzfälle. "
                "Suche IMMER auf Deutsch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Deutsche Suchanfrage für Gerichtsentscheide",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explore_citations",
            "description": (
                "Erkunde das Zitationsnetzwerk eines bestimmten Gesetzes oder Entscheids. "
                "Ausgabe: Zwei Listen – was diese Entität am meisten zitiert und was sie am meisten zitiert wird. "
                "Verwende dieses Tool, um das Zitationsnetzwerk rund um ein bekanntes Gesetz oder einen Entscheid zu verstehen. "
                "Beispiel-Eingaben: 'Art. 41 Abs. 1 OR', 'BGE 145 II 32', 'Art. 1 ZGB'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "citation_id": {
                        "type": "string",
                        "description": "Zitations-ID (z.B. 'Art. 34 BV', 'BGE 139 I 2')",
                    }
                },
                "required": ["citation_id"],
            },
        },
    },
]


def _is_citation_relevant(query: str, citation: str, text: str) -> bool:
    """Check whether a single citation is relevant to the query.

    Args:
        query: The original legal query.
        citation: The citation identifier (e.g. "Art. 221 StPO").
        text: The full text / excerpt associated with this citation.

    Returns:
        True if the citation is relevant, False otherwise.
    """
    prompt = (
        f"Anfrage: {query}\n\n"
        f"Zitat: {citation}\n"
        f"Text:\n{text}\n\n"
        "Ist dieses Zitat thematisch relevant für die Anfrage? "
        "Antworte NUR mit JA oder NEIN."
    )

    try:
        print(f"[LLM CALL] _is_citation_relevant | model={CONFIG['model']} | citation={citation!r}")
        r = client.chat.completions.create(
            model=CONFIG["model"],
            messages=[
                {"role": "system", "content": "Du bist ein Relevanz-Filter für Schweizer Rechtsrecherche. Antworte ausschliesslich mit JA oder NEIN."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        answer = (r.choices[0].message.content or "").strip().upper()
        result = answer.startswith("JA")
        print(f"[LLM RESP] _is_citation_relevant | citation={citation!r} | answer={answer!r} | relevant={result}")
        delay = CONFIG.get("rate_limit_delay", 0)
        if delay > 0:
            time.sleep(delay)
        return result
    except Exception as e:
        print(f"[LLM ERR]  _is_citation_relevant | citation={citation!r} | {e}")
        warnings.warn(f"Relevance check failed for {citation} ({e}), keeping it.", stacklevel=2)
        return True  # keep on error


def _filter_relevant_citations(
    query: str,
    citation_text_pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """Verify each citation individually against the query.

    Args:
        query: The original legal query.
        citation_text_pairs: List of (citation_id, associated_text) tuples.

    Returns:
        Tuple of (verified_citations, discarded_citations).
    """
    verified: list[str] = []
    discarded: list[str] = []
    seen: set[str] = set()
    for citation, text in citation_text_pairs:
        if citation in seen:
            continue
        seen.add(citation)
        if _is_citation_relevant(query, citation, text):
            verified.append(citation)
        else:
            discarded.append(citation)
    return verified, discarded


def run_agent_with_precedent_analysis(
    query: str,
    law_tool: LawSearchTool,
    court_tool: CourtSearchTool,
    citation_explorer: CitationExplorerTool | None = None,
    verbose: bool = False,
) -> tuple[list[str], list[dict]]:
    """
    Run agent with standard OpenAI tool-calling for three-stage citation discovery:
    1. Initial retrieval (BM25 / embedding) → broad candidate set
    2. Court search + citation extraction from court texts
    3. Citation graph exploration via explore_citations

    Args:
        query: Legal query to research
        law_tool: LawSearchTool instance
        court_tool: CourtSearchTool instance
        citation_explorer: CitationExplorerTool instance for graph-based citation exploration
        verbose: Whether to print per-iteration details

    Returns:
        Tuple of (citations, logs) where logs contains detailed execution information
    """
    all_citations: set[str] = set()
    logs: list[dict] = []

    messages = [
        {"role": "system", "content": ENHANCED_AGENT_PROMPT},
        {"role": "user", "content": query},
    ]

    # Select available tools (exclude explore_citations if no graph DB)
    active_tools = TOOLS if citation_explorer else [
        t for t in TOOLS if t["function"]["name"] != "explore_citations"
    ]

    for iteration in range(CONFIG["max_iterations"]):
        # Get LLM response with tool definitions
        try:
            print(f"[LLM CALL] iteration={iteration+1} | model={CONFIG['model']}")
            response = client.chat.completions.create(
                model=CONFIG["model"],
                messages=messages,
                tools=active_tools,
                temperature=CONFIG["temperature"],
            )
            message = response.choices[0].message
            messages.append(message)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            logs.append({
                "iteration": iteration + 1,
                "action": "LLM Error",
                "error": str(e),
            })
            break

        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
            if message.content:
                content_preview = message.content[:500]
                print(content_preview + ("..." if len(message.content) > 500 else ""))
            if message.tool_calls:
                for tc in message.tool_calls:
                    print(f"  [Tool Call] {tc.function.name}({tc.function.arguments})")

        # No tool calls → final answer (LLM decided to respond directly)
        if not message.tool_calls:
            final_text = message.content or ""
            print(f"[LLM RESP] Final answer | len={len(final_text)} | preview={final_text[:200]!r}")

            # Extract citations from final answer text
            citation_patterns = [
                r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+[A-Za-z]{2,}',
                r'BGE\s+\d+\s+[IVX]+\s+\d+',
                r'\d[A-Z]_\d+/\d{4}',
            ]
            final_citations = []
            for pattern in citation_patterns:
                matches = re.findall(pattern, final_text)
                final_citations.extend(matches)

            # Verify relevance of final answer citations one by one
            if final_citations:
                final_pairs = [(c, final_text) for c in set(final_citations)]
                verified_final, discarded_final = _filter_relevant_citations(query, final_pairs)
                all_citations.update(verified_final)
            else:
                verified_final, discarded_final = [], []

            logs.append({
                "iteration": iteration + 1,
                "action": "Final Answer",
                "llm_response": final_text,
                "final_citations_raw": final_citations,
                "final_citations_verified": verified_final,
                "final_citations_discarded": discarded_final,
            })
            break

        # Execute each tool call (standard OpenAI tool-calling pattern)
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"[Tool] {fn_name}({fn_args})")

            observation = ""

            if fn_name == "search_laws":
                action_input = fn_args["query"]
                observation = law_tool.run(action_input)
                # Build (citation, full_text) pairs from last results
                citation_text_pairs = [
                    (doc.get("citation", ""), doc.get("text", ""))
                    for doc in law_tool._last_results
                    if doc.get("citation")
                ]

                # -- Relevance verification: one citation at a time --
                verified_citations, discarded_citations = _filter_relevant_citations(query, citation_text_pairs)
                all_citations.update(verified_citations)

                if verbose and discarded_citations:
                    print(f"  [Relevance filter] search_laws: kept {len(verified_citations)}/{len(citation_text_pairs)}, discarded: {discarded_citations}")

                logs.append({
                    "iteration": iteration + 1,
                    "action": "search_laws",
                    "input": action_input,
                    "observation": observation,
                    "found_citations": len(citation_text_pairs),
                    "citation_list": [c for c, _ in citation_text_pairs],
                    "verified_citations": verified_citations,
                    "discarded_citations": discarded_citations,
                })

            elif fn_name == "search_courts":
                action_input = fn_args["query"]
                results = court_tool.search_with_metadata(action_input)

                # Build (citation, full_text) pairs for court decisions
                court_pairs = [
                    (r.get("citation", ""), r.get("text", ""))
                    for r in results if r.get("citation")
                ]

                # Extract law citations from court decision texts
                extracted_pairs: list[tuple[str, str]] = []
                for court_doc in results[:10]:
                    court_text = court_doc.get("text", "")
                    law_cites = extract_citations_from_text(
                        court_text,
                        max_length=CONFIG["max_court_text_length"]
                    )
                    for lc in law_cites:
                        extracted_pairs.append((lc, court_text))

                observation = court_tool.run(action_input)
                extracted_law_citations = set(c for c, _ in extracted_pairs)
                if extracted_law_citations:
                    observation += f"\n\n[Extracted law citations from court texts: {', '.join(list(extracted_law_citations)[:5])}...]"

                # -- Relevance verification: one citation at a time --
                all_pairs = court_pairs + extracted_pairs
                verified_citations, discarded_citations = _filter_relevant_citations(query, all_pairs)
                all_citations.update(verified_citations)

                if verbose and discarded_citations:
                    print(f"  [Relevance filter] search_courts: kept {len(verified_citations)}/{len(all_pairs)}, discarded: {discarded_citations}")

                court_citation_list = [c for c, _ in court_pairs]
                logs.append({
                    "iteration": iteration + 1,
                    "action": "search_courts",
                    "input": action_input,
                    "observation": observation,
                    "found_court_citations": len(court_citation_list),
                    "court_citation_list": court_citation_list,
                    "extracted_law_citations": len(extracted_law_citations),
                    "extracted_law_citation_list": list(extracted_law_citations),
                    "verified_citations": verified_citations,
                    "discarded_citations": discarded_citations,
                })

            elif fn_name == "explore_citations":
                action_input = fn_args["citation_id"]
                if citation_explorer is not None:
                    observation = citation_explorer.run(action_input)
                    meta = citation_explorer.search_with_metadata(action_input)
                    explorer_pairs: list[tuple[str, str]] = []
                    for r in meta.get("inbound", []):
                        explorer_pairs.append((r.get("id", ""), r.get("text", "")))
                    for r in meta.get("outbound", []):
                        explorer_pairs.append((r.get("id", ""), r.get("text", "")))

                    verified_citations, discarded_citations = _filter_relevant_citations(query, explorer_pairs)
                    all_citations.update(verified_citations)

                    if verbose and discarded_citations:
                        print(f"  [Relevance filter] explore_citations: kept {len(verified_citations)}/{len(explorer_pairs)}, discarded: {discarded_citations}")

                    logs.append({
                        "iteration": iteration + 1,
                        "action": "explore_citations",
                        "input": action_input,
                        "observation": observation,
                        "found_citations": len(explorer_pairs),
                        "citation_list": [c for c, _ in explorer_pairs],
                        "verified_citations": verified_citations,
                        "discarded_citations": discarded_citations,
                    })
                else:
                    observation = "explore_citations tool is not available (no graph database configured)."
                    logs.append({
                        "iteration": iteration + 1,
                        "action": "explore_citations",
                        "input": action_input,
                        "observation": observation,
                    })

            else:
                observation = f"Unknown tool: {fn_name}"
                logs.append({
                    "iteration": iteration + 1,
                    "action": f"Unknown: {fn_name}",
                    "input": str(fn_args),
                    "observation": observation,
                })

            # Send tool result back to the LLM (standard tool-calling protocol)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": observation,
            })

        delay = CONFIG.get("rate_limit_delay", 0)
        if delay > 0:
            time.sleep(delay)

    return list(all_citations), logs


print("Agent implementation ready with three-stage retrieval (initial → embedding → agent)!")

# ======================================================================
# MAIN PROCESSING
# ======================================================================

def main():
    """Run the complete agentic precedent retrieval pipeline with semantic embedding."""

    global CONFIG, ENHANCED_AGENT_PROMPT

    # ------------------------------------------------------------------
    # 0. Create timestamped run output directory
    # ------------------------------------------------------------------
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = CONFIG["output_dir"] / f"run_{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run output directory: {run_output_dir}")

    # ------------------------------------------------------------------
    # 1. Load / build search indices (same as 03_)
    # ------------------------------------------------------------------
    LAWS_CORPUS_PATH = CONFIG["data_dir"] / "laws_de.csv"
    COURTS_CORPUS_PATH = CONFIG["data_dir"] / "court_considerations.csv"

    index_type = CONFIG.get("index_type", "bm25")
    enrich_meta = CONFIG.get("enrich_metadata", False)
    # Suffix for cache/output naming: bm25, embedding model name, or hybrid (+ embedding name).
    emb_model = EMBEDDING_MODEL
    emb_suffix = str(emb_model).replace("/", "_").replace(" ", "_")
    meta_tag = "_meta" if enrich_meta else ""
    if index_type == "embedding":
        suffix = emb_suffix + meta_tag
    elif index_type == "hybrid":
        suffix = f"hybrid_{emb_suffix}" + meta_tag
    else:
        suffix = "bm25"
    LAWS_INDEX_CACHE = CONFIG["cache_dir"] / f"laws_index_{suffix}.pkl"
    LAWS_INDEX_CACHE_BM25 = CONFIG["cache_dir"] / "laws_index_bm25.pkl"
    LAWS_INDEX_CACHE_EMBED = CONFIG["cache_dir"] / f"laws_index_{emb_suffix}{meta_tag}.pkl"
    COURTS_INDEX_CACHE = CONFIG["cache_dir"] / f"courts_index_bm25.pkl"
    SUMMARY_CACHE_PATH = CONFIG["cache_dir"] / "laws_summary_cache.json"

    print(f"Corpus files:")
    print(f"  Laws: {LAWS_CORPUS_PATH} (exists: {LAWS_CORPUS_PATH.exists()})")
    print(f"  Courts: {COURTS_CORPUS_PATH} (exists: {COURTS_CORPUS_PATH.exists()})")
    print(f"\nIndex type: {index_type}")
    print(f"Cache files:")
    print(f"  Laws index: {LAWS_INDEX_CACHE} (exists: {LAWS_INDEX_CACHE.exists()})")
    if index_type == "hybrid":
        print(f"  Laws BM25: {LAWS_INDEX_CACHE_BM25} (exists: {LAWS_INDEX_CACHE_BM25.exists()})")
        print(f"  Laws embedding: {LAWS_INDEX_CACHE_EMBED} (exists: {LAWS_INDEX_CACHE_EMBED.exists()})")
    print(f"  Courts index: {COURTS_INDEX_CACHE} (exists: {COURTS_INDEX_CACHE.exists()})")

    # Load or build laws index (single index or hybrid: BM25 + embedding)
    laws_index = None
    if index_type == "hybrid":
        # Hybrid: load or build both BM25 and embedding, then fuse with RRF
        laws_bm25 = None
        if LAWS_INDEX_CACHE_BM25.exists():
            try:
                print("Loading laws BM25 index from cache...")
                laws_bm25 = BM25Index.load(LAWS_INDEX_CACHE_BM25)
                print(f"Loaded BM25: {len(laws_bm25.documents)} documents.")
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"WARNING: laws BM25 cache corrupted ({e}), rebuilding...")
                LAWS_INDEX_CACHE_BM25.unlink(missing_ok=True)
        if laws_bm25 is None:
            print("Building laws BM25 index...")
            laws_df = pd.read_csv(LAWS_CORPUS_PATH)
            documents = laws_df.to_dict("records")
            laws_bm25 = BM25Index(text_field="text", citation_field="citation")
            laws_bm25.build(documents)
            laws_bm25.save(LAWS_INDEX_CACHE_BM25)
            print(f"Laws BM25 built and cached to {LAWS_INDEX_CACHE_BM25}")

        laws_embedding = None
        if LAWS_INDEX_CACHE_EMBED.exists():
            try:
                print("Loading laws embedding index from cache...")
                laws_embedding = EmbeddingIndex.load(
                    LAWS_INDEX_CACHE_EMBED, openai_client=openai_client_for_embeddings
                )
                print(f"Loaded embedding: {len(laws_embedding.documents)} documents.")
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"WARNING: laws embedding cache corrupted ({e}), rebuilding...")
                LAWS_INDEX_CACHE_EMBED.unlink(missing_ok=True)
        if laws_embedding is None:
            print("Building laws embedding index (for hybrid)...")
            laws_df = pd.read_csv(LAWS_CORPUS_PATH)
            documents = laws_df.to_dict("records")
            documents = deduplicate_laws(documents, text_field="text", citation_field="citation")

            # --- Metadata enrichment (per-document, interleaved with embedding) ---
            enrich_fn = None
            summary_cache = None
            if enrich_meta:
                summary_model = CONFIG.get("summary_model", CONFIG["model"])
                print(f"Enriching laws with metadata inline during embedding (model={summary_model})...")
                summary_cache = _load_summary_cache(SUMMARY_CACHE_PATH)
                enrich_fn = _make_single_doc_enricher(
                    llm_client=client,
                    model=summary_model,
                    text_field="text",
                    citation_field="citation",
                    cache=summary_cache,
                    rate_limit_delay=CONFIG.get("rate_limit_delay", 0),
                    add_summary=CONFIG.get("add_summary", False),
                )

            emb_model = EMBEDDING_MODEL
            laws_embedding = EmbeddingIndex(
                text_field="text",
                citation_field="citation",
                model=emb_model,
                openai_client=openai_client_for_embeddings,
                rate_limit_delay=CONFIG.get("rate_limit_delay", 0.5),
            )
            laws_embedding.build(documents, show_progress=True, progress_desc="Laws", preprocess_fn=enrich_fn)
            if summary_cache is not None:
                _save_summary_cache(SUMMARY_CACHE_PATH, summary_cache)
            laws_embedding.save(LAWS_INDEX_CACHE_EMBED)
            print(f"Laws embedding built and cached to {LAWS_INDEX_CACHE_EMBED}")

        laws_index = HybridIndex(
            bm25_index=laws_bm25,
            embedding_index=laws_embedding,
            rrf_k=CONFIG.get("hybrid_rrf_k", RRF_K),
        )
        print(f"Hybrid laws index ready (BM25 + dense, RRF k={laws_index.rrf_k}).")
    else:
        # Single index: BM25 or embedding
        if LAWS_INDEX_CACHE.exists():
            print("Loading laws index from cache...")
            try:
                if index_type == "embedding":
                    laws_index = EmbeddingIndex.load(
                        LAWS_INDEX_CACHE, openai_client=openai_client_for_embeddings
                    )
                else:
                    laws_index = BM25Index.load(LAWS_INDEX_CACHE)
                print(f"Loaded! Index contains {len(laws_index.documents)} documents.")
            except (EOFError, pickle.UnpicklingError, Exception) as e:
                print(f"WARNING: law Cache corrupted ({e}), rebuilding...")
                LAWS_INDEX_CACHE.unlink(missing_ok=True)
                laws_index = None

        if laws_index is None:
            print(f"Building laws index from corpus ({index_type})...")
            laws_df = pd.read_csv(LAWS_CORPUS_PATH)
            print(f"Loaded {len(laws_df)} law articles from {LAWS_CORPUS_PATH}")
            documents = laws_df.to_dict("records")
            if index_type == "embedding":
                documents = deduplicate_laws(documents, text_field="text", citation_field="citation")

            if index_type == "embedding":
                # --- Metadata enrichment (per-document, interleaved with embedding) ---
                enrich_fn = None
                summary_cache = None
                if enrich_meta:
                    summary_model = CONFIG.get("summary_model", CONFIG["model"])
                    print(f"Enriching laws with metadata inline during embedding (model={summary_model})...")
                    summary_cache = _load_summary_cache(SUMMARY_CACHE_PATH)
                    enrich_fn = _make_single_doc_enricher(
                        llm_client=client,
                        model=summary_model,
                        text_field="text",
                        citation_field="citation",
                        cache=summary_cache,
                        rate_limit_delay=CONFIG.get("rate_limit_delay", 0),
                        add_summary=CONFIG.get("add_summary", False),
                    )

                emb_model = EMBEDDING_MODEL
                print(
                    f"Building embedding index with model '{emb_model}' (this may take a while and use API credits)..."
                )
                laws_index = EmbeddingIndex(
                    text_field="text",
                    citation_field="citation",
                    model=emb_model,
                    openai_client=openai_client_for_embeddings,
                    rate_limit_delay=CONFIG.get("rate_limit_delay", 0.5),
                )
                laws_index.build(
                    documents,
                    show_progress=True,
                    progress_desc="Laws",
                    preprocess_fn=enrich_fn,
                )
                if summary_cache is not None:
                    _save_summary_cache(SUMMARY_CACHE_PATH, summary_cache)
            else:
                laws_index = BM25Index(text_field="text", citation_field="citation")
                laws_index.build(documents)
            print("Saving index to cache...")
            laws_index.save(LAWS_INDEX_CACHE)
            print(f"Laws index built and cached to {LAWS_INDEX_CACHE}")

    # Load or build courts index
    courts_index = None
    if COURTS_INDEX_CACHE.exists():
        print("Loading courts index from cache...")
        try:
            courts_index = BM25Index.load(COURTS_INDEX_CACHE)
            print(f"Loaded! Index contains {len(courts_index.documents)} documents.")
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            print(f"WARNING: courts Cache corrupted ({e}), rebuilding...")
            COURTS_INDEX_CACHE.unlink(missing_ok=True)
            courts_index = None

    if courts_index is None:
        print(f"Loading CSV file for courts index from corpus ({index_type})...")
        courts_df = pd.read_csv(COURTS_CORPUS_PATH)
        print(f"Loaded {len(courts_df)} court decisions from {COURTS_CORPUS_PATH}")
        documents = courts_df.to_dict("records")
        documents = deduplicate_laws(documents, text_field="text", citation_field="citation")
        courts_index = BM25Index(text_field="text", citation_field="citation")
        courts_index.build(documents)
        print("Saving index to cache...")
        courts_index.save(COURTS_INDEX_CACHE)
        print(f"Courts index built and cached to {COURTS_INDEX_CACHE}")

    # ------------------------------------------------------------------
    # 2. Create search tools
    # ------------------------------------------------------------------
    law_tool = LawSearchTool(
        index=laws_index,
        threshold=CONFIG.get("threshold_laws", 0.3),
    )

    court_tool = CourtSearchTool(
        index=courts_index,
        threshold=CONFIG.get("threshold_courts", 0.3),
    )

    # Citation explorer (requires Neo4j graph database)
    citation_explorer = None
    try:
        graph_retriever = CitationGraphRetriever()
        citation_explorer = CitationExplorerTool(graph_retriever=graph_retriever, top_k=5)
        print("  - Citation explorer: connected to Neo4j graph")
    except Exception as e:
        print(f"  - Citation explorer: not available ({e})")

    print("Search tools initialized:")
    print(f"  - Law search: top_k={CONFIG['top_k_laws']}")
    print(f"  - Court search: top_k={CONFIG['top_k_courts']}")
    print(f"  - Citation explorer: {'enabled' if citation_explorer else 'disabled'}")

    # Quick sanity test
    print("\nTesting law search:")
    print(law_tool("Vertrag Abschluss")[:500])

    print("\n" + "="*50)
    print("\nTesting court search:")
    print(court_tool("Meinungsfreiheit")[:500])

    # ------------------------------------------------------------------
    # 4. Agent prompt
    # ------------------------------------------------------------------
    backend = (CONFIG.get("llm_backend") or "openai").strip().lower()
    model_display = CONFIG.get("llm_model", CONFIG["model"]) if backend == "local_transformer" else CONFIG["model"]
    print(f"\nLLM client ready ({backend}). Model: {model_display}")
    ENHANCED_AGENT_PROMPT = """Du bist ein Schweizer Rechtsrecherche-Assistent. Du hast Zugang zu Such-Tools \
die du über Funktionsaufrufe nutzen kannst.

WICHTIG: Suche IMMER auf Deutsch, da die Dokumente auf Deutsch sind.

STRATEGIE:
1. Suche – Kandidaten aus dem Index abrufen (Gesetze UND Gerichtsentscheide)
2. Relevanzprüfung – Nach JEDEM Tool-Aufruf die Ergebnisse auf Relevanz zur Anfrage prüfen. NUR relevante Ergebnisse behalten.
3. Präzedenzfall-Analyse – Gesetzeszitate aus Gerichtsentscheiden extrahieren
4. Zitationsnetzwerk – Mit explore_citations verwandte Gesetze und Entscheide über den Zitationsgraphen entdecken

=== RELEVANZPRÜFUNG (KRITISCH) ===
Nach JEDEM Tool-Aufruf MUSST du die zurückgegebenen Ergebnisse überprüfen:
- Lies jeden zurückgegebenen Textauszug sorgfältig
- Prüfe: Behandelt dieses Ergebnis tatsächlich das gleiche Rechtsthema wie die Anfrage?
- BEHALTE nur Ergebnisse, deren Inhalt thematisch zur Anfrage passt
- VERWERFE Ergebnisse, die zwar Suchbegriffe enthalten, aber ein anderes Rechtsgebiet behandeln

Beispiel:
Query: "Voraussetzungen für Untersuchungshaft"
- RELEVANT: Art. 221 StPO über Haftgründe → behandelt direkt Untersuchungshaft
- IRRELEVANT: Art. 59 StGB über therapeutische Massnahmen → anderes Thema

WARUM PRÄZEDENZFALL-ANALYSE WICHTIG IST:
Gerichtsentscheide zitieren die Gesetze, die sie anwenden. Durch relevante Präzedenzfälle \
entdeckst du Gesetze, die nicht direkt mit den Suchbegriffen übereinstimmen, aber rechtlich relevant sind.

WARUM ZITATIONSEXPLORATION WICHTIG IST:
Mit explore_citations findest du verwandte Gesetze und Entscheide, die durch reine Textsuche \
nicht gefunden werden.

Anleitung:
- Durchsuche BEIDE: Gesetze UND Gerichtsentscheide
- Verwende mehrere Suchanfragen mit deutschen Rechtsbegriffen
- Achte auf Gesetzeszitate in den Gerichtsentscheiden (z.B. "Art. 221 StPO")
- Verwende explore_citations um das Netzwerk relevanter Zitate zu erweitern
- Rufe die Tools auf bis alle relevanten Quellen gefunden sind

Wenn du alle relevanten Quellen gesammelt hast, antworte mit einer Zusammenfassung \
aller gefundenen relevanten Zitate. Liste alle relevanten Gesetzesartikel und \
Gerichtsentscheide auf (z.B. Art. 221 StPO, BGE 137 IV 122)."""

    print("Enhanced agent prompt loaded (with three-stage retrieval strategy)")

    # ------------------------------------------------------------------
    # 5. Run the pipeline on validation queries
    # ------------------------------------------------------------------
    val_path = CONFIG["data_dir"] / "val.csv"
    val_df = pd.read_csv(val_path)
    print(f"Loaded {len(val_df)} validation queries from {val_path}")

    normalizer = CitationNormalizer()

    predictions = []
    detailed_logs = []

    print("Processing validation queries...")
    print(f"Estimated time: {len(val_df) * 2}-{len(val_df) * 5} minutes (2-5 min per query)\n")

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing queries"):
        query_id = row['query_id']
        query_text = row['query']

        print(f"\n[Query {query_id}] {query_text[:100]}...")

        start_time = time.time()
        raw_citations, logs = run_agent_with_precedent_analysis(
            query_text,
            law_tool=law_tool,
            court_tool=court_tool,
            citation_explorer=citation_explorer,
            verbose=False,
        )
        elapsed = time.time() - start_time

        normalized = normalizer.canonicalize_list(raw_citations)
        predicted = ";".join(normalized) if normalized else " "

        predictions.append({
            "query_id": query_id,
            "predicted_citations": predicted
        })

        detailed_logs.append({
            "query_id": query_id,
            "query": query_text,
            "raw_citations": raw_citations,
            "normalized_citations": normalized,
            "execution_time": elapsed,
            "num_iterations": len(logs),
            "conversation_tool_history": logs,
        })

        print(f"  Found {len(normalized)} citations in {elapsed:.1f}s")
        print(f"  Citations: {'; '.join(normalized[:5])}{'...' if len(normalized) > 5 else ''}")

    print("\nProcessing complete!")

    # ------------------------------------------------------------------
    # 6. Save & evaluate
    # ------------------------------------------------------------------
    predictions_df = pd.DataFrame(predictions)

    it = CONFIG.get("index_type", "bm25")
    emb_name = str(EMBEDDING_MODEL).replace("/", "_").replace(" ", "_")
    if it == "embedding":
        index_suffix = emb_name
    elif it == "hybrid":
        index_suffix = f"hybrid_{emb_name}"
    else:
        index_suffix = "bm25"
    output_path = run_output_dir / f"submission_embedding_{index_suffix}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"\nSubmission format preview:")
    print(predictions_df.head())

    if 'gold_citations' in val_df.columns:
        print("\nEvaluating predictions against gold citations...\n")

        metrics = evaluate_submission(predictions_df, val_df)

        print("\n" + "="*50)
        print("FINAL RESULTS - Agentic Precedent Retrieval + Semantic Embedding")
        print("="*50)
        print(f"Macro F1:        {metrics['macro_f1']:.4f}")
        print(f"Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
        if 'macro_ap' in metrics:
            print(f"Macro AP (MAP):  {metrics['macro_ap']:.4f}")
        print("="*50)
    else:
        print("\nNo gold citations available — skipping evaluation.")

    # Save summary CSV (without full conversation history for easy viewing)
    logs_summary = []
    for entry in detailed_logs:
        logs_summary.append({
            "query_id": entry["query_id"],
            "query": entry["query"],
            "raw_citations": entry["raw_citations"],
            "normalized_citations": entry["normalized_citations"],
            "execution_time": entry["execution_time"],
            "num_iterations": entry["num_iterations"],
        })
    logs_df = pd.DataFrame(logs_summary)
    logs_path = run_output_dir / f"execution_logs_embedding_{index_suffix}.csv"
    logs_df.to_csv(logs_path, index=False)
    print(f"Detailed execution logs saved to: {logs_path}")

    # Save full conversation & tool calling history as JSON
    full_history_path = run_output_dir / f"conversation_tool_history_{index_suffix}.json"
    with open(full_history_path, "w", encoding="utf-8") as f:
        json.dump(detailed_logs, f, ensure_ascii=False, indent=2, default=str)
    print(f"Full conversation & tool calling history saved to: {full_history_path}")

    print("\nExecution Statistics:")
    print(f"  Average execution time: {logs_df['execution_time'].mean():.2f}s")
    print(f"  Total time: {logs_df['execution_time'].sum():.2f}s ({logs_df['execution_time'].sum()/60:.1f} minutes)")
    print(f"  Average citations found: {logs_df['normalized_citations'].apply(len).mean():.1f}")
    print(f"  Average iterations: {logs_df['num_iterations'].mean():.1f}")

    if 'gold_citations' in val_df.columns:
        print("\nPer-Query Performance:")
        print("="*80)

        for idx, row in val_df.iterrows():
            query_id = row['query_id']
            gold = set(row['gold_citations'].split(';')) if row['gold_citations'].strip() else set()

            pred_row = predictions_df[predictions_df['query_id'] == query_id].iloc[0]
            pred = set(pred_row['predicted_citations'].split(';')) if pred_row['predicted_citations'].strip() else set()

            tp = len(gold & pred)
            fp = len(pred - gold)
            fn = len(gold - pred)

            precision = tp / len(pred) if len(pred) > 0 else 0
            recall = tp / len(gold) if len(gold) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"Query {query_id}: F1={f1:.3f} P={precision:.3f} R={recall:.3f} | "
                  f"Gold={len(gold)} Pred={len(pred)} TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    print("="*70)
    print("Agentic Precedent Retrieval + Semantic Embedding - Starting Pipeline")
    print("="*70)
    print()
    main()
    print()
    print("="*70)
    print("Pipeline Complete!")
    print("="*70)
