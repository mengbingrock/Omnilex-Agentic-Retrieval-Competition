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
import tiktoken

from omnilex.config import OPENAI_API_KEY, EMBEDDING_MODEL, get_semantic_embedding_config
from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.embedding_index import EmbeddingIndex
from omnilex.retrieval.tools import LawSearchTool, CourtSearchTool, SearchIndex
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
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Schweizer Rechtsassistent."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
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

        # Build enriched text: <metadata> block + original text
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

        # Build enriched text: <metadata> block + original text
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

# Model context window sizes (tokens)
_MODEL_CONTEXT_WINDOW: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
}


def warn_if_conversation_exceeds_context(conversation: str, model: str) -> None:
    """Print a warning if the conversation exceeds the model's context window."""
    max_tokens = _MODEL_CONTEXT_WINDOW.get(model, 128_000)
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    n_tokens = len(enc.encode(conversation))
    if n_tokens > max_tokens:
        warnings.warn(
            f"Conversation length ({n_tokens:,} tokens) exceeds {model} "
            f"context window ({max_tokens:,} tokens). "
            f"Response quality may degrade.",
            stacklevel=2,
        )


def _llm_completion(conversation: str) -> str:
    """Call OpenAI API and return assistant text. Applies rate limit delay."""
    r = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": ENHANCED_AGENT_PROMPT},
            {"role": "user", "content": conversation},
        ],
        temperature=CONFIG["temperature"],
        stop=["Observation:", "[INST]", "</s>"],
    )
    text = (r.choices[0].message.content or "").strip()
    delay = CONFIG.get("rate_limit_delay", 0)
    if delay > 0:
        time.sleep(delay)
    return text


def run_agent_with_precedent_analysis(
    query: str,
    law_tool: LawSearchTool,
    court_tool: CourtSearchTool,
    verbose: bool = False,
) -> tuple[list[str], list[dict]]:
    """
    Run ReAct agent with three-stage citation discovery:
    1. Initial retrieval (BM25 / embedding) → broad candidate set
    2. Semantic embedding → refined top results
    3. Court search + citation extraction from court texts

    Args:
        query: Legal query to research
        law_tool: LawSearchTool instance
        court_tool: CourtSearchTool instance
        verbose: Whether to print per-iteration details

    Returns:
        Tuple of (citations, logs) where logs contains detailed execution information
    """
    conversation = f"[INST] {ENHANCED_AGENT_PROMPT}\n\nQuery: {query}\n\nThought: [/INST]"
    all_citations = set()
    logs: list[dict] = []

    for iteration in range(CONFIG["max_iterations"]):
        # Warn if conversation exceeds model context window
        warn_if_conversation_exceeds_context(conversation, CONFIG["model"])

        # Get LLM response
        try:
            response = _llm_completion(conversation)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            logs.append({
                "iteration": iteration + 1,
                "action": "LLM Error",
                "error": str(e),
                "conversation_snapshot": conversation,
            })
            break

        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
            print(response[:500] + "..." if len(response) > 500 else response)

        # Parse action from response
        action_match = re.search(r'Action:\s*([\w_]+)', response)
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)

        # Check for Final Answer
        if "Final Answer:" in response:
            final_part = response.split("Final Answer:")[1]
            citation_patterns = [
                r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+[A-Za-z]{2,}',
                r'BGE\s+\d+\s+[IVX]+\s+\d+',
                r'\d[A-Z]_\d+/\d{4}',
            ]
            for pattern in citation_patterns:
                matches = re.findall(pattern, final_part)
                all_citations.update(matches)

            logs.append({
                "iteration": iteration + 1,
                "action": "Final Answer",
                "llm_response": response,
                "conversation_snapshot": conversation,
            })
            break

        # Execute action
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()

            observation = ""

            if action == "search_laws":
                observation = law_tool.run(action_input)
                law_citations = law_tool.get_last_citations()
                all_citations.update(law_citations)

                logs.append({
                    "iteration": iteration + 1,
                    "action": "search_laws",
                    "input": action_input,
                    "llm_response": response,
                    "observation": observation,
                    "found_citations": len(law_citations),
                    "citation_list": list(law_citations),
                })

            elif action == "search_courts":
                results = court_tool.search_with_metadata(action_input)

                court_citations = [r.get("citation", "") for r in results if r.get("citation")]
                all_citations.update(court_citations)

                # Extract law citations from court decision texts
                extracted_law_citations = set()
                for court_doc in results[:10]:
                    court_text = court_doc.get("text", "")
                    law_cites = extract_citations_from_text(
                        court_text,
                        max_length=CONFIG["max_court_text_length"]
                    )
                    extracted_law_citations.update(law_cites)

                all_citations.update(extracted_law_citations)

                observation = court_tool.run(action_input)
                if extracted_law_citations:
                    observation += f"\n\n[Extracted law citations from court texts: {', '.join(list(extracted_law_citations)[:5])}...]"

                logs.append({
                    "iteration": iteration + 1,
                    "action": "search_courts",
                    "input": action_input,
                    "llm_response": response,
                    "observation": observation,
                    "found_court_citations": len(court_citations),
                    "court_citation_list": court_citations,
                    "extracted_law_citations": len(extracted_law_citations),
                    "extracted_law_citation_list": list(extracted_law_citations),
                })
            else:
                observation = f"Unknown action: {action}"
                logs.append({
                    "iteration": iteration + 1,
                    "action": f"Unknown: {action}",
                    "input": action_input,
                    "llm_response": response,
                    "observation": observation,
                })

            conversation += f" {response}\nObservation: {observation[:1000]}\nThought:"
        else:
            logs.append({
                "iteration": iteration + 1,
                "action": "Parse Error",
                "llm_response": response,
                "conversation_snapshot": conversation,
            })
            break

    if verbose:
        print("\n=== Full conversation ===")
        print(conversation)

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
        top_k=CONFIG["top_k_laws"],
        max_excerpt_length=300,
    )

    court_tool = CourtSearchTool(
        index=courts_index,
        top_k=CONFIG["top_k_courts"],
        max_excerpt_length=300,
    )

    print("Search tools initialized:")
    print(f"  - Law search: top_k={CONFIG['top_k_laws']}")
    print(f"  - Court search: top_k={CONFIG['top_k_courts']}")

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
    ENHANCED_AGENT_PROMPT = """Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu zwei Such-Tools:

    1. search_laws(query): Durchsuche Schweizer Bundesgesetze (SR/Systematische Rechtssammlung)
       - Gibt relevante Gesetzesbestimmungen mit Zitaten und Textauszügen zurück
       - Verwende für Gesetzesrecht: Kodizes, Gesetze, Verordnungen

    2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide (BGE)
       - Gibt relevante Rechtsprechung mit Zitaten und Auszügen zurück
       - Verwende für Gerichtsentscheide und Präzedenzfälle

    WICHTIG: Suche IMMER auf Deutsch, da die Dokumente auf Deutsch sind.

    STRATEGIE:
    1. Suche – Kandidaten aus dem Index abrufen
    2. Präzedenzfall-Analyse – Gesetzeszitate aus Gerichtsentscheiden extrahieren

    WARUM PRÄZEDENZFALL-ANALYSE WICHTIG IST:
    Gerichtsentscheide zitieren die Gesetze, die sie anwenden. Durch das Finden relevanter 
    Präzedenzfälle entdeckst du, welche Gesetze auf ähnliche rechtliche Fragestellungen 
    anwendbar sind. Dies findet auch Gesetze, die nicht direkt mit den Suchbegriffen 
    übereinstimmen, aber rechtlich relevant sind.

    Anleitung:
    - Durchsuche BEIDE: Gesetze UND Gerichtsentscheide
    - Verwende mehrere Suchanfragen mit deutschen Rechtsbegriffen
    - Achte auf Gesetzeszitate in den Gerichtsentscheiden (z.B. "Art. 221 StPO")
    - Rufe die Tools auf bis alle relevanten Quellen gefunden sind

    Antwortformat:
    Thought: [Deine Überlegung zur nächsten Suche]
    Action: [tool_name]
    Action Input: [deutsche Suchanfrage]

    === BEISPIELE ===

    Beispiel 1 - Untersuchungshaft:
    Query: What are the conditions for pretrial detention?

    Thought: Ich suche nach Untersuchungshaft im Strafprozessrecht.
    Action: search_laws
    Action Input: Untersuchungshaft Voraussetzungen StPO

    Observation: - Art. 221 Abs. 1 StPO: Untersuchungshaft ist zulässig...

    Thought: Jetzt suche ich nach BGE-Entscheiden zur Untersuchungshaft, um weitere relevante Gesetze zu finden.
    Action: search_courts
    Action Input: Untersuchungshaft Haftgründe Verhältnismässigkeit

    Observation: - BGE 137 IV 122: Zur Untersuchungshaft gemäss Art. 221 StPO und Art. 222 StPO...

    Thought: Der BGE-Entscheid zitiert Art. 221 StPO und Art. 222 StPO. Ich habe alle drei Stufen abgeschlossen.
    Final Answer: Art. 221 StPO, Art. 222 StPO, BGE 137 IV 122

    Beispiel 2 - Vertragsrecht:
    Query: What are the requirements for a valid contract?

    Thought: Ich suche nach Vertragsvoraussetzungen im Obligationenrecht.
    Action: search_laws
    Action Input: Vertrag Abschluss Voraussetzungen OR

    Observation: - Art. 1 Abs. 1 OR: Zum Abschluss eines Vertrages...

    Thought: Jetzt suche ich nach Rechtsprechung zum Vertragsschluss.
    Action: search_courts
    Action Input: Vertragsabschluss Gültigkeit Willensmängel

    Observation: - BGE 127 III 248: Zum Vertragsschluss nach Art. 1 OR, Art. 11 OR...

    Thought: Der Entscheid zitiert Art. 1 OR und Art. 11 OR. Das sind relevante Gesetze.
    Final Answer: Art. 1 OR, Art. 11 OR, BGE 127 III 248

    Merke: Suche immer sowohl Gesetze ALS AUCH Gerichtsentscheide, um alle relevanten Zitate zu finden!
    """

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
