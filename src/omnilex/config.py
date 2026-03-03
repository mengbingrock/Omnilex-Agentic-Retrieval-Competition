"""Top-level configuration for Omnilex."""

import os
from pathlib import Path
from typing import Any

# Repo root resolved from src/omnilex/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "processed"
OUTPUT_AGENTIC_RERANKED_DIR = PROJECT_ROOT / "output_agentic_reranked"

# OpenAI key used by notebooks / scripts (falls back to OpenAI SDK default lookup).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Single source of defaults (env vars override these).
_CONFIG_DEFAULTS: dict[str, Any] = {
    "embedding_model": "Alibaba-NLP/gte-Qwen2-1.5B-instruct", # joelniklaus/legal-swiss-roberta-base
    "embedding_backend": "local_transformer",  # "auto" | "local_transformer"
    "llm_model": "gpt-4o",
    "llm_backend": "openai",  # "openai" | "local_transformer"
    "llm_local_model": "Alijeff1214/DeutscheLexAI_BGB_2.0",  # used when llm_backend == "local_transformer"
    "llm_adapter_path": "",# "output_cpt/final_adapter",  # path to PEFT/LoRA adapter dir; empty = no adapter
}

# Resolve from existing config: env override > config defaults.
EMBEDDING_MODEL = os.environ.get("OMNILEX_EMBEDDING_MODE", _CONFIG_DEFAULTS["embedding_model"])
EMBEDDING_BACKEND = os.environ.get("OMNILEX_EMBEDDING_BACKEND", _CONFIG_DEFAULTS["embedding_backend"])

# Main LLM settings (separate from embedding model).
LLM_MODEL = os.environ.get("OMNILEX_LLM_MODEL", _CONFIG_DEFAULTS["llm_model"])
LLM_BACKEND = os.environ.get("OMNILEX_LLM_BACKEND", _CONFIG_DEFAULTS["llm_backend"])
LLM_LOCAL_MODEL = os.environ.get("OMNILEX_LLM_LOCAL_MODEL", _CONFIG_DEFAULTS["llm_local_model"])
LLM_ADAPTER_PATH = os.environ.get("OMNILEX_LLM_ADAPTER_PATH", _CONFIG_DEFAULTS["llm_adapter_path"])

# Defaults for notebooks/04_semantic_embedding_pipeline.py
SEMANTIC_EMBEDDING_CONFIG_DEFAULTS: dict[str, Any] = {
    "model": LLM_MODEL,
    "llm_backend": LLM_BACKEND,  # "openai" | "local_transformer"
    "llm_model": LLM_LOCAL_MODEL,  # used when llm_backend == "local_transformer"
    "llm_adapter_path": LLM_ADAPTER_PATH,  # PEFT/LoRA adapter to merge into local model
    "llm_max_new_tokens": 512,
    "temperature": 0.1,
    "max_iterations": 5,
    "top_k_laws": 20,
    "top_k_courts": 20,
    "rerank_top_k_laws": 15,
    "rerank_top_k_courts": 10,
    # "reranker_model": "BAAI/bge-reranker-v2-m3",
    # "reranker_batch_size": 1,
    # "reranker_max_length": 1024,
    "max_court_text_length": 2000,
    "rate_limit_delay": 0.0,
    "index_type": "embedding",  # "bm25" | "embedding" | "hybrid"
    "hybrid_rrf_k": 60,  # RRF constant for hybrid retrieval (dense + sparse)
    "enrich_metadata": True,  # Prepend citation + LLM summary to text before embedding
    "add_summary": False,  # Generate and prepend LLM summary when enriching metadata
    "summary_model": LLM_MODEL,  # LLM model used to generate document summaries (or uses llm_model for local)
    "data_dir": DATA_DIR,
    "cache_dir": CACHE_DIR,
    "output_dir": OUTPUT_AGENTIC_RERANKED_DIR,
}


def get_semantic_embedding_config(create_dirs: bool = True) -> dict[str, Any]:
    """Return semantic-embedding pipeline config with optional directory setup."""
    config = dict(SEMANTIC_EMBEDDING_CONFIG_DEFAULTS)
    if create_dirs:
        config["output_dir"].mkdir(exist_ok=True, parents=True)
        config["cache_dir"].mkdir(exist_ok=True, parents=True)
    return config
