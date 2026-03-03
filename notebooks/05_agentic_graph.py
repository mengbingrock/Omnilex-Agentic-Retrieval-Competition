#!/usr/bin/env python3
"""
Agentic Precedent Retrieval for Omnilex Legal Retrieval Competition

Two-stage citation discovery system:
1. Direct law search using BM25
2. Precedent analysis - extracting law citations from court decision texts

Usage:
  python 03_agentic_precedent_retrieval.py

Requirements:
  - OpenAI API key set in environment: export OPENAI_API_KEY=sk-...
  - Data files in ../data/ directory

Converted from: notebooks/03_agentic_precedent_retrieval.ipynb
"""

# ======================================================================
# IMPORTS
# ======================================================================

import json
import re
import time
import warnings
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import tiktoken

from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.embedding_index import EmbeddingIndex
from omnilex.retrieval.graph_retrieval import GraphRetrievalIndex
from omnilex.retrieval.tools import LawSearchTool, CourtSearchTool
from omnilex.citations.normalizer import CitationNormalizer
from omnilex.evaluation.scorer import evaluate_submission



client = OpenAI(api_key="")


# ======================================================================
# CONFIGURATION
# ======================================================================

CONFIG = {
    "model": "gpt-4o",              # Agent reasoning model
    "temperature": 0.1,             # Low for consistency
    "max_iterations": 5,            # Allow multi-step reasoning
    "top_k_laws": 50,               # Direct law search
    "top_k_courts": 30,             # Precedent case search
    "max_court_text_length": 2000,  # Limit text for citation extraction
    "rate_limit_delay": 0.5,        # Delay between API calls (seconds)
    "index_type": "embedding",          # "bm25" or "embedding" (OpenAI embeddings)
    "embedding_model": "text-embedding-3-small",  # Used when index_type == "embedding"
    "data_dir": Path(__file__).resolve().parent.parent / "data",
    "cache_dir": Path(__file__).resolve().parent.parent / "data" / "processed",
    "output_dir": Path(__file__).resolve().parent.parent / "output_agentic_precedent",
}

# Create output directory
CONFIG["output_dir"].mkdir(exist_ok=True, parents=True)
CONFIG["cache_dir"].mkdir(exist_ok=True, parents=True)

print(f"Configuration:")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

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
    
    text = text[:max_length]  # Limit text length for efficiency

    # Pattern: Art. [number] [optional: Abs. number] [LAW_CODE]
    # Examples: Art. 221 StPO, Art. 1 Abs. 2 ZGB, Art. 41 OR
    # Note: Swiss law codes use mixed case (e.g., StPO, StGB), so we use [A-Za-z]
    pattern = r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+[A-Za-z]{2,}'
    citations = re.findall(pattern, text)

    return list(set(citations))  # Deduplicate


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
    graph_retrieval: GraphRetrievalIndex,
    verbose: bool = False,
) -> tuple[list[str], list[dict]]:
    """
    Run ReAct agent with two-stage citation discovery:
    1. Direct law search
    2. Court search + citation extraction from court texts

    Args:
        query: Legal query to research
        law_tool: LawSearchTool instance for search_laws
        court_tool: CourtSearchTool instance for search_courts
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
            break
        
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
            print(response[:500] + "..." if len(response) > 500 else response)
        
        # Parse action from response
        action_match = re.search(r'Action:\s*([\w_]+)', response)
        input_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', response, re.IGNORECASE)
        
        # Check for Final Answer
        if "Final Answer:" in response:
            # Extract citations from final answer
            final_part = response.split("Final Answer:")[1]
            # Find all potential citations
            citation_patterns = [
                r'Art\.\s+\d+[a-z]?(?:\s+Abs\.\s+\d+)?\s+[A-Za-z]{2,}',  # Law citations (mixed case)
                r'BGE\s+\d+\s+[IVX]+\s+\d+',  # BGE citations
                r'\d[A-Z]_\d+/\d{4}',  # Docket-style citations
            ]
            for pattern in citation_patterns:
                matches = re.findall(pattern, final_part)
                all_citations.update(matches)
            
            logs.append({
                "iteration": iteration + 1,
                "action": "Final Answer",
                "response": response
            })
            break
        
        # Execute action
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            
            observation = ""
            
            if action == "search_laws":
                observation = law_tool.run(action_input)
                # Collect citations from law search
                law_citations = law_tool.get_last_citations()
                all_citations.update(law_citations)
                
                logs.append({
                    "iteration": iteration + 1,
                    "action": "search_laws",
                    "input": action_input,
                    "found_citations": len(law_citations)
                })
                
            elif action == "search_courts":
                # Get court search results with full metadata
                results = court_tool.search_with_metadata(action_input)
                # use the graph retrieval to get the most related cases, top k.
                # added graph search here
                results = graph_retrieval.search(action_input, top_k=CONFIG["top_k_courts"])
                # Collect court citations
                court_citations = [r.get("citation", "") for r in results if r.get("citation")]
                all_citations.update(court_citations)
                
                # STAGE 2: Extract law citations from court decision texts
                extracted_law_citations = set()
                for court_doc in results[:10]:  # Analyze top 10 court decisions
                    court_text = court_doc.get("text", "")
                    law_cites = extract_citations_from_text(
                        court_text, 
                        max_length=CONFIG["max_court_text_length"]
                    )
                    extracted_law_citations.update(law_cites)
                
                all_citations.update(extracted_law_citations)
                
                # Format observation for agent
                observation = court_tool.run(action_input)
                if extracted_law_citations:
                    observation += f"\n\n[Extracted law citations from court texts: {', '.join(list(extracted_law_citations)[:5])}...]"
                
                logs.append({
                    "iteration": iteration + 1,
                    "action": "search_courts",
                    "input": action_input,
                    "found_court_citations": len(court_citations),
                    "extracted_law_citations": len(extracted_law_citations)
                })
            else:
                observation = f"Unknown action: {action}"
            
            # Append to conversation
            conversation += f" {response}\nObservation: {observation[:1000]}\nThought:"
        else:
            # Could not parse action, end iteration
            logs.append({
                "iteration": iteration + 1,
                "action": "Parse Error",
                "response": response
            })
            break
    
    return list(all_citations), logs


print("Agent implementation ready with two-stage precedent analysis!")

# ======================================================================
# MAIN PROCESSING
# ======================================================================

def main():
    """Run the complete agentic precedent retrieval pipeline."""
    
    # Import CONFIG from global scope
    global CONFIG, ENHANCED_AGENT_PROMPT
    
    # Define corpus paths using absolute data_dir from CONFIG
    LAWS_CORPUS_PATH = CONFIG["data_dir"] / "laws_de.csv"
    COURTS_CORPUS_PATH = CONFIG["data_dir"] / "court_considerations.csv"

    index_type = CONFIG.get("index_type", "bm25")
    suffix = "openai_embedding" if index_type == "embedding" else "bm25"
    LAWS_INDEX_CACHE = CONFIG["cache_dir"] / f"laws_index_{suffix}.pkl"
    COURTS_INDEX_CACHE = CONFIG["cache_dir"] / f"courts_index_{suffix}.pkl"

    print(f"Corpus files:")
    print(f"  Laws: {LAWS_CORPUS_PATH} (exists: {LAWS_CORPUS_PATH.exists()})")
    print(f"  Courts: {COURTS_CORPUS_PATH} (exists: {COURTS_CORPUS_PATH.exists()})")
    print(f"\nIndex type: {index_type}")
    print(f"Cache files:")
    print(f"  Laws index: {LAWS_INDEX_CACHE} (exists: {LAWS_INDEX_CACHE.exists()})")
    print(f"  Courts index: {COURTS_INDEX_CACHE} (exists: {COURTS_INDEX_CACHE.exists()})")

    # Load or build laws index
    if LAWS_INDEX_CACHE.exists():
        print("Loading laws index from cache...")
        if index_type == "embedding":
            laws_index = EmbeddingIndex.load(LAWS_INDEX_CACHE, openai_client=client)
        else:
            laws_index = BM25Index.load(LAWS_INDEX_CACHE)
        print(f"Loaded! Index contains {len(laws_index.documents)} documents.")
    else:
        print(f"Building laws index from corpus ({index_type})...")
        laws_df = pd.read_csv(LAWS_CORPUS_PATH)
        print(f"Loaded {len(laws_df)} law articles from {LAWS_CORPUS_PATH}")
        documents = laws_df.to_dict("records")
        if index_type == "embedding":
            print("Building OpenAI embedding index (this may take a while and use API credits)...")
            laws_index = EmbeddingIndex(
                text_field="text",
                citation_field="citation",
                model=CONFIG.get("embedding_model", "text-embedding-3-small"),
                openai_client=client,
                rate_limit_delay=CONFIG.get("rate_limit_delay", 0.5),
            )
            laws_index.build(
                documents,
                show_progress=True,
                progress_desc="Laws",
                checkpoint_path=CONFIG["cache_dir"] / "laws_embedding_checkpoint.npy",
            )
        else:
            laws_index = BM25Index(
                documents=documents,
                text_field="text",
                citation_field="citation",
            )
        print("Saving index to cache...")
        laws_index.save(LAWS_INDEX_CACHE)
        print(f"Laws index built and cached to {LAWS_INDEX_CACHE}")

    # Load or build courts index
    if COURTS_INDEX_CACHE.exists():
        print("Loading courts index from cache...")
        if index_type == "embedding":
            courts_index = EmbeddingIndex.load(COURTS_INDEX_CACHE, openai_client=client)
        else:
            courts_index = BM25Index.load(COURTS_INDEX_CACHE)
        print(f"Loaded! Index contains {len(courts_index.documents)} documents.")
    else:
        print(f"Building courts index from corpus ({index_type})...")
        courts_df = pd.read_csv(COURTS_CORPUS_PATH)
        print(f"Loaded {len(courts_df)} court decisions from {COURTS_CORPUS_PATH}")
        documents = courts_df.to_dict("records")
        if index_type == "embedding":
            print("Building OpenAI embedding index (this may take a long time and use API credits)...")
            courts_index = EmbeddingIndex(
                text_field="text",
                citation_field="citation",
                model=CONFIG.get("embedding_model", "text-embedding-3-small"),
                openai_client=client,
                rate_limit_delay=CONFIG.get("rate_limit_delay", 0.5),
            )
            courts_index.build(
                documents,
                show_progress=True,
                progress_desc="Courts",
                checkpoint_path=CONFIG["cache_dir"] / "courts_embedding_checkpoint.npy",
            )
        else:
            courts_index = BM25Index(
                documents=documents,
                text_field="text",
                citation_field="citation",
            )
        print("Saving index to cache...")
        courts_index.save(COURTS_INDEX_CACHE)
        print(f"Courts index built and cached to {COURTS_INDEX_CACHE}")
    # Create search tools with appropriate top_k values
    law_tool = LawSearchTool(
        index=laws_index,
        top_k=CONFIG["top_k_laws"],
        max_excerpt_length=300
    )

    court_tool = CourtSearchTool(
        index=courts_index,
        top_k=CONFIG["top_k_courts"],
        max_excerpt_length=300
    )

    graph_retrieval = GraphRetrievalIndex(
        documents=courts_index.documents,
        text_field="text",
        citation_field="citation",
        fallback_index=courts_index,
    )

    print("Search tools initialized:")
    print(f"  - Law search tool (top_k={CONFIG['top_k_laws']})")
    print(f"  - Court search tool (top_k={CONFIG['top_k_courts']})")
    print(f"  - Graph retrieval (max_hops={graph_retrieval.max_hops}, fallback={type(courts_index).__name__})")
    # Test tools
    print("Testing law search:")
    print(law_tool("Vertrag Abschluss")[:500])

    print("\n" + "="*50)
    print("\nTesting court search:")
    print(court_tool("Meinungsfreiheit")[:500])
    # Uses OPENAI_API_KEY from environment
    # Set it before running: export OPENAI_API_KEY=sk-...



    print(f"OpenAI client ready. Model: {CONFIG['model']}")
    ENHANCED_AGENT_PROMPT = """Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu zwei Such-Tools:

    1. search_laws(query): Durchsuche Schweizer Bundesgesetze (SR/Systematische Rechtssammlung)
       - Gibt relevante Gesetzesbestimmungen mit Zitaten und Textauszügen zurück
       - Verwende für Gesetzesrecht: Kodizes, Gesetze, Verordnungen

    2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide (BGE)
       - Gibt relevante Rechtsprechung mit Zitaten und Auszügen zurück
       - Verwende für Gerichtsentscheide und Präzedenzfälle

    WICHTIG: Suche IMMER auf Deutsch, da die Dokumente auf Deutsch sind.

    STRATEGIE FÜR ZWEI-STUFEN-SUCHE:
    1. Stufe 1: Direkte Gesetzessuche mit search_laws() - finde direkt relevante Gesetzesartikel
    2. Stufe 2: Präzedenzfall-Analyse mit search_courts() - finde relevante Gerichtsentscheide
    3. Extrahiere Gesetzeszitate aus den gefundenen Gerichtsentscheiden

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

    Thought: Der BGE-Entscheid zitiert Art. 221 StPO und Art. 222 StPO. Ich habe beide Stufen abgeschlossen.
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

    print("Enhanced agent prompt loaded (with two-stage retrieval strategy)")
    # Load validation data
    val_path = CONFIG["data_dir"] / "val.csv"
    val_df = pd.read_csv(val_path)
    print(f"Loaded {len(val_df)} validation queries from {val_path}")
    # Initialize normalizer for citation standardization
    normalizer = CitationNormalizer()

    # Process all validation queries
    predictions = []
    detailed_logs = []
    all_agent_logs = []  # per-iteration agent logs across all queries

    print("Processing validation queries...")
    print(f"Estimated time: {len(val_df) * 2}-{len(val_df) * 5} minutes (2-5 min per query)\n")

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing queries"):
        query_id = row['query_id']
        query_text = row['query']

        print(f"\n[Query {query_id}] {query_text[:100]}...")

        # Run agent
        start_time = time.time()
        raw_citations, logs = run_agent_with_precedent_analysis(
            query_text,
            law_tool=law_tool,
            court_tool=court_tool,
            graph_retrieval=graph_retrieval,
            verbose=False,
        )
        elapsed = time.time() - start_time

        # Normalize citations
        normalized = normalizer.canonicalize_list(raw_citations)

        # Format for submission (semicolon-separated)
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
            "num_iterations": len(logs)
        })

        # Tag each per-iteration log with query_id and accumulate
        for log_entry in logs:
            log_entry["query_id"] = query_id
            all_agent_logs.append(log_entry)

        print(f"  Found {len(normalized)} citations in {elapsed:.1f}s")
        print(f"  Citations: {'; '.join(normalized[:5])}{'...' if len(normalized) > 5 else ''}")

    print("\nProcessing complete!")
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Save to CSV (different file per index type: embedding vs bm25)
    index_suffix = "openai_embedding" if CONFIG.get("index_type") == "embedding" else "bm25"
    output_path = CONFIG["output_dir"] / f"submission_agentic_{index_suffix}.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(f"\nSubmission format preview:")
    print(predictions_df.head())
    # Evaluate predictions
    print("\nEvaluating predictions against gold citations...\n")

    metrics = evaluate_submission(predictions_df, val_df)

    print("\n" + "="*50)
    print("FINAL RESULTS - Agentic Precedent Retrieval")
    print("="*50)
    print(f"Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall:    {metrics['macro_recall']:.4f}")
    if 'macro_ap' in metrics:
        print(f"Macro AP (MAP):  {metrics['macro_ap']:.4f}")
    print("="*50)
    # Save detailed logs for analysis (different file per index type)
    logs_df = pd.DataFrame(detailed_logs)
    logs_path = CONFIG["output_dir"] / f"execution_logs_{index_suffix}.csv"
    logs_df.to_csv(logs_path, index=False)
    print(f"Detailed execution logs saved to: {logs_path}")

    # Save per-iteration agent logs (action, input, citations per step)
    agent_logs_path = CONFIG["output_dir"] / f"agent_step_logs_{index_suffix}.jsonl"
    with open(agent_logs_path, "w", encoding="utf-8") as f:
        for entry in all_agent_logs:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Per-iteration agent logs saved to: {agent_logs_path}")

    # Summary statistics
    print("\nExecution Statistics:")
    print(f"  Average execution time: {logs_df['execution_time'].mean():.2f}s")
    print(f"  Total time: {logs_df['execution_time'].sum():.2f}s ({logs_df['execution_time'].sum()/60:.1f} minutes)")
    print(f"  Average citations found: {logs_df['normalized_citations'].apply(len).mean():.1f}")
    print(f"  Average iterations: {logs_df['num_iterations'].mean():.1f}")
    # Per-query performance comparison (if gold citations available)
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
    print("Agentic Precedent Retrieval - Starting Pipeline")
    print("="*70)
    print()
    main()
    print()
    print("="*70)
    print("Pipeline Complete!")
    print("="*70)
