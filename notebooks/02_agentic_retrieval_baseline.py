# Agentic Retrieval Baseline for Omnilex Legal Retrieval

"""
This script implements an **agentic retrieval approach** using a ReAct-style agent with search tools.

## Approach
1. Use the OpenAI API (no local model)
2. Build BM25 search indices for laws and court decisions
3. Create search tools the agent can use
4. For each query, run a ReAct agent that:
   - Reasons about what to search
   - Uses tools to search laws and court decisions
   - Extracts citations from search results
   - Provides final answer with all found citations

## Advantages over Direct Generation
- Grounded in actual legal documents
- Less hallucination of non-existent citations
- Can iterate on searches to find more relevant sources

## Requirements
- `openai` package and **OPENAI_API_KEY** set in your environment
- rank-bm25
"""

# === 1. Setup & Configuration ===

import os
import sys
from pathlib import Path

# === CONFIGURATION ===
# Choose which dataset to run on: "val" or "test"
DATASET_MODE = "val"  # Change to "test" for final submission

# Set to True to rebuild indices from CSV (required on first run)
# Set to False to load cached indices (faster for subsequent runs)
FORCE_REBUILD_INDICES = False

# Detect environment
KAGGLE_ENV = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if KAGGLE_ENV:
    # Kaggle paths
    DATA_PATH = Path("/kaggle/input/omnilex-data")
    MODEL_PATH = Path("/kaggle/input/llama-model")
    OUTPUT_PATH = Path("/kaggle/working")
    INDEX_PATH = Path("/kaggle/input/omnilex-indices")
    sys.path.insert(0, "/kaggle/input/omnilex-utils")
else:
    # Local development paths
    REPO_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(REPO_ROOT / "src"))
    DATA_PATH = REPO_ROOT / "data"
    MODEL_PATH = REPO_ROOT / "models"
    OUTPUT_PATH = REPO_ROOT / "output"
    INDEX_PATH = REPO_ROOT / "data" / "processed"

# CSV corpus files for index building
LAWS_CSV = DATA_PATH / "laws_de.csv"
COURTS_CSV = DATA_PATH / "court_considerations.csv"

# Index cache paths
LAWS_INDEX_PATH = INDEX_PATH / "laws_index_bm25.pkl"
COURTS_INDEX_PATH = INDEX_PATH / "courts_index_bm25.pkl"

# Derived paths based on DATASET_MODE
QUERY_FILE = DATA_PATH / f"{DATASET_MODE}.csv"
IS_VALIDATION_MODE = DATASET_MODE == "val"

# Create output directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
INDEX_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if KAGGLE_ENV else 'Local'}")
print(f"Dataset mode: {DATASET_MODE}")
print(f"Query file: {QUERY_FILE}")
print(f"Validation mode: {IS_VALIDATION_MODE}")
print(f"Force rebuild indices: {FORCE_REBUILD_INDICES}")
print(f"\nCorpus files:")
print(f"  Laws CSV: {LAWS_CSV} ({LAWS_CSV.stat().st_size / 1e6:.1f} MB)" if LAWS_CSV.exists() else f"  Laws CSV: {LAWS_CSV} (NOT FOUND)")
print(f"  Courts CSV: {COURTS_CSV} ({COURTS_CSV.stat().st_size / 1e9:.2f} GB)" if COURTS_CSV.exists() else f"  Courts CSV: {COURTS_CSV} (NOT FOUND)")
print(f"\nIndex cache: {INDEX_PATH}")

# Configuration
CONFIG = {
    # OpenAI model (e.g. gpt-4o, gpt-4o-mini, gpt-4-turbo)
    "model": "gpt-4o",
    # Agent settings
    "max_iterations": 3,   # Max agent iterations per query
    "max_tokens": 512,
    "temperature": 0.1,
    "rate_limit_delay": 1.0,   # Seconds between API calls (0 = no delay)
    "max_observation_chars": 1200,  # Reduced from 2000 to prevent context overflow
    "max_conversation_chars": 28000,  # Safety net: truncate if conversation exceeds this
    
    # Retrieval settings
    "top_k_laws": 40,       # Results per law search
    "top_k_courts": 40,     # Results per court search
    
    # Paths
    "test_file": "test.csv",
}

# # 2. Load Corpora and Build/Load Indices

import pandas as pd
import re
from tqdm import tqdm
import pickle

from omnilex.retrieval.bm25_index import BM25Index


def load_csv_corpus(
    csv_path: Path,
    chunk_size: int = 100_000,
    max_rows: int | None = None
) -> list[dict]:
    """Load CSV corpus into list of dicts with progress bar.
    
    Args:
        csv_path: Path to CSV file with 'citation' and 'text' columns
        chunk_size: Rows to process per chunk (for memory efficiency)
        max_rows: Optional limit on rows (for testing with smaller corpus)
    
    Returns:
        List of {"citation": str, "text": str} dicts
    """
    documents = []
    
    # Count rows for progress bar (fast line count)
    print(f"Counting rows in {csv_path.name}...")
    with open(csv_path, encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # minus header
    
    if max_rows:
        total_rows = min(total_rows, max_rows)
    print(f"Total rows to load: {total_rows:,}")
    
    rows_loaded = 0
    with tqdm(total=total_rows, desc=f"Loading {csv_path.name}") as pbar:
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                if max_rows and rows_loaded >= max_rows:
                    break
                documents.append({
                    "citation": str(row["citation"]),
                    "text": str(row["text"]) if pd.notna(row["text"]) else ""
                })
                rows_loaded += 1
            pbar.update(min(len(chunk), total_rows - pbar.n))
            if max_rows and rows_loaded >= max_rows:
                break
    
    return documents


def get_or_build_index(
    name: str,
    csv_path: Path,
    index_path: Path,
    force_rebuild: bool = False,
    max_rows: int | None = None
) -> BM25Index:
    """Load cached index or build from CSV.
    
    Args:
        name: Index name for logging
        csv_path: Path to corpus CSV
        index_path: Path to cache index pickle
        force_rebuild: If True, rebuild even if cache exists
        max_rows: Optional row limit (for testing with smaller corpus)
    
    Returns:
        BM25Index instance
    """
    # Use cached index if available and not forcing rebuild
    if index_path.exists() and not force_rebuild:
        print(f"Loading cached {name} index from {index_path}")
        index = BM25Index.load(index_path)
        print(f"  Loaded {len(index.documents):,} documents")
        return index
    
    # Check CSV exists
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating empty index.")
        return BM25Index(documents=[])
    
    # Load corpus from CSV
    print(f"\n{'='*50}")
    print(f"Building {name} index from {csv_path}")
    print(f"{'='*50}")
    documents = load_csv_corpus(csv_path, max_rows=max_rows)
    
    if not documents:
        print(f"Warning: No documents loaded. Creating empty index.")
        return BM25Index(documents=[])
    
    # Build BM25 index
    print(f"\nBuilding BM25 index for {len(documents):,} documents...")
    index = BM25Index(
        documents=documents,
        text_field="text",
        citation_field="citation"
    )
    print(f"Index built successfully!")
    
    # Cache index for future runs
    if not KAGGLE_ENV:
        print(f"Saving index to {index_path}...")
        index.save(index_path)
        print(f"Index cached.")
    
    return index

# Load or build laws index
# Laws CSV: ~45MB, ~269K rows
# Build time: ~30 seconds | Load from cache: <1 second

laws_index = get_or_build_index(
    name="laws",
    csv_path=LAWS_CSV,
    index_path=LAWS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    # max_rows=10000  # Uncomment to test with smaller corpus
)
print(f"\nLaws index: {len(laws_index.documents):,} documents")

# Test search
test_results = laws_index.search("Vertrag", top_k=3)
print(f"\nTest search 'Vertrag': {len(test_results)} results")
if test_results:
    print(f"  Top result: {test_results[0].get('citation', 'N/A')}")

# Load or build courts index
# Courts CSV: ~2.3GB, ~2.5M rows
# Full corpus build time: ~15-20 minutes | Load from cache: ~10 seconds
# Full corpus can have peak memory during build: ~8-16GB

courts_index = get_or_build_index(
    name="courts",
    csv_path=COURTS_CSV,
    index_path=COURTS_INDEX_PATH,
    force_rebuild=FORCE_REBUILD_INDICES,
    max_rows=100000  # Change to use bigger corpus
)
print(f"\nCourts index: {len(courts_index.documents):,} documents")

# Test search
test_results = courts_index.search("Meinungsfreiheit", top_k=3)
print(f"\nTest search 'Meinungsfreiheit': {len(test_results)} results")
if test_results:
    print(f"  Top result: {test_results[0].get('citation', 'N/A')}")

# # 3. Define Search Tools

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
        index: BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize law search tool.

        Args:
            index: BM25Index for federal laws corpus
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
        index: BM25Index,
        top_k: int = 5,
        max_excerpt_length: int = 300,
    ):
        """Initialize court search tool.

        Args:
            index: BM25Index for court decisions corpus
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


# Create tools
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

# Tool registry
TOOLS = {
    "search_laws": law_tool,
    "search_courts": court_tool,
}

print("Tools registered:")
for name, tool in TOOLS.items():
    print(f"  - {name}: {tool.description.split(chr(10))[0]}")

# Test tools
print("Testing law search:")
print(law_tool("Vertrag Abschluss"))

print("\nTesting court search:")
print(court_tool("Meinungsfreiheit"))

# # 4. Setup OpenAI Client

import time
from openai import OpenAI

# Uses OPENAI_API_KEY from environment. Set it before running:
#   export OPENAI_API_KEY=sk-...
# client = OpenAI()
client = OpenAI(api_key="")

print(f"OpenAI client ready. Model: {CONFIG['model']}")

# # 5. Define ReAct Agent

import re

AGENT_SYSTEM_PROMPT = """Du bist ein Schweizer Rechtsrecherche-Assistent mit Zugang zu zwei Such-Tools:

1. search_laws(query): Durchsuche Schweizer Bundesgesetze (SR/Systematische Rechtssammlung)
   - Gibt relevante Gesetzesbestimmungen mit Zitaten und Textauszügen zurück
   - Verwende für Gesetzesrecht: Kodizes, Gesetze, Verordnungen

2. search_courts(query): Durchsuche Schweizer Bundesgerichtsentscheide (BGE)
   - Gibt relevante Rechtsprechung mit Zitaten und Auszügen zurück
   - Verwende für Gerichtsentscheide und Präzedenzfälle

WICHTIG: Suche IMMER auf Deutsch, da die Dokumente auf Deutsch sind.

Deine Aufgabe: Rufe die Such-Tools auf, um relevante Schweizer Rechtszitate zu finden.

Anleitung:
- Durchsuche BEIDE: Gesetze UND Gerichtsentscheide
- Verwende mehrere Suchanfragen mit deutschen Rechtsbegriffen
- Rufe die Tools auf bis alle relevanten Quellen gefunden sind

Antwortformat:
Thought: [Deine Überlegung zur nächsten Suche]
Action: [tool_name]
Action Input: [deutsche Suchanfrage]

=== BEISPIELE ===

Beispiel 1 - Vertragsrecht:
Query: What are the requirements for a valid contract?

Thought: Ich suche nach Vertragsvoraussetzungen im Obligationenrecht.
Action: search_laws
Action Input: Vertrag Abschluss Voraussetzungen OR

Observation: - Art. 1 Abs. 1 OR: Zum Abschluss eines Vertrages...

Thought: Jetzt suche ich nach BGE-Entscheiden zum Vertragsschluss.
Action: search_courts
Action Input: Vertragsabschluss Gültigkeit Voraussetzungen

Observation: - BGE 127 III 248 E. 3.1: Die Voraussetzungen...

Thought: Ich suche nach weiteren Aspekten wie Willensmängel.
Action: search_laws
Action Input: Irrtum Täuschung Vertrag OR

Beispiel 2 - Strafrecht:
Query: Under what conditions can someone be held liable for negligent homicide?

Thought: Ich suche nach fahrlässiger Tötung im Strafgesetzbuch.
Action: search_laws
Action Input: fahrlässige Tötung StGB

Observation: - Art. 117 StGB: Wer fahrlässig den Tod...

Thought: Ich suche nach Rechtsprechung zur Sorgfaltspflicht.
Action: search_courts
Action Input: fahrlässige Tötung Sorgfaltspflicht

Observation: - BGE 135 IV 56 E. 2.1: Die Sorgfaltspflicht...

Thought: Ich suche nach weiteren BGE zu Fahrlässigkeitsmassstäben.
Action: search_courts
Action Input: Fahrlässigkeit Verschulden Massstab

Beispiel 3 - Familienrecht:
Query: How is child custody determined after divorce?

Thought: Ich suche nach Sorgerecht bei Scheidung im ZGB.
Action: search_laws
Action Input: Scheidung Sorgerecht Kinder ZGB

Observation: - Art. 133 Abs. 1 ZGB: Das Gericht regelt...

Thought: Ich suche nach BGE-Entscheiden zum Kindeswohl.
Action: search_courts
Action Input: Kindeswohl Obhut Zuteilung

Observation: - BGE 142 III 481 E. 2.6: Das Kindeswohl...

Thought: Ich suche nach weiteren Bestimmungen zur elterlichen Sorge.
Action: search_laws
Action Input: elterliche Sorge Zuteilung ZGB

Beispiel 4 - Mietrecht:
Query: When can a landlord terminate a lease?

Thought: Ich suche nach Kündigungsrecht im Mietrecht.
Action: search_laws
Action Input: Mietvertrag Kündigung Vermieter OR

Observation: - Art. 266a OR: Die Kündigung ist...

Thought: Ich suche nach BGE zur missbräuchlichen Kündigung.
Action: search_courts
Action Input: Miete Kündigung missbräuchlich

Observation: - BGE 140 III 496 E. 4.1: Eine Kündigung ist...

Thought: Ich suche nach Kündigungsschutz.
Action: search_laws
Action Input: Kündigungsschutz Miete OR

=== ENDE BEISPIELE ===

Suche IMMER auf Deutsch. Rufe beide Tools (search_laws UND search_courts) auf."""


def parse_all_agent_actions(response: str) -> list[tuple[str, str]]:
    """
    Parse ALL action/input pairs from agent response.
    
    The LLM may output multiple actions in one response. This function
    extracts all of them.
    
    Args:
        response: Full LLM response text
        
    Returns:
        List of (action, action_input) tuples
    """
    actions = []
    
    # Find all "Action:" lines
    action_pattern = r"Action:\s*(\w+)"
    input_pattern = r"Action Input:\s*(.+?)(?=\nAction:|$)"
    
    # Find all action matches with their positions
    action_matches = list(re.finditer(action_pattern, response, re.IGNORECASE))
    
    for i, action_match in enumerate(action_matches):
        action = action_match.group(1).strip()
        
        # Find the corresponding Action Input
        # Start search after the Action line
        start_pos = action_match.end()
        # End search at next Action or end of string
        if i + 1 < len(action_matches):
            end_pos = action_matches[i + 1].start()
        else:
            end_pos = len(response)
        
        input_text = response[start_pos:end_pos]
        input_match = re.search(input_pattern, input_text, re.IGNORECASE | re.DOTALL)
        
        if input_match:
            action_input = input_match.group(1).strip()
            actions.append((action, action_input))
    
    return actions


def extract_citations_from_text(text: str) -> list[str]:
    """Extract citations from any text (tool output or final answer)."""
    citations = []
    
    # SR pattern: SR followed by number (optionally with article)
    sr_matches = re.findall(
        r"SR\s*\d{3}(?:\.\d+)?(?:\s+Art\.?\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(sr_matches)
    
    # BGE pattern: BGE volume section page
    bge_matches = re.findall(
        r"BGE\s+\d{1,3}\s+[IVX]+[a-z]?\s+\d+(?:\s+E\.\s*\d+[a-z]?)?",
        text,
        re.IGNORECASE
    )
    citations.extend(bge_matches)
    
    # Art. pattern: Art. X LAW (e.g., Art. 1 ZGB, Art. 41 OR)
    art_matches = re.findall(
        r"Art\.?\s+\d+[a-z]?\s+(?:Abs\.?\s*\d+\s+)?[A-Z]{2,}",
        text,
        re.IGNORECASE
    )
    citations.extend(art_matches)
    
    return list(set(citations))


def truncate_observation_for_llm(observation: str, max_chars: int = 1200) -> str:
    """Truncate observation text for LLM context, preserving data elsewhere.
    
    This truncates only the text sent to the LLM in the conversation.
    Full observations remain in logs and are used for citation extraction.
    
    Args:
        observation: Full observation text
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated observation text
    """
    if len(observation) <= max_chars:
        return observation
    
    # Truncate and add indicator
    return observation[:max_chars] + f"\n... (truncated, {len(observation) - max_chars} chars remaining)"


def truncate_conversation(conversation: str, max_chars: int) -> str:
    """Truncate conversation to fit within token budget, keeping system prompt and recent context.
    
    Args:
        conversation: Full conversation text
        max_chars: Maximum characters to keep
        
    Returns:
        Truncated conversation text
    """
    if len(conversation) <= max_chars:
        return conversation
    
    # Find the system prompt end marker and keep it
    inst_end = conversation.find("[/INST]")
    if inst_end == -1:
        # Fallback: keep last max_chars
        return "..." + conversation[-max_chars:]
    
    system_part = conversation[:inst_end + 7]  # Include [/INST]
    remaining_budget = max_chars - len(system_part) - 100  # Buffer for truncation marker
    
    if remaining_budget <= 0:
        # System prompt itself is too long, just truncate from end
        return conversation[-max_chars:]
    
    # Keep the most recent conversation
    rest = conversation[inst_end + 7:]
    if len(rest) > remaining_budget:
        rest = "\n...[earlier conversation truncated]...\n" + rest[-remaining_budget:]
    
    return system_part + rest


def _llm_completion(conversation: str) -> str:
    """Call OpenAI API and return assistant text. Applies rate limit delay."""
    r = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": conversation},
        ],
        #max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
        stop=["Observation:", "[INST]", "</s>"],
    )
    text = (r.choices[0].message.content or "").strip()
    delay = CONFIG.get("rate_limit_delay", 0)
    if delay > 0:
        time.sleep(delay)
    return text


def run_agent(query: str, verbose: bool = False) -> tuple[list[str], list[dict]]:
    """Run ReAct agent to retrieve citations.
    
    Returns:
        Tuple of (citations, logs) where logs contains detailed execution information
    """
    # Format with Mistral Instruct tags (OpenAI accepts as user message)
    conversation = f"[INST] {AGENT_SYSTEM_PROMPT}\n\nQuery: {query}\n\nThought: [/INST]"
    all_citations = []
    logs: list[dict] = []
    
    for iteration in range(CONFIG["max_iterations"]):
        # Truncate conversation if too long to avoid context window overflow
        max_conv_chars = CONFIG.get("max_conversation_chars", 28000)
        conversation = truncate_conversation(conversation, max_conv_chars)
        
        # Get LLM response with error handling for context overflow
        try:
            response = _llm_completion(conversation)
        except Exception as e:
            error_str = str(e).lower()
            if "context" in error_str or "length" in error_str or "token" in error_str:
                # Aggressively truncate and retry once
                conversation = truncate_conversation(conversation, max_chars=20000)
                try:
                    response = _llm_completion(conversation)
                except Exception as retry_error:
                    logs.append({
                        "type": "error",
                        "iteration": iteration + 1,
                        "error": f"Context overflow after retry: {retry_error}",
                    })
                    break
            else:
                raise
        
        # For subsequent turns, we need to handle the conversation format
        if iteration == 0:
            conversation = f"[INST] {AGENT_SYSTEM_PROMPT}\n\nQuery: {query} [/INST]\n\nThought:{response}"
        else:
            conversation += response
        
        # Log LLM output
        logs.append({
            "type": "llm_response",
            "iteration": iteration + 1,
            "response": response,
            "response_trunc": response[:500] if len(response) > 500 else response,
        })
        
        if verbose:
            print(f"\n[Iteration {iteration + 1}] LLM output (trunc):")
            print(response[:500])
        
        # Parse all actions from response
        actions = parse_all_agent_actions(response)
        
        # Log parsed actions
        if actions:
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "actions_count": len(actions),
                "actions": actions,
            })
            if verbose:
                print(f"\n[Iteration {iteration + 1}] Parsed {len(actions)} action(s):")
                for action, action_input in actions:
                    print(f"  Action: {action}, Input: {action_input[:100]}")
        
        # Execute all actions
        observations = []
        for action, action_input in actions:
            action_lower = action.lower()
            
            if action_lower in TOOLS:
                tool = TOOLS[action_lower]
                observation = tool(action_input)
                
                # Extract citations from full observation (before truncation)
                obs_citations = tool.get_last_citations()
                all_citations.extend(obs_citations)
                
                # Truncate observation only for LLM conversation (preserve full data in logs)
                obs_truncated = truncate_observation_for_llm(observation, CONFIG["max_observation_chars"])
                observations.append(f"Tool {action_lower}: {obs_truncated}")
                
                # Log tool execution with full observation
                logs.append({
                    "type": "tool_execution",
                    "iteration": iteration + 1,
                    "tool": action,
                    "query": action_input,
                    "citations_found": obs_citations,
                    "citations_count": len(obs_citations),
                    "observation": observation,
                    "observation_trunc": observation[:500] if len(observation) > 500 else observation,
                })
                
                if verbose:
                    print(f"\n[Tool: {action}]")
                    print(f"  Query: {action_input}")
                    print(f"  Citations found: {len(obs_citations)}")
                    if obs_citations:
                        print(f"  Citations: {obs_citations[:5]}")
                    print(f"  Observation (trunc): {observation[:300]}")
            else:
                error_msg = f"Unknown tool '{action}'. Available: search_laws, search_courts"
                observations.append(f"Tool {action_lower}: {error_msg}")
                logs.append({
                    "type": "tool_error",
                    "iteration": iteration + 1,
                    "tool": action,
                    "error": error_msg,
                })
        
        # Add all observations to conversation
        if observations:
            conversation += "\n" + "\n".join(observations) + "\n\n[INST] Continue your analysis. [/INST]\n\nThought:"
        
        # Check for final answer AFTER executing all actions
        if "Final Answer:" in response:
            final_text = response.split("Final Answer:")[-1].strip()
            citations = extract_citations_from_text(final_text)
            all_citations.extend(citations)
            
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "status": "final_answer_seen",
            })
            
            if verbose:
                print(f"\n[Iteration {iteration + 1}] Final Answer detected")
            break
        
        # If no actions found and no final answer, try to extract citations from response
        if not actions and "Final Answer:" not in response:
            citations = extract_citations_from_text(response)
            all_citations.extend(citations)
            logs.append({
                "type": "parse",
                "iteration": iteration + 1,
                "status": "no_actions_found",
                "citations_extracted": citations,
            })
            break
    
    # Deduplicate citations
    unique_citations = list(set(all_citations))
    
    logs.append({
        "type": "summary",
        "total_iterations": len(logs),
        "total_citations": len(unique_citations),
        "citations": unique_citations,
    })
    
    if verbose:
        print("\n" + "="*50)
        print("Found citations:")
        for c in unique_citations:
            print(f"  - {c}")
    
    return unique_citations, logs

# Test agent with a sample query
test_query = "What are the requirements for a valid contract under Swiss law?"
print(f"Query: {test_query}")
print("\nRunning agent...\n")

citations, logs = run_agent(test_query, verbose=True)

print("\n" + "="*50)
print("Found citations:")
for c in citations:
    print(f"  - {c}")

# # 6. Load Test Data

import pandas as pd

# Load queries from the configured query file
if not QUERY_FILE.exists():
    raise FileNotFoundError(f"Query file not found: {QUERY_FILE}")

test_df = pd.read_csv(QUERY_FILE)

print(f"Loaded {len(test_df)} queries from {QUERY_FILE}")
print(f"Columns: {list(test_df.columns)}")

if IS_VALIDATION_MODE and "gold_citations" in test_df.columns:
    print(f"Gold citations available for evaluation")

test_df.head()

# # 7. Generate Predictions

from tqdm import tqdm

# Generate predictions
predictions = []
all_logs = []  # Store logs for all queries

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Running agent"):
    query_id = row["query_id"]
    query_text = row["query"]
    
    # Run agent
    raw_citations, logs = run_agent(query_text, verbose=False)
    
    # Store logs with query_id
    all_logs.append({
        "query_id": query_id,
        "query": query_text,
        "logs": logs,
    })
    
    predictions.append({
        "query_id": query_id,
        "predicted_citations": ";".join(raw_citations),
    })

print(f"\nGenerated predictions for {len(predictions)} queries")
print(f"Collected logs for {len(all_logs)} queries")

# Preview predictions
predictions_df = pd.DataFrame(predictions)
predictions_df.head(10)

# # 8. Create Submission

# Save submission
submission_path = OUTPUT_PATH / "submission.csv"
predictions_df.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Total predictions: {len(predictions_df)}")

# Show sample
print("\nSample submission:")
print(predictions_df.head())

from collections.abc import Sequence


def citation_f1(
    predicted: Sequence[str],
    gold: Sequence[str],
) -> dict[str, float]:
    """Compute F1 score for citation overlap on a single query.

    Args:
        predicted: List of predicted canonical citation IDs
        gold: List of ground truth canonical citation IDs

    Returns:
        Dictionary with precision, recall, and F1
    """
    pred_set = set(predicted)
    gold_set = set(gold)

    # Edge case: both empty
    if len(pred_set) == 0 and len(gold_set) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    # Edge case: prediction empty but gold not
    if len(pred_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Edge case: gold empty but prediction not
    if len(gold_set) == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute Macro F1: average F1 across all queries.

    This is the PRIMARY competition metric.

    Args:
        predictions: List of predicted citation lists (one per query)
        gold: List of gold citation lists (one per query)

    Returns:
        Dictionary with macro precision, recall, and F1
    """
    if len(predictions) != len(gold):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold")

    if len(predictions) == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    precision_scores = []
    recall_scores = []
    f1_scores = []

    for pred, g in zip(predictions, gold):
        scores = citation_f1(pred, g)
        precision_scores.append(scores["precision"])
        recall_scores.append(scores["recall"])
        f1_scores.append(scores["f1"])

    n = len(f1_scores)
    return {
        "macro_precision": sum(precision_scores) / n,
        "macro_recall": sum(recall_scores) / n,
        "macro_f1": sum(f1_scores) / n,
    }


def micro_f1(
    predictions: Sequence[Sequence[str]],
    gold: Sequence[Sequence[str]],
) -> dict[str, float]:
    """Compute Micro F1: aggregate TP/FP/FN across all queries.

    Args:
        predictions: List of predicted citation lists (one per query)
        gold: List of gold citation lists (one per query)

    Returns:
        Dictionary with micro precision, recall, and F1
    """
    if len(predictions) != len(gold):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold)} gold")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for pred, g in zip(predictions, gold):
        pred_set = set(pred)
        gold_set = set(g)

        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    if total_tp + total_fp == 0:
        precision = 0.0
    else:
        precision = total_tp / (total_tp + total_fp)

    if total_tp + total_fn == 0:
        recall = 0.0
    else:
        recall = total_tp / (total_tp + total_fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def evaluate_submission(
    submission_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Evaluate a submission DataFrame against gold DataFrame.

    Args:
        submission_df: DataFrame with query_id and predicted_citations
        gold_df: DataFrame with query_id and gold_citations
        metrics: List of metrics to compute (default: all)

    Returns:
        Dictionary with requested metric scores
    """
    citation_separator = ";"
    
    def parse_citations(citation_string: str) -> list[str]:
        """Parse citation string into list (citations are already normalized)."""
        if not citation_string or citation_string.strip() == "":
            return []
        return [c.strip() for c in citation_string.split(citation_separator) if c.strip()]

    # Merge DataFrames
    merged = pd.merge(
        submission_df,
        gold_df,
        on="query_id",
        how="inner",
    )

    # Parse citations
    predictions = [
        parse_citations(row.get("predicted_citations", "")) for _, row in merged.iterrows()
    ]
    gold = [parse_citations(row.get("gold_citations", "")) for _, row in merged.iterrows()]

    # Compute all scores
    all_scores = {}

    macro_scores = macro_f1(predictions, gold)
    micro_scores = micro_f1(predictions, gold)

    all_scores.update(macro_scores)
    all_scores.update(micro_scores)

    # Log per-sample TP/FP/FN for each query
    print("\n" + "="*50)
    print("PER-SAMPLE EVALUATION RESULTS")
    print("="*50)
    for idx, (_, row) in enumerate(merged.iterrows()):
        query_id = row["query_id"]
        pred_set = set(predictions[idx])
        gold_set = set(gold[idx])
        
        true_positives = list(pred_set & gold_set)
        false_positives = list(pred_set - gold_set)
        false_negatives = list(gold_set - pred_set)
        
        print(f"\nQuery ID: {query_id}")
        print(f"  True Positives ({len(true_positives)}): {true_positives}")
        print(f"  False Positives ({len(false_positives)}): {false_positives}")
        print(f"  False Negatives ({len(false_negatives)}): {false_negatives}")
    
    print("\n" + "="*50)

    # Filter to requested metrics
    if metrics:
        metric_mapping = {
            "f1": "macro_f1",
            "precision": "macro_precision",
            "recall": "macro_recall",
            "macro_f1": "macro_f1",
            "micro_f1": "micro_f1",
        }
        filtered = {}
        for m in metrics:
            key = metric_mapping.get(m, m)
            if key in all_scores:
                filtered[m] = all_scores[key]
        return filtered

    return all_scores

# # 9. Local Evaluation (Optional)

# Evaluate if in validation mode with gold labels
if IS_VALIDATION_MODE and "gold_citations" in test_df.columns:
    # Join predictions with gold citations from the same file
    eval_df = predictions_df.merge(
        test_df[["query_id", "gold_citations"]],
        on="query_id",
        how="inner"
    )
    
    if len(eval_df) > 0:
        scores = evaluate_submission(
            eval_df[["query_id", "predicted_citations"]],
            eval_df[["query_id", "gold_citations"]],
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Queries evaluated: {len(eval_df)}")
        print(f"\nMacro F1 (PRIMARY): {scores['macro_f1']:.4f}")
        print(f"Macro Precision:    {scores['macro_precision']:.4f}")
        print(f"Macro Recall:       {scores['macro_recall']:.4f}")
        print(f"\nMicro F1:           {scores['micro_f1']:.4f}")
        print(f"Micro Precision:    {scores['micro_precision']:.4f}")
        print(f"Micro Recall:       {scores['micro_recall']:.4f}")
    else:
        print("No overlapping queries for evaluation.")
else:
    print("Skipping evaluation (not in validation mode or no gold labels available)")

# # Summary
# This agentic retrieval baseline demonstrates a more sophisticated approach:
# 1. **Tool-augmented generation**: The LLM can search actual legal corpora rather than relying solely on parametric knowledge.
# 2. **ReAct-style reasoning**: The agent reasons about what to search, executes searches, observes results, and iterates.
# 3. **Grounded citations**: Citations are extracted from actual search results, reducing hallucination.
# 4. **Comprehensive search**: The agent searches both laws and court decisions for complete results.
# # Potential Improvements
# - **Better search**: Use semantic search (embeddings) instead of BM25
# - **Query expansion**: Generate multiple search queries in different languages
# - **Relevance filtering**: Add a step to verify citations are actually relevant
# - **Citation validation**: Check that generated citations exist in the corpus
# - **Multi-hop reasoning**: Follow citation chains to find related sources

# Load test set
TEST_QUERY_FILE = DATA_PATH / "test.csv"

if TEST_QUERY_FILE.exists():
    print(f"Loading test set from {TEST_QUERY_FILE}")
    test_set_df = pd.read_csv(TEST_QUERY_FILE)
    print(f"Loaded {len(test_set_df)} test queries")
    print(f"Columns: {list(test_set_df.columns)}")
    
    # Generate predictions for test set
    test_predictions = []
    test_all_logs = []  # Store logs for all test queries
    
    print("\n" + "="*50)
    print("RUNNING AGENT ON TEST SET")
    print("="*50)
    
    for _, row in tqdm(test_set_df.iterrows(), total=len(test_set_df), desc="Running agent on test set"):
        query_id = row["query_id"]
        query_text = row["query"]
        
        # Run agent
        raw_citations, logs = run_agent(query_text, verbose=False)
        
        # Store logs with query_id
        test_all_logs.append({
            "query_id": query_id,
            "query": query_text,
            "logs": logs,
        })
        
        test_predictions.append({
            "query_id": query_id,
            "predicted_citations": ";".join(raw_citations),
        })
    
    print(f"\nGenerated predictions for {len(test_predictions)} test queries")
    print(f"Collected logs for {len(test_all_logs)} test queries")
    
    # Create DataFrame and save test submission
    test_predictions_df = pd.DataFrame(test_predictions)
    test_submission_path = OUTPUT_PATH / "test_submission.csv"
    test_predictions_df.to_csv(test_submission_path, index=False)
    
    print(f"\nTest submission saved to: {test_submission_path}")
    print(f"Total test predictions: {len(test_predictions_df)}")
    print("\nSample test submission:")
    print(test_predictions_df.head())
else:
    print(f"Test set file not found: {TEST_QUERY_FILE}")
    print("Skipping test set processing.")
