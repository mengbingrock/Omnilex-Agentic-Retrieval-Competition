# Direct Generation Baseline for Omnilex Legal Retrieval

"""
This script implements a **direct generation approach** where we prompt an LLM (OpenAI API) 
to generate Swiss legal citations based on the query.

## Approach
1. Use the OpenAI API (no local model)
2. For each query, prompt the LLM to directly generate relevant citations
3. Parse and normalize the generated citations
4. Create submission file

## Requirements
- `openai` package and **OPENAI_API_KEY** set in your environment
- pandas, tqdm
"""

# === 1. Setup & Configuration ===

import os
import sys
from pathlib import Path

# === CONFIGURATION ===
# Choose which dataset to run on: "val" or "test"
DATASET_MODE = "val"  # Change to "test" for final submission

# Detect environment
KAGGLE_ENV = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

if KAGGLE_ENV:
    # Kaggle paths
    DATA_PATH = Path("/kaggle/input/omnilex-data")
    MODEL_PATH = Path("/kaggle/input/llama-model")
    OUTPUT_PATH = Path("/kaggle/working")
    sys.path.insert(0, "/kaggle/input/omnilex-utils")
else:
    # Local development paths
    REPO_ROOT = Path(".").resolve().parent
    DATA_PATH = REPO_ROOT / "data"
    MODEL_PATH = REPO_ROOT / "models"
    OUTPUT_PATH = REPO_ROOT / "output_openai_no_index"
    sys.path.insert(0, str(REPO_ROOT / "src"))

# Derived paths based on DATASET_MODE
QUERY_FILE = DATA_PATH / f"{DATASET_MODE}.csv"
IS_VALIDATION_MODE = DATASET_MODE == "val"

# Create output directory
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print(f"Environment: {'Kaggle' if KAGGLE_ENV else 'Local'}")
print(f"Dataset mode: {DATASET_MODE}")
print(f"Query file: {QUERY_FILE}")
print(f"Validation mode: {IS_VALIDATION_MODE}")
print(f"Data path: {DATA_PATH}")
print(f"Output path: {OUTPUT_PATH}")

# Configuration
CONFIG = {
    # OpenAI model (e.g. gpt-4o, gpt-4o-mini, gpt-4-turbo)
    "model": "gpt-5.2",
    # Generation settings
    "max_tokens": 512,
    "temperature": 0.0,    # Low temperature for consistency
    "rate_limit_delay": 10.0,   # Seconds between API calls (0 = no delay)
    # Paths
    "test_file": "test.csv",
    "train_file": "train.csv",  # For local evaluation
}

# === 2. Setup OpenAI Client ===

from openai import OpenAI

# Uses OPENAI_API_KEY from environment (or .env). Set it before running.
# Uses OPENAI_API_KEY from environment. Set it before running:
#   export OPENAI_API_KEY=sk-...
client = OpenAI(api_key="")
print(f"OpenAI client ready. Model: {CONFIG['model']}")

# === 3. Define Generation Prompt ===

import time

SYSTEM_PROMPT = """You are a Swiss legal citation expert. Output ONLY a Python list of citations.

CITATION FORMATS:
- Federal laws: "Art. X ABBREV" where ABBREV is ZGB, OR, StGB, BV, etc.
- Court decisions: "BGE X Y Z" or "BGE X Y Z E. N" with consideration number

OUTPUT FORMAT: Python list like ["citation1", "citation2", ...]

EXAMPLES:

Query: What are the requirements for a valid contract under Swiss law?
["Art. 1 OR", "Art. 11 OR", "Art. 12 OR", "BGE 119 II 449 E. 2", "BGE 127 III 248 E. 3.1"]

Query: When can a marriage be annulled in Switzerland?
["Art. 104 ZGB", "Art. 105 ZGB", "Art. 106 ZGB", "BGE 121 III 38 E. 2b"]

Query: What constitutes negligent homicide under Swiss criminal law?
["Art. 117 StGB", "Art. 12 StGB", "BGE 116 IV 306 E. 1a"]

Query: What are the grounds for divorce in Swiss law?
["Art. 111 ZGB", "Art. 112 ZGB", "Art. 114 ZGB", "Art. 115 ZGB", "BGE 130 III 585 E. 2.1"]

Query: How is inheritance distributed under Swiss law?
["Art. 457 ZGB", "Art. 462 ZGB", "Art. 471 ZGB", "BGE 132 III 305 E. 3.2"]

Now answer:"""

# Common Swiss law abbreviations for regex matching
LAW_ABBREVS = (
    "ZGB|OR|StGB|BV|SchKG|ZPO|StPO|BGG|VwVG|IPRG|KG|DSG|MSchG|URG|PatG|"
    "DesG|UWG|PrSG|FINMAG|BankG|VAG|KAG|GwG|BEHG|FinfraG|FIDLEG|FINIG|"
    "ATSG|AHV|IV|EO|ALV|KVG|UVG|BVG|ArG|GlG|USG|RPG|WaG|JSG|TSchG|"
    "LwG|PBG|EBG|SVG|LFG|SebG|SpG|BoeB|EMRK|SR|AS|BBl|ParlG|RVOG|RVOV|"
    "MG|BPG|BPV|VBGÖ|VDSG|MWSTG|DBG|StHG|VStG|StG|ZG|CO|CP|CC|CPC|CPP|"
    "LEtr|LAsi|LN|LDIP|LCart|LDA|LPM|LBI|LDes|LCD|LFINMA|LB|LSA|LPCC|"
    "LBA|LBVM|LIMF|LSFin|LEFin|LAVS|LAI|LAPG|LACI|LAMal|LAA|LPP|LTr|"
    "LEg|LPD|LPE|LAT|LFo|LChP|LPN|LAgr|LTV|LCdF|LNA|LPTh|LTAF|LTF"
)


def extract_citations(raw_output: str) -> list[str]:
    """Extract citations from raw LLM output using regex patterns."""
    import re

    citations = []

    # Pattern for BGE citations: BGE 141 II 345 E. 3.2
    # Matches: BGE + volume + part (roman) + page + optional consideration
    bge_pattern = r'BGE\s+(\d+)\s+([IVX]+[a-z]?)\s+(\d+)(?:\s+E\.\s*([\d.a-z/]+))?'
    for match in re.finditer(bge_pattern, raw_output):
        vol, part, page, consid = match.groups()
        if consid:
            citations.append(f"BGE {vol} {part} {page} E. {consid}")
        else:
            citations.append(f"BGE {vol} {part} {page}")

    # Pattern for Art. citations with Abs./lit./Ziff.
    # Matches: Art. 221 Abs. 1 lit. b StPO, Art. 364 Abs. 1 OR, Art. 1 ZGB
    art_pattern = rf'Art\.?\s*(\d+[a-z]?)(?:\s+(Abs\.?\s*\d+))?(?:\s+(lit\.?\s*[a-z]))?(?:\s+(Ziff\.?\s*\d+))?\s+({LAW_ABBREVS})\b'
    for match in re.finditer(art_pattern, raw_output, re.IGNORECASE):
        art_num, abs_part, lit_part, ziff_part, abbrev = match.groups()
        parts = [f"Art. {art_num}"]
        if abs_part:
            # Normalize "Abs1" or "Abs 1" to "Abs. 1"
            abs_normalized = re.sub(r'Abs\.?\s*', 'Abs. ', abs_part)
            parts.append(abs_normalized.strip())
        if lit_part:
            lit_normalized = re.sub(r'lit\.?\s*', 'lit. ', lit_part)
            parts.append(lit_normalized.strip())
        if ziff_part:
            ziff_normalized = re.sub(r'Ziff\.?\s*', 'Ziff. ', ziff_part)
            parts.append(ziff_normalized.strip())
        parts.append(abbrev.upper())
        citations.append(" ".join(parts))

    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


def generate_citations(query: str) -> list[str]:
    """Generate citations using OpenAI API."""
    response = client.chat.completions.create(
        model=CONFIG["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        #max_tokens=CONFIG["max_tokens"],
        temperature=CONFIG["temperature"],
    )
    raw_output = (response.choices[0].message.content or "").strip()
    print(f"Raw output: {raw_output}\n")

    # Parse Python list format first
    import re
    import ast
    
    citations = []
    
    # Try to parse as Python list
    try:
        list_match = re.search(r'\[.*?\]', raw_output, re.DOTALL)
        if list_match:
            parsed = ast.literal_eval(list_match.group())
            if isinstance(parsed, list):
                # Extract citations from parsed list items
                for item in parsed:
                    item_str = str(item).strip()
                    # Extract citations from each item (may have descriptions in parens)
                    extracted = extract_citations(item_str)
                    citations.extend(extracted)
    except (ValueError, SyntaxError):
        pass
    
    # If no citations found from list parsing, try extracting from full output
    if not citations:
        citations = extract_citations(raw_output)

    # Rate limit: wait before next API call
    delay = CONFIG.get("rate_limit_delay", 0)
    if delay > 0:
        time.sleep(delay)

    return citations

# Test generation with a sample query
if __name__ == "__main__":
    test_query = "What are the requirements for a valid contract under Swiss law?"
    print(f"Query: {test_query}")
    raw_citations = generate_citations(test_query)
    print("\nGenerated citations:")
    for c in raw_citations:
        print(f"  - {c}")

# === 4. Load Test Data ===

import pandas as pd

# Load queries from the configured query file
if not QUERY_FILE.exists():
    raise FileNotFoundError(f"Query file not found: {QUERY_FILE}")

test_df = pd.read_csv(QUERY_FILE)

print(f"Loaded {len(test_df)} queries from {QUERY_FILE}")
print(f"Columns: {list(test_df.columns)}")

if IS_VALIDATION_MODE and "gold_citations" in test_df.columns:
    print(f"Gold citations available for evaluation")

print(test_df.head())

# === 5. Generate Predictions ===

from tqdm import tqdm

# Generate predictions
predictions = []

assert test_df is not None, "test_df must be loaded before generating predictions"

for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating predictions"):
    query_id = row["query_id"]
    query_text = row["query"]

    # Generate citations using regex extraction (no external normalizer needed)
    raw_citations = generate_citations(query_text)

    # Use " " (single space) when no citations found, otherwise join with ";"
    predicted = ";".join(raw_citations) if raw_citations else " "

    predictions.append({
        "query_id": query_id,
        "predicted_citations": predicted,
    })

print(f"\nGenerated predictions for {len(predictions)} queries")

# Preview predictions
predictions_df = pd.DataFrame(predictions)
print(predictions_df.head(10))

# === 6. Create Submission ===

# Save submission
submission_path = OUTPUT_PATH / "submission.csv"
predictions_df.to_csv(submission_path, index=False)

print(f"Submission saved to: {submission_path}")
print(f"Total predictions: {len(predictions_df)}")

# Show sample
print("\nSample submission:")
print(predictions_df.head())

"""
## Summary

This baseline script demonstrates a simple direct generation approach:

1. **Prompt engineering**: We use a structured prompt that asks the LLM to generate Swiss legal citations in standard format.

2. **Citation normalization**: The generated citations are normalized to canonical form for consistent evaluation.

3. **Limitations**:
   - The LLM may hallucinate non-existent citations
   - No access to actual legal documents for verification
   - Relies entirely on the LLM's training data knowledge

For better results, see the **Agentic Retrieval Baseline** notebook which uses search tools to ground the generation in actual legal documents.
"""
