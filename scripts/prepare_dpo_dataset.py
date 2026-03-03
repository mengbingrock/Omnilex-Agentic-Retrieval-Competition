"""
Prepare a DPO (Direct Preference Optimization) dataset from train.csv.

Task: Relevance judgment – given a legal case and a law/court decision,
determine whether the legal provision is relevant to the case.

For each training query:
  - POSITIVE sample: gold_citation text (relevant) -> chosen = "relevant",
                     rejected = "not relevant".
  - NEGATIVE sample: random non-gold citation text (irrelevant) ->
                     chosen = "not relevant", rejected = "relevant".

Produces two outputs:
  1. data/dpo_dataset.json          – full DPO dataset (shuffled)
  2. data/dpo_dataset_hf/           – HuggingFace Dataset on disk (for TRL)

Each sample has the schema expected by TRL's DPOTrainer:
  { "prompt": str, "chosen": str, "rejected": str }
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "train.csv"
LAWS_CSV = DATA_DIR / "laws_de.csv"
COURT_CSV = DATA_DIR / "court_considerations.csv"

# ---------------------------------------------------------------------------
# Prompt template – relevance judgment
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Du bist ein juristischer Assistent. Dir wird ein Rechtsfall und eine "
    "Gesetzesbestimmung oder ein Gerichtsentscheid gegeben. "
    "Beurteile, ob die Bestimmung bzw. der Entscheid für den Rechtsfall "
    "einschlägig ist. Antworte mit 'Ja, diese Bestimmung ist einschlägig.' "
    "oder 'Nein, diese Bestimmung ist nicht einschlägig.'"
)

RELEVANT = "Ja, diese Bestimmung ist einschlägig."
NOT_RELEVANT = "Nein, diese Bestimmung ist nicht einschlägig."


def build_prompt(citation: str, citation_text: str, case: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Rechtsfall:\n{case}\n\n"
        f"{'-' * 72}\n\n"
        f"Bestimmung: {citation}\n{citation_text}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    random.seed(args.seed)

    # ---- Load source data ------------------------------------------------
    print("Loading train.csv …")
    train_df = pd.read_csv(TRAIN_CSV)

    print("Loading laws_de.csv …")
    laws_df = pd.read_csv(LAWS_CSV)

    print("Loading court_considerations.csv (this may take a moment) …")
    court_df = pd.read_csv(COURT_CSV)

    # ---- Build citation -> text lookup -----------------------------------
    print("Building citation lookup …")
    citation_to_text: dict[str, str] = {}

    for _, row in tqdm(
        laws_df.iterrows(),
        total=len(laws_df),
        desc="Indexing laws_de",
    ):
        cit = str(row["citation"]).strip()
        txt = str(row["text"]).strip() if pd.notna(row["text"]) else ""
        if cit and txt:
            citation_to_text[cit] = txt

    for _, row in tqdm(
        court_df.iterrows(),
        total=len(court_df),
        desc="Indexing court_considerations",
    ):
        cit = str(row["citation"]).strip()
        txt = str(row["text"]).strip() if pd.notna(row["text"]) else ""
        if cit and txt:
            citation_to_text[cit] = txt

    all_citations = list(citation_to_text.keys())
    print(f"  Total citations in lookup: {len(all_citations):,}")

    # ---- Build DPO samples -----------------------------------------------
    print("Building DPO samples …")
    dpo_positive: list[dict] = []  # relevant citation   -> chosen = RELEVANT
    dpo_negative: list[dict] = []  # irrelevant citation -> chosen = NOT_RELEVANT

    skipped_no_gold = 0
    skipped_no_text = 0

    for _, row in tqdm(
        train_df.iterrows(),
        total=len(train_df),
        desc="Building DPO samples",
    ):
        query = str(row["query"]).strip()
        gold_raw = str(row["gold_citations"]).strip() if pd.notna(row["gold_citations"]) else ""

        if not gold_raw:
            skipped_no_gold += 1
            continue

        gold_cits = [c.strip() for c in gold_raw.split(";") if c.strip()]
        gold_set = set(gold_cits)

        # --- Positive samples: each gold citation that has text ------------
        found_any = False
        for cit in gold_cits:
            if cit not in citation_to_text:
                continue
            found_any = True
            prompt = build_prompt(cit, citation_to_text[cit], query)
            dpo_positive.append({
                "prompt": prompt,
                "chosen": RELEVANT,
                "rejected": NOT_RELEVANT,
            })

        if not found_any:
            skipped_no_text += 1
            continue

        # --- Negative sample: one random non-gold citation -----------------
        non_gold = [c for c in all_citations if c not in gold_set]
        if not non_gold:
            continue

        wrong_cit = random.choice(non_gold)
        prompt = build_prompt(wrong_cit, citation_to_text[wrong_cit], query)
        dpo_negative.append({
            "prompt": prompt,
            "chosen": NOT_RELEVANT,
            "rejected": RELEVANT,
        })

    print(f"  Positive (relevant)   samples: {len(dpo_positive)}")
    print(f"  Negative (irrelevant) samples: {len(dpo_negative)}")
    print(f"  Skipped (no gold_citations):   {skipped_no_gold}")
    print(f"  Skipped (no text found):       {skipped_no_text}")

    # ---- Merge and shuffle ------------------------------------------------
    dpo_all = dpo_positive + dpo_negative
    random.shuffle(dpo_all)
    print(f"  Total DPO samples: {len(dpo_all)}")

    # ---- Save JSON --------------------------------------------------------
    out_json = DATA_DIR / "dpo_dataset.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dpo_all, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON  -> {out_json}")

    # ---- Save HuggingFace Dataset -----------------------------------------
    out_hf = DATA_DIR / "dpo_dataset_hf"
    ds = Dataset.from_list(dpo_all)
    ds.save_to_disk(str(out_hf))
    print(f"Saved HF Dataset -> {out_hf}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DPO dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    main(parser.parse_args())
