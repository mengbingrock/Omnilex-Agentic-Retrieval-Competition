#!/usr/bin/env python3
"""
Continued pre-training of DeutscheLexAI_BGB_2.0 on Swiss/German legal texts.

Model:   Qwen2ForCausalLM (~3B params) from local hub cache
Dataset: data/laws_de.csv  (272K law articles: citation, text, title)
Method:  QLoRA (4-bit NF4 quantisation + LoRA adapters) so it fits in 16 GB VRAM

Usage:
    python scripts/continue_pretraining.py                       # defaults
    python scripts/continue_pretraining.py --epochs 3 --lr 2e-4  # override
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = (
    PROJECT_ROOT
    / "hub"
    / "models--Alijeff1214--DeutscheLexAI_BGB_2.0"
    / "snapshots"
    / "66e75e513c06f5e0fd17e07a721439ecfcebe666"
)
DATA_PATH = PROJECT_ROOT / "data" / "laws_de.csv"
OUTPUT_DIR = PROJECT_ROOT / "output_cpt"  # continued-pre-training artefacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Continue pre-training DeutscheLexAI on legal texts")
    p.add_argument("--model_path", type=str, default=str(MODEL_PATH))
    p.add_argument("--data_path", type=str, default=str(DATA_PATH))
    p.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    p.add_argument("--max_length", type=int, default=2048, help="Max token sequence length")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    p.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    p.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_split", type=float, default=0.02, help="Fraction of data for validation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_legal_texts(data_path: str) -> list[str]:
    """Read laws CSV and format each row as a training document."""
    log.info("Loading dataset from %s", data_path)
    df = pd.read_csv(data_path, dtype=str).fillna("")

    documents: list[str] = []
    for _, row in df.iterrows():
        parts = []
        if row.get("title"):
            parts.append(row["title"])
        if row.get("citation"):
            parts.append(row["citation"])
        if row.get("text"):
            parts.append(row["text"])
        doc = "\n".join(parts).strip()
        if doc:
            documents.append(doc)

    log.info("Loaded %d documents", len(documents))
    return documents


def tokenize_and_chunk(
    documents: list[str],
    tokenizer,
    max_length: int,
) -> Dataset:
    """Tokenise documents and pack them into fixed-length chunks for CLM.

    We concatenate all token IDs with EOS separators, then split into
    non-overlapping chunks of ``max_length``.  This avoids wasting compute
    on padding and maximises throughput.
    """
    log.info("Tokenising %d documents (max_length=%d) ...", len(documents), max_length)
    eos_id = tokenizer.eos_token_id

    # Tokenise in batches for speed
    all_ids: list[int] = []
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        encoded = tokenizer(batch, add_special_tokens=False)
        for ids in encoded["input_ids"]:
            all_ids.extend(ids)
            all_ids.append(eos_id)

    total_tokens = len(all_ids)
    n_chunks = total_tokens // max_length
    log.info("Total tokens: %d  →  %d chunks of %d", total_tokens, n_chunks, max_length)

    # Trim tail that doesn't fill a complete chunk
    all_ids = all_ids[: n_chunks * max_length]

    input_ids = [all_ids[i * max_length : (i + 1) * max_length] for i in range(n_chunks)]

    ds = Dataset.from_dict({"input_ids": input_ids})
    return ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_path: str, lora_cfg: LoraConfig):
    """Load the base model in 4-bit quantisation and attach LoRA adapters."""

    log.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model in 4-bit (NF4) quantisation ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    log.info("Arguments: %s", vars(args))

    # ---- LoRA config ----
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ---- Load model & tokenizer ----
    model, tokenizer = load_model_and_tokenizer(args.model_path, lora_config)

    # ---- Load & tokenise data ----
    documents = load_legal_texts(args.data_path)
    dataset = tokenize_and_chunk(documents, tokenizer, args.max_length)

    # Train / validation split
    split = dataset.train_test_split(test_size=args.val_split, seed=args.seed)
    train_ds = split["train"]
    val_ds = split["test"]
    log.info("Train chunks: %d  |  Val chunks: %d", len(train_ds), len(val_ds))

    # ---- Data collator (CLM: labels = input_ids, shifted internally) ----
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM
    )

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        seed=args.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # ---- Train ----
    log.info("Starting continued pre-training ...")
    trainer.train()

    # ---- Save final adapter weights ----
    final_dir = os.path.join(args.output_dir, "final_adapter")
    log.info("Saving LoRA adapter to %s", final_dir)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log.info("Done!  Adapter saved to %s", final_dir)
    log.info(
        "To load later:\n"
        "  from peft import AutoPeftModelForCausalLM\n"
        '  model = AutoPeftModelForCausalLM.from_pretrained("%s")',
        final_dir,
    )


if __name__ == "__main__":
    main()
