#!/usr/bin/env python3
"""
DPO fine-tuning script for a legal LLM.

Uses TRL's DPOTrainer with LoRA (PEFT) following the approach from
thirdparty/meta-enriched-rag-for-legal-llms/dpo/.

Usage:
    python scripts/train_dpo.py                          # defaults
    python scripts/train_dpo.py --model_name meta-llama/Llama-3.2-1B-Instruct
    python scripts/train_dpo.py --bf16                   # use bf16 on Ampere+
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "dpo_dataset_hf"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "output_dpo"


def main(args):
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from {args.dataset} …")
    dataset = load_from_disk(str(args.dataset))
    print(f"  Samples: {len(dataset)}")

    # Convert raw-string prompt/chosen/rejected into conversational format
    # so that TRL uses the chat template for tokenization, avoiding BPE
    # boundary mismatches between tokenize(prompt) and tokenize(prompt+response).
    def to_conversational(example):
        example["prompt"] = [{"role": "user", "content": example["prompt"]}]
        example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
        example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
        return example

    dataset = dataset.map(to_conversational)

    # ------------------------------------------------------------------
    # 2. Load tokenizer & model
    # ------------------------------------------------------------------
    # Detect if model_name points to a PEFT adapter (from continued pre-training)
    adapter_config_path = Path(args.model_name) / "adapter_config.json"
    is_peft_adapter = adapter_config_path.exists()

    if is_peft_adapter:
        import json as _json
        with open(adapter_config_path) as _f:
            _adapter_cfg = _json.load(_f)
        base_model_path = _adapter_cfg["base_model_name_or_path"]
        print(f"Detected PEFT adapter at {args.model_name}")
        print(f"  Base model: {base_model_path}")
        print(f"  Merging CPT adapter into base model before DPO …")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model_name)
        model = model.merge_and_unload()
        print("  CPT adapter merged successfully.")
    else:
        print(f"Loading model: {args.model_name} …")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    # ------------------------------------------------------------------
    # 3. LoRA configuration (mirrors reference repo)
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
    )

    # ------------------------------------------------------------------
    # 4. Training arguments
    # ------------------------------------------------------------------
    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to="none",
        max_length=args.max_length,
        remove_unused_columns=False,
        beta=args.beta,
        # --- Memory optimizations for single-GPU ---
        precompute_ref_log_probs=True,      # compute ref logprobs upfront, then unload ref model
        gradient_checkpointing=True,         # trade compute for memory on activations
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ------------------------------------------------------------------
    # 5. DPO Trainer
    # ------------------------------------------------------------------
    print("Initialising DPOTrainer …")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("Starting DPO training …")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    print(f"Saving model to {args.output_dir} …")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    # Save training args for reproducibility
    with open(Path(args.output_dir) / "training_args.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DPO fine-tuning for legal LLM")
    p.add_argument("--model_name", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "output_cpt" / "final_adapter"),
                    help="HuggingFace model name, path, or PEFT adapter dir")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="Path to HF dataset on disk")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT,
                    help="Directory to save LoRA adapters")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--logging_steps", type=int, default=32)
    p.add_argument("--save_steps", type=int, default=120)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1536)
    p.add_argument("--beta", type=float, default=0.1,
                    help="DPO beta (KL penalty weight)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true",
                    help="Use bfloat16 instead of fp16")
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
