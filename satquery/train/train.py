"""
satquery/train/train.py

SatQuery AI — LoRA fine-tuning for TerraQ-VL (LLaVA-1.5-7B).

Strategy (CPU-friendly):
  - Uses LlavaForConditionalGeneration (correct class for LLaVA)
  - LoRA targets ONLY the language model's attention layers (q_proj, v_proj)
    → ~0.5% of total params trained — everything else frozen
  - Text-only instruction tuning (no images needed — injects land-cover vocabulary)
  - Small batch size + gradient accumulation to fit on CPU RAM
"""
from __future__ import annotations

import argparse
import json
import os
import sys


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="SatQuery AI LoRA fine-tuning")
    parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--data_path",          type=str, required=True)
    parser.add_argument("--output_dir",         type=str, default="checkpoints/terraq-vl-bigearthnet-lora")
    parser.add_argument("--lora_enable",        action="store_true")
    parser.add_argument("--lora_r",             type=int,   default=8,    help="LoRA rank (lower = faster, less VRAM)")
    parser.add_argument("--lora_alpha",         type=int,   default=16)
    parser.add_argument("--lora_dropout",       type=float, default=0.05)
    parser.add_argument("--num_train_epochs",   type=int,   default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate",      type=float, default=2e-4)
    parser.add_argument("--max_seq_length",     type=int,   default=256,  help="Max tokens per sample")
    parser.add_argument("--max_samples",        type=int,   default=5000, help="Cap training samples (0=all)")
    return parser.parse_args()


# ── Hardware ──────────────────────────────────────────────────────────────────

def detect_hardware() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        return "CPU"
    except ImportError:
        return "CPU (torch not installed)"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_instruction_data(path: str) -> list[dict]:
    """Load .jsonl (one JSON per line) or .json (array) instruction file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: Skipping malformed line {i}: {e}")
        else:
            data = json.load(f)
    return data


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(name: str):
    """
    Load LLaVA-1.5 using the correct class: LlavaForConditionalGeneration.
    Returns (processor, model).
    """
    try:
        import torch
        from transformers import LlavaForConditionalGeneration, AutoProcessor
    except ImportError as e:
        print(f"BLOCKED: Missing required packages: {e}")
        print("Run: pip install transformers torch")
        sys.exit(2)

    on_gpu = torch.cuda.is_available()
    dtype  = torch.float16  # Force float16 to save ~14GB of RAM even on CPU
    device = "cuda" if on_gpu else "cpu"

    print(f"Loading processor from {name} …")
    processor = AutoProcessor.from_pretrained(name)

    print(f"Loading model from {name} (dtype={dtype}, device={device}) …")
    print("  This will take a few minutes on CPU …")
    model = LlavaForConditionalGeneration.from_pretrained(
        name,
        dtype=dtype,              # fixed: was torch_dtype (deprecated)
        device_map=device,
        low_cpu_mem_usage=True,
    )

    return processor, model


# ── LoRA attachment ───────────────────────────────────────────────────────────

def attach_lora(model, r: int, alpha: int, dropout: float):
    """
    Apply LoRA only to the language model's attention layers.
    This is the 'fine-tune a layer' approach — freezes vision encoder completely.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        print("BLOCKED: peft not installed. Run: pip install peft")
        sys.exit(2)

    # Target only the language model's q/v projection layers
    # (NOT the vision encoder — that stays completely frozen)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total if total else 0
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}  ({pct:.2f}%) ← LoRA layers only")
    assert trainable > 0, "LoRA attachment failed — 0 trainable params"

    return model


# ── Dataset wrapper ───────────────────────────────────────────────────────────

class InstructionDataset:
    """Wraps JSONL instruction pairs for HuggingFace Trainer."""
    def __init__(self, data: list[dict], processor, max_length: int):
        self.data       = data
        self.processor  = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item  = self.data[idx]
        convs = item.get("conversations", [])

        user_msg  = convs[0]["value"] if len(convs) > 0 else ""
        asst_msg  = convs[1]["value"] if len(convs) > 1 else ""

        # Format as LLaVA-style conversation (text-only, no image token)
        text = f"USER: {user_msg}\nASSISTANT: {asst_msg}"

        tokenizer = self.processor.tokenizer
        enc = tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: mask the USER portion with -100 so loss is only on ASSISTANT tokens
        user_only = f"USER: {user_msg}\nASSISTANT:"
        user_enc  = tokenizer(user_only, return_tensors="pt")
        user_len  = user_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:user_len] = -100          # mask prompt
        labels[attention_mask == 0] = -100  # mask padding

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ── Trainer setup ─────────────────────────────────────────────────────────────

def build_trainer(model, dataset, args):
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as e:
        print(f"BLOCKED: {e}")
        sys.exit(2)

    on_cpu = detect_hardware().startswith("CPU")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        use_cpu=on_cpu,
        fp16=not on_cpu,       # fp16 on GPU only
        report_to="none",      # no wandb/tensorboard
        dataloader_num_workers=0,  # avoid multiprocessing issues on Windows
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    args = parse_args()

    # 1. Hardware
    hw = detect_hardware()
    print(f"Hardware: {hw}")
    if hw.startswith("CPU"):
        print("  NOTE: CPU training is slow. A small --max_samples cap is recommended.")

    # 2. Load data
    try:
        raw_data = load_instruction_data(args.data_path)
    except Exception as e:
        print(f"Failed to load dataset '{args.data_path}': {e}")
        sys.exit(1)

    # Cap samples for CPU feasibility
    if args.max_samples and args.max_samples > 0:
        raw_data = raw_data[:args.max_samples]

    print(f"Loaded {len(raw_data):,} training examples")

    # 3. Load model
    processor, model = load_model(args.model_name_or_path)

    # 4. Attach LoRA (language-layer only)
    if args.lora_enable:
        print("Attaching LoRA adapters to language attention layers …")
        model = attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)
    else:
        print("WARNING: --lora_enable not set. Full model fine-tuning is very slow on CPU.")

    # 5. Build dataset
    dataset = InstructionDataset(raw_data, processor, max_length=args.max_seq_length)

    # 6. Build trainer
    trainer = build_trainer(model, dataset, args)

    # 7. Train
    print(f"\nStarting training: {args.num_train_epochs} epoch(s), "
          f"{len(dataset):,} samples, batch={args.per_device_train_batch_size} "
          f"× accum={args.gradient_accumulation_steps}")
    try:
        train_result = trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        sys.exit(1)

    print(f"Training loss: {train_result.training_loss:.4f}")

    # 8. Save LoRA adapter only (not full model — much smaller)
    if args.lora_enable:
        model.save_pretrained(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"LoRA adapter saved to: {args.output_dir}")
    else:
        trainer.save_model(args.output_dir)
        print(f"Model saved to: {args.output_dir}")

    print("\nDone! Fine-tuning complete.")


if __name__ == "__main__":
    train()
