#!/usr/bin/env python3
"""
scripts/lora_smoke_test.py

LoRA Training Smoke Test for SatQuery AI.

Verifies the complete LoRA training pipeline using a tiny model
(distilgpt2 or bert-base-uncased) on synthetic data.

Does NOT require GPU or actual BigEarthNet data.
Does NOT require the full TerraQ-VL checkpoint.

Expected output:
  Total parameters:     NN,NNN,NNN
  Trainable parameters: NNN,NNN
  Trainable percentage: X.XX%
  Initial loss: X.XXXX
  Final loss: X.XXXX  
  Checkpoint: /path/to/checkpoint
  
  ✓ Dataset loaded
  ✓ Model loaded  
  ✓ LoRA attached
  ✓ Trainable parameters > 0
  ✓ Forward pass
  ✓ Finite loss
  ✓ Backward pass
  ✓ Optimizer step
  ✓ Checkpoint saved
  
  ALL CHECKS PASSED
"""

import os
import sys
import tempfile
import time

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    print('BLOCKED: pip install peft transformers torch')
    sys.exit(2)

def main():
    print("Starting LoRA Smoke Test...")
    model_id = "distilgpt2"
    
    # 1. Dataset mock
    print("Loading synthetic dataset...")
    synthetic_texts = [
        "This Sentinel-2 patch shows Pastures and Coniferous forest.",
        "The dominant land-cover types visible here are urban fabric.",
        "This tile is characterised by water bodies."
    ]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        tokenized_data = []
        for text in synthetic_texts:
            encoded = tokenizer(text, truncation=True, max_length=32, padding="max_length", return_tensors="pt")
            tokenized_data.append({
                "input_ids": encoded.input_ids[0],
                "attention_mask": encoded.attention_mask[0],
                "labels": encoded.input_ids[0]  # causal LM uses input as labels
            })
        print("✓ Dataset loaded")
    except Exception as e:
        print(f"✗ Dataset preparation failed: {e}")
        sys.exit(1)
        
    # 2. Model loading
    print(f"Loading {model_id} model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")
        print("✓ Model loaded")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        sys.exit(1)
        
    # 3. LoRA Attachment
    print("Attaching LoRA...")
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["c_attn"], # distilgpt2 attention block
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        print("✓ LoRA attached")
    except Exception as e:
        print(f"✗ LoRA attachment failed: {e}")
        sys.exit(1)
        
    # 4. Parameter checks
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = 100 * trainable_params / total_params
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable percentage: {trainable_pct:.2f}%")
    
    if trainable_params > 0:
        print("✓ Trainable parameters > 0")
    else:
        print("✗ Trainable parameters check failed")
        sys.exit(1)
        
    # 5. Forward Pass and Loss
    try:
        input_ids = tokenized_data[0]["input_ids"].unsqueeze(0)
        attention_mask = tokenized_data[0]["attention_mask"].unsqueeze(0)
        labels = tokenized_data[0]["labels"].unsqueeze(0)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        initial_loss = outputs.loss.item()
        
        print("✓ Forward pass")
        
        import math
        if math.isfinite(initial_loss):
            print("✓ Finite loss")
            print(f"Initial loss: {initial_loss:.4f}")
        else:
            print("✗ Finite loss failed (loss is nan/inf)")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        sys.exit(1)
        
    # 6. Backward Pass and Optimizer step
    try:
        outputs.loss.backward()
        print("✓ Backward pass")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        optimizer.zero_grad()
        
        # Second forward pass to show loss reduction
        outputs2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        final_loss = outputs2.loss.item()
        print(f"Final loss: {final_loss:.4f}")
        print("✓ Optimizer step")
        
    except Exception as e:
        print(f"✗ Backward pass/Optimizer step failed: {e}")
        sys.exit(1)
        
    # 7. Checkpoint saving
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            model.save_pretrained(temp_dir)
            print(f"Checkpoint: {temp_dir}")
            if os.path.exists(os.path.join(temp_dir, "adapter_model.safetensors")) or os.path.exists(os.path.join(temp_dir, "adapter_model.bin")):
                print("✓ Checkpoint saved")
            else:
                print("✗ Checkpoint file not found in temp dir")
                sys.exit(1)
    except Exception as e:
        print(f"✗ Checkpoint saving failed: {e}")
        sys.exit(1)
        
    print("\nALL CHECKS PASSED")

if __name__ == "__main__":
    main()
