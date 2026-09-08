#!/usr/bin/env python3
"""Standalone TerraQ-VL model downloader for SatQuery AI (PS 26167).

This script downloads and caches the base VLM model weights.
Run via download_terraqvl.bat — do NOT import this from the main app.
"""
import os
import sys

# Resolve project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FLAG_FILE = os.path.join(PROJECT_ROOT, 'vlm_ready.flag')

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
    print('[INFO] Loaded .env configuration.')
except ImportError:
    print('[WARN] python-dotenv not installed; reading OS environment only.')

model_path = os.environ.get('VLM_MODEL_PATH', 'llava-hf/llava-1.5-7b-hf')
print(f'[INFO] Target model: {model_path}')

# Check dependencies
try:
    import torch
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print('[INFO] torch and transformers are available.')
except ImportError as e:
    print(f'[ERROR] Missing dependency: {e}')
    print('[ERROR] Please run install.bat to install all dependencies first.')
    sys.exit(1)

# Select device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)
    vram = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    print(f'[INFO] GPU detected: {gpu_name} ({vram} GB VRAM)')
else:
    print('[INFO] No GPU detected. Model will run on CPU (slow but functional).')

# Download processor
print('[STEP 1/2] Downloading processor and tokenizer...')
try:
    processor = AutoProcessor.from_pretrained(model_path)
    print('[STEP 1/2] Processor downloaded OK.')
except Exception as e:
    print(f'[ERROR] Failed to download processor: {e}')
    sys.exit(1)

# Download model weights
print('[STEP 2/2] Downloading model weights (this is the large file ~14 GB)...')
print('[INFO] Progress bars will appear below. Do not close this window.')
try:
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    print('[STEP 2/2] Model downloaded and loaded OK.')
except Exception as e:
    print(f'[ERROR] Failed to download model weights: {e}')
    sys.exit(1)

# Write success flag
try:
    with open(FLAG_FILE, 'w') as f:
        f.write('READY')
    print(f'[INFO] Flag written: {FLAG_FILE}')
except Exception as e:
    print(f'[WARN] Could not write flag file: {e}')

print()
print('[SUCCESS] TerraQ-VL model is ready for local inference!')
print(f'[INFO] Model cached at: {os.path.expanduser("~/.cache/huggingface")}')
print('[INFO] Start the application with start_app.bat')
sys.exit(0)
