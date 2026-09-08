@echo off
REM ================================================================
REM  SatQuery AI — BigEarthNet LoRA Fine-Tuning Pipeline
REM  PS 26167 / ISRO
REM ================================================================

REM Always run from the project root (one level above scripts\)
cd /d "%~dp0.."

echo.
echo  SATQUERY AI - BIGEARTHNET LORA FINE-TUNING
echo  ============================================
echo  Working directory: %CD%
echo.

REM Step 1: Prepare instruction data from metadata.csv
echo [1/3] Preparing BigEarthNet instruction data...
python -m satquery.train.prepare_bigearthnet_csv ^
    --csv_path data/metadata.csv ^
    --out data/bigearthnet_instructions.jsonl ^
    --max_patches 10000 ^
    --split train

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Data preparation failed.
    pause
    exit /b 1
)
echo       Done. Instruction data ready.
echo.

REM Step 2: Run LoRA fine-tuning
echo [2/3] Starting LoRA fine-tuning (this may take a while on CPU)...
echo       Model: llava-hf/llava-1.5-7b-hf
echo       Output: checkpoints\terraq-vl-bigearthnet-lora
echo.

python -m satquery.train.train ^
    --model_name_or_path llava-hf/llava-1.5-7b-hf ^
    --data_path data/bigearthnet_instructions.jsonl ^
    --lora_enable ^
    --num_train_epochs 1 ^
    --per_device_train_batch_size 1 ^
    --gradient_accumulation_steps 8 ^
    --learning_rate 2e-4 ^
    --output_dir checkpoints/terraq-vl-bigearthnet-lora

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo WARNING: Training returned non-zero exit code.
    echo This may mean GPU/VRAM is insufficient. Check output above.
    echo The fine-tuned adapter may not have been saved.
    pause
    exit /b 1
)
echo       Fine-tuning complete!
echo.

REM Step 3: Update .env to point to the new LoRA adapter
echo [3/3] Updating .env with LoRA adapter path...

REM Check if VLM_LORA_PATH already exists in .env
findstr /C:"VLM_LORA_PATH" .env >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    REM Replace existing line
    powershell -Command "(Get-Content .env) -replace '^VLM_LORA_PATH=.*', 'VLM_LORA_PATH=checkpoints/terraq-vl-bigearthnet-lora' | Set-Content .env"
) else (
    REM Append new line
    echo VLM_LORA_PATH=checkpoints/terraq-vl-bigearthnet-lora >> .env
)

echo       .env updated: VLM_LORA_PATH=checkpoints/terraq-vl-bigearthnet-lora
echo.
echo ================================================================
echo  FINE-TUNING COMPLETE!
echo  Restart the server to load the LoRA adapter:
echo    python api_app.py
echo ================================================================
echo.
pause
