@echo off
title Hybrid Multi-Strategy Copilot Scanner (XAUUSD / EURUSD / USDJPY)
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

echo ============================================================================
echo   [1/3] Syncing latest Data-Engine updates from GitHub...
echo ============================================================================
git rebase --abort 2>nul
git fetch origin main
git reset --hard origin/main

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv .venv
)

echo ============================================================================
echo   [2/3] Verifying RTX 5060 (Blackwell sm_120 / CUDA 12.8) ^& MT5 Bridge...
echo ============================================================================
".venv\Scripts\python.exe" -c "import torch; assert torch.cuda.is_available(); x = torch.zeros(1, device='cuda:0') + 1" 2>nul
if errorlevel 1 (
    echo Installing PyTorch CUDA 12.8 ^(cu128^) for NVIDIA RTX 5060...
    ".venv\Scripts\pip.exe" install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
    ".venv\Scripts\pip.exe" install -r requirements.txt
    ".venv\Scripts\pip.exe" install huggingface_hub transformers safetensors einops
)

".venv\Scripts\python.exe" -c "import MetaTrader5" 2>nul
if errorlevel 1 (
    echo Installing MetaTrader5 Python bridge...
    ".venv\Scripts\pip.exe" install MetaTrader5
)

if not exist "kronos_repo\model" (
    echo Cloning official Kronos model wrapper...
    git clone https://github.com/shiyu-coder/Kronos.git kronos_repo
)

echo ============================================================================
echo   [3/3] Starting Live M15 Copilot Scanner...
echo ============================================================================
".venv\Scripts\python.exe" run_scanner.py
pause
