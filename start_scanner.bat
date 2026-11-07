@echo off
title Hybrid Multi-Strategy Copilot Scanner (XAUUSD / EURUSD / USDJPY)
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_scanner.py
) else (
    python run_scanner.py
)
pause
