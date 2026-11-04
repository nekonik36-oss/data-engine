# Home PC Setup Guide (RTX 5060 + MetaTrader 5 + Telegram + TradingView)

Follow these 5 quick steps when you get home to your Personal PC.

---

## Step 0: Before Leaving — Push from Laptop to GitHub
Open a PowerShell terminal on your laptop and run:
```powershell
cd C:\Users\nikmuhammadshahiran\projects\data-engine
git push -u origin main
```
*(Approve the one-time GitHub browser sign-in prompt if asked).*

---

## Step 1: Clone & Create Virtual Environment on Your Home PC
Open PowerShell on your **Personal PC (RTX 5060)** and run:
```powershell
git clone https://github.com/nekonik36-oss/data-engine.git
cd data-engine
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install MetaTrader5 huggingface_hub safetensors
```

### Install PyTorch with CUDA (for your RTX 5060 8GB VRAM)
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```
*(Optional: To enable native `NeoQuasar/Kronos-base` weights, clone `https://github.com/shiyu-coder/Kronos.git` alongside or into `PYTHONPATH`—`src/strategies/kronos_adapter.py` will automatically detect and run `NeoQuasar/Kronos-base` on `cuda:0`!).*

---

## Step 2: Connect Your Telegram Bot (`setup_telegram.py`)
1. On your phone's **Telegram app**, message **`@BotFather`** -> send `/newbot`.
2. Copy the **HTTP API Token**, tap the link to your new bot (`t.me/your_bot`), and tap **Start**.
3. On your Personal PC, run:
```powershell
python setup_telegram.py
```
- Paste your token when prompted (typing is hidden and saved locally to `.env`).
- It will automatically detect your `TELEGRAM_CHAT_ID` and send a **Test Copilot Signal Card** to your phone!

---

## Step 3: Run the Strategy Tournament & Verify Everything
```powershell
python -m tests.test_engine
python run_backtest.py
```
Open `reports/tournament_report.html` in your browser to view the full Strategy Leaderboard across **`XAUUSD`**, **`EURUSD`**, and **`USDJPY`**.

---

## Step 4: Start the Live Copilot Scanner with MetaTrader 5
1. Open your **MetaTrader 5** terminal on your Personal PC and log into your **Demo Account**.
2. Make sure `XAUUSD`, `EURUSD`, and `USDJPY` are visible in the MT5 **Market Watch** window.
3. Run the scanner in default `signal_only` Copilot mode:
```powershell
python run_scanner.py
```
Whenever a high-confluence setup triggers, you will see the **Copilot Signal Card** in your terminal AND receive an instant push notification on **Telegram** with the exact `Entry`, `Stop Loss`, `TP1 (1.5R)`, `TP2 (3.0R)`, and `Lot Size`!

---

## Step 5: Add the Companion Indicator to TradingView
1. Open [`pine/hybrid_multi_strategy.pine`](pine/hybrid_multi_strategy.pine) in any text editor and copy the code.
2. In **TradingView**, open the **Pine Editor** tab at the bottom, paste the script, and click **Add to chart**.
3. You will see the **Supply & Demand (SND) zones**, **ICT Liquidity Sweeps & Fair Value Gaps**, **Asian Session range**, **EMA 20/50/200 ribbon**, and **BUY/SELL labels** with `SL`/`TP1`/`TP2` levels plotted directly on your chart.
