# Hybrid Multi-Strategy Signal, Tournament Backtester & Execution Engine

Multi-strategy algorithmic market scanner, walk-forward tournament backtester, and MetaTrader 5 / TradingView copilot for **`XAUUSD`**, **`EURUSD`**, and **`USDJPY`**.

## Architecture & Included Strategies
All **5 strategy & AI modules** are built-in and evaluated simultaneously:
1. **`SND` (`src/strategies/snd.py`)**: Institutional Supply & Demand zones (`RBR`, `DBR`, `RBD`, `DBD`) with causal mitigation tracking.
2. **`ICT` (`src/strategies/ict.py`)**: Session Buy-Side / Sell-Side Liquidity (BSL/SSL) Stop-Hunt Sweeps + Market Structure Shift (MSS) + 3-Candle Fair Value Gap (FVG).
3. **`BREAKOUT` (`src/strategies/breakout.py`)**: Asian Session range compression followed by London/NY Open volatility expansion breakout.
4. **`TREND_PULLBACK` (`src/strategies/trend_pullback.py`)**: Multi-EMA (`20 / 50 / 200`) alignment + `RSI` / Bollinger Band value pullback resumption.
5. **`KRONOS` (`src/strategies/kronos_adapter.py`)**: Local GPU K-line foundation model (`NeoQuasar/Kronos-base`, 102.3M parameters) running `sample_count=10` Monte-Carlo future candle trajectories on **NVIDIA RTX 5060 (8GB VRAM)**.

## Quick Start

### 1. Run Unit Tests & Automated Strategy Tournament (Offline / Any Machine)
```powershell
python -m tests.test_engine
python run_backtest.py
```
This outputs the **Strategy Tournament Leaderboard** (comparing each strategy alone vs. `ALL_4_NAIVE` vs. `HYBRID_REGIME_KRONOS`), runs a **70% In-Sample / 30% Out-of-Sample Walk-Forward validation**, and exports `reports/tournament_report.html`.

### 2. Run the Copilot Market Scanner (`signal_only` Default Mode)
```powershell
python run_scanner.py
```
- **`signal_only` (Default)**: Prints the **Copilot Signal Card** (`Entry`, `SL`, `TP1 1.5R`, `TP2 3.0R`, `Recommended Lot Size`, and per-strategy vote breakdown) without placing any automated trade.
- **`confirm`**: Prompts `[y/N]` in the terminal before sending the order to MetaTrader 5.
- **`auto`**: Automatically executes high-confluence setups on your connected MetaTrader 5 account.

### 3. TradingView Visual Companion
Open [`pine/hybrid_multi_strategy.pine`](pine/hybrid_multi_strategy.pine) in TradingView's Pine Editor and click **Add to Chart** to visualize the SND zones, ICT sweeps/FVGs, Asian compression boxes, EMA ribbon, and `SL`/`TP1`/`TP2` signal labels directly on your chart.
