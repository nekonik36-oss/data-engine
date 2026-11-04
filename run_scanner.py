"""
Live / Demo Market Scanner & Copilot Alert Engine (`signal_only` default mode).

Scans `XAUUSD`, `EURUSD`, and `USDJPY` on M15 using MetaTrader 5 (or simulated feed if
running offline) and prints actionable Copilot Signal Cards with exact Entry, SL, TP1, TP2,
and Lot Size whenever a high-confluence setup triggers.
"""

from __future__ import annotations

from src.confluence import ConfluenceEngine
from src.config import EngineConfig, INSTRUMENTS
from src.mt5_connector import MT5Connector


def scan_once() -> None:
    config = EngineConfig()
    engine = ConfluenceEngine(config)
    mt5_conn = MT5Connector(execution_mode=config.execution_mode)
    mt5_conn.connect()

    mode_banner = (
        "MT5 Live/Demo Connected"
        if mt5_conn.connected
        else "Offline Simulation Feed (Connect MT5 on Personal PC for Live Bars)"
    )
    print("=" * 74)
    print(f"  HYBRID MULTI-STRATEGY COPILOT SCANNER | Mode: {config.execution_mode}")
    print(f"  Data Source: {mode_banner}")
    print(f"  Kronos Model: {config.kronos.model_id} (Device: {config.kronos.device})")
    print("=" * 74)

    for symbol in INSTRUMENTS:
        candles = mt5_conn.fetch_recent_candles(symbol=symbol, timeframe="M15", count=250)
        series = engine.precompute(candles)

        # Check most recent closed bars for active setup
        found = False
        for idx in range(len(candles) - 1, max(50, len(candles) - 25), -1):
            signal = engine.evaluate_bar(
                symbol=symbol,
                timeframe="M15",
                candles=candles,
                idx=idx,
                series=series,
                preset_name="HYBRID_REGIME_KRONOS",
            )
            if signal is not None:
                status_msg = mt5_conn.handle_signal(signal)
                print(f"  {status_msg}\n")
                found = True
                break
        if not found:
            print(f"  [{symbol}] No high-confluence setup on current bar. Waiting patiently...")


if __name__ == "__main__":
    scan_once()
