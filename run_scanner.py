"""
Live / Demo Market Scanner & Copilot Alert Engine (`signal_only` default mode).

Supports two modes:
  1. Single Scan (`python run_scanner.py --once`): Scans recent bars and sends a sample setup.
  2. Continuous Live Watch (`python run_scanner.py`): Watches `XAUUSD`, `EURUSD`, and `USDJPY`
     every 60 seconds (synchronized to M15 candle closes), deduplicates alerts per bar,
     reads live MT5 account equity, and pushes high-confluence Signal Cards to Telegram!
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import time
from typing import Dict

from src.confluence import ConfluenceEngine
from src.config import EngineConfig, INSTRUMENTS
from src.mt5_connector import MT5Connector
from src.remote_sync import write_and_push_telemetry


def run_scanner(watch: bool = True, poll_seconds: int = 60, sync_github: bool = True) -> None:
    console_buffer: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        console_buffer.extend(msg.splitlines() if msg else [""])

    config = EngineConfig()
    engine = ConfluenceEngine(config)
    mt5_conn = MT5Connector(execution_mode=config.execution_mode)
    mt5_conn.connect()

    live_equity = mt5_conn.get_account_equity(config.default_equity)
    mode_banner = (
        f"MT5 Live/Demo Connected (Equity: ${live_equity:,.2f})"
        if mt5_conn.connected
        else "Offline Simulation Feed (Open MetaTrader 5 on PC for Live Broker Bars)"
    )
    tg_banner = (
        "Configured (Ready)"
        if mt5_conn.telegram.is_configured
        else "Not configured (Run `python setup_telegram.py`)"
    )

    log("=" * 78)
    log(f"  HYBRID MULTI-STRATEGY COPILOT SCANNER | Execution Mode: {config.execution_mode}")
    log(f"  Data Source    : {mode_banner}")
    log(f"  Telegram Alert : {tg_banner}")
    log(f"  Kronos Model   : {config.kronos.model_id} (Device: {config.kronos.device})")
    log("=" * 78)

    last_alerted_bar: Dict[str, datetime] = {}
    cycle_count = 0

    while True:
        cycle_count += 1
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        live_equity = mt5_conn.get_account_equity(config.default_equity)
        log(f"\n[{now_str}] Scanning XAUUSD, EURUSD, USDJPY (M15)...")

        latest_prices: Dict[str, float] = {}
        new_signal_fired = False

        for symbol in INSTRUMENTS:
            candles = mt5_conn.fetch_recent_candles(symbol=symbol, timeframe="M15", count=260)
            if len(candles) < 60:
                log(f"  [{symbol}] Insufficient bars ({len(candles)}).")
                continue

            latest_prices[symbol] = candles[-1].close
            series = engine.precompute(candles)
            # In live MT5 mode, index -1 is the currently forming bar and -2 is the just-closed M15 bar.
            # In single-scan / demo preview, check the most recent closed bars.
            lookback_bars = 2 if (mt5_conn.connected and watch and last_alerted_bar) else 20

            found = False
            for idx in range(len(candles) - 1, max(50, len(candles) - 1 - lookback_bars), -1):
                bar_ts = candles[idx].timestamp
                if last_alerted_bar.get(symbol) == bar_ts:
                    found = True
                    break

                signal = engine.evaluate_bar(
                    symbol=symbol,
                    timeframe="M15",
                    candles=candles,
                    idx=idx,
                    series=series,
                    preset_name="HYBRID_REGIME_KRONOS",
                    equity=live_equity,
                )
                if signal is not None:
                    last_alerted_bar[symbol] = bar_ts
                    console_buffer.extend(signal.format_signal_card().splitlines())
                    status_msg = mt5_conn.handle_signal(signal)
                    log(f"  {status_msg}")
                    found = True
                    new_signal_fired = True
                    break

            if not found:
                last_c = candles[-1]
                log(
                    f"  [{symbol:<6}] Price={last_c.close:<9} | No new setup on latest bar. Waiting for A+ confluence..."
                )

        # Auto-sync Home PC telemetry & console log to GitHub on startup or when a new signal fires
        if cycle_count == 1 or new_signal_fired or not watch:
            pushed = write_and_push_telemetry(
                console_lines=console_buffer,
                extra_status={
                    "mt5_connected": mt5_conn.connected,
                    "live_equity": live_equity,
                    "telegram_configured": mt5_conn.telegram.is_configured,
                    "kronos_active_backend": engine.kronos._active_backend,
                    "latest_prices": latest_prices,
                    "cycle_count": cycle_count,
                },
                push_to_github=sync_github,
            )
            if pushed:
                print("  [GitHub Sync] Home PC status & console log synced to GitHub for remote inspection.")

        if not watch:
            break

        print(f"\n  Sleeping {poll_seconds}s until next scan cycle (Press Ctrl+C to stop)...")
        try:
            time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("\n  Scanner stopped by user.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Multi-Strategy Copilot Scanner")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan pass and exit instead of continuous watching.",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Disable automatic GitHub telemetry push.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds for continuous watch mode (default: 60).",
    )
    args = parser.parse_args()
    run_scanner(watch=not args.once, poll_seconds=args.interval, sync_github=not args.no_sync)


if __name__ == "__main__":
    main()
