"""
Autonomous Multi-Strategy MT5 Auto-Trader & Telegram Result Notifier.

Modes of Operation:
  1. Continuous Auto-Trader (`python run_scanner.py`):
     - Evaluates closed M15 candles on `XAUUSD`, `EURUSD`, and `USDJPY`.
     - Automatically places risk-managed orders (`0.50%` equity risk) in MetaTrader 5 when
       a high-confluence `HYBRID_REGIME_KRONOS` setup triggers, and sends a Telegram execution alert.
     - Monitors open trades every 60 seconds:
       * At `TP1 (1.5R)`: locks 50% partial profit (if lot >= 0.02), moves SL to Break-Even,
         and notifies Telegram.
       * On trade close (`TP2`, `SL`, `Break-Even`, or manual close): computes exact realized
         P&L ($ USD and R-multiple) from MT5 deal history and sends the Telegram Result Card!
  2. Copilot Signal-Only Mode (`python run_scanner.py --signal-only`):
     - Sends Signal Cards without placing orders.
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


def run_scanner(
    watch: bool = True,
    poll_seconds: int = 60,
    sync_github: bool = True,
    force_mode: str | None = "auto",
) -> None:
    console_buffer: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        console_buffer.extend(msg.splitlines() if msg else [""])

    config = EngineConfig()
    if force_mode is not None:
        config.execution_mode = force_mode

    engine = ConfluenceEngine(config)
    mt5_conn = MT5Connector(execution_mode=config.execution_mode)
    mt5_conn.connect()

    live_equity = mt5_conn.get_account_equity(config.default_equity)
    mode_banner = (
        f"MT5 Live/Demo Connected (Equity: ${live_equity:,.2f})"
        if mt5_conn.connected
        else f"Offline Simulation Feed (MT5 Status: {mt5_conn.last_error or 'Not Connected'})"
    )
    tg_banner = (
        "Configured (Auto-Trade & Result Alerts Ready)"
        if mt5_conn.telegram.is_configured
        else "Not configured (Run `python setup_telegram.py`)"
    )

    log("=" * 78)
    log(f"  HYBRID MULTI-STRATEGY AUTO-TRADER BOT | Execution Mode: {config.execution_mode.upper()}")
    log(f"  Data Source    : {mode_banner}")
    log(f"  Telegram Alert : {tg_banner}")
    log(f"  Kronos Model   : {config.kronos.model_id} (Device: {config.kronos.device})")
    log("=" * 78)

    last_alerted_bar: Dict[str, datetime] = {}
    cycle_count = 0
    was_connected = mt5_conn.connected

    while True:
        cycle_count += 1
        just_connected = False
        if not mt5_conn.connected and cycle_count > 1:
            if mt5_conn.connect():
                just_connected = True
                live_equity = mt5_conn.get_account_equity(config.default_equity)
                log(f"  [MT5 Connected] MetaTrader 5 is now online! Account Equity: ${live_equity:,.2f}")
                last_alerted_bar.clear()

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        live_equity = mt5_conn.get_account_equity(config.default_equity)

        # 1. Monitor open positions for TP1 (1.5R -> Break-Even) & Closed Trade Results
        trade_events = mt5_conn.manage_open_trades()
        for ev in trade_events:
            log(f"  {ev}")

        open_count = len(mt5_conn.active_trades)
        log(
            f"\n[{now_str}] Scanning XAUUSD, EURUSD, USDJPY (M15) | "
            f"Open Bot Trades: {open_count} | Equity: ${live_equity:,.2f}..."
        )

        latest_prices: Dict[str, float] = {}
        new_signal_fired = bool(trade_events)

        for symbol in INSTRUMENTS:
            candles = mt5_conn.fetch_recent_candles(symbol=symbol, timeframe="M15", count=260)
            if len(candles) < 60:
                log(f"  [{symbol}] Insufficient bars ({len(candles)}).")
                continue

            latest_prices[symbol] = candles[-1].close
            series = engine.precompute(candles)

            # In live MT5 Auto-Trader mode, index -1 is the currently forming M15 bar and
            # index -2 is the just-closed M15 bar. Only trade newly closed bars!
            if mt5_conn.connected and config.execution_mode == "auto":
                candidate_indices = [len(candles) - 2]
            else:
                lookback_bars = 2 if (mt5_conn.connected and watch and last_alerted_bar) else 20
                candidate_indices = list(
                    range(len(candles) - 1, max(50, len(candles) - 1 - lookback_bars), -1)
                )

            found = False
            for idx in candidate_indices:
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
                    f"  [{symbol:<6}] Price={last_c.close:<9} | No new setup on closed M15 bar. Waiting for A+ confluence..."
                )

        # Auto-sync Home PC telemetry & console log to GitHub on startup, trade event, or new signal
        if (
            cycle_count == 1
            or just_connected
            or new_signal_fired
            or not watch
            or (mt5_conn.connected != was_connected)
        ):
            was_connected = mt5_conn.connected
            pushed = write_and_push_telemetry(
                console_lines=console_buffer,
                extra_status={
                    "execution_mode": config.execution_mode,
                    "mt5_connected": mt5_conn.connected,
                    "mt5_error": mt5_conn.last_error,
                    "live_equity": live_equity,
                    "telegram_configured": mt5_conn.telegram.is_configured,
                    "kronos_active_backend": engine.kronos._active_backend,
                    "latest_prices": latest_prices,
                    "active_trades": mt5_conn.active_trades,
                    "bot_stats": mt5_conn.stats,
                    "cycle_count": cycle_count,
                },
                push_to_github=sync_github,
            )
            if pushed:
                print("  [GitHub Sync] Home PC status & trade telemetry synced to GitHub.")

        if not watch:
            break

        print(f"\n  Sleeping {poll_seconds}s until next scan cycle (Press Ctrl+C to stop)...")
        try:
            time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("\n  Auto-Trader stopped by user.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Multi-Strategy MT5 Auto-Trader Bot")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan pass and exit instead of continuous watching.",
    )
    parser.add_argument(
        "--signal-only",
        action="store_true",
        help="Run in Copilot signal-only mode (do not place automated MT5 orders).",
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
    mode = "signal_only" if args.signal_only else "auto"
    run_scanner(
        watch=not args.once,
        poll_seconds=args.interval,
        sync_github=not args.no_sync,
        force_mode=mode,
    )


if __name__ == "__main__":
    main()
