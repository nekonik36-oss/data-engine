"""
Automated Strategy Tournament & Walk-Forward Backtest CLI.

Automatically pulls REAL historical M15 candles from your open MetaTrader 5 terminal
if connected on your Home PC (or falls back to multi-regime simulated data when offline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.backtester import WalkForwardBacktester
from src.config import EngineConfig
from src.models import BacktestMetrics
from src.mt5_connector import MT5Connector, generate_realistic_market_candles


def main() -> None:
    config = EngineConfig()
    backtester = WalkForwardBacktester(config)
    mt5_conn = MT5Connector(execution_mode="signal_only")
    mt5_connected = mt5_conn.connect()

    source_label = (
        "LIVE METATRADER 5 BROKER HISTORY (M15)"
        if mt5_connected
        else "MULTI-REGIME SIMULATED HISTORY (Open MT5 to backtest on real broker data)"
    )

    all_results: Dict[str, Dict[str, BacktestMetrics]] = {}
    print("=" * 102)
    print("  HYBRID MULTI-STRATEGY ENGINE | AUTOMATED TOURNAMENT & WALK-FORWARD VALIDATION")
    print(f"  Data Source     : {source_label}")
    print(f"  Kronos AI Model : {config.kronos.model_id} (Device: {config.kronos.device})")
    print("=" * 102)

    for symbol in ("XAUUSD", "EURUSD", "USDJPY"):
        if mt5_connected:
            candles = mt5_conn.fetch_recent_candles(symbol=symbol, timeframe="M15", count=2000)
        else:
            candles = generate_realistic_market_candles(symbol=symbol, count=1100)

        tournament_res = backtester.run_tournament(symbol=symbol, candles=candles)
        all_results[symbol] = tournament_res
        print(backtester.format_tournament_table(symbol, tournament_res))

        # Chronological Walk-Forward Check (70% In-Sample vs 30% Out-of-Sample)
        split_idx = int(len(candles) * 0.70)
        precomputed = backtester.engine.precompute(candles)
        is_metrics = backtester.run_backtest(
            symbol=symbol,
            candles=candles,
            preset_name="HYBRID_REGIME_KRONOS",
            start_idx=50,
            end_idx=split_idx,
            split_label="IN_SAMPLE (70%)",
            precomputed=precomputed,
        )
        oos_metrics = backtester.run_backtest(
            symbol=symbol,
            candles=candles,
            preset_name="HYBRID_REGIME_KRONOS",
            start_idx=split_idx,
            end_idx=len(candles) - 1,
            split_label="OUT_OF_SAMPLE (30%)",
            precomputed=precomputed,
        )
        print(
            f"  [Walk-Forward Validation - {symbol} HYBRID_REGIME_KRONOS]\n"
            f"    -> In-Sample (70%)     : {is_metrics.total_trades} trades | "
            f"WinRate={is_metrics.win_rate_pct:.1f}% | TP1 Hit={is_metrics.tp1_hit_rate_pct:.1f}% | "
            f"PF={is_metrics.profit_factor:.2f} | Expectancy={is_metrics.expectancy_r:+.2f}R\n"
            f"    -> Out-of-Sample (30%) : {oos_metrics.total_trades} trades | "
            f"WinRate={oos_metrics.win_rate_pct:.1f}% | TP1 Hit={oos_metrics.tp1_hit_rate_pct:.1f}% | "
            f"PF={oos_metrics.profit_factor:.2f} | Expectancy={oos_metrics.expectancy_r:+.2f}R"
        )

    report_path = backtester.export_html_report(
        all_results, Path(__file__).parent / "reports" / "tournament_report.html"
    )
    print(f"\n  [Report Saved] Interactive HTML Tournament Leaderboard: {report_path}")


if __name__ == "__main__":
    main()
