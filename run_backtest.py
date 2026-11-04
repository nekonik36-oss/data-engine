"""
Automated Strategy Tournament & Walk-Forward Backtest CLI.

Runs all 4 individual strategies (`SND_ONLY`, `ICT_ONLY`, `BREAKOUT_ONLY`, `TREND_PULLBACK_ONLY`)
and multi-strategy ensembles (`SND_PLUS_ICT`, `ALL_4_NAIVE`, `HYBRID_REGIME_KRONOS`) across
`XAUUSD`, `EURUSD`, and `USDJPY`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.backtester import WalkForwardBacktester
from src.config import EngineConfig, INSTRUMENTS
from src.models import BacktestMetrics
from src.mt5_connector import generate_realistic_market_candles


def main() -> None:
    config = EngineConfig()
    backtester = WalkForwardBacktester(config)

    all_results: Dict[str, Dict[str, BacktestMetrics]] = {}
    print("=" * 102)
    print("  HYBRID MULTI-STRATEGY ENGINE | AUTOMATED TOURNAMENT & WALK-FORWARD VALIDATION")
    print("  Hardware Profile: RTX 5060 (8GB VRAM) -> Kronos Model: NeoQuasar/Kronos-base (102.3M)")
    print("=" * 102)

    for symbol in ("XAUUSD", "EURUSD", "USDJPY"):
        candles = generate_realistic_market_candles(symbol=symbol, count=1100)
        tournament_res = backtester.run_tournament(symbol=symbol, candles=candles)
        all_results[symbol] = tournament_res
        print(backtester.format_tournament_table(symbol, tournament_res))

        # Also run chronological Walk-Forward Check (70% In-Sample vs 30% Out-of-Sample)
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

    # Display a live example of the Copilot Signal Card on XAUUSD
    gold_candles = generate_realistic_market_candles(symbol="XAUUSD", count=300)
    gold_series = backtester.engine.precompute(gold_candles)
    sample_signal = None
    for idx in range(len(gold_candles) - 1, 50, -1):
        sig = backtester.engine.evaluate_bar(
            symbol="XAUUSD",
            timeframe="M15",
            candles=gold_candles,
            idx=idx,
            series=gold_series,
            preset_name="HYBRID_REGIME_KRONOS",
        )
        if sig is not None:
            sample_signal = sig
            break

    if sample_signal is not None:
        print("\n  SAMPLE COPILOT SIGNAL CARD (What you see when a high-probability setup triggers):")
        print(sample_signal.format_signal_card())

    report_path = backtester.export_html_report(
        all_results, Path(__file__).parent / "reports" / "tournament_report.html"
    )
    print(f"\n  [Report Saved] Interactive HTML Tournament Leaderboard: {report_path}")


if __name__ == "__main__":
    main()
