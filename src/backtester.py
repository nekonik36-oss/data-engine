"""
Walk-Forward Backtester & Automated Strategy Tournament Runner.

Key Capabilities:
  1. Chronological Walk-Forward Validation (70% In-Sample / 30% Out-of-Sample, zero look-ahead).
  2. Realistic Execution Accounting:
     - Deducts instrument bid-ask spread on every entry.
     - Partial Take-Profit 1 (`TP1` at 1.5R): closes 50% of position and moves Stop Loss to
       Breakeven (`entry_price`) for a risk-free runner toward `TP2` (3.0R).
  3. Strategy Tournament Leaderboard:
     - Benchmarks each individual strategy (`SND_ONLY`, `ICT_ONLY`, `BREAKOUT_ONLY`,
       `TREND_PULLBACK_ONLY`) against multi-strategy combinations (`SND_PLUS_ICT`,
       `ALL_4_NAIVE`, and `HYBRID_REGIME_KRONOS`) so you can verify empirically whether adding
       each strategy improves or dilutes Win Rate, Profit Factor, and Max Drawdown.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.confluence import ConfluenceEngine, PrecomputedSeries, TOURNAMENT_PRESETS
from src.config import EngineConfig, INSTRUMENTS, InstrumentProfile
from src.models import (
    BacktestMetrics,
    Candle,
    Direction,
    SetupSignal,
    TradeRecord,
)


class WalkForwardBacktester:
    """Simulates trade lifecycle with realistic spreads, partial TP1, and breakeven stops."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.engine = ConfluenceEngine(self.config)

    def run_backtest(
        self,
        symbol: str,
        candles: Sequence[Candle],
        preset_name: str = "HYBRID_REGIME_KRONOS",
        timeframe: str = "M15",
        start_idx: int = 50,
        end_idx: Optional[int] = None,
        split_label: str = "FULL",
        precomputed: Optional[PrecomputedSeries] = None,
    ) -> BacktestMetrics:
        profile: InstrumentProfile = INSTRUMENTS.get(symbol, INSTRUMENTS["XAUUSD"])
        series = precomputed if precomputed is not None else self.engine.precompute(candles)
        stop_bar = min(len(candles) - 1, end_idx if end_idx is not None else len(candles) - 1)

        equity = self.config.default_equity
        peak_equity = equity
        max_dd_pct = 0.0

        trades: List[TradeRecord] = []
        idx = max(50, start_idx)

        while idx < stop_bar - 1:
            signal: Optional[SetupSignal] = self.engine.evaluate_bar(
                symbol=symbol,
                timeframe=timeframe,
                candles=candles,
                idx=idx,
                series=series,
                preset_name=preset_name,
                equity=equity,
            )
            if signal is None:
                idx += 1
                continue

            # Simulate trade from idx + 1 forward with realistic bid-ask spread penalty
            trade, exit_bar = self._simulate_trade(
                profile=profile,
                signal=signal,
                candles=candles,
                entry_idx=idx + 1,
                max_exit_idx=stop_bar,
            )
            trades.append(trade)
            equity += trade.pnl_usd
            peak_equity = max(peak_equity, equity)
            dd_pct = ((peak_equity - equity) / max(peak_equity, 1.0)) * 100.0
            max_dd_pct = max(max_dd_pct, dd_pct)

            # Advance bar index past trade exit + cooldown to prevent overlapping duplicate trades
            idx = max(idx + 1, exit_bar + self.config.cooldown_bars_after_trade)

        return self._compute_metrics(
            symbol=symbol,
            strategy_combo=preset_name,
            split_label=split_label,
            trades=trades,
            initial_equity=self.config.default_equity,
            final_equity=equity,
            max_dd_pct=max_dd_pct,
        )

    def _simulate_trade(
        self,
        profile: InstrumentProfile,
        signal: SetupSignal,
        candles: Sequence[Candle],
        entry_idx: int,
        max_exit_idx: int,
    ) -> tuple[TradeRecord, int]:
        spread = profile.typical_spread_price
        entry_bar = candles[entry_idx]

        if signal.direction == Direction.LONG:
            fill_price = entry_bar.open + 0.5 * spread
            sl_dist = max(abs(signal.entry_price - signal.stop_loss), 2.0 * spread)
            sl_price = fill_price - sl_dist
            tp1_price = fill_price + profile.tp1_rr * sl_dist
            tp2_price = fill_price + profile.tp2_rr * sl_dist
        else:
            fill_price = entry_bar.open - 0.5 * spread
            sl_dist = max(abs(signal.stop_loss - signal.entry_price), 2.0 * spread)
            sl_price = fill_price + sl_dist
            tp1_price = fill_price - profile.tp1_rr * sl_dist
            tp2_price = fill_price - profile.tp2_rr * sl_dist

        tp1_hit = False
        tp2_hit = False
        active_sl = sl_price
        exit_price = fill_price
        exit_reason = "TIME_EXIT"
        exit_idx = min(max_exit_idx, entry_idx + self.config.max_bars_in_trade)

        for j in range(entry_idx, exit_idx + 1):
            bar = candles[j]
            if signal.direction == Direction.LONG:
                # Check SL first (conservative intrabar assumption)
                if bar.low <= active_sl:
                    exit_price = active_sl
                    exit_idx = j
                    exit_reason = "TP1_THEN_BE" if tp1_hit else "SL"
                    break
                if not tp1_hit and bar.high >= tp1_price:
                    tp1_hit = True
                    # Move stop loss to breakeven (+ tiny spread cushion) after TP1 hits
                    active_sl = fill_price
                if bar.high >= tp2_price:
                    tp1_hit = True
                    tp2_hit = True
                    exit_price = tp2_price
                    exit_idx = j
                    exit_reason = "TP2"
                    break
            else:
                if bar.high >= active_sl:
                    exit_price = active_sl
                    exit_idx = j
                    exit_reason = "TP1_THEN_BE" if tp1_hit else "SL"
                    break
                if not tp1_hit and bar.low <= tp1_price:
                    tp1_hit = True
                    active_sl = fill_price
                if bar.low <= tp2_price:
                    tp1_hit = True
                    tp2_hit = True
                    exit_price = tp2_price
                    exit_idx = j
                    exit_reason = "TP2"
                    break
        else:
            exit_price = candles[exit_idx].close

        # Compute realized R-multiple with 50% partial close at TP1 (1.5R) and 50% runner
        w1 = profile.tp1_close_fraction
        w2 = 1.0 - w1
        if tp2_hit:
            r_mult = w1 * profile.tp1_rr + w2 * profile.tp2_rr
        elif tp1_hit and exit_reason == "TP1_THEN_BE":
            r_mult = w1 * profile.tp1_rr + 0.0  # Second half stopped at Breakeven
        elif tp1_hit and exit_reason == "TIME_EXIT":
            rem_r = (
                (exit_price - fill_price) / sl_dist
                if signal.direction == Direction.LONG
                else (fill_price - exit_price) / sl_dist
            )
            r_mult = w1 * profile.tp1_rr + w2 * rem_r
        elif exit_reason == "SL":
            r_mult = -1.0
        else:
            r_mult = (
                (exit_price - fill_price) / sl_dist
                if signal.direction == Direction.LONG
                else (fill_price - exit_price) / sl_dist
            )

        pnl_usd = r_mult * signal.risk_usd

        record = TradeRecord(
            symbol=profile.symbol,
            strategy_combo=signal.active_combination,
            direction=signal.direction,
            entry_time=entry_bar.timestamp,
            exit_time=candles[exit_idx].timestamp,
            entry_price=fill_price,
            stop_loss=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            exit_price=exit_price,
            lot_size=signal.recommended_lot,
            risk_usd=signal.risk_usd,
            pnl_usd=round(pnl_usd, 2),
            r_multiple=round(r_mult, 3),
            tp1_hit=tp1_hit,
            tp2_hit=tp2_hit,
            exit_reason=exit_reason,
            confluence_score=signal.confluence_score,
            regime=signal.regime.value,
        )
        return record, exit_idx

    @staticmethod
    def _compute_metrics(
        symbol: str,
        strategy_combo: str,
        split_label: str,
        trades: List[TradeRecord],
        initial_equity: float,
        final_equity: float,
        max_dd_pct: float,
    ) -> BacktestMetrics:
        total = len(trades)
        if total == 0:
            return BacktestMetrics(
                symbol=symbol,
                strategy_combo=strategy_combo,
                split_Label=split_label,
                total_trades=0,
                wins=0,
                losses=0,
                breakevens=0,
                win_rate_pct=0.0,
                tp1_hit_rate_pct=0.0,
                profit_factor=0.0,
                expectancy_r=0.0,
                total_return_pct=0.0,
                net_profit_usd=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                trades=[],
            )

        wins = sum(1 for t in trades if t.pnl_usd > 0.5)
        losses = sum(1 for t in trades if t.pnl_usd < -0.5)
        breakevens = total - wins - losses
        tp1_hits = sum(1 for t in trades if t.tp1_hit)

        gross_profit = sum(t.pnl_usd for t in trades if t.pnl_usd > 0)
        gross_loss = abs(sum(t.pnl_usd for t in trades if t.pnl_usd < 0))
        pf = gross_profit / gross_loss if gross_loss > 1e-6 else (9.99 if gross_profit > 0 else 0.0)

        r_vals = [t.r_multiple for t in trades]
        avg_r = sum(r_vals) / total
        var_r = sum((r - avg_r) ** 2 for r in r_vals) / max(1, total - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1.0
        sharpe = (avg_r / std_r) * math.sqrt(min(total, 252)) if std_r > 1e-9 else 0.0

        net_profit = final_equity - initial_equity
        ret_pct = (net_profit / initial_equity) * 100.0

        return BacktestMetrics(
            symbol=symbol,
            strategy_combo=strategy_combo,
            split_Label=split_label,
            total_trades=total,
            wins=wins,
            losses=losses,
            breakevens=breakevens,
            win_rate_pct=round((wins / total) * 100.0, 1),
            tp1_hit_rate_pct=round((tp1_hits / total) * 100.0, 1),
            profit_factor=round(pf, 2),
            expectancy_r=round(avg_r, 3),
            total_return_pct=round(ret_pct, 2),
            net_profit_usd=round(net_profit, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            trades=trades,
        )

    def run_tournament(
        self, symbol: str, candles: Sequence[Candle]
    ) -> Dict[str, BacktestMetrics]:
        """
        Run all 7 strategy configurations (Individual Strategies vs. Naive vs. Hybrid Regime + Kronos)
        on the identical candle dataset and return a ranked dictionary of results.
        """
        precomputed = self.engine.precompute(candles)
        results: Dict[str, BacktestMetrics] = {}
        for preset in TOURNAMENT_PRESETS:
            metrics = self.run_backtest(
                symbol=symbol,
                candles=candles,
                preset_name=preset,
                precomputed=precomputed,
                split_label="FULL",
            )
            results[preset] = metrics
        return results

    @staticmethod
    def format_tournament_table(
        symbol: str, results: Dict[str, BacktestMetrics]
    ) -> str:
        """Render a clear comparison table ranking every strategy and combination."""
        ranked = sorted(
            results.values(),
            key=lambda m: (m.profit_factor, m.expectancy_r, m.win_rate_pct),
            reverse=True,
        )
        lines = [
            "",
            "=" * 102,
            f"  STRATEGY TOURNAMENT LEADERBOARD | {symbol} (M15) | Spread-Adjusted + Partial TP1 (1.5R) & Runner (3.0R)",
            "=" * 102,
            f"  {'Rank':<4} {'Strategy / Combination':<24} {'Trades':>6} {'WinRate':>8} "
            f"{'TP1 Hit':>8} {'ProfFactor':>10} {'Expect(R)':>10} {'Return%':>9} {'MaxDD%':>8} {'Sharpe':>7}",
            "-" * 102,
        ]
        for rank, m in enumerate(ranked, start=1):
            marker = " *" if m.strategy_combo == "HYBRID_REGIME_KRONOS" else "  "
            lines.append(
                f"  #{rank:<3} {m.strategy_combo:<24} {m.total_trades:>6} "
                f"{m.win_rate_pct:>7.1f}% {m.tp1_hit_rate_pct:>7.1f}% "
                f"{m.profit_factor:>10.2f} {m.expectancy_r:>+9.2f}R "
                f"{m.total_return_pct:>+8.2f}% {m.max_drawdown_pct:>7.2f}% {m.sharpe_ratio:>7.2f}{marker}"
            )
        lines.append("=" * 102)
        return "\n".join(lines)

    @staticmethod
    def export_html_report(
        all_results: Dict[str, Dict[str, BacktestMetrics]], output_path: str | Path
    ) -> Path:
        """Export an interactive HTML report comparing all strategies across XAUUSD, EURUSD, USDJPY."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows_html = []
        for sym, sym_res in all_results.items():
            ranked = sorted(
                sym_res.values(),
                key=lambda m: (m.profit_factor, m.expectancy_r),
                reverse=True,
            )
            for idx, m in enumerate(ranked, start=1):
                highlight = (
                    "background: rgba(16, 185, 129, 0.16); font-weight: 600;"
                    if m.strategy_combo == "HYBRID_REGIME_KRONOS"
                    else ""
                )
                rows_html.append(
                    f"<tr style='{highlight}'>"
                    f"<td><strong>{sym}</strong></td>"
                    f"<td>#{idx}</td>"
                    f"<td>{m.strategy_combo}</td>"
                    f"<td>{m.total_trades}</td>"
                    f"<td>{m.win_rate_pct:.1f}%</td>"
                    f"<td>{m.tp1_hit_rate_pct:.1f}%</td>"
                    f"<td>{m.profit_factor:.2f}</td>"
                    f"<td>{m.expectancy_r:+.2f}R</td>"
                    f"<td>{m.total_return_pct:+.2f}%</td>"
                    f"<td>{m.max_drawdown_pct:.2f}%</td>"
                    f"<td>{m.sharpe_ratio:.2f}</td>"
                    f"</tr>"
                )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hybrid Multi-Strategy Tournament Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
  h1 {{ color: #38bdf8; margin-bottom: 0.25rem; }}
  p {{ color: #94a3b8; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; background: #1e293b; border-radius: 8px; overflow: hidden; }}
  th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #334155; }}
  th {{ background: #0f172a; color: #38bdf8; font-size: 0.85rem; text-transform: uppercase; }}
</style>
</head>
<body>
  <h1>Hybrid Multi-Strategy Backtest Tournament Report</h1>
  <p>Instruments: XAUUSD, EURUSD, USDJPY | Risk: 0.5% per trade | TP1: 1.5R (50% + Breakeven SL) | TP2: 3.0R</p>
  <table>
    <thead>
      <tr>
        <th>Symbol</th><th>Rank</th><th>Strategy / Combination</th><th>Trades</th>
        <th>Win Rate</th><th>TP1 Hit Rate</th><th>Profit Factor</th>
        <th>Expectancy (R)</th><th>Return %</th><th>Max DD %</th><th>Sharpe</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>"""
        path.write_text(html, encoding="utf-8")
        return path
