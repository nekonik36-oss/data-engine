"""ICT Strategy: Liquidity Pool Sweep + Market Structure Shift (MSS) + Fair Value Gap (FVG)."""

from __future__ import annotations

from typing import Optional, Sequence

from src.models import Candle, Direction, PriceZone, StrategyVote


class ICTStrategy:
    """
    Implements core Inner Circle Trader (ICT) / Smart Money Concepts mechanics:
      1. Liquidity Sweep (Stop Hunt): Price sweeps below a recent Session/Swing Low
         (Sell-Side Liquidity / SSL) or above a recent Session/Swing High
         (Buy-Side Liquidity / BSL) within the last `sweep_window` bars, then closes back inside.
      2. Market Structure Shift (MSS) & Displacement: Energetic candle body displacement
         in the reversal direction.
      3. Fair Value Gap (FVG): 3-candle price inefficiency (c[i-2].high < c[i].low for Bullish FVG,
         or c[i-2].low > c[i].high for Bearish FVG) or immediate post-sweep displacement entry.
    """

    name = "ICT"

    def __init__(self, liquidity_lookback: int = 32, recent_sweep_bars: int = 6) -> None:
        self.liquidity_lookback = liquidity_lookback
        self.recent_sweep_bars = recent_sweep_bars

    @staticmethod
    def detect_recent_fvg(
        candles: Sequence[Candle], idx: int, direction: Direction
    ) -> Optional[PriceZone]:
        """Look for a 3-candle Fair Value Gap formed within the last 5 bars."""
        start = max(2, idx - 4)
        for k in range(idx, start - 1, -1):
            c0 = candles[k - 2]
            c1 = candles[k - 1]
            c2 = candles[k]
            if direction == Direction.LONG and c2.low > c0.high and c1.is_bullish:
                return PriceZone(
                    zone_type="BULL_FVG",
                    pattern="ICT_FVG",
                    top=c2.low,
                    bottom=c0.high,
                    created_idx=k,
                    created_at=c2.timestamp,
                )
            if direction == Direction.SHORT and c2.high < c0.low and c1.is_bearish:
                return PriceZone(
                    zone_type="BEAR_FVG",
                    pattern="ICT_FVG",
                    top=c0.low,
                    bottom=c2.high,
                    created_idx=k,
                    created_at=c2.timestamp,
                )
        return None

    def evaluate(
        self, candles: Sequence[Candle], atr: Sequence[float], idx: int
    ) -> StrategyVote:
        if idx < self.liquidity_lookback + self.recent_sweep_bars + 2:
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Warming up ICT liquidity pools",
            )

        curr = candles[idx]
        cur_atr = max(atr[idx], 1e-9)
        p_fmt = "{:.5f}" if curr.close < 10.0 else ("{:.3f}" if curr.close < 500.0 else "{:.2f}")

        # Define prior liquidity pool before the recent sweep window
        pool_end = idx - self.recent_sweep_bars
        pool_start = max(0, pool_end - self.liquidity_lookback)
        pool_slice = candles[pool_start:pool_end]
        if not pool_slice:
            return StrategyVote(self.name, Direction.NEUTRAL, 0.0, "Empty liquidity pool")

        bsl_high = max(c.high for c in pool_slice)  # Buy-Side Liquidity (above swing highs)
        ssl_low = min(c.low for c in pool_slice)    # Sell-Side Liquidity (below swing lows)

        recent_slice = candles[pool_end : idx + 1]
        lowest_recent = min(c.low for c in recent_slice)
        highest_recent = max(c.high for c in recent_slice)

        # Check Bullish ICT Setup: Sell-Side Liquidity (SSL) swept + bullish displacement / MSS + FVG
        ssl_swept = lowest_recent < ssl_low and curr.close > ssl_low
        if ssl_swept and curr.is_bullish:
            fvg = self.detect_recent_fvg(candles, idx, Direction.LONG)
            displacement = curr.body >= 0.55 * cur_atr or (
                curr.close - lowest_recent >= 0.85 * cur_atr
            )
            if displacement or fvg is not None:
                fvg_bonus = 0.14 if fvg is not None else 0.0
                killzone_bonus = 0.08 if curr.timestamp.hour in (7, 8, 9, 12, 13, 14) else 0.0
                conf = min(0.95, 0.68 + fvg_bonus + killzone_bonus)
                sl = lowest_recent - 0.45 * cur_atr
                fvg_note = (
                    f" + Bullish FVG [{p_fmt.format(fvg.bottom)}-{p_fmt.format(fvg.top)}]"
                    if fvg
                    else " + MSS Displacement"
                )
                return StrategyVote(
                    strategy_name=self.name,
                    direction=Direction.LONG,
                    confidence=conf,
                    reason=(
                        f"SSL Liquidity Sweep below {p_fmt.format(ssl_low)} "
                        f"(low {p_fmt.format(lowest_recent)}){fvg_note}"
                    ),
                    suggested_sl=sl,
                    zone=fvg,
                )

        # Check Bearish ICT Setup: Buy-Side Liquidity (BSL) swept + bearish displacement / MSS + FVG
        bsl_swept = highest_recent > bsl_high and curr.close < bsl_high
        if bsl_swept and curr.is_bearish:
            fvg = self.detect_recent_fvg(candles, idx, Direction.SHORT)
            displacement = curr.body >= 0.55 * cur_atr or (
                highest_recent - curr.close >= 0.85 * cur_atr
            )
            if displacement or fvg is not None:
                fvg_bonus = 0.14 if fvg is not None else 0.0
                killzone_bonus = 0.08 if curr.timestamp.hour in (7, 8, 9, 12, 13, 14) else 0.0
                conf = min(0.95, 0.68 + fvg_bonus + killzone_bonus)
                sl = highest_recent + 0.45 * cur_atr
                fvg_note = (
                    f" + Bearish FVG [{p_fmt.format(fvg.bottom)}-{p_fmt.format(fvg.top)}]"
                    if fvg
                    else " + MSS Displacement"
                )
                return StrategyVote(
                    strategy_name=self.name,
                    direction=Direction.SHORT,
                    confidence=conf,
                    reason=(
                        f"BSL Liquidity Sweep above {p_fmt.format(bsl_high)} "
                        f"(high {p_fmt.format(highest_recent)}){fvg_note}"
                    ),
                    suggested_sl=sl,
                    zone=fvg,
                )

        return StrategyVote(
            strategy_name=self.name,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=(
                f"Waiting for BSL ({p_fmt.format(bsl_high)}) or "
                f"SSL ({p_fmt.format(ssl_low)}) liquidity sweep"
            ),
        )
