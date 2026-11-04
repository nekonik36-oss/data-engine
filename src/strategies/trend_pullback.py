"""EMA 20/50/200 Trend Pullback + RSI & Bollinger Band Exhaustion Strategy."""

from __future__ import annotations

from typing import Sequence

from src.models import Candle, Direction, StrategyVote


class TrendPullbackStrategy:
    """
    Combines institutional trend following with mean-reversion entry timing:
      - Requires multi-EMA alignment (EMA 20 > EMA 50 >= EMA 200 for LONG,
        or EMA 20 < EMA 50 <= EMA 200 for SHORT) + expanding EMA20 slope.
      - Waits for a multi-bar pullback toward EMA 20/50 value zone during active
        London/NY session hours (07:00-16:00 UTC) followed by an impulsive resumption candle.
    """

    name = "TREND_PULLBACK"

    def evaluate(
        self,
        candles: Sequence[Candle],
        ema20: Sequence[float],
        ema50: Sequence[float],
        ema200: Sequence[float],
        rsi: Sequence[float],
        bb_upper: Sequence[float],
        bb_lower: Sequence[float],
        atr: Sequence[float],
        idx: int,
    ) -> StrategyVote:
        if idx < 35:
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Warming up EMA 20/50/200 & RSI/Bollinger indicators",
            )

        curr = candles[idx]
        prev = candles[idx - 1]
        cur_atr = max(atr[idx], 1e-9)
        e20, e50, e200 = ema20[idx], ema50[idx], ema200[idx]
        cur_rsi = rsi[idx]
        in_active_session = 7 <= curr.timestamp.hour <= 16

        if not in_active_session:
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Outside London/NY trend continuation window (07:00-16:00 UTC)",
            )

        # 1. Bullish Trend Pullback
        bull_trend = (
            e20 > e50
            and e50 >= e200
            and (e20 - ema20[idx - 3]) > 0.15 * cur_atr
        )
        if bull_trend:
            pulled_back = prev.is_bearish and (
                curr.low <= e20 + 0.35 * cur_atr or prev.low <= e20 + 0.35 * cur_atr
            )
            resumed_up = (
                curr.is_bullish
                and curr.body >= 0.65 * cur_atr
                and curr.close > prev.high
                and curr.close > e20
            )
            rsi_healthy = 40.0 <= cur_rsi <= 63.0

            if pulled_back and resumed_up and rsi_healthy:
                conf = min(0.92, 0.74 + (0.06 if curr.lower_wick >= 0.25 * curr.range else 0.0))
                sl = min(curr.low, prev.low) - 0.55 * cur_atr
                return StrategyVote(
                    strategy_name=self.name,
                    direction=Direction.LONG,
                    confidence=conf,
                    reason=(
                        f"Bullish EMA20/50 pullback impulse resumption "
                        f"(EMA20={e20:.2f} > EMA50={e50:.2f}, RSI={cur_rsi:.1f})"
                    ),
                    suggested_sl=sl,
                )

        # 2. Bearish Trend Pullback
        bear_trend = (
            e20 < e50
            and e50 <= e200
            and (ema20[idx - 3] - e20) > 0.15 * cur_atr
        )
        if bear_trend:
            pulled_back = prev.is_bullish and (
                curr.high >= e20 - 0.35 * cur_atr or prev.high >= e20 - 0.35 * cur_atr
            )
            resumed_down = (
                curr.is_bearish
                and curr.body >= 0.65 * cur_atr
                and curr.close < prev.low
                and curr.close < e20
            )
            rsi_healthy = 37.0 <= cur_rsi <= 60.0

            if pulled_back and resumed_down and rsi_healthy:
                conf = min(0.92, 0.74 + (0.06 if curr.upper_wick >= 0.25 * curr.range else 0.0))
                sl = max(curr.high, prev.high) + 0.55 * cur_atr
                return StrategyVote(
                    strategy_name=self.name,
                    direction=Direction.SHORT,
                    confidence=conf,
                    reason=(
                        f"Bearish EMA20/50 pullback impulse resumption "
                        f"(EMA20={e20:.2f} < EMA50={e50:.2f}, RSI={cur_rsi:.1f})"
                    ),
                    suggested_sl=sl,
                )

        return StrategyVote(
            strategy_name=self.name,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=f"No aligned EMA pullback trigger (RSI={cur_rsi:.1f})",
        )
