"""Asian Session Compression & London/NY Volatility Breakout Strategy."""

from __future__ import annotations

from typing import Sequence

from src.models import Candle, Direction, StrategyVote


class SessionBreakoutStrategy:
    """
    Measures the Asian Session consolidation box (or recent N-bar compression box)
    and triggers when a London/NY session candle decisively breaks and closes outside
    the box with strong body displacement and expanding ATR.
    """

    name = "BREAKOUT"

    def __init__(self, box_lookback: int = 20) -> None:
        self.box_lookback = box_lookback

    def evaluate(
        self,
        candles: Sequence[Candle],
        atr: Sequence[float],
        adx: Sequence[float],
        idx: int,
        is_24_7: bool = False,
    ) -> StrategyVote:
        if idx < self.box_lookback + 5:
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Warming up Session Breakout range box",
            )

        curr = candles[idx]
        prev = candles[idx - 1]
        cur_atr = max(atr[idx], 1e-9)
        cur_adx = adx[idx]

        # Build the preceding consolidation box (excluding current bar)
        box_bars = candles[idx - self.box_lookback : idx]
        box_high = max(b.high for b in box_bars)
        box_low = min(b.low for b in box_bars)
        box_height = box_high - box_low

        # Ensure the preceding box was reasonably compressed (not already over-extended)
        is_compressed = box_height <= 5.5 * cur_atr
        strong_body = curr.body >= 0.60 * cur_atr and curr.body >= 0.55 * curr.range

        if not is_compressed:
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=f"Prior range too wide for clean breakout ({box_height / cur_atr:.1f}x ATR)",
            )

        # Bullish Expansion Breakout above box high
        if curr.close > box_high and prev.close <= box_high and strong_body:
            session_bonus = 0.12 if (is_24_7 or (7 <= curr.timestamp.hour <= 16)) else 0.0
            adx_bonus = 0.08 if cur_adx >= 20.0 else 0.0
            conf = min(0.92, 0.66 + session_bonus + adx_bonus)
            sl = max(box_low, curr.low - 1.2 * cur_atr)
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.LONG,
                confidence=conf,
                reason=(
                    f"Bullish volatility breakout above compression high {box_high:.2f} "
                    f"(body={curr.body / cur_atr:.2f}x ATR, ADX={cur_adx:.1f})"
                ),
                suggested_sl=sl,
            )

        # Bearish Expansion Breakout below box low
        if curr.close < box_low and prev.close >= box_low and strong_body:
            session_bonus = 0.12 if (is_24_7 or (7 <= curr.timestamp.hour <= 16)) else 0.0
            adx_bonus = 0.08 if cur_adx >= 20.0 else 0.0
            conf = min(0.92, 0.66 + session_bonus + adx_bonus)
            sl = min(box_high, curr.high + 1.2 * cur_atr)
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.SHORT,
                confidence=conf,
                reason=(
                    f"Bearish volatility breakout below compression low {box_low:.2f} "
                    f"(body={curr.body / cur_atr:.2f}x ATR, ADX={cur_adx:.1f})"
                ),
                suggested_sl=sl,
            )

        return StrategyVote(
            strategy_name=self.name,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=f"Inside compression box [{box_low:.2f} - {box_high:.2f}]",
        )
