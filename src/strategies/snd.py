"""Supply & Demand (SND) Unmitigated Institutional Zone Strategy."""

from __future__ import annotations

from typing import List, Sequence

from src.models import Candle, Direction, PriceZone, StrategyVote


class SupplyDemandStrategy:
    """
    Identifies institutional Supply and Demand zones created by explosive
    imbalance departures from a tight base:
      - Demand (DBR / RBR): Tight base candle followed by explosive bullish impulse (>1.20x ATR)
      - Supply (RBD / DBD): Tight base candle followed by explosive bearish impulse (>1.20x ATR)

    Tracks zone mitigation causally and votes LONG/SHORT when price returns to
    an unmitigated (fresh or 1st-touch) zone and prints a decisive displacement rejection.
    """

    name = "SND"

    def __init__(self, lookback_bars: int = 85, max_touches: int = 1) -> None:
        self.lookback_bars = lookback_bars
        self.max_touches = max_touches

    def extract_active_zones(
        self, candles: Sequence[Candle], atr: Sequence[float], idx: int
    ) -> List[PriceZone]:
        """Extract unmitigated Supply & Demand zones prior to bar `idx`."""
        start = max(2, idx - self.lookback_bars)
        zones: List[PriceZone] = []

        for i in range(start, idx - 1):
            base = candles[i - 1]
            impulse = candles[i]
            cur_atr = max(atr[i], 1e-9)

            if base.range > 1.20 * cur_atr:
                continue

            # Bullish Impulse Departure -> DEMAND Zone (DBR or RBR)
            if (
                impulse.is_bullish
                and impulse.body >= 1.15 * cur_atr
                and impulse.close > base.high
            ):
                pattern = "DBR" if base.is_bearish else "RBR"
                z_top = max(base.open, base.close)
                z_bottom = base.low
                strength = min(1.0, impulse.body / (1.7 * cur_atr))
                zones.append(
                    PriceZone(
                        zone_type="DEMAND",
                        pattern=pattern,
                        top=z_top,
                        bottom=z_bottom,
                        created_idx=i,
                        created_at=impulse.timestamp,
                        strength=strength,
                    )
                )

            # Bearish Impulse Departure -> SUPPLY Zone (RBD or DBD)
            elif (
                impulse.is_bearish
                and impulse.body >= 1.15 * cur_atr
                and impulse.close < base.low
            ):
                pattern = "RBD" if base.is_bullish else "DBD"
                z_top = base.high
                z_bottom = min(base.open, base.close)
                strength = min(1.0, impulse.body / (1.7 * cur_atr))
                zones.append(
                    PriceZone(
                        zone_type="SUPPLY",
                        pattern=pattern,
                        top=z_top,
                        bottom=z_bottom,
                        created_idx=i,
                        created_at=impulse.timestamp,
                        strength=strength,
                    )
                )

        active: List[PriceZone] = []
        for z in zones:
            for j in range(z.created_idx + 1, idx):
                bar = candles[j]
                if z.zone_type == "DEMAND":
                    if bar.close < z.bottom:
                        z.mitigated = True
                        break
                    if bar.low <= z.top:
                        z.touch_count += 1
                else:
                    if bar.close > z.top:
                        z.mitigated = True
                        break
                    if bar.high >= z.bottom:
                        z.touch_count += 1
            if not z.mitigated and z.touch_count <= self.max_touches:
                active.append(z)

        return active

    def evaluate(
        self, candles: Sequence[Candle], atr: Sequence[float], idx: int
    ) -> StrategyVote:
        if idx < 25:
            return StrategyVote(
                strategy_name=self.name,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Warming up SND zone history",
            )

        curr = candles[idx]
        prev = candles[idx - 1]
        cur_atr = max(atr[idx], 1e-9)
        active_zones = self.extract_active_zones(candles, atr, idx)

        for z in reversed(active_zones):
            buffer = 0.20 * cur_atr
            if z.zone_type == "DEMAND":
                touched_now = (curr.low <= z.top + buffer and curr.close >= z.bottom) or (
                    prev.low <= z.top + buffer and prev.close >= z.bottom
                )
                bullish_rejection = (
                    curr.is_bullish
                    and curr.close > prev.high
                    and (curr.lower_wick >= 0.25 * curr.range or curr.body >= 0.55 * cur_atr)
                )
                if touched_now and bullish_rejection:
                    fresh_bonus = 0.15 if z.touch_count == 0 else 0.05
                    conf = min(0.95, 0.68 + 0.18 * z.strength + fresh_bonus)
                    sl = min(z.bottom, curr.low) - 0.50 * cur_atr
                    return StrategyVote(
                        strategy_name=self.name,
                        direction=Direction.LONG,
                        confidence=conf,
                        reason=(
                            f"Rejected unmitigated {z.pattern} Demand Zone "
                            f"[{z.bottom:.2f}-{z.top:.2f}] (touches={z.touch_count})"
                        ),
                        suggested_sl=sl,
                        zone=z,
                    )

            elif z.zone_type == "SUPPLY":
                touched_now = (curr.high >= z.bottom - buffer and curr.close <= z.top) or (
                    prev.high >= z.bottom - buffer and prev.close <= z.top
                )
                bearish_rejection = (
                    curr.is_bearish
                    and curr.close < prev.low
                    and (curr.upper_wick >= 0.25 * curr.range or curr.body >= 0.55 * cur_atr)
                )
                if touched_now and bearish_rejection:
                    fresh_bonus = 0.15 if z.touch_count == 0 else 0.05
                    conf = min(0.95, 0.68 + 0.18 * z.strength + fresh_bonus)
                    sl = max(z.top, curr.high) + 0.50 * cur_atr
                    return StrategyVote(
                        strategy_name=self.name,
                        direction=Direction.SHORT,
                        confidence=conf,
                        reason=(
                            f"Rejected unmitigated {z.pattern} Supply Zone "
                            f"[{z.bottom:.2f}-{z.top:.2f}] (touches={z.touch_count})"
                        ),
                        suggested_sl=sl,
                        zone=z,
                    )

        return StrategyVote(
            strategy_name=self.name,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=f"No active SND zone rejection ({len(active_zones)} unmitigated zones tracked)",
        )
