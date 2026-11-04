"""Market Regime Detector and Adaptive Strategy Weight Router."""

from __future__ import annotations

from typing import Dict, Sequence

from src.models import Candle, MarketRegime


class RegimeDetector:
    """
    Classifies the current market environment into one of 5 regimes:
      - TRENDING_UP: Strong directional bullish trend (ADX >= 23, EMA20 > EMA50 > EMA200)
      - TRENDING_DOWN: Strong directional bearish trend (ADX >= 23, EMA20 < EMA50 < EMA200)
      - SESSION_SWEEP_WINDOW: London Open (07:00-10:00 UTC) or NY Open (12:00-15:00 UTC)
        where ICT Liquidity Sweeps and Asian Session Breakouts have highest win rate
      - RANGING: Low ADX (< 20) mean-reverting / zone-bound market
      - HIGH_VOL_CHOP: Extreme ATR spike (> 2.3x average ATR) with conflicting EMAs
    """

    def classify(
        self,
        candles: Sequence[Candle],
        idx: int,
        ema20: Sequence[float],
        ema50: Sequence[float],
        ema200: Sequence[float],
        atr: Sequence[float],
        adx: Sequence[float],
    ) -> MarketRegime:
        if idx < 30:
            return MarketRegime.RANGING

        c = candles[idx]
        cur_atr = atr[idx]
        avg_atr = sum(atr[max(0, idx - 20) : idx + 1]) / min(21, idx + 1)
        atr_ratio = cur_atr / max(avg_atr, 1e-9)

        e20, e50, e200 = ema20[idx], ema50[idx], ema200[idx]
        cur_adx = adx[idx]
        hour_utc = c.timestamp.hour

        # 1. Detect dangerous erratic volatility spikes with tangled EMAs
        ema_tangled = abs(e20 - e50) < 0.25 * cur_atr
        if atr_ratio > 2.35 and ema_tangled:
            return MarketRegime.HIGH_VOL_CHOP

        # 2. Detect strong directional trends
        if cur_adx >= 22.0 and e20 > e50 >= e200 and c.close >= e50:
            return MarketRegime.TRENDING_UP
        if cur_adx >= 22.0 and e20 < e50 <= e200 and c.close <= e50:
            return MarketRegime.TRENDING_DOWN

        # 3. Detect prime London/NY opening liquidity sweep & breakout window
        if (7 <= hour_utc <= 10) or (12 <= hour_utc <= 15):
            return MarketRegime.SESSION_SWEEP_WINDOW

        return MarketRegime.RANGING

    @staticmethod
    def regime_multipliers(regime: MarketRegime) -> Dict[str, float]:
        """
        Return adaptive weight multipliers for each strategy family in the given regime.
        This prevents counter-regime strategies from diluting high-quality setups.
        """
        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            return {
                "SND": 1.15,            # Continuation RBR / DBD zones work well in trends
                "ICT": 1.00,            # Pro-trend FVG retracements
                "BREAKOUT": 1.25,       # Volatility expansion thrives in strong trends
                "TREND_PULLBACK": 1.35, # Primary habitat for EMA + RSI pullback
                "KRONOS": 1.10,
            }
        if regime == MarketRegime.SESSION_SWEEP_WINDOW:
            return {
                "SND": 1.25,            # Fresh HTF supply/demand tested at session open
                "ICT": 1.40,            # Prime habitat for Asian/London Sweep + MSS + FVG
                "BREAKOUT": 1.30,       # Prime habitat for Asian Range Breakout
                "TREND_PULLBACK": 0.90,
                "KRONOS": 1.15,
            }
        if regime == MarketRegime.RANGING:
            return {
                "SND": 1.30,            # Range boundaries respect Supply & Demand zones
                "ICT": 1.15,            # Range-edge liquidity sweeps work well
                "BREAKOUT": 0.45,       # Suppress breakouts inside low-ADX ranges (avoids fakeouts!)
                "TREND_PULLBACK": 0.70, # Lower weight when EMAs are flat
                "KRONOS": 1.05,
            }
        # HIGH_VOL_CHOP: require extreme selectivity across all strategies
        return {
            "SND": 0.65,
            "ICT": 0.65,
            "BREAKOUT": 0.40,
            "TREND_PULLBACK": 0.50,
            "KRONOS": 0.80,
        }
