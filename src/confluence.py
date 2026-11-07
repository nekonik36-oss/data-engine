"""
Multi-Strategy Confluence Engine & Regime-Gated Voter.

Supports both:
  - Individual Strategy Modes (`SND_ONLY`, `ICT_ONLY`, `BREAKOUT_ONLY`, `TREND_PULLBACK_ONLY`)
  - Multi-Strategy Combinations (`SND_PLUS_ICT`, `ALL_4_NAIVE`, `HYBRID_REGIME_KRONOS`)
so the Automated Backtest Tournament can benchmark every strategy alone vs. combined.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from src.config import EngineConfig, INSTRUMENTS, InstrumentProfile
from src.indicators import (
    compute_adx,
    compute_atr,
    compute_bollinger_bands,
    compute_ema,
    compute_rsi,
)
from src.models import (
    Candle,
    Direction,
    MarketRegime,
    SetupSignal,
    StrategyVote,
)
from src.regime import RegimeDetector
from src.risk_guard import RiskGuard
from src.strategies import (
    ICTStrategy,
    KronosForecaster,
    SessionBreakoutStrategy,
    SupplyDemandStrategy,
    TrendPullbackStrategy,
)


TOURNAMENT_PRESETS: Dict[str, Set[str]] = {
    "SND_ONLY": {"SND"},
    "ICT_ONLY": {"ICT"},
    "BREAKOUT_ONLY": {"BREAKOUT"},
    "TREND_PULLBACK_ONLY": {"TREND_PULLBACK"},
    "SND_PLUS_ICT": {"SND", "ICT"},
    "ALL_4_NAIVE": {"SND", "ICT", "BREAKOUT", "TREND_PULLBACK"},
    "HYBRID_REGIME_KRONOS": {"SND", "ICT", "BREAKOUT", "TREND_PULLBACK", "KRONOS"},
}


class PrecomputedSeries:
    """Holds strictly causal indicator arrays for fast scanning and backtesting."""

    def __init__(self, candles: Sequence[Candle]) -> None:
        closes = [c.close for c in candles]
        self.atr = compute_atr(candles, period=14)
        self.ema20 = compute_ema(closes, period=20)
        self.ema50 = compute_ema(closes, period=50)
        self.ema200 = compute_ema(closes, period=200)
        self.rsi = compute_rsi(closes, period=14)
        self.bb_mid, self.bb_upper, self.bb_lower = compute_bollinger_bands(
            closes, period=20, num_std=2.0
        )
        self.adx = compute_adx(candles, period=14)


class ConfluenceEngine:
    """
    Evaluates all 4 price-action strategy modules + Market Regime + Kronos AI Forecaster,
    computes a normalized 0-100 Confluence Score, and builds an actionable `SetupSignal`
    with ATR-buffered SL, TP1 (1.5R), TP2 (3.0R), and risk-managed lot size.
    """

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.regime_detector = RegimeDetector()
        self.risk_guard = RiskGuard(self.config)

        self.snd = SupplyDemandStrategy()
        self.ict = ICTStrategy()
        self.breakout = SessionBreakoutStrategy()
        self.trend_pullback = TrendPullbackStrategy()
        self.kronos = KronosForecaster(self.config.kronos)

    def precompute(self, candles: Sequence[Candle]) -> PrecomputedSeries:
        return PrecomputedSeries(candles)

    def evaluate_bar(
        self,
        symbol: str,
        timeframe: str,
        candles: Sequence[Candle],
        idx: int,
        series: Optional[PrecomputedSeries] = None,
        preset_name: str = "HYBRID_REGIME_KRONOS",
        equity: Optional[float] = None,
    ) -> Optional[SetupSignal]:
        """
        Evaluate bar `idx` under the chosen strategy preset (`preset_name`).
        Returns a `SetupSignal` if confluence passes threshold, else `None`.
        """
        if idx < 45:
            return None

        profile: InstrumentProfile = INSTRUMENTS.get(symbol, INSTRUMENTS["XAUUSD"])
        acct_equity = equity if equity is not None else self.config.default_equity
        ind = series if series is not None else self.precompute(candles[: idx + 1])

        curr = candles[idx]
        cur_atr = max(ind.atr[idx], 1e-9)

        # 1. Classify Market Regime
        regime = self.regime_detector.classify(
            candles, idx, ind.ema20, ind.ema50, ind.ema200, ind.atr, ind.adx
        )

        use_regime_gate = preset_name == "HYBRID_REGIME_KRONOS"
        if use_regime_gate:
            # Block entries during HIGH_VOL_CHOP or dead illiquid rollover hours (21:00-06:00 UTC)
            if regime == MarketRegime.HIGH_VOL_CHOP:
                return None
            if not (6 <= curr.timestamp.hour <= 18):
                return None

        active_set = TOURNAMENT_PRESETS.get(
            preset_name, TOURNAMENT_PRESETS["HYBRID_REGIME_KRONOS"]
        )

        # 2. Gather votes from enabled price-action strategies
        votes: List[StrategyVote] = []
        if "SND" in active_set:
            votes.append(self.snd.evaluate(candles, ind.atr, idx))
        if "ICT" in active_set:
            votes.append(self.ict.evaluate(candles, ind.atr, idx))
        if "BREAKOUT" in active_set:
            votes.append(self.breakout.evaluate(candles, ind.atr, ind.adx, idx))
        if "TREND_PULLBACK" in active_set:
            votes.append(
                self.trend_pullback.evaluate(
                    candles,
                    ind.ema20,
                    ind.ema50,
                    ind.ema200,
                    ind.rsi,
                    ind.bb_upper,
                    ind.bb_lower,
                    ind.atr,
                    idx,
                )
            )

        pa_votes = [
            v
            for v in votes
            if v.strategy_name != "KRONOS" and v.direction != Direction.NEUTRAL
        ]
        if not pa_votes:
            return None

        kronos_fc = None
        if "KRONOS" in active_set and self.config.kronos.enabled:
            k_vote, kronos_fc = self.kronos.evaluate_vote(candles, ind.atr, idx)
            votes.append(k_vote)

        # 3. Score LONG vs SHORT with Regime-Adaptive Multipliers
        multipliers = (
            self.regime_detector.regime_multipliers(regime)
            if use_regime_gate
            else {k: 1.0 for k in self.config.base_weights}
        )

        long_pa_points = 0.0
        short_pa_points = 0.0
        for v in pa_votes:
            w = self.config.base_weights.get(v.strategy_name, 25.0) * multipliers.get(
                v.strategy_name, 1.0
            )
            if v.direction == Direction.LONG:
                long_pa_points += w * v.confidence
            elif v.direction == Direction.SHORT:
                short_pa_points += w * v.confidence

        if long_pa_points >= short_pa_points:
            direction = Direction.LONG
            agreeing_pa = [v for v in pa_votes if v.direction == Direction.LONG]
            opposing_pa = [v for v in pa_votes if v.direction == Direction.SHORT]
        else:
            direction = Direction.SHORT
            agreeing_pa = [v for v in pa_votes if v.direction == Direction.SHORT]
            opposing_pa = [v for v in pa_votes if v.direction == Direction.LONG]

        if not agreeing_pa:
            return None

        agreeing_names = {v.strategy_name for v in agreeing_pa}

        # Single-strategy presets vs Multi-strategy ensemble rules
        if len(active_set) == 1:
            confluence_score = agreeing_pa[0].confidence * 100.0
            threshold = 66.0
        elif preset_name == "SND_PLUS_ICT":
            # Require either both SND + ICT agreeing, or high-conviction ICT/SND with no conflict
            if opposing_pa:
                return None
            best_conf = max(v.confidence for v in agreeing_pa)
            synergy = 15.0 if len(agreeing_pa) >= 2 else 0.0
            confluence_score = min(100.0, best_conf * 100.0 + synergy)
            threshold = 78.0
        elif preset_name == "ALL_4_NAIVE":
            # Unfiltered combination: fires whenever any of the 4 strategies triggers
            best_conf = max(v.confidence for v in agreeing_pa)
            confluence_score = best_conf * 100.0
            threshold = 66.0
        else:
            # HYBRID_REGIME_KRONOS: Intelligent Regime-Switched Router + Kronos AI Filter
            # Rule A: Never trade if another price-action strategy opposes the direction
            if opposing_pa:
                return None

            # Rule B: Regime-specific strategy gating (prevents low-winrate strategies from diluting results)
            if agreeing_names == {"BREAKOUT"} and ind.adx[idx] < 20.0:
                # Require ADX >= 20 for standalone breakout to avoid low-momentum fakeouts
                return None

            if regime == MarketRegime.RANGING:
                # In ranging markets, block standalone BREAKOUT (avoids range fakeouts)
                if agreeing_names == {"BREAKOUT"}:
                    return None
                # Require either 2+ agreeing strategies OR a high-confidence ICT Sweep + FVG
                if len(agreeing_pa) < 2 and "ICT" not in agreeing_names:
                    return None

            if (
                regime == MarketRegime.TRENDING_UP
                and direction == Direction.SHORT
                and len(agreeing_pa) < 2
            ):
                return None
            if (
                regime == MarketRegime.TRENDING_DOWN
                and direction == Direction.LONG
                and len(agreeing_pa) < 2
            ):
                return None

            # Rule C: Kronos AI Trajectory Filter (veto setups where Kronos predicts counter-drift)
            kronos_boost = 0.0
            if kronos_fc is not None:
                if direction == Direction.LONG:
                    if kronos_fc.bullish_prob < 0.52:
                        return None
                    kronos_boost = (kronos_fc.bullish_prob - 0.50) * 35.0
                else:
                    if kronos_fc.bullish_prob > 0.48:
                        return None
                    kronos_boost = (0.50 - kronos_fc.bullish_prob) * 35.0

            best_pa_conf = max(v.confidence for v in agreeing_pa)
            pa_synergy = 12.0 * (len(agreeing_pa) - 1)
            confluence_score = min(
                100.0, best_pa_conf * 88.0 + pa_synergy + kronos_boost
            )
            threshold = self.config.min_confluence_score

        if confluence_score < threshold:
            return None

        # 4. Calculate Structural + ATR Stop Loss, TP1 (1.5R), TP2 (3.0R), and Lot Size
        entry_price = curr.close
        candidate_sls = [
            v.suggested_sl for v in agreeing_pa if v.suggested_sl is not None
        ]
        min_sl_dist = 0.90 * cur_atr
        max_sl_dist = 2.60 * cur_atr

        if direction == Direction.LONG:
            raw_sl = (
                min(candidate_sls)
                if candidate_sls
                else entry_price - profile.atr_sl_multiplier * cur_atr
            )
            sl_dist = max(min_sl_dist, min(max_sl_dist, entry_price - raw_sl))
            stop_loss = entry_price - sl_dist
            tp1_price = entry_price + profile.tp1_rr * sl_dist
            tp2_price = entry_price + profile.tp2_rr * sl_dist
        else:
            raw_sl = (
                max(candidate_sls)
                if candidate_sls
                else entry_price + profile.atr_sl_multiplier * cur_atr
            )
            sl_dist = max(min_sl_dist, min(max_sl_dist, raw_sl - entry_price))
            stop_loss = entry_price + sl_dist
            tp1_price = entry_price - profile.tp1_rr * sl_dist
            tp2_price = entry_price - profile.tp2_rr * sl_dist

        lot_size, risk_usd = self.risk_guard.calculate_lot_size(
            profile, acct_equity, entry_price, stop_loss
        )

        return SetupSignal(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=curr.timestamp,
            direction=direction,
            regime=regime,
            confluence_score=round(confluence_score, 1),
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            atr=cur_atr,
            rr_tp1=profile.tp1_rr,
            rr_tp2=profile.tp2_rr,
            recommended_lot=lot_size,
            risk_usd=round(risk_usd, 2),
            risk_pct=self.config.risk_per_trade_pct,
            votes=votes,
            kronos=kronos_fc,
            active_combination=preset_name,
        )
