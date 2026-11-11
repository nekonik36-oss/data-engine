"""
Multi-Strategy Confluence Engine, Regime-Gated Voter & Online Self-Learning Adapter.

Supports both:
  - Individual Strategy Modes (`SND_ONLY`, `ICT_ONLY`, `BREAKOUT_ONLY`, `TREND_PULLBACK_ONLY`)
  - Multi-Strategy Combinations (`SND_PLUS_ICT`, `ALL_4_NAIVE`, `HYBRID_REGIME_KRONOS`)
and integrates `AdaptiveLearner` so closed trade outcomes dynamically refine strategy
weights (`0.60x`–`1.40x`) and per-symbol confluence thresholds in live trading.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

from src.adaptive_learner import AdaptiveLearner
from src.config import ALL_INSTRUMENTS, EngineConfig, INSTRUMENTS, InstrumentProfile
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
        vols = [c.volume for c in candles]
        n = len(candles)
        self.atr = compute_atr(candles, period=14)
        self.ema20 = compute_ema(closes, period=20)
        self.ema50 = compute_ema(closes, period=50)
        self.ema60 = compute_ema(closes, period=60)
        self.ema200 = compute_ema(closes, period=200)
        self.ema240 = compute_ema(closes, period=240)
        self.rsi = compute_rsi(closes, period=14)
        self.rsi2 = compute_rsi(closes, period=2)
        self.bb_mid, self.bb_upper, self.bb_lower = compute_bollinger_bands(
            closes, period=20, num_std=2.0
        )
        self.adx = compute_adx(candles, period=14)
        self.vol_ma20 = [0.0] * n
        for i in range(n):
            s = max(0, i - 20)
            self.vol_ma20[i] = sum(vols[s : i + 1]) / max(1, i - s + 1)


class ConfluenceEngine:
    """
    Evaluates all 4 price-action strategy modules + Market Regime + Kronos AI Forecaster
    + Online Self-Learning weights (`AdaptiveLearner`), computes a normalized 0-100
    Confluence Score, and builds an actionable `SetupSignal`.
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        learner: Optional[AdaptiveLearner] = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.regime_detector = RegimeDetector()
        self.risk_guard = RiskGuard(self.config)
        self.learner = learner or AdaptiveLearner()

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
        record_shadow: bool = False,
    ) -> Optional[SetupSignal]:
        """
        Evaluate bar `idx` under the chosen strategy preset (`preset_name`).
        Returns a `SetupSignal` if confluence passes threshold, else `None`.
        """
        if idx < 45:
            return None

        profile: InstrumentProfile = ALL_INSTRUMENTS.get(symbol, INSTRUMENTS["XAUUSD"])
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
            if not profile.is_crypto and regime == MarketRegime.HIGH_VOL_CHOP:
                return None
            if not profile.is_crypto and not (6 <= curr.timestamp.hour <= 18):
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
            votes.append(
                self.breakout.evaluate(
                    candles, ind.atr, ind.adx, idx, is_24_7=profile.is_crypto
                )
            )
        if "TREND_PULLBACK" in active_set:
            if use_regime_gate and profile.is_crypto and idx >= 50:
                # Engine 1 (Crypto M5 Momentum Pullback Break — 66.7%-69.2% Real M5 Win Rate):
                # H1 Multi-Timeframe Trend (EMA60 vs EMA240) + EMA50/200 Stack + 3-Bar Connors RSI(2) Pullback + Break
                prev = candles[idx - 1]
                e50, e60, e200, e240 = (
                    ind.ema50[idx],
                    ind.ema60[idx],
                    ind.ema200[idx],
                    ind.ema240[idx],
                )
                min_rsi2 = min(ind.rsi2[idx], ind.rsi2[idx - 1], ind.rsi2[idx - 2])
                max_rsi2 = max(ind.rsi2[idx], ind.rsi2[idx - 1], ind.rsi2[idx - 2])
                h1_bull = e60 > e240 and (e60 - ind.ema60[idx - 12]) > 0
                h1_bear = e60 < e240 and (ind.ema60[idx - 12] - e60) > 0
                recent_max_high = max(c.high for c in candles[max(0, idx - 3) : idx + 1])
                recent_min_low = min(c.low for c in candles[max(0, idx - 3) : idx + 1])
                ema20_slope3 = ind.ema20[idx] - ind.ema20[max(0, idx - 3)]
                if (
                    h1_bull
                    and ind.ema20[idx] > e50
                    and e50 > e200
                    and curr.close > e60
                    and min_rsi2 < 20.0
                    and curr.is_bullish
                    and curr.close > prev.high
                    and recent_min_low >= e50 - 1.15 * cur_atr
                    and ema20_slope3 >= -0.25 * cur_atr
                ):
                    votes.append(
                        StrategyVote(
                            strategy_name="TREND_PULLBACK",
                            direction=Direction.LONG,
                            confidence=0.88,
                            reason=f"H1 Bull Trend + EMA20>50>200 + Connors RSI(2)={min_rsi2:.1f} Pullback Break",
                            suggested_sl=min(curr.low, prev.low) - 1.20 * cur_atr,
                        )
                    )
                elif (
                    h1_bear
                    and ind.ema20[idx] < e50
                    and e50 < e200
                    and curr.close < e60
                    and max_rsi2 > 80.0
                    and curr.is_bearish
                    and curr.close < prev.low
                    and recent_max_high <= e50 + 1.15 * cur_atr
                    and ema20_slope3 <= 0.25 * cur_atr
                ):
                    votes.append(
                        StrategyVote(
                            strategy_name="TREND_PULLBACK",
                            direction=Direction.SHORT,
                            confidence=0.88,
                            reason=f"H1 Bear Trend + EMA50<200 + Connors RSI(2)={max_rsi2:.1f} Pullback Break",
                            suggested_sl=max(curr.high, prev.high) + 1.20 * cur_atr,
                        )
                    )
                else:
                    votes.append(
                        StrategyVote(
                            strategy_name="TREND_PULLBACK",
                            direction=Direction.NEUTRAL,
                            confidence=0.0,
                            reason="Waiting for H1-aligned Connors RSI(2) pullback break",
                        )
                    )
            else:
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
                        is_24_7=profile.is_crypto,
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

        if record_shadow and use_regime_gate:
            sh_sl_dist = max(
                1.50 * cur_atr,
                (curr.close * 0.0022) if profile.is_crypto else (1.50 * cur_atr),
            )
            for sv in votes:
                if sv.direction != Direction.NEUTRAL and sv.confidence >= 0.65:
                    self.learner.open_shadow_trade(
                        symbol=symbol,
                        strategy_name=sv.strategy_name,
                        direction=sv.direction.value,
                        regime=regime.value,
                        entry_price=curr.close,
                        sl_dist=sh_sl_dist,
                    )

        # 3. Score LONG vs SHORT with Regime-Adaptive + Online Self-Learned Multipliers
        multipliers = (
            self.regime_detector.regime_multipliers(regime)
            if use_regime_gate
            else {k: 1.0 for k in self.config.base_weights}
        )

        long_pa_points = 0.0
        short_pa_points = 0.0
        for v in pa_votes:
            learned_mult = (
                self.learner.get_strategy_multiplier(v.strategy_name, regime.value)
                if use_regime_gate
                else 1.0
            )
            w = (
                self.config.base_weights.get(v.strategy_name, 25.0)
                * multipliers.get(v.strategy_name, 1.0)
                * learned_mult
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
            if opposing_pa:
                return None
            best_conf = max(v.confidence for v in agreeing_pa)
            synergy = 15.0 if len(agreeing_pa) >= 2 else 0.0
            confluence_score = min(100.0, best_conf * 100.0 + synergy)
            threshold = 78.0
        elif preset_name == "ALL_4_NAIVE":
            best_conf = max(v.confidence for v in agreeing_pa)
            confluence_score = best_conf * 100.0
            threshold = 66.0
        else:
            # HYBRID_REGIME_KRONOS: 3-Engine Per-Instrument Specialized Quant Architecture + Online Learner
            if symbol in {"BTCUSD", "SOLUSD"}:
                # Engine 1 (Momentum Leaders BTCUSD & SOLUSD):
                # Path A: H1 Trend + EMA50/200 Stack + Connors RSI(2) Pullback confirmed by >=1 peer module (SND/ICT/BREAKOUT/KRONOS)
                # Path B: Multi-Strategy Structural Consensus (>=2 of SND/ICT/BREAKOUT aligned with H1 trend)
                tpb_votes = [
                    v
                    for v in pa_votes
                    if v.strategy_name == "TREND_PULLBACK" and v.confidence >= 0.85
                ]
                if tpb_votes:
                    direction = tpb_votes[0].direction
                    if any(v.direction != direction and v.confidence >= 0.72 for v in pa_votes):
                        return None
                    if kronos_fc is not None:
                        if direction == Direction.SHORT and kronos_fc.bullish_prob >= 0.68:
                            return None
                        if direction == Direction.LONG and kronos_fc.bullish_prob <= 0.32:
                            return None
                    agreeing_pa = [v for v in pa_votes if v.direction == direction]
                    kronos_confirms = (
                        kronos_fc is not None
                        and kronos_fc.direction == direction
                        and kronos_fc.confidence >= 0.35
                    )
                    # Ban unconfirmed solo TREND_PULLBACK & block TREND_PULLBACK inside SESSION_SWEEP_WINDOW (0.853x RL stop-hunt trap)
                    if len(agreeing_pa) < 2 and not kronos_confirms:
                        return None
                    if regime == MarketRegime.SESSION_SWEEP_WINDOW:
                        return None
                    avg_learned_m = sum(
                        self.learner.get_strategy_multiplier(v.strategy_name, regime.value)
                        for v in agreeing_pa
                    ) / len(agreeing_pa)
                    synergy_bonus = 5.0 if len(agreeing_pa) >= 2 else 0.0
                    confluence_score = min(100.0, 88.0 * avg_learned_m + synergy_bonus)
                    threshold = 80.0 + self.learner.get_symbol_threshold_offset(symbol)
                else:
                    # Path B: Multi-Strategy Consensus from SND (100% RL WR), BREAKOUT (58.3% RL WR), ICT (55.2% RL WR)
                    if any(v.confidence >= 0.72 for v in opposing_pa):
                        return None
                    strong_agree = [v for v in agreeing_pa if v.confidence >= 0.75]
                    if len(strong_agree) < 2:
                        return None
                    e60, e240 = ind.ema60[idx], ind.ema240[idx]
                    if direction == Direction.LONG and e60 < e240:
                        return None
                    if direction == Direction.SHORT and e60 > e240:
                        return None
                    if kronos_fc is not None:
                        if direction == Direction.SHORT and kronos_fc.bullish_prob >= 0.68:
                            return None
                        if direction == Direction.LONG and kronos_fc.bullish_prob <= 0.32:
                            return None
                    agreeing_pa = strong_agree
                    best_pa_conf = max(
                        v.confidence
                        * self.learner.get_strategy_multiplier(v.strategy_name, regime.value)
                        for v in agreeing_pa
                    )
                    confluence_score = min(100.0, best_pa_conf * 100.0 + 12.0)
                    threshold = 80.0 + self.learner.get_symbol_threshold_offset(symbol)

            elif symbol == "ETHUSD":
                # Engine 2 (ETHUSD Dual-Engine: Liquidity-Sweep + Volume Climax OR Confirmed H1 Trend Pullback):
                tpb_votes = [
                    v
                    for v in pa_votes
                    if v.strategy_name == "TREND_PULLBACK" and v.confidence >= 0.85
                ]
                rev_votes = [
                    v
                    for v in pa_votes
                    if v.strategy_name in {"SND", "ICT"} and v.confidence >= 0.80
                ]
                if tpb_votes:
                    direction = tpb_votes[0].direction
                    if any(v.direction != direction and v.confidence >= 0.72 for v in pa_votes):
                        return None
                    agreeing_pa = [v for v in pa_votes if v.direction == direction]
                    kronos_confirms = (
                        kronos_fc is not None
                        and kronos_fc.direction == direction
                        and kronos_fc.confidence >= 0.35
                    )
                    # Ban unconfirmed solo TREND_PULLBACK & block TREND_PULLBACK inside SESSION_SWEEP_WINDOW (0.853x RL stop-hunt trap)
                    if len(agreeing_pa) < 2 and not kronos_confirms:
                        return None
                    if regime == MarketRegime.SESSION_SWEEP_WINDOW:
                        return None
                    avg_learned_m = sum(
                        self.learner.get_strategy_multiplier(v.strategy_name, regime.value)
                        for v in agreeing_pa
                    ) / len(agreeing_pa)
                    synergy_bonus = 5.0 if len(agreeing_pa) >= 2 else 0.0
                    confluence_score = min(100.0, 88.0 * avg_learned_m + synergy_bonus)
                    threshold = 80.0 + self.learner.get_symbol_threshold_offset(symbol)
                elif rev_votes:
                    long_rev = [v for v in rev_votes if v.direction == Direction.LONG]
                    short_rev = [v for v in rev_votes if v.direction == Direction.SHORT]
                    if long_rev and short_rev:
                        return None
                    direction = Direction.LONG if long_rev else Direction.SHORT
                    agreeing_pa = long_rev if long_rev else short_rev

                    prev = candles[idx - 1]
                    vol_ratio = curr.volume / max(ind.vol_ma20[idx], 1e-9)
                    prev_vol_ratio = prev.volume / max(ind.vol_ma20[idx - 1], 1e-9)
                    if max(vol_ratio, prev_vol_ratio) < 1.20:
                        return None

                    e60, e240 = ind.ema60[idx], ind.ema240[idx]
                    h1_bull = e60 > e240 and (e60 - ind.ema60[idx - 12]) > 0
                    h1_bear = e60 < e240 and (ind.ema60[idx - 12] - e60) > 0
                    rsi_val = ind.rsi[idx]
                    adx_val = ind.adx[idx]

                    if direction == Direction.LONG:
                        if not (curr.is_bullish and curr.close > prev.open and rsi_val < 55.0):
                            return None
                        if h1_bear and adx_val > 25.0:
                            return None
                        if len(agreeing_pa) < 2 and (regime == MarketRegime.TRENDING_DOWN or e60 < e240):
                            return None
                    else:
                        if not (curr.is_bearish and curr.close < prev.open and rsi_val > 45.0):
                            return None
                        if h1_bull and adx_val > 25.0:
                            return None
                        if len(agreeing_pa) < 2 and (regime == MarketRegime.TRENDING_UP or e60 > e240):
                            return None

                    best_pa_conf = max(
                        v.confidence
                        * self.learner.get_strategy_multiplier(v.strategy_name, regime.value)
                        for v in agreeing_pa
                    )
                    synergy = 14.0 if len(agreeing_pa) >= 2 else 8.0
                    confluence_score = min(100.0, best_pa_conf * 100.0 + synergy)
                    threshold = 80.0 + self.learner.get_symbol_threshold_offset(symbol)
                else:
                    return None
                if kronos_fc is not None:
                    if direction == Direction.SHORT and kronos_fc.bullish_prob >= 0.68:
                        return None
                    if direction == Direction.LONG and kronos_fc.bullish_prob <= 0.32:
                        return None

            else:
                # Engine 3: Gold & Forex M15 5-Module Regime-Gated + Kronos AI Sniper (78.2%-82.4% Win Rate)
                if opposing_pa:
                    max_opp_weight = max(
                        v.confidence * multipliers.get(v.strategy_name, 1.0)
                        for v in opposing_pa
                    )
                    if len(agreeing_pa) < 2 or max_opp_weight >= 0.70:
                        return None

                if agreeing_names == {"BREAKOUT"}:
                    if ind.adx[idx] < 20.0:
                        return None
                    # Block solo counter-trend BREAKOUT traps (e.g. Judas Swing sweeps in SESSION_SWEEP_WINDOW)
                    if direction == Direction.SHORT and ind.ema50[idx] > ind.ema200[idx]:
                        return None
                    if direction == Direction.LONG and ind.ema50[idx] < ind.ema200[idx]:
                        return None
                    if regime == MarketRegime.SESSION_SWEEP_WINDOW and ind.adx[idx] < 25.0:
                        return None

                if regime == MarketRegime.RANGING:
                    if agreeing_names == {"BREAKOUT"}:
                        return None
                    if len(agreeing_pa) < 2 and "ICT" not in agreeing_names:
                        return None

                # Block solo counter-trend entries in Engine 3 (Gold & Forex M15) across all regimes
                # including SESSION_SWEEP_WINDOW where regime overrides TRENDING_UP / TRENDING_DOWN.
                if len(agreeing_pa) < 2:
                    if direction == Direction.SHORT and (
                        regime == MarketRegime.TRENDING_UP
                        or ind.ema50[idx] > ind.ema200[idx]
                    ):
                        return None
                    if direction == Direction.LONG and (
                        regime == MarketRegime.TRENDING_DOWN
                        or ind.ema50[idx] < ind.ema200[idx]
                    ):
                        return None

                kronos_boost = 0.0
                if kronos_fc is not None:
                    k_learned = self.learner.get_strategy_multiplier(
                        "KRONOS", regime.value
                    )
                    if direction == Direction.LONG:
                        if kronos_fc.bullish_prob < 0.52:
                            return None
                        kronos_boost = (
                            (kronos_fc.bullish_prob - 0.50) * 35.0 * k_learned
                        )
                    else:
                        if kronos_fc.bullish_prob > 0.48:
                            return None
                        kronos_boost = (
                            (0.50 - kronos_fc.bullish_prob) * 35.0 * k_learned
                        )

                best_pa_conf = max(
                    v.confidence
                    * self.learner.get_strategy_multiplier(v.strategy_name, regime.value)
                    for v in agreeing_pa
                )
                pa_synergy = 12.0 * (len(agreeing_pa) - 1)
                confluence_score = min(
                    100.0, best_pa_conf * 88.0 + pa_synergy + kronos_boost
                )
                threshold = (
                    max(self.config.min_confluence_score, 78.0)
                    + self.learner.get_symbol_threshold_offset(symbol)
                )

        if confluence_score < threshold:
            return None

        # 4. Calculate Structural + ATR Stop Loss (with Self-Tuned Anti-Wick Crypto Floor), TP1, TP2, and Lot Size
        entry_price = curr.close
        candidate_sls = [
            v.suggested_sl for v in agreeing_pa if v.suggested_sl is not None
        ]
        sl_floor_mult = self.learner.get_symbol_sl_floor_mult(symbol) if use_regime_gate else 1.0
        min_sl_dist = max(
            1.50 * cur_atr * sl_floor_mult,
            (entry_price * 0.0022 * sl_floor_mult) if profile.is_crypto else (1.50 * cur_atr * sl_floor_mult),
        )
        max_sl_dist = max(3.20 * cur_atr, min_sl_dist * 1.35)

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
