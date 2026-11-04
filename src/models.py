"""Core data structures for candles, zones, strategy votes, signals, and backtest results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class MarketRegime(str, Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    SESSION_SWEEP_WINDOW = "SESSION_SWEEP_WINDOW"
    RANGING = "RANGING"
    HIGH_VOL_CHOP = "HIGH_VOL_CHOP"


@dataclass(frozen=True)
class Candle:
    """Single OHLCV bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 1.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return max(self.high - self.low, 1e-9)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class PriceZone:
    """Represents a Supply/Demand zone or Fair Value Gap (FVG)."""

    zone_type: str  # "DEMAND", "SUPPLY", "BULL_FVG", "BEAR_FVG"
    pattern: str    # e.g., "DBR", "RBR", "RBD", "DBD", "ICT_FVG"
    top: float
    bottom: float
    created_idx: int
    created_at: datetime
    strength: float = 1.0
    touch_count: int = 0
    mitigated: bool = False

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.top + self.bottom)


@dataclass
class StrategyVote:
    """Output of a single strategy module or AI forecaster on the current bar."""

    strategy_name: str
    direction: Direction
    confidence: float  # 0.0 to 1.0
    reason: str
    suggested_sl: Optional[float] = None
    zone: Optional[PriceZone] = None
    metadata: Dict[str, float | str] = field(default_factory=dict)


@dataclass
class KronosForecast:
    """Probabilistic forecast summary from Kronos-base (or statistical fallback)."""

    model_name: str
    direction: Direction
    bullish_prob: float       # 0.0 to 1.0 across Monte-Carlo samples
    expected_return_atr: float
    upside_excursion_atr: float
    downside_excursion_atr: float
    confidence: float         # 0.0 to 1.0
    summary: str


@dataclass
class SetupSignal:
    """Actionable high-confluence trade setup presented to the user (or MT5 executor)."""

    symbol: str
    timeframe: str
    timestamp: datetime
    direction: Direction
    regime: MarketRegime
    confluence_score: float  # 0 to 100
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    atr: float
    rr_tp1: float
    rr_tp2: float
    recommended_lot: float
    risk_usd: float
    risk_pct: float
    votes: List[StrategyVote]
    kronos: Optional[KronosForecast] = None
    active_combination: str = "HYBRID_ENSEMBLE"

    def format_signal_card(self) -> str:
        """Render a clear, human-readable Copilot Signal Card for terminal/Telegram."""
        arrow = "BUY (LONG)" if self.direction == Direction.LONG else "SELL (SHORT)"
        decimals = 2 if self.symbol == "XAUUSD" else (3 if self.symbol == "USDJPY" else 5)
        fmt = f"{{:.{decimals}f}}"

        lines = [
            "=" * 74,
            f"  HIGH-CONFLUENCE SETUP DETECTED | {self.symbol} ({self.timeframe}) | {arrow}",
            "=" * 74,
            f"  Timestamp (UTC)   : {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}",
            f"  Market Regime     : {self.regime.value}",
            f"  Confluence Score  : {self.confluence_score:.1f} / 100.0  [{self.active_combination}]",
            "-" * 74,
            f"  Entry Price       : {fmt.format(self.entry_price)}",
            f"  Stop Loss (SL)    : {fmt.format(self.stop_loss)}  (Invalidation + ATR Buffer)",
            f"  Take Profit 1     : {fmt.format(self.tp1_price)}  ({self.rr_tp1:.1f}R -> Close 50% & Move SL to BE)",
            f"  Take Profit 2     : {fmt.format(self.tp2_price)}  ({self.rr_tp2:.1f}R -> Runner Target)",
            f"  Suggested Lot     : {self.recommended_lot:.2f} lots  (Risk: ${self.risk_usd:.2f} / {self.risk_pct:.2f}%)",
            "-" * 74,
            "  STRATEGY BREAKDOWN (Why this setup triggered):",
        ]
        for v in self.votes:
            status = (
                "AGREES"
                if v.direction == self.direction
                else ("NEUTRAL" if v.direction == Direction.NEUTRAL else "OPPOSES")
            )
            lines.append(
                f"    [{status:^7}] {v.strategy_name:<15} (conf {v.confidence * 100:4.0f}%): {v.reason}"
            )
        if self.kronos is not None:
            lines.append("-" * 74)
            lines.append(
                f"  KRONOS AI FORECAST ({self.kronos.model_name}): {self.kronos.summary}"
            )
        lines.append("=" * 74)
        return "\n".join(lines)


@dataclass
class TradeRecord:
    """Single simulated or executed trade with partial TP1 and TP2 tracking."""

    symbol: str
    strategy_combo: str
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    stop_loss: float
    tp1_price: float
    tp2_price: float
    exit_price: float
    lot_size: float
    risk_usd: float
    pnl_usd: float
    r_multiple: float
    tp1_hit: bool
    tp2_hit: bool
    exit_reason: str  # "TP2", "TP1_THEN_BE", "SL", "TIME_EXIT"
    confluence_score: float
    regime: str


@dataclass
class BacktestMetrics:
    """Performance metrics for a single strategy or ensemble configuration."""

    symbol: str
    strategy_combo: str
    split_Label: str  # "FULL", "IN_SAMPLE", "OUT_OF_SAMPLE"
    total_trades: int
    wins: int
    losses: int
    breakevens: int
    win_rate_pct: float
    tp1_hit_rate_pct: float
    profit_factor: float
    expectancy_r: float
    total_return_pct: float
    net_profit_usd: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[TradeRecord] = field(default_factory=list)
