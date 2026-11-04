"""Configuration and instrument profiles for the Hybrid Multi-Strategy Engine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class InstrumentProfile:
    """Market specification and risk parameters for a single tradable instrument."""

    symbol: str
    display_name: str
    pip_size: float
    contract_size: float
    typical_spread_price: float
    max_spread_price: float
    atr_sl_multiplier: float
    tp1_rr: float = 1.5
    tp2_rr: float = 3.0
    tp1_close_fraction: float = 0.5
    min_lot: float = 0.01
    lot_step: float = 0.01


INSTRUMENTS: Dict[str, InstrumentProfile] = {
    "XAUUSD": InstrumentProfile(
        symbol="XAUUSD",
        display_name="Gold vs US Dollar (Spot)",
        pip_size=0.10,
        contract_size=100.0,  # 1 standard lot = 100 oz ($1 move = $100 per 1.00 lot, $1 per 0.01 lot)
        typical_spread_price=0.25,  # $0.25 spread on ECN/Raw accounts
        max_spread_price=0.60,
        atr_sl_multiplier=1.5,
        tp1_rr=1.5,
        tp2_rr=3.0,
    ),
    "EURUSD": InstrumentProfile(
        symbol="EURUSD",
        display_name="Euro vs US Dollar",
        pip_size=0.0001,
        contract_size=100_000.0,  # 1 standard lot = 100,000 units ($10/pip per 1.00 lot)
        typical_spread_price=0.00010,  # 1.0 pip
        max_spread_price=0.00022,
        atr_sl_multiplier=1.4,
        tp1_rr=1.5,
        tp2_rr=3.0,
    ),
    "USDJPY": InstrumentProfile(
        symbol="USDJPY",
        display_name="US Dollar vs Japanese Yen",
        pip_size=0.01,
        contract_size=100_000.0,  # Approx $6.50-$7.00/pip per 1.00 lot (converted via price)
        typical_spread_price=0.012,  # 1.2 pips
        max_spread_price=0.025,
        atr_sl_multiplier=1.4,
        tp1_rr=1.5,
        tp2_rr=3.0,
    ),
}


@dataclass
class KronosConfig:
    """Configuration for the local Kronos K-line Foundation Model."""

    enabled: bool = True
    # Recommended for NVIDIA RTX 5060 (8GB VRAM): NeoQuasar/Kronos-base (102.3M params)
    # Uses ~400MB VRAM in FP16/BF16 and runs 10 Monte-Carlo trajectories in ~60ms.
    model_id: str = "NeoQuasar/Kronos-base"
    tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base"
    fallback_model_id: str = "NeoQuasar/Kronos-small"
    device: str = "cuda"  # Automatically falls back to "cpu" if CUDA is unavailable
    use_fp16: bool = True
    lookback: int = 256  # Up to 512 supported by Kronos-base
    pred_len: int = 12  # Forecast next 12 candles (e.g. 3 hours on M15)
    sample_count: int = 10  # Number of Monte-Carlo paths averaged for probability scoring
    temperature: float = 1.0
    top_p: float = 0.9


@dataclass
class EngineConfig:
    """Global configuration for scanning, confluence scoring, and risk control."""

    execution_mode: str = field(
        default_factory=lambda: os.getenv("EXECUTION_MODE", "signal_only")
    )
    default_equity: float = field(
        default_factory=lambda: float(os.getenv("ACCOUNT_EQUITY_DEFAULT", "10000.0"))
    )
    risk_per_trade_pct: float = field(
        default_factory=lambda: float(os.getenv("RISK_PER_TRADE_PCT", "0.5"))
    )
    max_daily_drawdown_pct: float = field(
        default_factory=lambda: float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "2.0"))
    )
    min_confluence_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_CONFLUENCE_SCORE", "65.0"))
    )
    max_bars_in_trade: int = 36  # Max holding horizon on M15 (9 hours)
    cooldown_bars_after_trade: int = 4
    kronos: KronosConfig = field(default_factory=KronosConfig)

    # Base weights for each strategy family (before Regime-Adaptive multiplier)
    base_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "SND": 28.0,            # Supply & Demand Unmitigated Zone
            "ICT": 28.0,            # Session Liquidity Sweep + MSS + FVG
            "BREAKOUT": 22.0,       # Asian Range Compression -> London/NY Expansion
            "TREND_PULLBACK": 22.0, # EMA 20/50/200 Alignment + RSI/BB Pullback
            "KRONOS": 25.0,         # Kronos-base Probabilistic Trajectory Vote
        }
    )
