"""Risk Management, Dynamic Lot Sizing, Spread Guard, and Daily Drawdown Circuit Breaker."""

from __future__ import annotations

from datetime import datetime
import math
from typing import Optional, Tuple

from src.config import EngineConfig, InstrumentProfile
from src.fundamental_guard import MacroNewsShield


class RiskGuard:
    """
    Enforces institutional risk management rules:
      1. Fixed fractional risk per trade (default 0.5% of account equity).
      2. Dynamic lot size calculation based on exact Stop Loss distance and instrument contract spec.
      3. Daily max drawdown circuit breaker (default 2.0% daily loss lock).
      4. Live bid-ask spread guard.
      5. Tier-1 Red-Folder Macro News Blackout Shield (-20m / +15m around CPI, PPI, NFP, FOMC, etc.).
    """

    def __init__(self, config: EngineConfig, allow_network_news: bool = True) -> None:
        self.config = config
        self.daily_realized_pnl_usd: float = 0.0
        self.current_day: str = ""
        self.news_shield = MacroNewsShield(allow_network=allow_network_news)

    def is_news_blackout(
        self, symbol: str, now_utc: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        return self.news_shield.is_blackout_active(symbol, now_utc=now_utc)

    def update_day(self, ts: datetime) -> None:
        day_str = ts.strftime("%Y-%m-%d")
        if day_str != self.current_day:
            self.current_day = day_str
            self.daily_realized_pnl_usd = 0.0

    def record_trade_pnl(self, ts: datetime, pnl_usd: float) -> None:
        self.update_day(ts)
        self.daily_realized_pnl_usd += pnl_usd

    def is_daily_drawdown_breached(self, equity: float) -> bool:
        max_loss_usd = equity * (self.config.max_daily_drawdown_pct / 100.0)
        return self.daily_realized_pnl_usd <= -max_loss_usd

    @staticmethod
    def is_spread_acceptable(profile: InstrumentProfile, current_spread_price: float) -> bool:
        return current_spread_price <= profile.max_spread_price

    def calculate_lot_size(
        self,
        profile: InstrumentProfile,
        equity: float,
        entry_price: float,
        stop_loss: float,
    ) -> tuple[float, float]:
        """
        Calculate (recommended_lot_size, risk_usd) for `equity`.
        Automatically scales fractional risk for Small Accounts ($10-$200) vs Large Accounts,
        and handles USD-quoted pairs (`XAUUSD`, `EURUSD`) and JPY-quoted pairs (`USDJPY`).
        """
        if equity < 35.0:
            eff_risk_pct = max(self.config.risk_per_trade_pct, 4.0)
        elif equity < 100.0:
            eff_risk_pct = max(self.config.risk_per_trade_pct, 3.0)
        elif equity < 250.0:
            eff_risk_pct = max(self.config.risk_per_trade_pct, 1.5)
        else:
            eff_risk_pct = self.config.risk_per_trade_pct

        risk_usd = equity * (eff_risk_pct / 100.0)
        sl_dist = abs(entry_price - stop_loss)
        if sl_dist <= 1e-9:
            return profile.min_lot, risk_usd

        if profile.symbol.endswith("JPY"):
            # For USDJPY, 1.00 lot = 100,000 USD; value of 1 JPY move per 1.00 lot is 100,000 / entry_price USD
            usd_per_unit_per_lot = profile.contract_size / max(entry_price, 1.0)
        else:
            # For XAUUSD (100 oz) and EURUSD (100,000 EUR), 1 price unit move per 1.00 lot = contract_size USD
            usd_per_unit_per_lot = profile.contract_size

        loss_per_lot = sl_dist * usd_per_unit_per_lot
        if loss_per_lot <= 1e-9:
            return profile.min_lot, risk_usd

        raw_lots = risk_usd / loss_per_lot
        stepped_lots = math.floor((raw_lots / profile.lot_step) + 1e-9) * profile.lot_step
        final_lots = max(profile.min_lot, round(stepped_lots, 2))
        actual_risk_usd = final_lots * loss_per_lot
        return final_lots, actual_risk_usd
