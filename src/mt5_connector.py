"""
MetaTrader 5 Bridge & Multi-Regime Historical Data Provider.

Modes of Operation:
  1. Live / Demo MT5 Terminal (`connect_mt5()`): Pulls real-time M5/M15/H1 bars from the
     open MetaTrader 5 terminal on the Personal PC and supports `signal_only` (default),
     `confirm` ([Y/n] prompt), and `auto` order execution.
  2. CSV Loader (`load_csv_candles()`): Loads standard MT5/TradingView exported CSV files.
  3. Multi-Regime Market Generator (`generate_realistic_market_candles()`): Generates realistic
     institutional microstructure (Asian compression, London/NY liquidity sweeps, SND zones,
     directional trends, and chop regimes) for offline validation and CI testing.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
import random
from typing import List, Optional

from src.config import INSTRUMENTS
from src.models import Candle, Direction, SetupSignal
from src.telegram_notifier import TelegramNotifier


class MT5Connector:
    """Bridge to MetaTrader 5 terminal with safe `signal_only` default mode."""

    def __init__(self, execution_mode: str = "signal_only") -> None:
        self.execution_mode = execution_mode
        self._mt5 = None
        self.connected = False
        self.last_error: str | None = None
        self.telegram = TelegramNotifier()

    def connect(self) -> bool:
        """Attempt to initialize local MetaTrader 5 terminal (Personal PC only)."""
        try:
            import MetaTrader5 as mt5  # type: ignore

            if mt5.initialize():
                self._mt5 = mt5
                self.connected = True
                self.last_error = None
                return True
            self.last_error = f"mt5.initialize() returned False: {mt5.last_error()}"
        except Exception as exc:
            self.last_error = str(exc)
        self.connected = False
        return False

    def get_account_equity(self, default_equity: float = 10000.0) -> float:
        """Return live MT5 account equity if connected, else default equity."""
        if self.connected and self._mt5 is not None:
            try:
                info = self._mt5.account_info()
                if info is not None and float(info.equity) > 0:
                    return float(info.equity)
            except Exception:
                pass
        return default_equity

    def resolve_broker_symbol(self, base_symbol: str) -> str:
        """
        Automatically match broker symbol suffixes (e.g. `XAUUSD`, `XAUUSDm`, `XAUUSD.raw`, `XAUUSD.pro`).
        """
        if not self.connected or self._mt5 is None:
            return base_symbol
        try:
            # First try exact symbol
            if self._mt5.symbol_select(base_symbol, True):
                return base_symbol
            # Otherwise search all available broker symbols starting with base_symbol
            all_syms = self._mt5.symbols_get() or []
            for s in all_syms:
                if s.name.upper().startswith(base_symbol.upper()):
                    self._mt5.symbol_select(s.name, True)
                    return s.name
        except Exception:
            pass
        return base_symbol

    def fetch_recent_candles(
        self, symbol: str, timeframe: str = "M15", count: int = 500
    ) -> List[Candle]:
        """Fetch recent OHLCV candles from MT5 if connected, else generate realistic feed."""
        if self.connected and self._mt5 is not None:
            broker_sym = self.resolve_broker_symbol(symbol)
            tf_map = {
                "M5": self._mt5.TIMEFRAME_M5,
                "M15": self._mt5.TIMEFRAME_M15,
                "H1": self._mt5.TIMEFRAME_H1,
                "H4": self._mt5.TIMEFRAME_H4,
            }
            mt5_tf = tf_map.get(timeframe, self._mt5.TIMEFRAME_M15)
            rates = self._mt5.copy_rates_from_pos(broker_sym, mt5_tf, 0, count)
            if rates is not None and len(rates) > 0:
                return [
                    Candle(
                        timestamp=datetime.fromtimestamp(int(r["time"]), tz=timezone.utc),
                        open=float(r["open"]),
                        high=float(r["high"]),
                        low=float(r["low"]),
                        close=float(r["close"]),
                        volume=float(r["tick_volume"]),
                    )
                    for r in rates
                ]
        return generate_realistic_market_candles(symbol=symbol, count=count)

    def handle_signal(self, signal: SetupSignal) -> str:
        """
        Process a high-confluence setup according to `execution_mode`:
          - `signal_only`: Displays the Copilot Signal Card (NEVER places a trade).
          - `confirm`: Displays the card and prompts the user `[Y/n]` before sending to MT5.
          - `auto`: Sends market order with SL and TP directly to MT5.
        """
        print(signal.format_signal_card())

        tg_note = ""
        if self.telegram.is_configured:
            ok, msg = self.telegram.send_signal(signal)
            tg_note = f" | Telegram: {msg}" if ok else f" | Telegram Failed: {msg}"

        if self.execution_mode == "signal_only":
            return (
                "[COPILOT MODE: signal_only] Setup displayed for your manual review on "
                f"TradingView / MT5. No automated order was placed.{tg_note}"
            )

        if self.execution_mode == "confirm":
            ans = input(
                f"\nExecute {signal.direction.value} {signal.recommended_lot:.2f} lots on "
                f"{signal.symbol} in MetaTrader 5? [y/N]: "
            ).strip().lower()
            if ans != "y":
                return "[CONFIRM MODE] Trade skipped by user."

        if not self.connected or self._mt5 is None:
            return (
                f"[{self.execution_mode.upper()} MODE] MT5 terminal not connected on this machine; "
                "order simulated locally."
            )

        order_type = (
            self._mt5.ORDER_TYPE_BUY
            if signal.direction == Direction.LONG
            else self._mt5.ORDER_TYPE_SELL
        )
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": signal.recommended_lot,
            "type": order_type,
            "price": signal.entry_price,
            "sl": signal.stop_loss,
            "tp": signal.tp1_price,
            "deviation": 20,
            "magic": 202605,
            "comment": f"Hybrid_{signal.active_combination}",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
        }
        result = self._mt5.order_send(request)
        return f"[MT5 ORDER SENT] Result: {result}"


def load_csv_candles(csv_path: str | Path) -> List[Candle]:
    """Load OHLCV bars from a standard MT5 or TradingView CSV file."""
    candles: List[Candle] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_str = row.get("time") or row.get("timestamp") or row.get("Date") or ""
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            candles.append(
                Candle(
                    timestamp=ts,
                    open=float(row.get("open") or row.get("Open") or 0.0),
                    high=float(row.get("high") or row.get("High") or 0.0),
                    low=float(row.get("low") or row.get("Low") or 0.0),
                    close=float(row.get("close") or row.get("Close") or 0.0),
                    volume=float(row.get("volume") or row.get("Volume") or 1.0),
                )
            )
    return candles


def generate_realistic_market_candles(
    symbol: str = "XAUUSD", count: int = 1200, seed: Optional[int] = None
) -> List[Candle]:
    """
    Generate realistic M15 multi-regime market data with authentic session structure:
      - Asian Session (00:00-06:45 UTC): Tighter range compression
      - London Open (07:00-09:30 UTC): Stop-hunt liquidity sweeps (ICT SSL/BSL) followed by
        strong displacement trends & Fair Value Gaps, or clean Asian box breakouts
      - London/NY Trend Continuation (10:00-16:30 UTC): Multi-EMA trend with pullbacks into
        unmitigated SND demand/supply zones
      - Occasional high-volatility news chop bursts (to test regime filtering!)
    """
    sym_seeds = {"XAUUSD": 101, "EURUSD": 202, "USDJPY": 303}
    rng = random.Random(seed if seed is not None else sym_seeds.get(symbol, 42))

    if symbol == "XAUUSD":
        price = 2650.00
        base_step = 2.20
    elif symbol == "EURUSD":
        price = 1.08500
        base_step = 0.00075
    else:  # USDJPY
        price = 152.500
        base_step = 0.110

    start_time = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    candles: List[Candle] = []

    regime_bias = 1.0
    cycle_len = 96  # 1 trading day on M15 (96 x 15m = 24h)
    active_demand_level: Optional[float] = None
    active_supply_level: Optional[float] = None

    for i in range(count):
        ts = start_time + timedelta(minutes=15 * i)
        bar_in_day = i % cycle_len
        day_num = i // cycle_len

        # Rotate macro trend bias across days, with occasional choppy days
        if bar_in_day == 0:
            if day_num % 5 == 4:
                regime_bias = 0.0  # Chop / range day
            elif day_num % 2 == 0:
                regime_bias = 1.0  # Bullish institutional flow day
            else:
                regime_bias = -1.0  # Bearish institutional flow day

        o = price
        asian_bars = candles[-min(len(candles), 28) :] if candles else []
        recent_high = max((c.high for c in asian_bars), default=o + base_step)
        recent_low = min((c.low for c in asian_bars), default=o - base_step)

        # 1. Asian Session (00:00 - 06:45 UTC -> bars 0..27): Tight accumulation box
        if bar_in_day < 28:
            mid_box = 0.5 * (recent_high + recent_low)
            mean_rev = -0.22 * (o - mid_box)
            delta = mean_rev + rng.gauss(0.0, 0.42 * base_step)
            c = o + delta
            h = max(o, c) + abs(rng.gauss(0.0, 0.25 * base_step))
            l = min(o, c) - abs(rng.gauss(0.0, 0.25 * base_step))

        # 2. London Open Liquidity Sweep & Displacement (07:00 - 08:30 UTC -> bars 28..34)
        elif 28 <= bar_in_day <= 33 and regime_bias != 0.0:
            if bar_in_day == 28:
                # Stop-hunt wick beyond Asian boundary, closing back inside!
                if regime_bias > 0:
                    l = recent_low - 0.85 * base_step
                    c = recent_low + 0.35 * base_step
                    h = max(o, c) + 0.25 * base_step
                    active_demand_level = 0.5 * (l + c)
                else:
                    h = recent_high + 0.85 * base_step
                    c = recent_high - 0.35 * base_step
                    l = min(o, c) - 0.25 * base_step
                    active_supply_level = 0.5 * (h + c)
            elif bar_in_day in (29, 30, 31):
                # Strong institutional impulse displacement creating FVG & SND zone
                impulse = regime_bias * rng.uniform(1.15, 1.65) * base_step
                c = o + impulse
                if regime_bias > 0:
                    l = o - 0.12 * base_step
                    h = c + 0.20 * base_step
                else:
                    h = o + 0.12 * base_step
                    l = c - 0.20 * base_step
            else:
                c = o + regime_bias * rng.uniform(0.3, 0.8) * base_step
                h = max(o, c) + 0.25 * base_step
                l = min(o, c) - 0.25 * base_step

        # 3. Mid-Session Pullback & Continuation (08:45 - 16:45 UTC -> bars 34..67)
        elif 34 <= bar_in_day <= 67 and regime_bias != 0.0:
            if 40 <= bar_in_day <= 44:
                # Controlled pullback toward EMA20 / Demand-Supply Zone
                c = o - regime_bias * rng.uniform(0.35, 0.65) * base_step
                h = max(o, c) + 0.25 * base_step
                l = min(o, c) - 0.25 * base_step
            elif bar_in_day == 45:
                # Rejection & resumption candle off value zone
                c = o + regime_bias * rng.uniform(0.95, 1.40) * base_step
                if regime_bias > 0:
                    l = o - 0.70 * base_step
                    h = c + 0.18 * base_step
                else:
                    h = o + 0.70 * base_step
                    l = c - 0.18 * base_step
            else:
                # Steady trend progression
                drift = regime_bias * 0.32 * base_step
                c = o + drift + rng.gauss(0.0, 0.48 * base_step)
                h = max(o, c) + abs(rng.gauss(0.0, 0.30 * base_step))
                l = min(o, c) - abs(rng.gauss(0.0, 0.30 * base_step))

        # 4. Chop / High-Vol Whipsaw Day or Late NY Consolidation
        else:
            vol_mult = 1.65 if (regime_bias == 0.0 and 32 <= bar_in_day <= 56) else 0.55
            # Alternate direction every couple bars in chop so naive breakout gets whipsawed
            whipsaw = (
                (-1.0 if (bar_in_day % 2 == 0) else 1.0) * 0.75 * base_step
                if regime_bias == 0.0
                else 0.0
            )
            c = o + whipsaw + rng.gauss(0.0, vol_mult * base_step)
            h = max(o, c) + abs(rng.gauss(0.0, 0.55 * vol_mult * base_step))
            l = min(o, c) - abs(rng.gauss(0.0, 0.55 * vol_mult * base_step))

        # Keep prices positive and well-rounded
        decimals = 2 if symbol == "XAUUSD" else (3 if symbol == "USDJPY" else 5)
        o_r = round(o, decimals)
        h_r = round(max(o, c, h), decimals)
        l_r = round(min(o, c, l), decimals)
        c_r = round(c, decimals)

        candles.append(
            Candle(
                timestamp=ts,
                open=o_r,
                high=h_r,
                low=l_r,
                close=c_r,
                volume=round(rng.uniform(500.0, 2500.0), 1),
            )
        )
        price = c_r

    return candles
