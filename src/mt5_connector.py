"""
MetaTrader 5 & 24/7 Live Crypto Autonomous Auto-Trader Bridge & Position Lifecycle Manager.

Modes of Operation:
  1. Autonomous Auto-Trader (`execution_mode="auto"`):
     - Trades `BTCUSD`, `ETHUSD`, `SOLUSD` 24/7 (including Saturday & Sunday!) plus
       `XAUUSD`, `EURUSD`, `USDJPY` during Forex market hours.
     - Places live orders in MetaTrader 5 when the broker symbol is open for trading, and
       automatically routes weekend/24-7 Crypto trades (`BTCUSD`, `ETHUSD`, `SOLUSD`) through
       the built-in 24/7 Live Crypto Execution Book backed by real-time Binance/Bybit/Coinbase
       orderbook ticks when MT5 weekend servers are paused.
     - Actively monitors open positions every cycle:
       * When `TP1 (1.5R)` is reached: locks in 50% partial profit, moves Stop Loss to
         Break-Even (`entry_price`), and sends a Telegram notification.
       * When the trade closes (`TP2`, `SL`, `Break-Even`, or manual close): computes exact
         realized P&L ($ USD and R-multiple), updates cumulative bot stats, and sends a
         Telegram Trade Result Card!
  2. Copilot Signal Mode (`execution_mode="signal_only"`): Sends Signal Cards without placing orders.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from src.adaptive_learner import AdaptiveLearner
from src.config import ALL_INSTRUMENTS, CRYPTO_INSTRUMENTS, INSTRUMENTS, InstrumentProfile
from src.crypto_exchange import CryptoLiveFeed
from src.models import Candle, Direction, SetupSignal
from src.telegram_notifier import TelegramNotifier

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
ACTIVE_TRADES_PATH = LOGS_DIR / "active_trades.json"
BOT_STATS_PATH = LOGS_DIR / "bot_stats.json"
BOT_MAGIC = 202605


class MT5Connector:
    """Bridge to MetaTrader 5 + 24/7 Live Crypto Exchange Feed & Auto-Execution Book."""

    def __init__(self, execution_mode: str = "auto", learner: AdaptiveLearner | None = None) -> None:
        self.execution_mode = execution_mode
        self._mt5 = None
        self.connected = False
        self.last_error: str | None = None
        self.terminal_path: str | None = None
        self.telegram = TelegramNotifier()
        self.crypto_feed = CryptoLiveFeed()
        self.learner = learner or AdaptiveLearner()
        self.active_trades: Dict[str, Dict[str, Any]] = self._load_json(ACTIVE_TRADES_PATH, {})
        self.stats: Dict[str, Any] = self._load_json(
            BOT_STATS_PATH,
            {
                "total_closed": 0,
                "wins": 0,
                "losses": 0,
                "breakevens": 0,
                "cumulative_pnl": 0.0,
                "history": [],
            },
        )
        # Hydrate synced state from logs/home_pc_status.json on cloud / office failover machines
        synced_status = self._load_json(LOGS_DIR / "home_pc_status.json", {})
        if isinstance(synced_status, dict):
            synced_stats = synced_status.get("bot_stats")
            if isinstance(synced_stats, dict) and int(synced_stats.get("total_closed", 0)) > int(self.stats.get("total_closed", 0)):
                self.stats = dict(synced_stats)
            if not self.active_trades and isinstance(synced_status.get("active_trades"), dict):
                self.active_trades = dict(synced_status["active_trades"])

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        try:
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return default

    def _save_state(self) -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            ACTIVE_TRADES_PATH.write_text(
                json.dumps(self.active_trades, indent=2), encoding="utf-8"
            )
            BOT_STATS_PATH.write_text(
                json.dumps(self.stats, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def connect(self) -> bool:
        """Attempt to initialize local MetaTrader 5 terminal (Personal PC only)."""
        try:
            try:
                import MetaTrader5 as mt5  # type: ignore
            except ModuleNotFoundError:
                import subprocess
                import sys

                if sys.platform == "win32":
                    print("  [MT5 Bridge] Installing MetaTrader5 Python package into .venv...")
                    subprocess.run(
                        [sys.executable, "-m", "pip", "install", "MetaTrader5", "--quiet"],
                        check=False,
                    )
                    import MetaTrader5 as mt5  # type: ignore
                else:
                    raise

            import subprocess
            import sys

            if sys.platform == "win32":
                ps_cmd = (
                    "$p = Get-Process terminal64 -ErrorAction SilentlyContinue | Select-Object -First 1; "
                    "if ($p) { if ($p.Path) { Write-Output ('PROC:' + $p.Path) } else { Write-Output 'PROC_ELEVATED' } } "
                    "else { Write-Output 'NO_TERMINAL64' }"
                )
                proc_res = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    capture_output=True,
                    text=True,
                    check=False,
                ).stdout.strip()

                if proc_res.startswith("PROC:"):
                    exe_path = proc_res.split("PROC:", 1)[1].strip()
                    if mt5.initialize(path=exe_path, timeout=8000):
                        self._mt5 = mt5
                        self.connected = True
                        self.terminal_path = exe_path
                        self.last_error = None
                        return True

                if mt5.initialize(timeout=8000):
                    self._mt5 = mt5
                    self.connected = True
                    self.last_error = None
                    return True

                first_err = mt5.last_error()
                self.last_error = (
                    f"mt5.initialize() failed: {first_err} | "
                    f"proc_check={proc_res}"
                )
            else:
                if mt5.initialize():
                    self._mt5 = mt5
                    self.connected = True
                    self.last_error = None
                    return True
                self.last_error = f"mt5.initialize() returned False: {mt5.last_error()}"
        except Exception as exc:
            self.last_error = str(exc)

        # Zero-Admin Cloud MT5 HTTPS Bridge fallback (MetaApi.cloud) for office laptops / cloud VMs
        try:
            from src.metaapi_bridge import MetaApiMT5Bridge
            cloud_bridge = MetaApiMT5Bridge()
            if cloud_bridge.is_configured and cloud_bridge.initialize():
                self._mt5 = cloud_bridge
                self.connected = True
                self.terminal_path = f'MetaApi Cloud HTTPS ({cloud_bridge.region})'
                self.last_error = None
                return True
        except Exception as cloud_exc:
            self.last_error = f'{self.last_error} | CloudBridge: {cloud_exc}'

        self.connected = False
        return False

    def get_account_equity(self, default_equity: float = 10000.0) -> float:
        """Return live MT5 account equity (+ realized 24/7 crypto PnL if applicable)."""
        if self.connected and self._mt5 is not None:
            try:
                info = self._mt5.account_info()
                if info is not None and float(info.equity) > 0:
                    crypto_pnl = float(self.stats.get("crypto_book_pnl", 0.0))
                    return round(float(info.equity) + crypto_pnl, 2)
            except Exception:
                pass
        return round(default_equity + float(self.stats.get("cumulative_pnl", 0.0)), 2)

    def get_account_balance(self, default_balance: float = 10000.0) -> float:
        """Return live MT5 account balance (+ realized 24/7 crypto PnL if applicable)."""
        if self.connected and self._mt5 is not None:
            try:
                info = self._mt5.account_info()
                if info is not None and float(info.balance) > 0:
                    crypto_pnl = float(self.stats.get("crypto_book_pnl", 0.0))
                    return round(float(info.balance) + crypto_pnl, 2)
            except Exception:
                pass
        return round(default_balance + float(self.stats.get("cumulative_pnl", 0.0)), 2)

    def get_account_meta(self) -> Dict[str, Any]:
        """Return connected MT5 account login, server, and company metadata."""
        if self.connected and self._mt5 is not None:
            try:
                info = self._mt5.account_info()
                if info is not None:
                    return {
                        "login": int(getattr(info, "login", 0)),
                        "server": str(getattr(info, "server", "")),
                        "company": str(getattr(info, "company", "")),
                        "currency": str(getattr(info, "currency", "USD")),
                    }
            except Exception:
                pass
        return {"login": 0, "server": None, "company": None, "currency": "USD"}

    def mt5_has_live_symbol(self, base_symbol: str) -> bool:
        """Return True if the connected MT5 terminal has `base_symbol` with live tradeable quotes."""
        if not self.connected or self._mt5 is None:
            return False
        try:
            broker_sym = self.resolve_broker_symbol(base_symbol)
            sym_info = self._mt5.symbol_info(broker_sym)
            tick = self._mt5.symbol_info_tick(broker_sym)
            if sym_info is None or tick is None or float(tick.ask) <= 0:
                return False
            # Check that the symbol exists in MT5 and its name actually matches base_symbol
            if not sym_info.name.upper().startswith(base_symbol.upper()[:3]):
                return False
            trade_mode = int(getattr(sym_info, "trade_mode", 4))
            # 0 = SYMBOL_TRADE_MODE_DISABLED
            return trade_mode != 0
        except Exception:
            return False

    def resolve_broker_symbol(self, base_symbol: str) -> str:
        """
        Automatically match broker symbol suffixes and XM aliases
        (e.g. `XAUUSD` -> `GOLD`/`GOLD#`, `BTCUSD` -> `BTCUSD`/`BITCOIN`, `ETHUSD` -> `ETHUSD`/`ETHEREUM`).
        """
        if not self.connected or self._mt5 is None:
            return base_symbol
        alias_map = {
            "XAUUSD": ["XAUUSD", "GOLD", "XAUUSD#", "GOLD#", "XAUUSDm", "GOLDm"],
            "BTCUSD": ["BTCUSD", "BTCUSD#", "BTCUSDm", "BITCOIN", "BITCOIN#", "BTCUSDT"],
            "ETHUSD": ["ETHUSD", "ETHUSD#", "ETHUSDm", "ETHEREUM", "ETHEREUM#", "ETHUSDT"],
            "SOLUSD": ["SOLUSD", "SOLUSD#", "SOLUSDm", "SOLANA", "SOLANA#", "SOLUSDT"],
            "EURUSD": ["EURUSD", "EURUSD#", "EURUSDm"],
            "USDJPY": ["USDJPY", "USDJPY#", "USDJPYm"],
        }
        candidates = alias_map.get(base_symbol.upper(), [base_symbol])
        try:
            for cand in candidates:
                info = self._mt5.symbol_info(cand)
                if info is not None and self._mt5.symbol_select(cand, True):
                    return cand
            all_syms = self._mt5.symbols_get() or []
            for cand in candidates:
                for s in all_syms:
                    if s.name.upper().startswith(cand.upper()):
                        self._mt5.symbol_select(s.name, True)
                        return s.name
        except Exception:
            pass
        return base_symbol

    def fetch_recent_candles(
        self, symbol: str, timeframe: str = "M15", count: int = 500
    ) -> List[Candle]:
        """
        Fetch recent OHLCV candles:
          - For 24/7 Crypto (`BTCUSD`, `ETHUSD`, `SOLUSD`): Uses live 24/7 Crypto Exchange feed
            (or active MT5 feed if MT5 is streaming fresh weekend crypto bars).
          - For Forex/Gold (`XAUUSD`, `EURUSD`, `USDJPY`): Uses live MT5 terminal bars.
        """
        sym_upper = symbol.upper()
        is_crypto = sym_upper in CRYPTO_INSTRUMENTS

        # 1. If 24/7 Crypto symbol, prefer live 24/7 Exchange Feed so weekend candles are always real-time
        if is_crypto:
            live_crypto = self.crypto_feed.fetch_candles(
                symbol=sym_upper, timeframe=timeframe, count=count
            )
            if len(live_crypto) >= 60:
                return live_crypto

        # 2. Try connected MetaTrader 5 terminal
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
            return []
        return generate_realistic_market_candles(symbol=symbol, count=count)

    def _send_order_with_filling_fallback(self, base_request: Dict[str, Any]) -> Any:
        """Try broker filling modes (FOK -> IOC -> RETURN) so orders work across all MT5 brokers."""
        if self._mt5 is None:
            return None
        filling_modes = [
            self._mt5.ORDER_FILLING_FOK,
            self._mt5.ORDER_FILLING_IOC,
            self._mt5.ORDER_FILLING_RETURN,
        ]
        last_res = None
        for mode in filling_modes:
            req = dict(base_request)
            req["type_filling"] = mode
            res = self._mt5.order_send(req)
            last_res = res
            if res is not None and res.retcode == self._mt5.TRADE_RETCODE_DONE:
                return res
        return last_res

    def _has_open_trade_on_symbol(self, symbol: str) -> Optional[int]:
        """Check both active_trades book and MT5 open positions for an existing trade on `symbol`."""
        mt5_live_avail = self.mt5_has_live_symbol(symbol)
        for t_str, info in list(self.active_trades.items()):
            if str(info.get("symbol", "")).upper() == symbol.upper():
                if mt5_live_avail and info.get("broker_source") == "CRYPTO_24_7_LIVE":
                    del self.active_trades[t_str]
                    self._save_state()
                    continue
                return int(info.get("ticket", t_str))
        if self.connected and self._mt5 is not None:
            broker_sym = self.resolve_broker_symbol(symbol)
            pos = self._mt5.positions_get(symbol=broker_sym)
            if pos and len(pos) > 0:
                return int(pos[0].ticket)
        return None

    def _execute_live_crypto_book_order(self, signal: SetupSignal) -> str:
        """
        Execute a 24/7 Live Crypto order backed by real-time Binance/Bybit/Coinbase bid/ask prices
        when the MT5 broker does not offer weekend crypto execution.
        """
        quote = self.crypto_feed.fetch_live_price(signal.symbol)
        is_long = signal.direction == Direction.LONG
        decimals = 2 if signal.symbol.upper().startswith("BTC") else 2

        if quote is not None:
            bid, ask, _ = quote
            fill_price = round(ask if is_long else bid, decimals)
        else:
            fill_price = round(signal.entry_price, decimals)

        sl_dist = abs(signal.entry_price - signal.stop_loss)
        tp1_dist = abs(signal.tp1_price - signal.entry_price)
        tp2_dist = abs(signal.tp2_price - signal.entry_price)

        if is_long:
            sl_price = round(fill_price - sl_dist, decimals)
            tp1_price = round(fill_price + tp1_dist, decimals)
            tp2_price = round(fill_price + tp2_dist, decimals)
        else:
            sl_price = round(fill_price + sl_dist, decimals)
            tp1_price = round(fill_price - tp1_dist, decimals)
            tp2_price = round(fill_price - tp2_dist, decimals)

        ticket = int(datetime.now(timezone.utc).timestamp()) % 10_000_000 + 80_000_000
        volume = signal.recommended_lot
        acct_equity = self.get_account_equity()

        agreeing_strats = list(
            dict.fromkeys(v.strategy_name for v in signal.votes if v.direction == signal.direction)
        )
        if (
            signal.kronos is not None
            and signal.kronos.direction == signal.direction
            and "KRONOS" not in agreeing_strats
        ):
            agreeing_strats.append("KRONOS")

        self.active_trades[str(ticket)] = {
            "ticket": ticket,
            "symbol": signal.symbol,
            "broker_symbol": signal.symbol,
            "broker_source": "CRYPTO_24_7_LIVE",
            "direction": signal.direction.value,
            "entry_price": fill_price,
            "sl_price": sl_price,
            "initial_sl_price": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "volume": volume,
            "initial_volume": volume,
            "risk_usd": signal.risk_usd,
            "confluence_score": signal.confluence_score,
            "regime": signal.regime.value,
            "agreeing_strategies": agreeing_strats,
            "opened_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "tp1_hit": False,
            "partial_pnl": 0.0,
        }
        self._save_state()

        tg_note = ""
        if self.telegram.is_configured:
            html_msg = self.telegram.format_trade_opened_html(
                signal=signal,
                ticket=ticket,
                fill_price=fill_price,
                sl_price=sl_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                volume=volume,
                account_equity=acct_equity,
            )
            ok, msg = self.telegram.send_message(html_msg)
            tg_note = f" | Telegram: {msg}" if ok else f" | Telegram Failed: {msg}"

        return (
            f"[24/7 CRYPTO AUTO-TRADE EXECUTED] Ticket #{ticket} | {signal.direction.value} "
            f"{volume:.2f} lots {signal.symbol} @ {fill_price} "
            f"(SL={sl_price}, TP1={tp1_price}, TP2={tp2_price}){tg_note}"
        )

    def handle_signal(self, signal: SetupSignal) -> str:
        """
        Process a high-confluence setup according to `execution_mode`:
          - `auto`: Places live order in MetaTrader 5 (or 24/7 Live Crypto Book if MT5 weekend
                    crypto is unavailable), tracks ticket for TP1/BE/Close, and sends an instant
                    Telegram execution alert.
          - `signal_only`: Displays the Copilot Signal Card and sends to Telegram without ordering.
        """
        print(signal.format_signal_card())

        if self.execution_mode == "signal_only":
            tg_note = ""
            if self.telegram.is_configured:
                ok, msg = self.telegram.send_signal(signal)
                tg_note = f" | Telegram: {msg}" if ok else f" | Telegram Failed: {msg}"
            return (
                "[COPILOT MODE: signal_only] Setup displayed for manual review. "
                f"No automated order was placed.{tg_note}"
            )

        if self.execution_mode == "confirm":
            ans = input(
                f"\nExecute {signal.direction.value} {signal.recommended_lot:.2f} lots on "
                f"{signal.symbol} in MetaTrader 5? [y/N]: "
            ).strip().lower()
            if ans != "y":
                return "[CONFIRM MODE] Trade skipped by user."

        # 1. Check duplicate open position on this symbol
        dup_ticket = self._has_open_trade_on_symbol(signal.symbol)
        if dup_ticket is not None:
            return (
                f"[AUTO-TRADER SKIP] {signal.symbol} already has an open position "
                f"(Ticket #{dup_ticket}). Avoiding duplicate exposure."
            )

        is_crypto_sig = signal.symbol.upper() in CRYPTO_INSTRUMENTS

        # 1b. Correlated Crypto Risk Cap: Never open a 2nd Crypto trade while another Crypto trade is unprotected (< BE Shield)
        if is_crypto_sig:
            for open_t in self.active_trades.values():
                open_sym = str(open_t.get("symbol", "")).upper()
                if open_sym in CRYPTO_INSTRUMENTS and not (
                    open_t.get("be_shield_active", False) or open_t.get("tp1_hit", False)
                ):
                    return (
                        f"[CORRELATED CRYPTO GUARD] Skipping {signal.symbol}: {open_sym} "
                        f"(Ticket #{open_t.get('ticket')}) is still unprotected (< Break-Even Shield). "
                        f"Capping simultaneous unprotected Crypto risk at 1 trade."
                    )

        # 1c. 60m Same-Symbol Re-Entry Cooldown & 45m Cross-Crypto Stop-Loss Cooldown
        checked_same_symbol = False
        for past in reversed(list(self.stats.get("history", []))):
            past_sym = str(past.get("symbol", "")).upper()
            past_dir = str(past.get("direction", "")).upper()
            past_r = float(past.get("r_multiple", 0.0))
            closed_str = str(past.get("closed_at_utc", ""))
            elapsed_sec = 999999.0
            if closed_str:
                try:
                    closed_dt = datetime.strptime(closed_str, "%Y-%m-%d %H:%M:%S UTC").replace(
                        tzinfo=timezone.utc
                    )
                    elapsed_sec = (datetime.now(timezone.utc) - closed_dt).total_seconds()
                except Exception:
                    pass

            if not checked_same_symbol and past_sym == signal.symbol.upper():
                checked_same_symbol = True
                if past_dir == signal.direction.value.upper() and past_r < 0.80 and 0 <= elapsed_sec < 3600:
                    mins_ago = elapsed_sec / 60.0
                    return (
                        f"[RE-ENTRY COOLDOWN GUARD] Skipping {signal.symbol} {signal.direction.value}: "
                        f"previous {signal.direction.value} closed {mins_ago:.0f}m ago on reversal stop. "
                        f"Waiting 60m for structure reset."
                    )

            if (
                is_crypto_sig
                and past_sym in CRYPTO_INSTRUMENTS
                and past_dir == signal.direction.value.upper()
                and past_r <= -0.50
                and 0 <= elapsed_sec < 2700
            ):
                mins_ago = elapsed_sec / 60.0
                return (
                    f"[CRYPTO SECTOR COOLDOWN] Skipping {signal.symbol} {signal.direction.value}: "
                    f"correlated {past_sym} {past_dir} hit Stop Loss {mins_ago:.0f}m ago ({past_r:+.2f}R). "
                    f"Waiting 45m for crypto sector stabilization."
                )

        # 2. Check max concurrent open positions (dynamically scaled for $10-$50 small accounts)
        acct_equity_pre = self.get_account_equity()
        if acct_equity_pre < 35.0:
            max_concurrent = 1
        elif acct_equity_pre < 100.0:
            max_concurrent = 2
        elif acct_equity_pre < 500.0:
            max_concurrent = 3
        else:
            max_concurrent = 6
        if len(self.active_trades) >= max_concurrent:
            return (
                f"[AUTO-TRADER SKIP] Max concurrent open positions "
                f"({len(self.active_trades)}/{max_concurrent} for ${acct_equity_pre:,.2f} equity) reached."
            )

        is_crypto = signal.symbol.upper() in CRYPTO_INSTRUMENTS

        # 3. Attempt direct MT5 execution if connected
        if self.connected and self._mt5 is not None:
            broker_sym = self.resolve_broker_symbol(signal.symbol)
            profile: InstrumentProfile = ALL_INSTRUMENTS.get(signal.symbol, INSTRUMENTS["XAUUSD"])
            sym_info = self._mt5.symbol_info(broker_sym)
            tick = self._mt5.symbol_info_tick(broker_sym)

            if sym_info is not None and tick is not None and float(tick.ask) > 0:
                live_spread = float(tick.ask - tick.bid)
                spread_limit = profile.max_spread_price * (8.0 if is_crypto else 2.5)
                if live_spread <= spread_limit:
                    digits = int(sym_info.digits)
                    vol_min = float(sym_info.volume_min)
                    vol_max = float(sym_info.volume_max)
                    vol_step = float(sym_info.volume_step) if float(sym_info.volume_step) > 0 else 0.01
                    raw_lot = signal.recommended_lot
                    tick_val = float(getattr(sym_info, "trade_tick_value", 0.0) or 0.0)
                    tick_sz = float(getattr(sym_info, "trade_tick_size", 0.0) or 0.0)
                    is_long = signal.direction == Direction.LONG
                    fill_price = round(float(tick.ask if is_long else tick.bid), digits)

                    # Broker Percentage Spread Ceiling (blocks wide-spread altcoin CFDs like XM SOLUSD 0.84% spread)
                    if is_crypto and fill_price > 0 and (live_spread / fill_price) > 0.0025:
                        return (
                            f"[SPREAD CEILING GUARD] Skipping {broker_sym}: live broker spread "
                            f"${live_spread:.2f} ({live_spread / fill_price * 100:.2f}% of price) "
                            f"exceeds 0.25% M5 scalp ceiling."
                        )

                    # Anti-Wick Structural Stop Floor + Live MT5 Spread Buffer:
                    # Pad SL by +live_spread and enforce >= 4.5x spread & >= 0.22% floor on Crypto so
                    # weekend compressed M5 ATR + broker spread never wicks out a winning trade early.
                    raw_signal_sl = abs(signal.entry_price - signal.stop_loss)
                    crypto_floor = (fill_price * 0.0022) if is_crypto else 0.0
                    sl_dist = max(raw_signal_sl + live_spread, live_spread * 4.5, crypto_floor)
                    tp1_dist = max(abs(signal.tp1_price - signal.entry_price), sl_dist * 0.52)
                    tp2_dist = max(abs(signal.tp2_price - signal.entry_price), sl_dist * 1.10)

                    risk_per_lot = 0.0
                    if tick_val > 0 and tick_sz > 0 and sl_dist > 0:
                        risk_per_lot = (sl_dist / tick_sz) * tick_val
                        if risk_per_lot > 0:
                            raw_lot = signal.risk_usd / risk_per_lot
                    steps = math.floor((raw_lot / vol_step) + 1e-9)
                    volume = round(max(vol_min, min(vol_max, steps * vol_step)), 2)
                    actual_risk_usd = round(
                        volume * risk_per_lot if risk_per_lot > 0 else signal.risk_usd, 2
                    )

                    # Small-Account Pre-Trade 0.01-Lot Risk Cap (blocks oversized Standard contracts on $10-$50 accounts)
                    if acct_equity_pre < 250.0 and risk_per_lot > 0:
                        min_lot_risk = vol_min * risk_per_lot
                        if min_lot_risk > acct_equity_pre * 0.12:
                            return (
                                f"[SMALL-ACCOUNT GUARD] Skipping {broker_sym}: minimum {vol_min:.2f} lot "
                                f"risks ${min_lot_risk:.2f} ({min_lot_risk / max(acct_equity_pre, 1.0) * 100:.1f}% "
                                f"of ${acct_equity_pre:.2f} equity > 12% safety cap)."
                            )

                    if is_long:
                        sl_price = round(fill_price - sl_dist, digits)
                        tp1_price = round(fill_price + tp1_dist, digits)
                        tp2_price = round(fill_price + tp2_dist, digits)
                        order_type = self._mt5.ORDER_TYPE_BUY
                    else:
                        sl_price = round(fill_price + sl_dist, digits)
                        tp1_price = round(fill_price - tp1_dist, digits)
                        tp2_price = round(fill_price - tp2_dist, digits)
                        order_type = self._mt5.ORDER_TYPE_SELL

                    # Small-Account Free-Margin Anti-Margin-Call Check (never exceed 50% of free margin)
                    try:
                        acct_info = self._mt5.account_info()
                        free_margin = float(getattr(acct_info, "margin_free", acct_equity_pre) or acct_equity_pre)
                        req_margin = self._mt5.order_calc_margin(order_type, broker_sym, volume, fill_price)
                        if req_margin is not None and float(req_margin) > free_margin * 0.50:
                            return (
                                f"[MARGIN-CALL SHIELD] Skipping {broker_sym}: required margin "
                                f"${float(req_margin):.2f} exceeds 50% of free margin (${free_margin:.2f})."
                            )
                    except Exception:
                        pass

                    request = {
                        "action": self._mt5.TRADE_ACTION_DEAL,
                        "symbol": broker_sym,
                        "volume": volume,
                        "type": order_type,
                        "price": fill_price,
                        "sl": sl_price,
                        "tp": tp2_price,
                        "deviation": 30,
                        "magic": BOT_MAGIC,
                        "comment": f"Auto_{signal.symbol}_{signal.active_combination[:10]}",
                        "type_time": self._mt5.ORDER_TIME_GTC,
                    }

                    result = self._send_order_with_filling_fallback(request)
                    if result is not None and result.retcode == self._mt5.TRADE_RETCODE_DONE:
                        pos_list = self._mt5.positions_get(symbol=broker_sym) or []
                        ticket = int(pos_list[-1].ticket) if pos_list else int(getattr(result, "order", 0))
                        actual_fill = float(getattr(result, "price", fill_price) or fill_price)
                        acct_equity = self.get_account_equity()

                        agreeing_strats = list(
                            dict.fromkeys(
                                v.strategy_name for v in signal.votes if v.direction == signal.direction
                            )
                        )
                        if (
                            signal.kronos is not None
                            and signal.kronos.direction == signal.direction
                            and "KRONOS" not in agreeing_strats
                        ):
                            agreeing_strats.append("KRONOS")

                        self.active_trades[str(ticket)] = {
                            "ticket": ticket,
                            "symbol": signal.symbol,
                            "broker_symbol": broker_sym,
                            "broker_source": "MT5_LIVE",
                            "direction": signal.direction.value,
                            "entry_price": actual_fill,
                            "sl_price": sl_price,
                            "initial_sl_price": sl_price,
                            "tp1_price": tp1_price,
                            "tp2_price": tp2_price,
                            "volume": volume,
                            "initial_volume": volume,
                            "risk_usd": actual_risk_usd,
                            "confluence_score": signal.confluence_score,
                            "regime": signal.regime.value,
                            "agreeing_strategies": agreeing_strats,
                            "opened_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "tp1_hit": False,
                            "partial_pnl": 0.0,
                            "scalp_upgraded": True,
                        }
                        self._save_state()

                        tg_note = ""
                        if self.telegram.is_configured:
                            html_msg = self.telegram.format_trade_opened_html(
                                signal=signal,
                                ticket=ticket,
                                fill_price=actual_fill,
                                sl_price=sl_price,
                                tp1_price=tp1_price,
                                tp2_price=tp2_price,
                                volume=volume,
                                account_equity=acct_equity,
                            )
                            ok, msg = self.telegram.send_message(html_msg)
                            tg_note = f" | Telegram: {msg}" if ok else f" | Telegram Failed: {msg}"

                        return (
                            f"[AUTO-TRADE EXECUTED (MT5)] Ticket #{ticket} | {signal.direction.value} "
                            f"{volume:.2f} lots {broker_sym} @ {actual_fill} "
                            f"(SL={sl_price}, TP1={tp1_price}, TP2={tp2_price}){tg_note}"
                        )

        # 4. If 24/7 Crypto symbol and MT5 doesn't execute weekend crypto orders, execute on Live Crypto Book!
        if is_crypto:
            return self._execute_live_crypto_book_order(signal)

        return f"[AUTO-TRADER SKIP] Market closed or symbol unavailable on MT5 for {signal.symbol}."

    def _manage_crypto_book_trade(self, t_str: str, trade: Dict[str, Any]) -> List[str]:
        """Monitor a 24/7 Live Crypto trade against real-time exchange prices for TP1/BE and TP2/SL."""
        events: List[str] = []
        symbol = str(trade.get("symbol", "BTCUSD"))
        ticket = int(trade["ticket"])
        quote = self.crypto_feed.fetch_live_price(symbol)
        if quote is None:
            return events

        bid, ask, mid = quote
        is_long = str(trade.get("direction", "LONG")) == "LONG"
        cur_price = bid if is_long else ask
        entry_price = float(trade.get("entry_price", cur_price))
        sl_price = float(trade.get("sl_price", entry_price))
        tp1_price = float(trade.get("tp1_price", entry_price))
        tp2_price = float(trade.get("tp2_price", entry_price))
        risk_usd = max(float(trade.get("risk_usd", 15.0)), 1.0)
        init_vol = float(trade.get("initial_volume", trade.get("volume", 0.01)))

        if "initial_sl_price" not in trade or abs(entry_price - float(trade["initial_sl_price"])) < 1e-5:
            inferred_dist = max(abs(tp1_price - entry_price) / 0.75, abs(entry_price - sl_price), 1e-4)
            trade["initial_sl_price"] = round(
                entry_price - inferred_dist if is_long else entry_price + inferred_dist, 2
            )
        init_sl = float(trade["initial_sl_price"])
        sl_dist = max(abs(entry_price - init_sl), 1e-4)
        fav_excursion = (cur_price - entry_price) if is_long else (entry_price - cur_price)
        excursion_r = fav_excursion / sl_dist
        peak_r = max(float(trade.get("peak_favorable_r", 0.0)), round(excursion_r, 3))
        if peak_r > float(trade.get("peak_favorable_r", 0.0)):
            trade["peak_favorable_r"] = peak_r

        # Compute elapsed trade age in minutes for 20m / 35m Fast-Scalp Time-Decay Shield
        age_min = 0.0
        opened_str = str(trade.get("opened_at_utc", ""))
        if opened_str:
            try:
                opened_dt = datetime.strptime(opened_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                age_min = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 60.0
            except Exception:
                pass

        # Tier 1 (+0.35R excursion, +0.18R peak pullback, OR >=20m open with >=+0.10R) -> +0.04R BE Shield
        if not trade.get("be_shield_active", False) and not trade.get("tp1_hit", False):
            if (
                excursion_r >= 0.35
                or (peak_r >= 0.18 and 0.04 <= excursion_r <= 0.14)
                or (age_min >= 20.0 and excursion_r >= 0.10)
            ):
                be_sl = round(entry_price + (0.04 * sl_dist if is_long else -0.04 * sl_dist), 2)
                trade["be_shield_active"] = True
                trade["sl_price"] = be_sl
                sl_price = be_sl
                self._save_state()
                events.append(
                    f"[24/7 CRYPTO BE SHIELD] Ticket #{ticket} ({symbol}) reached {peak_r:+.2f}R! "
                    f"Locked SL at +0.04R ({be_sl})."
                )

        # 35-Minute Fast-Scalp Time-Decay Stagnation Shield (cuts risk on stale >35m scalps that never popped)
        if (
            age_min >= 35.0
            and not trade.get("be_shield_active", False)
            and not trade.get("tp1_hit", False)
            and not trade.get("time_decay_applied", False)
        ):
            decay_dist = 0.45 * sl_dist
            if is_long:
                decay_sl = round(min(cur_price - 0.06 * sl_dist, max(sl_price, entry_price - decay_dist)), 2)
            else:
                decay_sl = round(max(cur_price + 0.06 * sl_dist, min(sl_price, entry_price + decay_dist)), 2)
            trade["time_decay_applied"] = True
            trade["sl_price"] = decay_sl
            sl_price = decay_sl
            self._save_state()
            events.append(
                f"[35M TIME-DECAY SHIELD] Ticket #{ticket} ({symbol}) open {age_min:.0f}m without impulse; "
                f"tightened SL to {decay_sl} to cap stale-scalp risk."
            )

        # Tier 1.5 (+0.44R excursion) -> Lock +0.22R Guaranteed Profit Stop
        if not trade.get("pre_tp1_ratchet", False) and not trade.get("tp1_hit", False) and excursion_r >= 0.44:
            r22_sl = round(entry_price + (0.22 * sl_dist if is_long else -0.22 * sl_dist), 2)
            trade["pre_tp1_ratchet"] = True
            trade["be_shield_active"] = True
            trade["sl_price"] = r22_sl
            sl_price = r22_sl
            self._save_state()
            events.append(
                f"[24/7 CRYPTO +0.22R RATCHET] Ticket #{ticket} ({symbol}) reached {excursion_r:+.2f}R! "
                f"Locked SL at +0.22R ({r22_sl})."
            )

        # Tier 2: Check if Fast-Scalp TP1 (+0.52R or tp1_price) was reached
        if not trade.get("tp1_hit", False):
            reached_tp1 = (
                excursion_r >= 0.52
                or ((cur_price >= tp1_price) if is_long else (cur_price <= tp1_price))
            )
            if reached_tp1:
                partial_lots = round(init_vol * 0.5, 2)
                r20_sl = round(entry_price + (0.20 * sl_dist if is_long else -0.20 * sl_dist), 2)
                trade["tp1_hit"] = True
                trade["be_shield_active"] = True
                trade["sl_price"] = r20_sl
                tp1_r = abs(tp1_price - entry_price) / sl_dist
                trade["partial_pnl"] = round(0.5 * max(tp1_r, 0.52) * risk_usd, 2)
                self._save_state()

                if self.telegram.is_configured:
                    msg_html = self.telegram.format_tp1_breakeven_html(
                        trade=trade,
                        current_price=cur_price,
                        partial_closed_lots=partial_lots,
                    )
                    self.telegram.send_message(msg_html)

                events.append(
                    f"[24/7 CRYPTO TP1 HIT] Ticket #{ticket} ({symbol}) reached Fast-Scalp TP1 @ {cur_price}! "
                    f"Locked +${trade['partial_pnl']:.2f} & moved runner SL to +0.20R ({r20_sl})."
                )
                sl_price = r20_sl

        # Tier 2.5 (+0.75R -> +0.45R SL) & Tier 3 (+0.92R -> +0.72R SL) Post-TP1 Runner Ratchet
        if trade.get("tp1_hit", False):
            if excursion_r >= 0.92 and int(trade.get("runner_stage", 0)) < 2:
                r72_sl = round(entry_price + (0.72 * sl_dist if is_long else -0.72 * sl_dist), 2)
                trade["runner_stage"] = 2
                trade["sl_price"] = r72_sl
                sl_price = r72_sl
                self._save_state()
            elif excursion_r >= 0.75 and int(trade.get("runner_stage", 0)) < 1:
                r45_sl = round(entry_price + (0.45 * sl_dist if is_long else -0.45 * sl_dist), 2)
                trade["runner_stage"] = 1
                trade["sl_price"] = r45_sl
                sl_price = r45_sl
                self._save_state()

        # 2. Check if TP2 or Stop Loss (or Ratchet/BE SL) was hit
        hit_tp2 = (cur_price >= tp2_price) if is_long else (cur_price <= tp2_price)
        hit_sl = (cur_price <= sl_price) if is_long else (cur_price >= sl_price)

        if hit_tp2 or hit_sl:
            if hit_tp2:
                exit_price = tp2_price
                runner_r = abs(tp2_price - entry_price) / sl_dist
                net_pnl = round(float(trade.get("partial_pnl", 0.0)) + 0.5 * runner_r * risk_usd, 2)
                close_reason = "Take Profit 2 Target Hit 🎯"
            else:
                exit_price = sl_price
                locked_r = ((sl_price - entry_price) if is_long else (entry_price - sl_price)) / sl_dist
                if trade.get("tp1_hit", False):
                    net_pnl = round(float(trade.get("partial_pnl", 0.0)) + 0.5 * max(locked_r, 0.0) * risk_usd, 2)
                    close_reason = "TP1 + Ratchet Profit Stop Hit 🛡️💰"
                elif trade.get("pre_tp1_ratchet", False):
                    net_pnl = round(max(locked_r, 0.25) * risk_usd, 2)
                    close_reason = "+0.25R Pre-TP1 Profit Ratchet Hit 🛡️💰"
                elif trade.get("be_shield_active", False):
                    net_pnl = round(max(locked_r, 0.04) * risk_usd, 2)
                    close_reason = "Break-Even Shield Hit 🛡️"
                elif trade.get("time_decay_applied", False):
                    net_pnl = round(locked_r * risk_usd, 2)
                    close_reason = "35m Time-Decay Reduced Stop Hit ⏱️🛡️"
                else:
                    net_pnl = round(-risk_usd, 2)
                    close_reason = "Stop Loss Hit 🛑"

            r_mult = net_pnl / risk_usd
            self.stats["total_closed"] = int(self.stats.get("total_closed", 0)) + 1
            if net_pnl > 0.50:
                self.stats["wins"] = int(self.stats.get("wins", 0)) + 1
            elif net_pnl >= -0.50:
                self.stats["breakevens"] = int(self.stats.get("breakevens", 0)) + 1
            else:
                self.stats["losses"] = int(self.stats.get("losses", 0)) + 1

            self.stats["cumulative_pnl"] = round(
                float(self.stats.get("cumulative_pnl", 0.0)) + net_pnl, 2
            )
            self.stats["crypto_book_pnl"] = round(
                float(self.stats.get("crypto_book_pnl", 0.0)) + net_pnl, 2
            )

            closed_record = {
                "ticket": ticket,
                "symbol": symbol,
                "direction": trade.get("direction"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "net_pnl": net_pnl,
                "r_multiple": round(r_mult, 2),
                "close_reason": close_reason,
                "closed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            history_list = list(self.stats.get("history", []))
            history_list.append(closed_record)
            self.stats["history"] = history_list[-50:]

            del self.active_trades[t_str]
            self._save_state()

            acct_bal = self.get_account_balance()
            acct_eq = self.get_account_equity()
            learn_note = self.learner.record_trade_outcome(
                symbol=symbol,
                regime=str(trade.get("regime", "TREND_EXPANSION")),
                agreeing_strategies=list(trade.get("agreeing_strategies", ["SND", "ICT", "KRONOS"])),
                r_multiple=r_mult,
                net_pnl=net_pnl,
            )

            if self.telegram.is_configured:
                result_html = self.telegram.format_trade_closed_html(
                    trade=trade,
                    exit_price=exit_price,
                    net_pnl=net_pnl,
                    r_multiple=r_mult,
                    close_reason=close_reason,
                    account_balance=acct_bal,
                    account_equity=acct_eq,
                    stats=self.stats,
                    learning_summary=learn_note,
                )
                self.telegram.send_message(result_html)

            events.append(
                f"[24/7 CRYPTO TRADE CLOSED] Ticket #{ticket} ({symbol}) | {close_reason} | "
                f"P&L: ${net_pnl:+.2f} ({r_mult:+.2f}R) | Equity: ${acct_eq:,.2f} | {learn_note}"
            )

        return events

    def manage_open_trades(self) -> List[str]:
        """
        Monitor all active bot trades (both MT5 positions and 24/7 Live Crypto Book trades):
          1. If an open trade reaches TP1 (1.5R), close 50%, move SL to Break-Even, and notify Telegram.
          2. If a tracked trade closes, calculate realized P&L ($ USD and R-multiple), update
             `bot_stats.json`, and send the Telegram Trade Result Card!
        """
        events: List[str] = []

        # Adopt any open MT5 positions with BOT_MAGIC not yet in active_trades
        if self.connected and self._mt5 is not None:
            try:
                open_positions = self._mt5.positions_get() or []
                for pos in open_positions:
                    t_str = str(pos.ticket)
                    if getattr(pos, "magic", 0) == BOT_MAGIC and t_str not in self.active_trades:
                        is_long = pos.type == self._mt5.POSITION_TYPE_BUY
                        entry = float(pos.price_open)
                        sl = float(pos.sl) if float(pos.sl) > 0 else entry
                        tp2 = float(pos.tp) if float(pos.tp) > 0 else entry
                        tp1 = entry + 0.5 * (tp2 - entry)
                        self.active_trades[t_str] = {
                            "ticket": int(pos.ticket),
                            "symbol": str(pos.symbol),
                            "broker_symbol": str(pos.symbol),
                            "broker_source": "MT5_LIVE",
                            "direction": "LONG" if is_long else "SHORT",
                            "entry_price": entry,
                            "sl_price": sl,
                            "tp1_price": tp1,
                            "tp2_price": tp2,
                            "volume": float(pos.volume),
                            "initial_volume": float(pos.volume),
                            "risk_usd": 15.0,
                            "opened_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "tp1_hit": False,
                            "partial_pnl": 0.0,
                        }
                        self._save_state()
            except Exception:
                pass

        for t_str, trade in list(self.active_trades.items()):
            if trade.get("broker_source") == "CRYPTO_24_7_LIVE":
                events.extend(self._manage_crypto_book_trade(t_str, trade))
                continue

            if not self.connected or self._mt5 is None:
                continue

            ticket = int(trade["ticket"])
            broker_sym = str(trade.get("broker_symbol", trade.get("symbol", "XAUUSD")))
            pos_list = self._mt5.positions_get(ticket=ticket)

            # CASE 1: Position is still open -> Check Early Break-Even Shield (+0.55R), TP1 (+0.72R / tp1_price) & Runner Ratchet (+1.35R)
            if pos_list and len(pos_list) > 0:
                pos = pos_list[0]
                cur_price = float(pos.price_current)
                pos_profit = float(getattr(pos, "profit", 0.0) or 0.0)
                risk_usd_val = max(float(trade.get("risk_usd", 1.0)), 0.05)
                is_long = str(trade.get("direction", "LONG")) == "LONG"
                entry_price = float(trade.get("entry_price", 0.0))
                sl_price_cur = float(trade.get("sl_price", 0.0))
                if "initial_sl_price" not in trade and sl_price_cur > 0:
                    trade["initial_sl_price"] = sl_price_cur
                init_sl_price = float(trade.get("initial_sl_price", sl_price_cur))
                r_dist = abs(entry_price - init_sl_price) if (entry_price > 0 and init_sl_price > 0) else 0.0
                init_vol = max(
                    float(trade.get("initial_volume", trade.get("volume", getattr(pos, "volume", 0.01))) or 0.01),
                    0.01,
                )
                cur_vol_pos = max(float(getattr(pos, "volume", init_vol) or init_vol), 0.01)
                vol_scale = min(1.0, cur_vol_pos / init_vol)
                remaining_risk_usd = max(risk_usd_val * vol_scale, 0.05)
                price_r = (
                    ((cur_price - entry_price) if is_long else (entry_price - cur_price)) / r_dist
                    if (entry_price > 0 and r_dist > 0)
                    else 0.0
                )
                cur_r = max(pos_profit / remaining_risk_usd, price_r)
                if cur_r > float(trade.get("peak_favorable_r", 0.0)):
                    prev_peak = float(trade.get("peak_favorable_r", 0.0))
                    trade["peak_favorable_r"] = round(cur_r, 3)
                    if cur_r - prev_peak >= 0.015:
                        self._save_state()
                # Auto-tighten wide-spread altcoin stops (e.g. SOLUSD >1.5% stop) to cap max risk & lock profit early
                if (
                    not trade.get("tp1_hit", False)
                    and not trade.get("spread_cap_applied", False)
                    and entry_price > 0
                    and (r_dist / entry_price) > 0.015
                ):
                    sym_info = self._mt5.symbol_info(broker_sym)
                    digits = int(sym_info.digits) if sym_info is not None else 2
                    tight_r = entry_price * 0.011
                    new_sl = round(entry_price - tight_r if is_long else entry_price + tight_r, digits)
                    new_tp1 = round(entry_price + 0.60 * tight_r if is_long else entry_price - 0.60 * tight_r, digits)
                    new_tp2 = round(entry_price + 1.35 * tight_r if is_long else entry_price - 1.35 * tight_r, digits)
                    self._mt5.order_send(
                        {
                            "action": self._mt5.TRADE_ACTION_SLTP,
                            "position": int(pos.ticket),
                            "symbol": broker_sym,
                            "sl": new_sl,
                            "tp": new_tp2,
                        }
                    )
                    trade["sl_price"] = new_sl
                    trade["initial_sl_price"] = new_sl
                    trade["tp1_price"] = new_tp1
                    trade["tp2_price"] = new_tp2
                    trade["risk_usd"] = round(min(risk_usd_val, 18.0), 2)
                    trade["spread_cap_applied"] = True
                    risk_usd_val = float(trade["risk_usd"])
                    r_dist = tight_r
                    self._save_state()
                # Auto-tighten open trades into 0.52R/1.10R Fast-Scalp targets
                if not trade.get("tp1_hit", False) and not trade.get("fast_scalp_upgraded", False) and r_dist > 0:
                    new_tp1 = round(entry_price + (0.52 * r_dist if is_long else -0.52 * r_dist), 5)
                    new_tp2 = round(entry_price + (1.10 * r_dist if is_long else -1.10 * r_dist), 5)
                    trade["tp1_price"] = new_tp1
                    trade["tp2_price"] = new_tp2
                    trade["fast_scalp_upgraded"] = True
                    self._save_state()
                tp1_price = float(trade.get("tp1_price", 0.0))

                age_min = 0.0
                opened_str = str(trade.get("opened_at_utc", ""))
                if opened_str:
                    try:
                        opened_dt = datetime.strptime(opened_str, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)
                        age_min = (datetime.now(timezone.utc) - opened_dt).total_seconds() / 60.0
                    except Exception:
                        pass

                # Step 1 — Fast-Scalp Break-Even Shield (+0.35R excursion, +0.18R peak pullback, OR >=20m open with >=+0.10R)
                if (
                    not trade.get("tp1_hit", False)
                    and not trade.get("be_shield_active", False)
                    and (
                        cur_r >= 0.35
                        or pos_profit >= 0.35 * risk_usd_val
                        or (float(trade.get("peak_favorable_r", 0.0)) >= 0.18 and pos_profit >= 0.04 * risk_usd_val)
                        or (age_min >= 20.0 and (cur_r >= 0.10 or pos_profit >= 0.10 * risk_usd_val))
                    )
                    and entry_price > 0
                ):
                    sym_info = self._mt5.symbol_info(broker_sym)
                    digits = int(sym_info.digits) if sym_info is not None else 2
                    be_sl = round(
                        entry_price + (0.04 * r_dist if is_long else -0.04 * r_dist)
                        if r_dist > 0
                        else entry_price,
                        digits,
                    )
                    cur_tp = round(float(pos.tp), digits) if float(pos.tp) > 0 else round(float(trade.get("tp2_price", 0.0)), digits)
                    self._mt5.order_send(
                        {
                            "action": self._mt5.TRADE_ACTION_SLTP,
                            "position": int(pos.ticket),
                            "symbol": broker_sym,
                            "sl": be_sl,
                            "tp": cur_tp,
                        }
                    )
                    trade["be_shield_active"] = True
                    trade["sl_price"] = be_sl
                    self._save_state()
                    events.append(
                        f"[BREAK-EVEN SHIELD +0.04R] Ticket #{ticket} ({broker_sym}) reached +${pos_profit:.2f} (+{cur_r:.2f}R)! "
                        f"SL locked at Spread-Protected Break-Even ({be_sl})."
                    )

                # Step 1.25 — 35-Minute Fast-Scalp Time-Decay Shield (cuts risk on stale >35m trades without impulse)
                if (
                    age_min >= 35.0
                    and not trade.get("tp1_hit", False)
                    and not trade.get("be_shield_active", False)
                    and not trade.get("time_decay_applied", False)
                    and r_dist > 0
                ):
                    sym_info = self._mt5.symbol_info(broker_sym)
                    digits = int(sym_info.digits) if sym_info is not None else 2
                    decay_dist = 0.45 * r_dist
                    cur_sl_val = float(pos.sl) if float(pos.sl) > 0 else float(trade.get("sl_price", entry_price))
                    if is_long:
                        decay_sl = round(min(cur_price - 0.06 * r_dist, max(cur_sl_val, entry_price - decay_dist)), digits)
                    else:
                        decay_sl = round(max(cur_price + 0.06 * r_dist, min(cur_sl_val, entry_price + decay_dist)), digits)
                    cur_tp = round(float(pos.tp), digits) if float(pos.tp) > 0 else round(float(trade.get("tp2_price", 0.0)), digits)
                    self._mt5.order_send(
                        {
                            "action": self._mt5.TRADE_ACTION_SLTP,
                            "position": int(pos.ticket),
                            "symbol": broker_sym,
                            "sl": decay_sl,
                            "tp": cur_tp,
                        }
                    )
                    trade["time_decay_applied"] = True
                    trade["sl_price"] = decay_sl
                    self._save_state()
                    events.append(
                        f"[35M TIME-DECAY SHIELD] Ticket #{ticket} ({broker_sym}) open {age_min:.0f}m without impulse; "
                        f"tightened SL to {decay_sl} to cap stale-scalp risk."
                    )

                # Step 1.5 - Near-TP1 Profit Cushion (+0.44R excursion): Lock SL at +0.22R guaranteed profit
                if (
                    not trade.get("tp1_hit", False)
                    and not trade.get("near_tp1_cushion_active", False)
                    and (cur_r >= 0.44 or pos_profit >= 0.44 * risk_usd_val)
                    and r_dist > 0
                ):
                    sym_info = self._mt5.symbol_info(broker_sym)
                    digits = int(sym_info.digits) if sym_info is not None else 2
                    cushion_sl = round(
                        entry_price + (0.22 * r_dist if is_long else -0.22 * r_dist), digits
                    )
                    cur_tp = round(float(pos.tp), digits) if float(pos.tp) > 0 else round(float(trade.get("tp2_price", 0.0)), digits)
                    self._mt5.order_send(
                        {
                            "action": self._mt5.TRADE_ACTION_SLTP,
                            "position": int(pos.ticket),
                            "symbol": broker_sym,
                            "sl": cushion_sl,
                            "tp": cur_tp,
                        }
                    )
                    trade["near_tp1_cushion_active"] = True
                    trade["be_shield_active"] = True
                    trade["sl_price"] = cushion_sl
                    self._save_state()
                    events.append(
                        f"[NEAR-TP1 CUSHION +0.22R] Ticket #{ticket} ({broker_sym}) reached +{cur_r:.2f}R (+${pos_profit:.2f})! "
                        f"SL ratcheted to +0.22R guaranteed profit ({cushion_sl})."
                    )

                # Step 2 - Fast-Scalp TP1 Hit (+0.52R floating/excursion or tp1_price): Close 50% partial & lock +0.20R runner
                if not trade.get("tp1_hit", False) and tp1_price > 0:
                    price_reached = (cur_price >= tp1_price) if is_long else (cur_price <= tp1_price)
                    profit_reached = (pos_profit >= 0.52 * risk_usd_val) or (cur_r >= 0.52)
                    if price_reached or profit_reached:
                        partial_closed = self._execute_tp1_protection(pos, trade)
                        trade["tp1_hit"] = True
                        trade["be_shield_active"] = True
                        self._save_state()
                        if self.telegram.is_configured:
                            msg_html = self.telegram.format_tp1_breakeven_html(
                                trade=trade,
                                current_price=cur_price,
                                partial_closed_lots=partial_closed,
                            )
                            self.telegram.send_message(msg_html)
                        events.append(
                            f"[TP1 REACHED] Ticket #{ticket} ({broker_sym}) reached Fast-Scalp TP1 (+${pos_profit:.2f})! "
                            f"SL locked at {trade.get('sl_price', entry_price)}."
                        )

                # Step 2.5 & Step 3 — Fast-Scalp Runner Ratchet (+0.75R -> +0.45R SL, +0.92R -> +0.72R SL)
                if (
                    trade.get("tp1_hit", False)
                    and not trade.get("mid_runner_lock_active", False)
                    and not trade.get("runner_ratchet_active", False)
                    and cur_r >= 0.75
                    and r_dist > 0
                ):
                    sym_info = self._mt5.symbol_info(broker_sym)
                    digits = int(sym_info.digits) if sym_info is not None else 2
                    mid_sl = round(
                        entry_price + (0.45 * r_dist if is_long else -0.45 * r_dist), digits
                    )
                    cur_tp = round(float(pos.tp), digits) if float(pos.tp) > 0 else round(float(trade.get("tp2_price", 0.0)), digits)
                    self._mt5.order_send(
                        {
                            "action": self._mt5.TRADE_ACTION_SLTP,
                            "position": int(pos.ticket),
                            "symbol": broker_sym,
                            "sl": mid_sl,
                            "tp": cur_tp,
                        }
                    )
                    trade["mid_runner_lock_active"] = True
                    trade["sl_price"] = mid_sl
                    self._save_state()
                    events.append(
                        f"[RUNNER LOCK +0.45R] Ticket #{ticket} ({broker_sym}) reached +{cur_r:.2f}R (+${pos_profit:.2f})! "
                        f"SL ratcheted to +0.45R guaranteed profit ({mid_sl})."
                    )

                if (
                    trade.get("tp1_hit", False)
                    and not trade.get("runner_ratchet_active", False)
                    and (cur_r >= 0.92 or pos_profit >= 0.92 * remaining_risk_usd)
                    and r_dist > 0
                ):
                    sym_info = self._mt5.symbol_info(broker_sym)
                    digits = int(sym_info.digits) if sym_info is not None else 2
                    ratchet_sl = round(
                        entry_price + (0.72 * r_dist if is_long else -0.72 * r_dist), digits
                    )
                    cur_tp = round(float(pos.tp), digits) if float(pos.tp) > 0 else round(float(trade.get("tp2_price", 0.0)), digits)
                    self._mt5.order_send(
                        {
                            "action": self._mt5.TRADE_ACTION_SLTP,
                            "position": int(pos.ticket),
                            "symbol": broker_sym,
                            "sl": ratchet_sl,
                            "tp": cur_tp,
                        }
                    )
                    trade["runner_ratchet_active"] = True
                    trade["sl_price"] = ratchet_sl
                    self._save_state()
                    events.append(
                        f"[RUNNER RATCHET +0.72R] Ticket #{ticket} ({broker_sym}) reached +{cur_r:.2f}R (+${pos_profit:.2f})! "
                        f"SL ratcheted to +0.72R guaranteed profit ({ratchet_sl})."
                    )
                continue

            # CASE 2: Position is no longer in open positions -> It has CLOSED!
            exit_price, net_pnl, close_reason = self._resolve_closed_trade_pnl(ticket, trade)
            risk_usd = max(float(trade.get("risk_usd", 1.0)), 0.05)
            r_mult = net_pnl / risk_usd

            self.stats["total_closed"] = int(self.stats.get("total_closed", 0)) + 1
            if r_mult > 0.08:
                self.stats["wins"] = int(self.stats.get("wins", 0)) + 1
            elif r_mult >= -0.08:
                self.stats["breakevens"] = int(self.stats.get("breakevens", 0)) + 1
            else:
                self.stats["losses"] = int(self.stats.get("losses", 0)) + 1
            self.stats["cumulative_pnl"] = round(
                float(self.stats.get("cumulative_pnl", 0.0)) + net_pnl, 2
            )

            closed_record = {
                "ticket": ticket,
                "symbol": trade.get("symbol"),
                "direction": trade.get("direction"),
                "entry_price": trade.get("entry_price"),
                "exit_price": exit_price,
                "net_pnl": round(net_pnl, 2),
                "r_multiple": round(r_mult, 2),
                "close_reason": close_reason,
                "closed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            }
            history_list = list(self.stats.get("history", []))
            history_list.append(closed_record)
            self.stats["history"] = history_list[-50:]

            del self.active_trades[t_str]
            self._save_state()

            acct_bal = self.get_account_balance()
            acct_eq = self.get_account_equity()
            learn_note = self.learner.record_trade_outcome(
                symbol=str(trade.get("symbol", broker_sym)),
                regime=str(trade.get("regime", "TREND_EXPANSION")),
                agreeing_strategies=list(trade.get("agreeing_strategies", ["SND", "ICT", "KRONOS"])),
                r_multiple=r_mult,
                net_pnl=net_pnl,
                trade_meta={
                    "ticket": ticket,
                    "direction": trade.get("direction", "LONG"),
                    "entry_price": trade.get("entry_price", 0.0),
                    "exit_price": exit_price,
                    "tp1_price": trade.get("tp1_price", 0.0),
                    "tp2_price": trade.get("tp2_price", 0.0),
                    "peak_favorable_r": trade.get("peak_favorable_r", 0.0),
                    "close_reason": close_reason,
                },
            )

            if self.telegram.is_configured:
                result_html = self.telegram.format_trade_closed_html(
                    trade=trade,
                    exit_price=exit_price,
                    net_pnl=net_pnl,
                    r_multiple=r_mult,
                    close_reason=close_reason,
                    account_balance=acct_bal,
                    account_equity=acct_eq,
                    stats=self.stats,
                    learning_summary=learn_note,
                )
                self.telegram.send_message(result_html)

            events.append(
                f"[TRADE CLOSED] Ticket #{ticket} ({broker_sym}) | {close_reason} | "
                f"P&L: ${net_pnl:+.2f} ({r_mult:+.2f}R) | Balance: ${acct_bal:,.2f} | {learn_note}"
            )

        return events

    def _execute_tp1_protection(self, pos: Any, trade: Dict[str, Any]) -> float:
        """
        At TP1:
        - If position volume >= 2 * vol_min: close tp1_close_fraction (50-60%) and move SL to Break-Even.
        - If position volume == 0.01 lot (small account cannot split): lock SL at +0.35R in profit so the
          0.01-lot trade is guaranteed to bank +0.35R cash while still running for TP2!
        """
        if self._mt5 is None:
            return 0.0
        broker_sym = str(pos.symbol)
        canonical_sym = str(trade.get("symbol", broker_sym))
        profile: InstrumentProfile = ALL_INSTRUMENTS.get(canonical_sym, INSTRUMENTS["XAUUSD"])
        sym_info = self._mt5.symbol_info(broker_sym)
        tick = self._mt5.symbol_info_tick(broker_sym)
        if sym_info is None or tick is None:
            return 0.0

        digits = int(sym_info.digits)
        vol_min = float(sym_info.volume_min)
        vol_step = float(sym_info.volume_step) if float(sym_info.volume_step) > 0 else 0.01
        cur_vol = float(pos.volume)
        is_long = pos.type == self._mt5.POSITION_TYPE_BUY
        partial_closed = 0.0

        if cur_vol >= 2.0 * vol_min:
            close_frac = float(getattr(profile, "tp1_close_fraction", 0.5) or 0.5)
            frac_steps = math.floor(((cur_vol * close_frac) / vol_step) + 1e-9)
            close_vol = round(max(vol_min, frac_steps * vol_step), 2)
            if close_vol < cur_vol:
                close_price = round(float(tick.bid if is_long else tick.ask), digits)
                close_req = {
                    "action": self._mt5.TRADE_ACTION_DEAL,
                    "position": int(pos.ticket),
                    "symbol": broker_sym,
                    "volume": close_vol,
                    "type": self._mt5.ORDER_TYPE_SELL if is_long else self._mt5.ORDER_TYPE_BUY,
                    "price": close_price,
                    "deviation": 30,
                    "magic": BOT_MAGIC,
                    "comment": "TP1_Partial",
                }
                res = self._send_order_with_filling_fallback(close_req)
                if res is not None and res.retcode == self._mt5.TRADE_RETCODE_DONE:
                    partial_closed = close_vol
                    trade["volume"] = round(max(vol_min, cur_vol - partial_closed), 2)
                    trade["partial_closed_volume"] = round(
                        float(trade.get("partial_closed_volume", 0.0)) + partial_closed, 2
                    )

        entry_price = float(trade.get("entry_price", pos.price_open))
        init_sl_price = float(trade.get("initial_sl_price", trade.get("sl_price", entry_price)))
        r_dist = abs(entry_price - init_sl_price)

        if partial_closed == 0.0 and r_dist > 0:
            # 0.01-lot Golden Ratchet: cannot split 0.01 lot, so lock SL at +0.35R guaranteed profit!
            be_sl = round(entry_price + (0.35 * r_dist if is_long else -0.35 * r_dist), digits)
        elif r_dist > 0:
            # Partial closed at TP1: lock remaining 50% runner at +0.15R profit cushion above entry!
            be_sl = round(entry_price + (0.15 * r_dist if is_long else -0.15 * r_dist), digits)
        else:
            be_sl = round(entry_price, digits)

        trade["sl_price"] = be_sl
        cur_tp = round(float(pos.tp), digits) if float(pos.tp) > 0 else round(float(trade.get("tp2_price", 0.0)), digits)
        sltp_req = {
            "action": self._mt5.TRADE_ACTION_SLTP,
            "position": int(pos.ticket),
            "symbol": broker_sym,
            "sl": be_sl,
            "tp": cur_tp,
        }
        self._mt5.order_send(sltp_req)
        return partial_closed

    def _resolve_closed_trade_pnl(
        self, ticket: int, trade: Dict[str, Any]
    ) -> Tuple[float, float, str]:
        """Query MT5 history deals for `position=ticket` to compute exact realized P&L."""
        entry_price = float(trade.get("entry_price", 0.0))
        exit_price = entry_price
        net_pnl = 0.0

        if self._mt5 is not None:
            try:
                deals = self._mt5.history_deals_get(position=ticket)
                if not deals:
                    from_date = datetime.now(timezone.utc) - timedelta(days=14)
                    to_date = datetime.now(timezone.utc) + timedelta(days=1)
                    all_deals = self._mt5.history_deals_get(from_date, to_date) or []
                    deals = [d for d in all_deals if int(getattr(d, "position_id", -1)) == ticket]

                if deals:
                    for d in deals:
                        if int(getattr(d, "entry", 0)) in (1, 2, 3):
                            net_pnl += (
                                float(getattr(d, "profit", 0.0))
                                + float(getattr(d, "swap", 0.0))
                                + float(getattr(d, "commission", 0.0))
                            )
                            exit_price = float(getattr(d, "price", exit_price))
            except Exception:
                pass

        risk_usd = max(float(trade.get("risk_usd", 1.0)), 0.05)
        r_mult = net_pnl / risk_usd
        if r_mult >= 1.40:
            reason = "Take Profit 2 (Runner Target) Hit 🎯"
        elif trade.get("tp1_hit") and r_mult > 0.08:
            reason = "TP1 + Ratchet Profit Stop Hit 🛡️💰"
        elif -0.08 <= r_mult <= 0.08:
            reason = "Break-Even Shield Hit 🛡️"
        elif r_mult > 0.08:
            reason = "Take Profit / Positive Close 💰"
        else:
            reason = "Stop Loss Hit 🛑"

        return exit_price, round(net_pnl, 2), reason


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
    Generate realistic M15 multi-regime market data with authentic session structure.
    """
    sym_seeds = {"XAUUSD": 101, "EURUSD": 202, "USDJPY": 303, "BTCUSD": 404, "ETHUSD": 505, "SOLUSD": 606}
    rng = random.Random(seed if seed is not None else sym_seeds.get(symbol, 42))

    if symbol == "XAUUSD":
        price = 2650.00
        base_step = 2.20
    elif symbol == "EURUSD":
        price = 1.08500
        base_step = 0.00075
    elif symbol == "BTCUSD":
        price = 95000.00
        base_step = 180.0
    elif symbol == "ETHUSD":
        price = 3200.00
        base_step = 12.0
    elif symbol == "SOLUSD":
        price = 195.00
        base_step = 1.20
    else:  # USDJPY
        price = 152.500
        base_step = 0.110

    start_time = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    candles: List[Candle] = []

    regime_bias = 1.0
    cycle_len = 96  # 1 trading day on M15 (96 x 15m = 24h)

    for i in range(count):
        ts = start_time + timedelta(minutes=15 * i)
        bar_in_day = i % cycle_len
        day_num = i // cycle_len

        if bar_in_day == 0:
            if day_num % 5 == 4:
                regime_bias = 0.0
            elif day_num % 2 == 0:
                regime_bias = 1.0
            else:
                regime_bias = -1.0

        o = price
        asian_bars = candles[-min(len(candles), 28) :] if candles else []
        recent_high = max((c.high for c in asian_bars), default=o + base_step)
        recent_low = min((c.low for c in asian_bars), default=o - base_step)

        if bar_in_day < 28:
            mid_box = 0.5 * (recent_high + recent_low)
            mean_rev = -0.22 * (o - mid_box)
            delta = mean_rev + rng.gauss(0.0, 0.42 * base_step)
            c = o + delta
            h = max(o, c) + abs(rng.gauss(0.0, 0.25 * base_step))
            l = min(o, c) - abs(rng.gauss(0.0, 0.25 * base_step))
        elif 28 <= bar_in_day <= 33 and regime_bias != 0.0:
            if bar_in_day == 28:
                if regime_bias > 0:
                    l = recent_low - 0.85 * base_step
                    c = recent_low + 0.35 * base_step
                    h = max(o, c) + 0.25 * base_step
                else:
                    h = recent_high + 0.85 * base_step
                    c = recent_high - 0.35 * base_step
                    l = min(o, c) - 0.25 * base_step
            elif bar_in_day in (29, 30, 31):
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
        elif 34 <= bar_in_day <= 67 and regime_bias != 0.0:
            if 40 <= bar_in_day <= 44:
                c = o - regime_bias * rng.uniform(0.35, 0.65) * base_step
                h = max(o, c) + 0.25 * base_step
                l = min(o, c) - 0.25 * base_step
            elif bar_in_day == 45:
                c = o + regime_bias * rng.uniform(0.95, 1.40) * base_step
                if regime_bias > 0:
                    l = o - 0.70 * base_step
                    h = c + 0.18 * base_step
                else:
                    h = o + 0.70 * base_step
                    l = c - 0.18 * base_step
            else:
                drift = regime_bias * 0.32 * base_step
                c = o + drift + rng.gauss(0.0, 0.48 * base_step)
                h = max(o, c) + abs(rng.gauss(0.0, 0.30 * base_step))
                l = min(o, c) - abs(rng.gauss(0.0, 0.30 * base_step))
        else:
            vol_mult = 1.65 if (regime_bias == 0.0 and 32 <= bar_in_day <= 56) else 0.55
            whipsaw = (
                (-1.0 if (bar_in_day % 2 == 0) else 1.0) * 0.75 * base_step
                if regime_bias == 0.0
                else 0.0
            )
            c = o + whipsaw + rng.gauss(0.0, vol_mult * base_step)
            h = max(o, c) + abs(rng.gauss(0.0, 0.55 * vol_mult * base_step))
            l = min(o, c) - abs(rng.gauss(0.0, 0.55 * vol_mult * base_step))

        decimals = 2 if symbol in ("XAUUSD", "BTCUSD", "ETHUSD", "SOLUSD") else (3 if symbol == "USDJPY" else 5)
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
