"""
MetaTrader 5 Autonomous Auto-Trader Bridge, Position Lifecycle Manager & Data Provider.

Modes of Operation:
  1. Autonomous Auto-Trader (`execution_mode="auto"`):
     - Places live/demo market orders on MetaTrader 5 with automatic broker filling-mode
       detection (`FOK` / `IOC` / `RETURN`), live Ask/Bid price normalization, spread guard,
       and duplicate-position protection.
     - Actively monitors open positions every cycle:
       * When `TP1 (1.5R)` is reached: locks in 50% partial profit (if lot >= 0.02), moves
         Stop Loss to Break-Even (`entry_price`), and sends a Telegram notification.
       * When the trade closes (`TP2`, `SL`, `Break-Even`, or manual close): queries MT5 deal
         history for exact realized P&L ($ USD and R-multiple), updates cumulative bot stats,
         and sends a Telegram Trade Result Card!
  2. Copilot Signal Mode (`execution_mode="signal_only"`): Sends Signal Cards without placing orders.
  3. Multi-Regime Market Generator: Offline simulation for backtesting and unit tests.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from src.config import INSTRUMENTS, InstrumentProfile
from src.models import Candle, Direction, SetupSignal
from src.telegram_notifier import TelegramNotifier

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
ACTIVE_TRADES_PATH = LOGS_DIR / "active_trades.json"
BOT_STATS_PATH = LOGS_DIR / "bot_stats.json"
BOT_MAGIC = 202605


class MT5Connector:
    """Bridge to MetaTrader 5 terminal with full Auto-Trader execution & lifecycle tracking."""

    def __init__(self, execution_mode: str = "auto") -> None:
        self.execution_mode = execution_mode
        self._mt5 = None
        self.connected = False
        self.last_error: str | None = None
        self.terminal_path: str | None = None
        self.telegram = TelegramNotifier()
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

    def get_account_balance(self, default_balance: float = 10000.0) -> float:
        """Return live MT5 account balance if connected, else default balance."""
        if self.connected and self._mt5 is not None:
            try:
                info = self._mt5.account_info()
                if info is not None and float(info.balance) > 0:
                    return float(info.balance)
            except Exception:
                pass
        return default_balance

    def resolve_broker_symbol(self, base_symbol: str) -> str:
        """
        Automatically match broker symbol suffixes (e.g. `XAUUSD`, `XAUUSDm`, `XAUUSD.raw`, `GOLD`).
        """
        if not self.connected or self._mt5 is None:
            return base_symbol
        try:
            if self._mt5.symbol_select(base_symbol, True):
                return base_symbol
            all_syms = self._mt5.symbols_get() or []
            for s in all_syms:
                if s.name.upper().startswith(base_symbol.upper()):
                    self._mt5.symbol_select(s.name, True)
                    return s.name
            if base_symbol.upper() == "XAUUSD":
                for s in all_syms:
                    if s.name.upper().startswith("GOLD"):
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

    def handle_signal(self, signal: SetupSignal) -> str:
        """
        Process a high-confluence setup according to `execution_mode`:
          - `auto`: Places live/demo order in MetaTrader 5, tracks ticket for TP1/BE/Close,
                    and sends an instant Telegram execution alert.
          - `signal_only`: Displays the Copilot Signal Card and sends to Telegram without ordering.
          - `confirm`: Prompts `[y/N]` before placing order.
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

        if not self.connected or self._mt5 is None:
            return (
                f"[{self.execution_mode.upper()} MODE] MT5 terminal not connected; "
                "order simulated locally."
            )

        broker_sym = self.resolve_broker_symbol(signal.symbol)
        profile: InstrumentProfile = INSTRUMENTS.get(signal.symbol, INSTRUMENTS["XAUUSD"])

        # 1. Check duplicate open position on this symbol
        existing_pos = self._mt5.positions_get(symbol=broker_sym)
        if existing_pos and len(existing_pos) > 0:
            return (
                f"[AUTO-TRADER SKIP] {broker_sym} already has an open position "
                f"(Ticket #{existing_pos[0].ticket}). Avoiding duplicate exposure."
            )

        # 2. Check max concurrent open positions (max 3 across all symbols)
        all_pos = self._mt5.positions_get() or []
        if len(all_pos) >= 3:
            return f"[AUTO-TRADER SKIP] Max concurrent open positions ({len(all_pos)}/3) reached."

        # 3. Verify live tick, spread, and broker symbol digits/volume constraints
        sym_info = self._mt5.symbol_info(broker_sym)
        tick = self._mt5.symbol_info_tick(broker_sym)
        if sym_info is None or tick is None:
            return f"[AUTO-TRADER ERROR] Could not read live tick/symbol_info for {broker_sym}."

        live_spread = float(tick.ask - tick.bid)
        if live_spread > profile.max_spread_price * 1.5:
            return (
                f"[AUTO-TRADER SKIP] Live spread on {broker_sym} ({live_spread:.5f}) "
                f"exceeds safety limit ({profile.max_spread_price * 1.5:.5f})."
            )

        digits = int(sym_info.digits)
        vol_min = float(sym_info.volume_min)
        vol_max = float(sym_info.volume_max)
        vol_step = float(sym_info.volume_step) if float(sym_info.volume_step) > 0 else 0.01
        steps = math.floor((signal.recommended_lot / vol_step) + 1e-9)
        volume = round(max(vol_min, min(vol_max, steps * vol_step)), 2)

        is_long = signal.direction == Direction.LONG
        fill_price = round(float(tick.ask if is_long else tick.bid), digits)
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        tp1_dist = abs(signal.tp1_price - signal.entry_price)
        tp2_dist = abs(signal.tp2_price - signal.entry_price)

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

        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": broker_sym,
            "volume": volume,
            "type": order_type,
            "price": fill_price,
            "sl": sl_price,
            "tp": tp2_price,  # Hard TP at 3.0R; Bot manages TP1 (1.5R) partial + Break-Even
            "deviation": 30,
            "magic": BOT_MAGIC,
            "comment": f"Auto_{signal.symbol}_{signal.active_combination[:10]}",
            "type_time": self._mt5.ORDER_TIME_GTC,
        }

        result = self._send_order_with_filling_fallback(request)
        if result is None or result.retcode != self._mt5.TRADE_RETCODE_DONE:
            retcode = getattr(result, "retcode", "None")
            comment = getattr(result, "comment", "Unknown MT5 error")
            return f"[AUTO-TRADER ORDER FAILED] {broker_sym} retcode={retcode} ({comment})"

        # Find the opened position ticket
        pos_list = self._mt5.positions_get(symbol=broker_sym) or []
        ticket = int(pos_list[-1].ticket) if pos_list else int(getattr(result, "order", 0))
        actual_fill = float(getattr(result, "price", fill_price) or fill_price)
        acct_equity = self.get_account_equity()

        self.active_trades[str(ticket)] = {
            "ticket": ticket,
            "symbol": signal.symbol,
            "broker_symbol": broker_sym,
            "direction": signal.direction.value,
            "entry_price": actual_fill,
            "sl_price": sl_price,
            "tp1_price": tp1_price,
            "tp2_price": tp2_price,
            "volume": volume,
            "initial_volume": volume,
            "risk_usd": signal.risk_usd,
            "confluence_score": signal.confluence_score,
            "regime": signal.regime.value,
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
            f"[AUTO-TRADE EXECUTED] Ticket #{ticket} | {signal.direction.value} {volume:.2f} lots "
            f"{broker_sym} @ {actual_fill} (SL={sl_price}, TP1={tp1_price}, TP2={tp2_price}){tg_note}"
        )

    def manage_open_trades(self) -> List[str]:
        """
        Monitor all active bot trades on every cycle:
          1. If an open trade reaches TP1 (1.5R), close 50% (if lot >= 2*min_lot), move SL to
             Break-Even (`entry_price`), and send a Telegram update.
          2. If a tracked trade has closed in MT5 (TP2, SL, BE, or manual close), calculate
             realized P&L ($ USD and R-multiple), update `bot_stats.json`, and send the
             Telegram Trade Result Card!
        """
        events: List[str] = []
        if not self.connected or self._mt5 is None:
            return events

        # Also adopt any open MT5 positions with BOT_MAGIC not yet in active_trades
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
            ticket = int(trade["ticket"])
            broker_sym = str(trade.get("broker_symbol", trade.get("symbol", "XAUUSD")))
            pos_list = self._mt5.positions_get(ticket=ticket)

            # CASE 1: Position is still open -> Check if TP1 (1.5R) was reached
            if pos_list and len(pos_list) > 0:
                pos = pos_list[0]
                cur_price = float(pos.price_current)
                is_long = str(trade.get("direction", "LONG")) == "LONG"
                tp1_price = float(trade.get("tp1_price", 0.0))
                entry_price = float(trade.get("entry_price", 0.0))

                if not trade.get("tp1_hit", False) and tp1_price > 0:
                    reached_tp1 = (cur_price >= tp1_price) if is_long else (cur_price <= tp1_price)
                    if reached_tp1:
                        partial_closed = self._execute_tp1_protection(pos, trade)
                        trade["tp1_hit"] = True
                        self._save_state()
                        if self.telegram.is_configured:
                            msg_html = self.telegram.format_tp1_breakeven_html(
                                trade=trade,
                                current_price=cur_price,
                                partial_closed_lots=partial_closed,
                            )
                            self.telegram.send_message(msg_html)
                        events.append(
                            f"[TP1 REACHED] Ticket #{ticket} ({broker_sym}) hit 1.5R! "
                            f"SL moved to Break-Even ({entry_price})."
                        )
                continue

            # CASE 2: Position is no longer in open positions -> It has CLOSED!
            exit_price, net_pnl, close_reason = self._resolve_closed_trade_pnl(ticket, trade)
            risk_usd = max(float(trade.get("risk_usd", 15.0)), 1.0)
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
                )
                self.telegram.send_message(result_html)

            events.append(
                f"[TRADE CLOSED] Ticket #{ticket} ({broker_sym}) | {close_reason} | "
                f"P&L: ${net_pnl:+.2f} ({r_mult:+.2f}R) | Balance: ${acct_bal:,.2f}"
            )

        return events

    def _execute_tp1_protection(self, pos: Any, trade: Dict[str, Any]) -> float:
        """Close 50% at TP1 (if volume allows) and move SL to Break-Even (`entry_price`)."""
        if self._mt5 is None:
            return 0.0
        broker_sym = str(pos.symbol)
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

        # 1. Partial 50% close if current volume >= 2 * vol_min
        if cur_vol >= 2.0 * vol_min:
            half_steps = math.floor(((cur_vol * 0.5) / vol_step) + 1e-9)
            close_vol = round(max(vol_min, half_steps * vol_step), 2)
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
                    "comment": "TP1_50pct",
                }
                res = self._send_order_with_filling_fallback(close_req)
                if res is not None and res.retcode == self._mt5.TRADE_RETCODE_DONE:
                    partial_closed = close_vol

        # 2. Move Stop Loss to Break-Even (Entry Price)
        be_sl = round(float(trade.get("entry_price", pos.price_open)), digits)
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
        tp2_price = float(trade.get("tp2_price", 0.0))
        sl_price = float(trade.get("sl_price", 0.0))
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
                        # DEAL_ENTRY_OUT (1) or DEAL_ENTRY_INOUT (2) or DEAL_ENTRY_OUT_BY (3)
                        if int(getattr(d, "entry", 0)) in (1, 2, 3):
                            net_pnl += (
                                float(getattr(d, "profit", 0.0))
                                + float(getattr(d, "swap", 0.0))
                                + float(getattr(d, "commission", 0.0))
                            )
                            exit_price = float(getattr(d, "price", exit_price))
            except Exception:
                pass

        risk_usd = max(float(trade.get("risk_usd", 15.0)), 1.0)
        r_mult = net_pnl / risk_usd
        if r_mult >= 2.0:
            reason = "Take Profit 2 (3.0R Target) Hit 🎯"
        elif trade.get("tp1_hit") and net_pnl > 0.50:
            reason = "TP1 (1.5R) Locked + Break-Even Stop Hit 🛡️"
        elif -0.50 <= net_pnl <= 0.50:
            reason = "Break-Even Stop Hit 🛡️"
        elif net_pnl > 0.50:
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
