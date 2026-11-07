"""
Telegram Bot Alert Notifier for Auto-Trader Execution & Trade Result Cards.

Uses Python's built-in `urllib.request` (zero third-party dependencies required) to:
  1. Send instant notifications when the Auto-Trader opens a live/demo MT5 position.
  2. Notify when TP1 (1.5R) is reached and Stop Loss is moved to Break-Even.
  3. Send a detailed Trade Result Card (Net P&L $, R-multiple, Exit Price, Updated Balance,
     and Session Win/Loss record) the moment any trade closes in MetaTrader 5.
"""

from __future__ import annotations

import html
import json
import os
import re
from typing import Any, Dict, Optional, Tuple
import urllib.error
import urllib.parse
import urllib.request

from src.env_loader import load_dotenv
from src.models import Direction, SetupSignal


def sanitize_token(raw: str) -> str:
    """Remove invisible control chars (like Ctrl+V \\x16 in Windows console) and whitespace."""
    cleaned = re.sub(r"[^0-9A-Za-z:_-]", "", raw.strip())
    return cleaned


class TelegramNotifier:
    """Sends formatted Auto-Trader execution & result cards to a Telegram chat or channel."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        load_dotenv()
        self.bot_token = sanitize_token(bot_token or os.getenv("TELEGRAM_BOT_TOKEN", ""))
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.timeout_sec = timeout_sec

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @staticmethod
    def _price_fmt(symbol: str) -> str:
        decimals = 2 if symbol.upper().startswith("XAU") else (3 if "JPY" in symbol.upper() else 5)
        return f"{{:.{decimals}f}}"

    @classmethod
    def format_telegram_html(cls, signal: SetupSignal, header_prefix: str = "") -> str:
        """Render a mobile-optimized HTML message for Telegram."""
        is_long = signal.direction == Direction.LONG
        header_icon = "🟢" if is_long else "🔴"
        dir_label = "BUY (LONG)" if is_long else "SELL (SHORT)"
        fmt = cls._price_fmt(signal.symbol)
        title_prefix = f"{html.escape(header_prefix)} | " if header_prefix else ""

        lines = [
            f"{header_icon} <b>{title_prefix}{html.escape(signal.symbol)} ({html.escape(signal.timeframe)}) — {dir_label}</b>",
            f"⏱ <code>{signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</code>",
            f"🧭 Regime: <b>{html.escape(signal.regime.value)}</b>",
            f"🎯 Confluence Score: <b>{signal.confluence_score:.1f} / 100</b>",
            "",
            "<b>📌 Execution Levels (MT5):</b>",
            f"• Entry: <code>{fmt.format(signal.entry_price)}</code>",
            f"• Stop Loss (SL): <code>{fmt.format(signal.stop_loss)}</code>",
            f"• TP1 ({signal.rr_tp1:.1f}R, 50% + BE): <code>{fmt.format(signal.tp1_price)}</code>",
            f"• TP2 ({signal.rr_tp2:.1f}R, Runner): <code>{fmt.format(signal.tp2_price)}</code>",
            f"• Lot Size: <b>{signal.recommended_lot:.2f} lots</b> (Risk ${signal.risk_usd:.2f} / {signal.risk_pct:.2f}%)",
            "",
            "<b>🧩 Strategy Votes:</b>",
        ]

        for v in signal.votes:
            if v.direction == signal.direction:
                badge = "✅"
            elif v.direction == Direction.NEUTRAL:
                badge = "⚪"
            else:
                badge = "❌"
            lines.append(
                f"{badge} <b>{html.escape(v.strategy_name)}</b> ({v.confidence * 100:.0f}%): "
                f"{html.escape(v.reason)}"
            )

        if signal.kronos is not None:
            lines.extend(
                [
                    "",
                    f"🧠 <b>Kronos AI ({html.escape(signal.kronos.model_name)}):</b>",
                    f"<code>{html.escape(signal.kronos.summary)}</code>",
                ]
            )

        return "\n".join(lines)

    @classmethod
    def format_trade_opened_html(
        cls,
        signal: SetupSignal,
        ticket: int,
        fill_price: float,
        sl_price: float,
        tp1_price: float,
        tp2_price: float,
        volume: float,
        account_equity: float,
    ) -> str:
        """Render Telegram HTML notification when the Auto-Trader opens a live MT5 position."""
        is_long = signal.direction == Direction.LONG
        header_icon = "🚀🟢" if is_long else "🚀🔴"
        dir_label = "BUY (LONG)" if is_long else "SELL (SHORT)"
        fmt = cls._price_fmt(signal.symbol)

        lines = [
            f"{header_icon} <b>AUTO-TRADE EXECUTED | {html.escape(signal.symbol)} ({html.escape(signal.timeframe)})</b>",
            f"🎫 Ticket: <code>#{ticket}</code> | Direction: <b>{dir_label}</b>",
            f"🧭 Regime: <b>{html.escape(signal.regime.value)}</b> | Score: <b>{signal.confluence_score:.1f}/100</b>",
            "",
            "<b>⚡ Live MT5 Fill Details:</b>",
            f"• Fill Price: <code>{fmt.format(fill_price)}</code>",
            f"• Stop Loss (SL): <code>{fmt.format(sl_price)}</code>",
            f"• TP1 ({signal.rr_tp1:.1f}R → 50% Close + BE): <code>{fmt.format(tp1_price)}</code>",
            f"• TP2 ({signal.rr_tp2:.1f}R → Final Target): <code>{fmt.format(tp2_price)}</code>",
            f"• Volume: <b>{volume:.2f} lots</b> (Risk: <b>${signal.risk_usd:.2f}</b> / {signal.risk_pct:.2f}%)",
            f"• Account Equity: <code>${account_equity:,.2f}</code>",
            "",
            "<b>🧩 Triggering Confluence:</b>",
        ]
        for v in signal.votes:
            if v.direction == signal.direction:
                lines.append(
                    f"✅ <b>{html.escape(v.strategy_name)}</b> ({v.confidence * 100:.0f}%): {html.escape(v.reason)}"
                )
        if signal.kronos is not None:
            lines.extend(
                [
                    "",
                    f"🧠 <b>Kronos GPU Forecast:</b>",
                    f"<code>{html.escape(signal.kronos.summary)}</code>",
                ]
            )
        return "\n".join(lines)

    @classmethod
    def format_tp1_breakeven_html(
        cls,
        trade: Dict[str, Any],
        current_price: float,
        partial_closed_lots: float,
    ) -> str:
        """Render Telegram HTML notification when TP1 (1.5R) is reached and SL moves to Break-Even."""
        symbol = str(trade.get("symbol", "XAUUSD"))
        fmt = cls._price_fmt(symbol)
        ticket = trade.get("ticket")
        direction = trade.get("direction", "LONG")
        entry_price = float(trade.get("entry_price", 0.0))
        tp2_price = float(trade.get("tp2_price", 0.0))

        partial_line = (
            f"• Partial Profit Locked: <b>Closed {partial_closed_lots:.2f} lots at +1.5R</b>\n"
            if partial_closed_lots > 0
            else ""
        )
        return (
            f"🎯🛡️ <b>TP1 (1.5R) HIT — RISK-FREE RUNNER | {html.escape(symbol)}</b>\n"
            f"🎫 Ticket: <code>#{ticket}</code> ({html.escape(direction)})\n"
            f"• Current Price: <code>{fmt.format(current_price)}</code>\n"
            f"{partial_line}"
            f"• Stop Loss Moved to Break-Even: <code>{fmt.format(entry_price)}</code>\n"
            f"• Runner Target (TP2 / 3.0R): <code>{fmt.format(tp2_price)}</code>"
        )

    @classmethod
    def format_trade_closed_html(
        cls,
        trade: Dict[str, Any],
        exit_price: float,
        net_pnl: float,
        r_multiple: float,
        close_reason: str,
        account_balance: float,
        account_equity: float,
        stats: Dict[str, Any],
    ) -> str:
        """Render Telegram HTML notification when a bot trade closes in MetaTrader 5."""
        symbol = str(trade.get("symbol", "XAUUSD"))
        fmt = cls._price_fmt(symbol)
        ticket = trade.get("ticket")
        direction = str(trade.get("direction", "LONG"))
        entry_price = float(trade.get("entry_price", 0.0))
        volume = float(trade.get("initial_volume", trade.get("volume", 0.01)))

        if net_pnl > 0.50:
            header = "🏆💰 <b>TRADE RESULT: WIN (PROFIT LOCKED)</b>"
        elif net_pnl >= -0.50:
            header = "🛡️⚖️ <b>TRADE RESULT: BREAK-EVEN</b>"
        else:
            header = "🛑📉 <b>TRADE RESULT: STOP LOSS HIT</b>"

        pnl_sign = "+" if net_pnl >= 0 else ""
        r_sign = "+" if r_multiple >= 0 else ""

        total_closed = int(stats.get("total_closed", 1))
        wins = int(stats.get("wins", 0))
        losses = int(stats.get("losses", 0))
        bes = int(stats.get("breakevens", 0))
        cum_pnl = float(stats.get("cumulative_pnl", net_pnl))
        win_rate = (wins / max(1, wins + losses)) * 100.0
        cum_sign = "+" if cum_pnl >= 0 else ""

        return "\n".join(
            [
                header,
                f"📊 <b>{html.escape(symbol)} ({html.escape(direction)})</b> | Ticket <code>#{ticket}</code>",
                f"🏁 Exit Trigger: <b>{html.escape(close_reason)}</b>",
                "",
                "<b>💵 Trade Performance:</b>",
                f"• Realized P&L: <b>{pnl_sign}${net_pnl:.2f} USD ({r_sign}{r_multiple:.2f}R)</b>",
                f"• Entry → Exit: <code>{fmt.format(entry_price)}</code> → <code>{fmt.format(exit_price)}</code>",
                f"• Position Size: <code>{volume:.2f} lots</code>",
                "",
                "<b>🏦 Account & Bot Scorecard:</b>",
                f"• New Balance: <code>${account_balance:,.2f}</code> (Equity: <code>${account_equity:,.2f}</code>)",
                f"• Bot Record: <b>{wins}W - {losses}L - {bes}BE</b> (Win Rate: <b>{win_rate:.1f}%</b> / {total_closed} trades)",
                f"• Cumulative Bot P&L: <b>{cum_sign}${cum_pnl:.2f} USD</b>",
            ]
        )

    def send_message(self, text: str, parse_mode: str = "HTML") -> Tuple[bool, str]:
        """Send raw or HTML message to the configured Telegram chat."""
        if not self.is_configured:
            return False, "Telegram credentials (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) not configured in .env"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = urllib.parse.urlencode(
            {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": "true",
            }
        ).encode("utf-8")

        req = urllib.request.Request(url, data=payload, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("ok"):
                    return True, "Telegram alert delivered to your phone!"
                return False, f"Telegram API error: {data.get('description', 'Unknown error')}"
        except urllib.error.HTTPError as http_err:
            try:
                err_body = json.loads(http_err.read().decode("utf-8"))
                desc = err_body.get("description", str(http_err))
            except Exception:
                desc = str(http_err)
            return False, f"Telegram HTTP {http_err.code}: {desc}"
        except Exception as exc:
            return False, f"Telegram request failed: {exc}"

    def send_signal(self, signal: SetupSignal) -> Tuple[bool, str]:
        """Format and send a `SetupSignal` card to Telegram."""
        html_msg = self.format_telegram_html(signal)
        return self.send_message(html_msg, parse_mode="HTML")

    @staticmethod
    def discover_chat_id(bot_token: str, timeout_sec: float = 10.0) -> Tuple[Optional[str], str]:
        """
        Query `/getUpdates` to automatically find the numeric `chat_id` of the user who most
        recently sent a message (e.g. `/start`) to the bot.
        """
        clean_tok = sanitize_token(bot_token)
        url = f"https://api.telegram.org/bot{clean_tok}/getUpdates"
        try:
            with urllib.request.urlopen(url, timeout=timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if not data.get("ok"):
                    return None, f"Telegram API returned error: {data.get('description')}"
                results = data.get("result") or []
                for update in reversed(results):
                    msg = (
                        update.get("message")
                        or update.get("edited_message")
                        or update.get("channel_post")
                        or {}
                    )
                    chat = msg.get("chat") or {}
                    cid = chat.get("id")
                    if cid is not None:
                        username = chat.get("username") or chat.get("first_name") or "User"
                        return str(cid), f"Found numeric chat_id={cid} ({username})"
                return (
                    None,
                    "No messages received by your bot yet.",
                )
        except urllib.error.HTTPError as http_err:
            try:
                err_body = json.loads(http_err.read().decode("utf-8"))
                desc = err_body.get("description", str(http_err))
            except Exception:
                desc = str(http_err)
            return None, f"Telegram HTTP {http_err.code} on /getUpdates: {desc}"
        except Exception as exc:
            return None, f"Could not connect to api.telegram.org: {exc}"
