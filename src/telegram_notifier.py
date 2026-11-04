"""
Telegram Bot Alert Notifier for Copilot Signal Cards.

Uses Python's built-in `urllib.request` (zero third-party dependencies required) to:
  1. Format `SetupSignal` into a clean mobile-friendly Telegram HTML card.
  2. Send instant push notifications to your phone when a high-confluence setup triggers.
  3. Auto-discover your `TELEGRAM_CHAT_ID` via `/getUpdates` during initial setup.
"""

from __future__ import annotations

import html
import json
import os
import urllib.parse
import urllib.request
from typing import Optional, Tuple

from src.env_loader import load_dotenv
from src.models import Direction, SetupSignal


class TelegramNotifier:
    """Sends formatted Copilot Signal Cards to a Telegram chat or channel."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_sec: float = 10.0,
    ) -> None:
        load_dotenv()
        self.bot_token = (bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
        self.chat_id = (chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.timeout_sec = timeout_sec

    @property
    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    @staticmethod
    def format_telegram_html(signal: SetupSignal) -> str:
        """Render a mobile-optimized HTML message for Telegram."""
        is_long = signal.direction == Direction.LONG
        header_icon = "🟢" if is_long else "🔴"
        dir_label = "BUY (LONG)" if is_long else "SELL (SHORT)"
        decimals = 2 if signal.symbol == "XAUUSD" else (3 if signal.symbol == "USDJPY" else 5)
        fmt = f"{{:.{decimals}f}}"

        lines = [
            f"{header_icon} <b>{html.escape(signal.symbol)} ({html.escape(signal.timeframe)}) — {dir_label}</b>",
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
                    return True, "Telegram alert delivered."
                return False, f"Telegram API error: {data.get('description', 'Unknown error')}"
        except Exception as exc:
            return False, f"Telegram request failed: {exc}"

    def send_signal(self, signal: SetupSignal) -> Tuple[bool, str]:
        """Format and send a `SetupSignal` card to Telegram."""
        html_msg = self.format_telegram_html(signal)
        return self.send_message(html_msg, parse_mode="HTML")

    @staticmethod
    def discover_chat_id(bot_token: str, timeout_sec: float = 10.0) -> Tuple[Optional[str], str]:
        """
        Query `/getUpdates` to automatically find the `chat_id` of the user who most
        recently sent a message (e.g. `/start`) to the bot.
        """
        url = f"https://api.telegram.org/bot{bot_token.strip()}/getUpdates"
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
                        return str(cid), f"Found chat_id={cid} ({username})"
                return (
                    None,
                    "No messages found yet. Open Telegram, send '/start' or 'hello' to your bot, and run again.",
                )
        except Exception as exc:
            return None, f"Could not connect to api.telegram.org: {exc}"
