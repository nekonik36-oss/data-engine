"""
Interactive Telegram Setup & Verification Utility (`setup_telegram.py`).

Fixes Windows PowerShell `Ctrl+V` control-character issues and automatically waits
for you to send `/start` or `hello` to your bot so it can discover your numeric
`TELEGRAM_CHAT_ID` automatically without manual lookup!
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import time

from src.env_loader import load_dotenv
from src.models import (
    Direction,
    KronosForecast,
    MarketRegime,
    SetupSignal,
    StrategyVote,
)
from src.telegram_notifier import TelegramNotifier, sanitize_token


ENV_PATH = Path(__file__).resolve().parent / ".env"
ENV_EXAMPLE_PATH = Path(__file__).resolve().parent / ".env.example"


def _upsert_env_var(env_file: Path, key: str, value: str) -> None:
    if not env_file.exists():
        if ENV_EXAMPLE_PATH.exists():
            env_file.write_text(ENV_EXAMPLE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            env_file.write_text("", encoding="utf-8")

    content = env_file.read_text(encoding="utf-8")
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    replacement = f"{key}={value}"
    if pattern.search(content):
        content = pattern.sub(replacement, content)
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        content += f"{replacement}\n"
    env_file.write_text(content, encoding="utf-8")


def build_sample_signal() -> SetupSignal:
    return SetupSignal(
        symbol="XAUUSD",
        timeframe="M15",
        timestamp=datetime.now(timezone.utc),
        direction=Direction.LONG,
        regime=MarketRegime.SESSION_SWEEP_WINDOW,
        confluence_score=91.5,
        entry_price=2658.50,
        stop_loss=2652.80,
        tp1_price=2667.05,
        tp2_price=2675.60,
        atr=3.80,
        rr_tp1=1.5,
        rr_tp2=3.0,
        recommended_lot=0.09,
        risk_usd=50.00,
        risk_pct=0.50,
        votes=[
            StrategyVote(
                "SND",
                Direction.LONG,
                0.86,
                "Rejected unmitigated DBR Demand Zone [2653.20-2656.80] (touches=0)",
            ),
            StrategyVote(
                "ICT",
                Direction.LONG,
                0.90,
                "SSL Liquidity Sweep below 2655.00 + Bullish FVG [2656.40-2657.90]",
            ),
            StrategyVote(
                "BREAKOUT",
                Direction.NEUTRAL,
                0.0,
                "Inside compression box [2652.80 - 2665.10]",
            ),
            StrategyVote(
                "TREND_PULLBACK",
                Direction.LONG,
                0.76,
                "Bullish EMA20/50 pullback impulse resumption (RSI=51.4)",
            ),
        ],
        kronos=KronosForecast(
            model_name="NeoQuasar/Kronos-base (CUDA)",
            direction=Direction.LONG,
            bullish_prob=0.88,
            expected_return_atr=1.45,
            upside_excursion_atr=2.10,
            downside_excursion_atr=0.45,
            confidence=0.88,
            summary="10-Path Forecast: Bullish=88.0%, Bearish=12.0% | Expected 12-Bar Drift=+1.45 ATR",
        ),
        active_combination="HYBRID_REGIME_KRONOS",
    )


def main() -> None:
    load_dotenv(ENV_PATH)
    print("=" * 74)
    print("  TELEGRAM COPILOT ALERT SETUP WIZARD")
    print("=" * 74)

    token = sanitize_token(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    if token and ":" in token:
        masked = token[:6] + "..." + token[-4:]
        print(f"  [✓] Current TELEGRAM_BOT_TOKEN in .env: {masked}")
        change = input("  Use this token? [Y/n]: ").strip().lower()
        if change == "n":
            token = ""
    else:
        token = ""

    if not token:
        raw_tok = input("  Paste your BotFather HTTP API Token (e.g. 123456:ABC-DEF...): ")
        token = sanitize_token(raw_tok)
        if not token or ":" not in token:
            print("  [!] Invalid token format (must contain ':'). Please run setup_telegram.py again.")
            return
        _upsert_env_var(ENV_PATH, "TELEGRAM_BOT_TOKEN", token)
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        print("  [✓] Saved clean TELEGRAM_BOT_TOKEN to .env")

    # Auto-discover numeric TELEGRAM_CHAT_ID via /getUpdates polling
    print("\n  Discovering your numeric TELEGRAM_CHAT_ID from your bot...")
    discovered_id, msg = TelegramNotifier.discover_chat_id(token)
    if not discovered_id:
        print(f"  -> {msg}")
        print("  👉 Action needed: Open your new bot in the Telegram app right now and send 'hello' or tap 'Start'!")
        for attempt in range(1, 16):
            time.sleep(2.5)
            discovered_id, msg = TelegramNotifier.discover_chat_id(token)
            if discovered_id:
                break
            print(f"     Waiting for your message on Telegram... ({attempt}/15)")

    if discovered_id:
        chat_id = discovered_id
        print(f"  [✓] {msg}")
    else:
        existing_cid = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if existing_cid.lstrip("-").isdigit():
            chat_id = existing_cid
        else:
            raw_cid = input("  Enter your numeric TELEGRAM_CHAT_ID (digits only, NOT @username): ").strip()
            if not raw_cid.lstrip("-").isdigit():
                print(
                    "  [!] TELEGRAM_CHAT_ID must be a number (e.g. 123456789), not a @username.\n"
                    "      Just send 'hello' to your bot in Telegram and run `python setup_telegram.py` again!"
                )
                return
            chat_id = raw_cid

    _upsert_env_var(ENV_PATH, "TELEGRAM_CHAT_ID", chat_id)
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    print(f"  [✓] Saved numeric TELEGRAM_CHAT_ID={chat_id} to .env")

    print("\n  Sending a live Test Copilot Signal Card to your Telegram...")
    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)
    ok, status = notifier.send_signal(build_sample_signal())
    print(f"  [{'✓' if ok else '!'}] {status}")


if __name__ == "__main__":
    main()
