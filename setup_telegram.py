"""
Interactive Telegram Setup & Verification Utility (`setup_telegram.py`).

Security Note:
  - Uses hidden terminal input (`getpass`) so your Bot Token is NEVER echoed on screen
    or leaked into chat logs.
  - Automatically discovers your `TELEGRAM_CHAT_ID` after you message your bot on Telegram.
  - Saves `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` into your local `.env` file (gitignored).
  - Sends a live test Copilot Signal Card to your phone to confirm everything works!
"""

from __future__ import annotations

from datetime import datetime, timezone
import getpass
import os
from pathlib import Path
import re

from src.env_loader import load_dotenv
from src.models import (
    Direction,
    KronosForecast,
    MarketRegime,
    SetupSignal,
    StrategyVote,
)
from src.telegram_notifier import TelegramNotifier


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
    print("  Step 1: Open Telegram and message @BotFather -> send /newbot")
    print("  Step 2: Copy the HTTP API Bot Token from @BotFather")
    print("  Step 3: Open a chat with your new bot and send '/start' or 'hello'\n")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if token:
        print("  [✓] Found existing TELEGRAM_BOT_TOKEN in .env")
        change = input("  Keep existing token? [Y/n]: ").strip().lower()
        if change == "n":
            token = ""

    if not token:
        token = getpass.getpass("  Paste TELEGRAM_BOT_TOKEN (input is hidden): ").strip()
        if not token:
            print("  [!] No token entered. Exiting.")
            return
        _upsert_env_var(ENV_PATH, "TELEGRAM_BOT_TOKEN", token)
        os.environ["TELEGRAM_BOT_TOKEN"] = token
        print("  [✓] Saved TELEGRAM_BOT_TOKEN to .env")

    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not chat_id:
        print("\n  Attempting to auto-discover your TELEGRAM_CHAT_ID from @BotFather updates...")
        discovered_id, msg = TelegramNotifier.discover_chat_id(token)
        print(f"  -> {msg}")
        if discovered_id:
            chat_id = discovered_id
        else:
            chat_id = input(
                "  Enter your numeric TELEGRAM_CHAT_ID manually (or press Enter after messaging your bot): "
            ).strip()
            if not chat_id:
                discovered_id, msg = TelegramNotifier.discover_chat_id(token)
                print(f"  -> {msg}")
                chat_id = discovered_id or ""

    if not chat_id:
        print("  [!] TELEGRAM_CHAT_ID not set yet. Send '/start' to your bot and re-run `python setup_telegram.py`.")
        return

    _upsert_env_var(ENV_PATH, "TELEGRAM_CHAT_ID", chat_id)
    os.environ["TELEGRAM_CHAT_ID"] = chat_id
    print(f"  [✓] Saved TELEGRAM_CHAT_ID={chat_id} to .env")

    print("\n  Sending a live Test Copilot Signal Card to your Telegram...")
    notifier = TelegramNotifier(bot_token=token, chat_id=chat_id)
    ok, status = notifier.send_signal(build_sample_signal())
    print(f"  [{'✓' if ok else '!'}] {status}")


if __name__ == "__main__":
    main()
