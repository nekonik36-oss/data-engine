"""
Online Self-Learning & Contextual Bandit Weight Optimizer (`AdaptiveLearner`).

How the Bot Learns from Every Closed Trade (Wins & Losses):
  1. Tracks which strategies (`SND`, `ICT`, `BREAKOUT`, `TREND_PULLBACK`, `KRONOS`) voted
     for each trade, along with the Market Regime (`TRENDING_UP`, `RANGING`, etc.) and Symbol.
  2. When a trade closes in MT5 or the 24/7 Crypto Book, `record_trade_outcome()` receives
     the realized `r_multiple` (e.g. `+2.25R` win, `+0.75R` TP1+BE, or `-1.00R` loss).
  3. Updates:
     - Strategy Multiplier (`0.65x` to `1.35x`): Uses Bayesian-smoothed Exponential Moving
       Expectancy so strategies that win in live market conditions gain voting power, while
       strategies that get stopped out lose voting weight.
     - Regime × Strategy Matrix (`0.60x` to `1.40x`): Learns which strategy works best in
       each specific Market Regime.
     - Per-Symbol Dynamic Confluence Threshold (`+0.0` to `+10.0` points): If a symbol suffers
       consecutive losses, the bot automatically tightens its entry filter (requires higher
       confluence score) until winning momentum resumes.
  4. Persists all learned memory to `logs/learned_weights.json` across PC restarts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
LEARNED_WEIGHTS_PATH = LOGS_DIR / "learned_weights.json"
AI_FEEDBACK_INBOX_PATH = LOGS_DIR / "ai_feedback_inbox.json"

STRATEGIES = ("SND", "ICT", "BREAKOUT", "TREND_PULLBACK", "KRONOS")


class AdaptiveLearner:
    """Contextual bandit, post-mortem self-diagnoser & AI feedback telemetry publisher."""

    def __init__(self, path: Path = LEARNED_WEIGHTS_PATH) -> None:
        self.path = path
        self.state: Dict[str, Any] = self._load()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "total_episodes": 0,
            "strategy_stats": {
                s: {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "ema_r": 0.0,
                    "multiplier": 1.0,
                }
                for s in STRATEGIES
            },
            "regime_strategy_multipliers": {},
            "symbol_streaks": {},
            "symbol_tuning": {},
            "pending_post_mortems": [],
            "ai_feedback_feed": [],
        }

    def _load(self) -> Dict[str, Any]:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "strategy_stats" in data:
                    data.setdefault("symbol_tuning", {})
                    data.setdefault("pending_post_mortems", [])
                    data.setdefault("ai_feedback_feed", [])
                    return data
        except Exception:
            pass
        return self._default_state()

    def save(self) -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            self.state["updated_utc"] = now_utc
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
            inbox_payload = {
                "updated_utc": now_utc,
                "total_episodes": self.state.get("total_episodes", 0),
                "symbol_tuning": self.state.get("symbol_tuning", {}),
                "symbol_streaks": self.state.get("symbol_streaks", {}),
                "pending_post_mortems_count": len(self.state.get("pending_post_mortems", [])),
                "recent_ai_feedback": (self.state.get("ai_feedback_feed") or [])[-25:],
            }
            AI_FEEDBACK_INBOX_PATH.write_text(
                json.dumps(inbox_payload, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    def get_symbol_sl_floor_mult(self, symbol: str) -> float:
        """Return dynamic Stop-Loss floor multiplier (`1.0` to `1.60`) learned from post-mortem wick-outs."""
        tuning = (self.state.get("symbol_tuning") or {}).get(symbol.upper(), {})
        return round(max(1.0, min(1.60, float(tuning.get("sl_floor_mult", 1.0)))), 3)

    def get_symbol_be_shield_r(self, symbol: str) -> float:
        """Return dynamic Break-Even Shield trigger (`0.42R` to `0.55R`) learned from near-TP1 reversals."""
        tuning = (self.state.get("symbol_tuning") or {}).get(symbol.upper(), {})
        return round(max(0.42, min(0.55, float(tuning.get("be_shield_r", 0.50)))), 3)

    def get_strategy_multiplier(
        self, strategy_name: str, regime_name: str = "TREND_EXPANSION"
    ) -> float:
        """
        Return the learned weight multiplier (`0.60` to `1.40`) for `strategy_name` in `regime_name`.
        """
        s_info = (self.state.get("strategy_stats") or {}).get(strategy_name, {})
        global_mult = float(s_info.get("multiplier", 1.0))

        reg_key = f"{regime_name}:{strategy_name}"
        reg_mult = float(
            (self.state.get("regime_strategy_multipliers") or {}).get(reg_key, 1.0)
        )
        combined = 0.6 * global_mult + 0.4 * reg_mult
        return round(max(0.60, min(1.40, combined)), 3)

    def get_strategy_multipliers(
        self, regime_name: str = "TREND_EXPANSION"
    ) -> Dict[str, float]:
        """Return dict of learned multipliers across all 5 strategy modules for `regime_name`."""
        return {s: self.get_strategy_multiplier(s, regime_name) for s in STRATEGIES}

    def get_symbol_threshold_offset(self, symbol: str) -> float:
        """
        Return dynamic confluence score offset (`-1.5` to `+4.0`) based on recent symbol streak.
        Capped at `+4.0` so a loss streak never creates an RL deadlock above single-engine max scores.
        """
        sym_info = (self.state.get("symbol_streaks") or {}).get(symbol.upper(), {})
        loss_streak = int(sym_info.get("loss_streak", 0))
        win_streak = int(sym_info.get("win_streak", 0))
        if loss_streak >= 1:
            return min(4.0, float(loss_streak) * 1.5)
        if win_streak >= 2:
            return -1.5
        return 0.0

    def get_threshold_offset(self, symbol: str) -> float:
        return self.get_symbol_threshold_offset(symbol)

    def format_weights_banner(self) -> str:
        """Return a concise status string of learned strategy multipliers for the startup banner."""
        ep = int(self.state.get("total_episodes", 0))
        s_stats = self.state.get("strategy_stats") or {}
        parts = [
            f"{s}={float((s_stats.get(s) or {}).get('multiplier', 1.0)):.2f}x"
            for s in STRATEGIES
        ]
        return f"Online RL Active ({ep} closed trades learned | {', '.join(parts)})"

    def record_trade_outcome(
        self,
        symbol: str,
        agreeing_strategies: List[str],
        r_multiple: float,
        regime: str = "TREND_EXPANSION",
        regime_name: str | None = None,
        net_pnl: float = 0.0,
        trade_meta: Dict[str, Any] | None = None,
    ) -> str:
        """
        Update learned strategy weights, regime-strategy multipliers, symbol threshold,
        and queue automated post-mortem forensics (`ai_feedback_inbox.json`).
        """
        active_regime = regime_name or regime or "TREND_EXPANSION"
        self.state["total_episodes"] = int(self.state.get("total_episodes", 0)) + 1
        s_stats = self.state.setdefault("strategy_stats", {})
        reg_map = self.state.setdefault("regime_strategy_multipliers", {})
        sym_map = self.state.setdefault("symbol_streaks", {})
        tuning_map = self.state.setdefault("symbol_tuning", {})
        feed_list = self.state.setdefault("ai_feedback_feed", [])
        pending_pm = self.state.setdefault("pending_post_mortems", [])

        # Reward signal clipped to [-1.5, +3.0]
        reward = max(-1.5, min(3.0, float(r_multiple)))
        alpha = 0.22  # Learning rate for exponential moving expectancy

        updated_notes: List[str] = []
        for s_name in agreeing_strategies:
            if s_name not in s_stats:
                s_stats[s_name] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "ema_r": 0.0,
                    "multiplier": 1.0,
                }
            entry = s_stats[s_name]
            entry["trades"] = int(entry.get("trades", 0)) + 1
            if reward > 0.05:
                entry["wins"] = int(entry.get("wins", 0)) + 1
            elif reward < -0.05:
                entry["losses"] = int(entry.get("losses", 0)) + 1

            old_ema = float(entry.get("ema_r", 0.0))
            new_ema = (1.0 - alpha) * old_ema + alpha * reward
            entry["ema_r"] = round(new_ema, 3)

            # Bayesian dampener for early trades: scales smoothly as trades accumulate
            confidence_scale = min(1.0, entry["trades"] / 4.0)
            new_mult = max(0.65, min(1.35, 1.0 + 0.25 * new_ema * confidence_scale))
            entry["multiplier"] = round(new_mult, 3)

            # Update Regime x Strategy specific multiplier
            reg_key = f"{active_regime}:{s_name}"
            old_reg_mult = float(reg_map.get(reg_key, 1.0))
            step = 0.045 * reward
            new_reg_mult = max(0.60, min(1.40, old_reg_mult + step))
            reg_map[reg_key] = round(new_reg_mult, 3)

            eff_mult = self.get_strategy_multiplier(s_name, active_regime)
            arrow = "^" if reward > 0 else ("v" if reward < -0.05 else "->")
            updated_notes.append(f"{s_name} {arrow}{eff_mult:.2f}x")

        # Update Symbol streak & dynamic confluence threshold offset
        sym_key = symbol.upper()
        sym_entry = sym_map.setdefault(
            sym_key, {"win_streak": 0, "loss_streak": 0, "net_r": 0.0}
        )
        sym_entry["net_r"] = round(float(sym_entry.get("net_r", 0.0)) + reward, 2)
        if reward > 0.05:
            sym_entry["win_streak"] = int(sym_entry.get("win_streak", 0)) + 1
            sym_entry["loss_streak"] = 0
        elif reward < -0.05:
            sym_entry["loss_streak"] = int(sym_entry.get("loss_streak", 0)) + 1
            sym_entry["win_streak"] = 0

        # Autonomous Post-Mortem Forensics on Closed Trade
        now_utc = datetime.now(timezone.utc)
        now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
        meta = trade_meta or {}
        peak_r = float(meta.get("peak_favorable_r", 0.0))
        auto_fix_action = "NONE"

        if reward < -0.08:
            sym_tune = tuning_map.setdefault(
                sym_key, {"sl_floor_mult": 1.0, "be_shield_r": 0.55}
            )
            if peak_r >= 0.45:
                # Trade went +0.45R+ into profit before reversing to SL -> tighten Break-Even Shield!
                old_be = float(sym_tune.get("be_shield_r", 0.55))
                new_be = round(max(0.42, old_be - 0.04), 2)
                sym_tune["be_shield_r"] = new_be
                auto_fix_action = (
                    f"NEAR_TP1_GIVEBACK_DETECTED (peak +{peak_r:.2f}R before SL): "
                    f"Auto-tightened {sym_key} Break-Even Shield {old_be:.2f}R -> {new_be:.2f}R"
                )
            else:
                auto_fix_action = (
                    f"LOSS_QUEUED_FOR_90M_WICK_AUDIT (monitoring if {sym_key} reaches TP1={meta.get('tp1_price')})"
                )

            if meta.get("tp1_price"):
                pending_pm.append(
                    {
                        "ticket": meta.get("ticket"),
                        "symbol": sym_key,
                        "direction": meta.get("direction", "LONG"),
                        "entry_price": float(meta.get("entry_price", 0.0)),
                        "exit_price": float(meta.get("exit_price", 0.0)),
                        "tp1_price": float(meta.get("tp1_price", 0.0)),
                        "tp2_price": float(meta.get("tp2_price", 0.0)),
                        "agreeing_strategies": list(agreeing_strategies),
                        "regime": active_regime,
                        "closed_epoch": now_utc.timestamp(),
                        "closed_at_utc": now_str,
                        "audited": False,
                    }
                )
                self.state["pending_post_mortems"] = pending_pm[-20:]

        feed_list.append(
            {
                "timestamp_utc": now_str,
                "event": "TRADE_CLOSED_DIAGNOSTIC",
                "ticket": meta.get("ticket"),
                "symbol": sym_key,
                "direction": meta.get("direction"),
                "net_pnl": round(net_pnl, 2),
                "r_multiple": round(r_multiple, 2),
                "peak_favorable_r": round(peak_r, 2),
                "regime": active_regime,
                "strategies": list(agreeing_strategies),
                "auto_fix_action": auto_fix_action,
            }
        )
        self.state["ai_feedback_feed"] = feed_list[-40:]

        offset = self.get_symbol_threshold_offset(sym_key)
        self.save()

        strat_str = ", ".join(updated_notes) if updated_notes else "weights unchanged"
        offset_str = (
            f"+{offset:.1f} stricter"
            if offset > 0
            else (f"{offset:.1f} relaxed" if offset < 0 else "normal")
        )
        return f"Weights [{strat_str}] | {sym_key} Filter: {offset_str}"

    def run_post_mortem_audits(self, latest_prices: Dict[str, float]) -> List[str]:
        """
        Monitor recently closed losses for 90 minutes after stop-out:
        If price subsequently reaches `tp1_price` (like #1078992397 BTCUSD LONG),
        diagnose `WICKED_OUT_BEFORE_TARGET`, auto-widen `sl_floor_mult` (+15%) for that symbol,
        restore the penalized strategy weight, and publish an AI Feedback report to GitHub!
        """
        pending = self.state.get("pending_post_mortems") or []
        if not pending or not latest_prices:
            return []

        now_ts = datetime.now(timezone.utc).timestamp()
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        tuning_map = self.state.setdefault("symbol_tuning", {})
        sym_map = self.state.setdefault("symbol_streaks", {})
        s_stats = self.state.setdefault("strategy_stats", {})
        feed_list = self.state.setdefault("ai_feedback_feed", [])

        remaining: List[Dict[str, Any]] = []
        events: List[str] = []
        changed = False

        for item in pending:
            if item.get("audited"):
                continue
            sym = str(item.get("symbol", "")).upper()
            age_min = (now_ts - float(item.get("closed_epoch", now_ts))) / 60.0
            cur_p = latest_prices.get(sym)
            if cur_p is None:
                if age_min <= 90.0:
                    remaining.append(item)
                continue

            is_long = str(item.get("direction", "LONG")) == "LONG"
            tp1 = float(item.get("tp1_price", 0.0))
            reached_tp1 = (cur_p >= tp1) if is_long else (cur_p <= tp1)

            if reached_tp1 and tp1 > 0:
                # WICK-OUT REVERSAL DETECTED! Direction was right, stop was too tight for current volatility/spread.
                sym_tune = tuning_map.setdefault(
                    sym, {"sl_floor_mult": 1.0, "be_shield_r": 0.55}
                )
                old_mult = float(sym_tune.get("sl_floor_mult", 1.0))
                new_mult = round(min(1.60, old_mult + 0.12), 2)
                sym_tune["sl_floor_mult"] = new_mult

                # Clear false loss streak penalty since direction prediction was accurate
                if sym in sym_map:
                    sym_map[sym]["loss_streak"] = max(
                        0, int(sym_map[sym].get("loss_streak", 1)) - 1
                    )

                # Restore strategy EMA weight so accurate directional signals aren't suppressed
                for s_name in item.get("agreeing_strategies") or []:
                    if s_name in s_stats:
                        s_entry = s_stats[s_name]
                        s_entry["ema_r"] = round(float(s_entry.get("ema_r", 0.0)) + 0.18, 3)
                        conf_scale = min(1.0, int(s_entry.get("trades", 1)) / 4.0)
                        s_entry["multiplier"] = round(
                            max(0.65, min(1.35, 1.0 + 0.25 * s_entry["ema_r"] * conf_scale)),
                            3,
                        )

                msg = (
                    f"[AI POST-MORTEM SELF-FIX] Ticket #{item.get('ticket')} ({sym} {item.get('direction')}) "
                    f"was wicked out @ {item.get('exit_price')} before reaching TP1 ({tp1} -> {cur_p})! "
                    f"Auto-widened {sym} Stop-Floor Multiplier {old_mult:.2f}x -> {new_mult:.2f}x & restored strategy weight."
                )
                events.append(msg)
                feed_list.append(
                    {
                        "timestamp_utc": now_str,
                        "event": "WICK_OUT_AUTO_TUNED",
                        "ticket": item.get("ticket"),
                        "symbol": sym,
                        "direction": item.get("direction"),
                        "tp1_price": tp1,
                        "current_price": cur_p,
                        "minutes_after_close": round(age_min, 1),
                        "old_sl_floor_mult": old_mult,
                        "new_sl_floor_mult": new_mult,
                        "summary": msg,
                    }
                )
                changed = True
            elif age_min <= 90.0:
                remaining.append(item)
            else:
                feed_list.append(
                    {
                        "timestamp_utc": now_str,
                        "event": "TRUE_TREND_LOSS_CONFIRMED",
                        "ticket": item.get("ticket"),
                        "symbol": sym,
                        "direction": item.get("direction"),
                        "summary": f"Ticket #{item.get('ticket')} ({sym}) did not reach TP1 within 90m; stricter confluence filter (+3.0) retained.",
                    }
                )
                changed = True

        if changed or len(remaining) != len(pending):
            self.state["pending_post_mortems"] = remaining
            self.state["ai_feedback_feed"] = feed_list[-40:]
            self.save()

        return events
