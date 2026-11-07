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

STRATEGIES = ("SND", "ICT", "BREAKOUT", "TREND_PULLBACK", "KRONOS")


class AdaptiveLearner:
    """Contextual bandit & exponentially-weighted reinforcement learner for strategy voting."""

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
        }

    def _load(self) -> Dict[str, Any]:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "strategy_stats" in data:
                    return data
        except Exception:
            pass
        return self._default_state()

    def save(self) -> None:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            self.state["updated_utc"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_strategy_multiplier(self, strategy_name: str, regime_name: str) -> float:
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

    def get_symbol_threshold_offset(self, symbol: str) -> float:
        """
        Return dynamic confluence score offset (`-2.0` to `+10.0`) based on recent symbol streak.
        Tightens entry threshold automatically after losses to prevent revenge/choppy trading.
        """
        sym_info = (self.state.get("symbol_streaks") or {}).get(symbol.upper(), {})
        loss_streak = int(sym_info.get("loss_streak", 0))
        win_streak = int(sym_info.get("win_streak", 0))
        if loss_streak >= 1:
            return min(10.0, float(loss_streak) * 3.0)
        if win_streak >= 2:
            return -1.5
        return 0.0

    def record_trade_outcome(
        self,
        symbol: str,
        regime_name: str,
        agreeing_strategies: List[str],
        r_multiple: float,
    ) -> str:
        """
        Update learned strategy weights, regime-strategy multipliers, and symbol threshold
        from a closed trade's realized `r_multiple`. Returns a human-readable summary for Telegram.
        """
        self.state["total_episodes"] = int(self.state.get("total_episodes", 0)) + 1
        s_stats = self.state.setdefault("strategy_stats", {})
        reg_map = self.state.setdefault("regime_strategy_multipliers", {})
        sym_map = self.state.setdefault("symbol_streaks", {})

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
            reg_key = f"{regime_name}:{s_name}"
            old_reg_mult = float(reg_map.get(reg_key, 1.0))
            step = 0.045 * reward
            new_reg_mult = max(0.60, min(1.40, old_reg_mult + step))
            reg_map[reg_key] = round(new_reg_mult, 3)

            eff_mult = self.get_strategy_multiplier(s_name, regime_name)
            arrow = "↑" if reward > 0 else ("↓" if reward < -0.05 else "→")
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

        offset = self.get_symbol_threshold_offset(sym_key)
        self.save()

        strat_str = ", ".join(updated_notes) if updated_notes else "weights unchanged"
        offset_str = f"+{offset:.1f} stricter" if offset > 0 else (f"{offset:.1f} relaxed" if offset < 0 else "normal")
        return f"Weights [{strat_str}] | {sym_key} Filter: {offset_str}"
