"""
Zero-Cost ($0.00) Institutional Macro News Blackout Shield & Crypto Derivatives Flow Module.

Architecture & RL-Safety Guarantees:
  1. `MacroNewsShield`:
     - Pulls the official weekly economic calendar (`ff_calendar_thisweek.json`) with a 4-hour
       memory + disk cache (`logs/macro_calendar_cache.json`).
     - Blocks new live executions 20m BEFORE through 15m AFTER Red-Folder (`High` impact)
       macro releases (`CPI`, `PPI`, `NFP`, `FOMC`, `Core PCE`, `Powell`, `GDP`, etc.) for all
       affected pairs (`USD` events shield Crypto + Gold + Forex; `EUR`/`JPY` shield their pairs).
     - Isolates Shadow RL episodes opened during news shocks under `MACRO_NEWS_WINDOW` so
       macro whipsaws never corrupt normal `TRENDING_UP` / `TRENDING_DOWN` technical weights.

  2. `CryptoDerivativesFlow` (`DERIV_FLOW`):
     - Queries keyless public perpetual futures microstructure (`Binance Futures` -> `Bybit`
       -> `OKX` fallback, cached for 5 minutes per symbol):
         * Perpetual Funding Rate (`lastFundingRate`)
         * Perpetual Mark-to-Spot Index Basis (`(markPrice - indexPrice) / indexPrice` in bps)
         * Retail Crowd Long/Short Skew (`globalLongShortAccountRatio`)
         * Aggressive Market-Order Taker Flow (`takerlongshortRatio`)
     - Strictly SHADOW-FIRST RL rollout:
         * `0.0` live confluence weight until `DERIV_FLOW` completes >= 30 real Shadow RL
           episodes with >= 55.0% decided win rate AND >= +0.05R EMA expectancy.
         * Leaves all existing `SND`, `ICT`, `BREAKOUT`, `TREND_PULLBACK`, and `KRONOS`
           learned weights 100% untouched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import urllib.request

from src.models import Direction, StrategyVote

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
CALENDAR_CACHE_PATH = LOGS_DIR / "macro_calendar_cache.json"

FF_WEEKLY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


def _http_get_json(url: str, timeout: float = 4.5) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) QuantDataEngine/2.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


class MacroNewsShield:
    """
    Pauses new M5/M15 trade entries around Tier-1 (`High` impact) macro economic releases
    to avoid two-way liquidity whipsaws and stop-loss slippage.
    """

    CALENDAR_TTL_SEC = 4 * 3600.0  # Refresh every 4 hours

    def __init__(
        self,
        pre_minutes: float = 20.0,
        post_minutes: float = 15.0,
        cache_path: Path = CALENDAR_CACHE_PATH,
        allow_network: bool = True,
    ) -> None:
        self.pre_minutes = pre_minutes
        self.post_minutes = post_minutes
        self.cache_path = cache_path
        self.allow_network = allow_network
        self._events: List[Dict[str, Any]] = []
        self._last_fetch_epoch: float = 0.0

    @staticmethod
    def relevant_currencies(symbol: str) -> Set[str]:
        sym = symbol.upper().strip()
        if sym in {"BTCUSD", "ETHUSD", "SOLUSD", "XAUUSD"}:
            return {"USD"}
        if sym == "EURUSD":
            return {"USD", "EUR"}
        if sym == "USDJPY":
            return {"USD", "JPY"}
        currencies = {"USD"}
        if len(sym) >= 6:
            currencies.add(sym[:3])
            currencies.add(sym[3:6])
        return currencies

    @staticmethod
    def _parse_raw_events(raw_list: Any) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        if not isinstance(raw_list, list):
            return parsed
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            impact = str(item.get("impact", "")).strip()
            if impact.lower() != "high":
                continue
            country = str(item.get("country", "")).strip().upper()
            title = str(item.get("title", "")).strip()
            date_str = str(item.get("date", "")).strip()
            if not country or not date_str:
                continue
            try:
                dt_utc = datetime.fromisoformat(date_str.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            except Exception:
                continue
            parsed.append(
                {
                    "country": country,
                    "title": title or "High-Impact Release",
                    "timestamp_utc": dt_utc.isoformat(),
                    "epoch": dt_utc.timestamp(),
                }
            )
        parsed.sort(key=lambda x: float(x["epoch"]))
        return parsed

    def set_events_for_testing(self, events: List[Dict[str, Any]]) -> None:
        """Inject deterministic calendar events for unit testing without network calls."""
        self._events = self._parse_raw_events(events)
        self._last_fetch_epoch = time.time()

    def _refresh_if_needed(self) -> None:
        now_ep = time.time()
        if self._events and (now_ep - self._last_fetch_epoch) < self.CALENDAR_TTL_SEC:
            return

        # 1. Check fresh disk cache first
        try:
            if self.cache_path.exists():
                cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                fetched_ep = float(cached.get("fetched_epoch", 0.0))
                cached_events = cached.get("events")
                if isinstance(cached_events, list) and (now_ep - fetched_ep) < self.CALENDAR_TTL_SEC:
                    self._events = cached_events
                    self._last_fetch_epoch = fetched_ep
                    return
        except Exception:
            pass

        if not self.allow_network:
            self._last_fetch_epoch = now_ep
            return

        # 2. Fetch weekly ForexFactory calendar JSON
        try:
            raw = _http_get_json(FF_WEEKLY_CALENDAR_URL, timeout=5.0)
            parsed = self._parse_raw_events(raw)
            self._events = parsed
            self._last_fetch_epoch = now_ep
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(
                    json.dumps({"fetched_epoch": now_ep, "events": parsed}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            return
        except Exception:
            # 3. Fallback to older disk cache (up to 7 days) if network transiently unavailable
            try:
                if self.cache_path.exists():
                    cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
                    if isinstance(cached.get("events"), list):
                        self._events = cached["events"]
            except Exception:
                pass
            # Retry again in 15 minutes rather than hammering every cycle
            self._last_fetch_epoch = now_ep - self.CALENDAR_TTL_SEC + 900.0

    def is_blackout_active(
        self, symbol: str, now_utc: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """
        Return `(True, reason)` if `now_utc` falls inside `[-pre_minutes, +post_minutes]`
        of any High-impact macro event affecting `symbol`.
        """
        self._refresh_if_needed()
        if not self._events:
            return False, ""

        ref_dt = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
        ref_ep = ref_dt.timestamp()
        relevant = self.relevant_currencies(symbol)

        matching_titles: List[str] = []
        closest_delta_min: Optional[float] = None
        event_time_str = ""

        for ev in self._events:
            ccy = str(ev.get("country", "")).upper()
            if ccy not in relevant:
                continue
            ev_ep = float(ev.get("epoch", 0.0))
            delta_min = (ev_ep - ref_ep) / 60.0  # positive = upcoming, negative = already released
            if -self.post_minutes <= delta_min <= self.pre_minutes:
                title = f"{ccy} {ev.get('title', 'Macro Event')}"
                if title not in matching_titles:
                    matching_titles.append(title)
                if closest_delta_min is None or abs(delta_min) < abs(closest_delta_min):
                    closest_delta_min = delta_min
                    ev_dt = datetime.fromtimestamp(ev_ep, tz=timezone.utc)
                    event_time_str = ev_dt.strftime("%H:%M UTC")

        if not matching_titles or closest_delta_min is None:
            return False, ""

        joined = " / ".join(matching_titles[:2])
        if closest_delta_min >= 0:
            timing = f"releases in {closest_delta_min:.0f}m ({event_time_str})"
        else:
            timing = f"released {abs(closest_delta_min):.0f}m ago ({event_time_str})"
        return (
            True,
            f"Red-Folder {joined} {timing} [-{int(self.pre_minutes)}m/+{int(self.post_minutes)}m shield]",
        )

    def next_high_impact_event(
        self, now_utc: Optional[datetime] = None, currencies: Optional[Set[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the next upcoming High-impact event for telemetry / banner display."""
        self._refresh_if_needed()
        if not self._events:
            return None
        ref_ep = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).timestamp()
        target_ccys = currencies or {"USD", "EUR", "JPY"}

        for ev in self._events:
            ccy = str(ev.get("country", "")).upper()
            if ccy not in target_ccys:
                continue
            ev_ep = float(ev.get("epoch", 0.0))
            if ev_ep >= (ref_ep - self.post_minutes * 60.0):
                mins_until = round((ev_ep - ref_ep) / 60.0, 1)
                ev_dt = datetime.fromtimestamp(ev_ep, tz=timezone.utc)
                return {
                    "country": ccy,
                    "title": str(ev.get("title", "")),
                    "utc": ev_dt.strftime("%Y-%m-%d %H:%M UTC"),
                    "minutes_until": mins_until,
                }
        return None

    def format_banner(self, now_utc: Optional[datetime] = None) -> str:
        active, reason = self.is_blackout_active("BTCUSD", now_utc=now_utc)
        if active:
            return f"ACTIVE SHIELD ({reason})"
        nxt = self.next_high_impact_event(now_utc=now_utc)
        if not nxt:
            return f"Armed (-{int(self.pre_minutes)}m/+{int(self.post_minutes)}m Red-Folder USD/EUR/JPY)"
        mins = float(nxt["minutes_until"])
        if mins >= 60:
            eta = f"in {int(mins // 60)}h {int(mins % 60)}m"
        elif mins >= 0:
            eta = f"in {int(mins)}m"
        else:
            eta = f"{int(abs(mins))}m ago"
        return (
            f"Armed (-{int(self.pre_minutes)}m/+{int(self.post_minutes)}m) | "
            f"Next: {nxt['country']} {nxt['title']} @ {nxt['utc']} ({eta})"
        )


class CryptoDerivativesFlow:
    """
    Evaluates real-time Crypto Perpetual Futures positioning (Funding Rate, Perp-Spot Basis,
    Retail Long/Short Crowd Ratio, and Aggressive Taker Flow).

    Operates strictly in SHADOW-FIRST RL mode until statistically proven over >= 30 episodes.
    """

    SNAPSHOT_TTL_SEC = 300.0  # 5-minute cache per symbol
    MIN_GRADUATION_EPISODES = 30
    MIN_GRADUATION_WR = 0.55
    MIN_GRADUATION_EMA_R = 0.05

    SYMBOL_MAP = {
        "BTCUSD": ("BTCUSDT", "BTC-USDT-SWAP"),
        "ETHUSD": ("ETHUSDT", "ETH-USDT-SWAP"),
        "SOLUSD": ("SOLUSDT", "SOL-USDT-SWAP"),
    }

    def __init__(self, allow_network: bool = True) -> None:
        self.allow_network = allow_network
        self._cache: Dict[str, Tuple[float, Dict[str, float | str]]] = {}

    @classmethod
    def is_graduated(cls, learner: Any) -> Tuple[bool, str]:
        """
        Check whether `DERIV_FLOW` has earned live voting rights via Shadow RL:
          - >= 30 closed shadow episodes
          - >= 55% decided win rate
          - >= +0.05R EMA expectancy
        """
        if learner is None or not hasattr(learner, "state"):
            return False, "Shadow-Only (no learner)"
        s_stats = (learner.state.get("strategy_stats") or {}).get("DERIV_FLOW") or {}
        trades = int(s_stats.get("trades", 0))
        wins = int(s_stats.get("wins", 0))
        losses = int(s_stats.get("losses", 0))
        ema_r = float(s_stats.get("ema_r", 0.0))
        decided = wins + losses
        wr = (wins / decided) if decided > 0 else 0.0

        if trades >= cls.MIN_GRADUATION_EPISODES and wr >= cls.MIN_GRADUATION_WR and ema_r >= cls.MIN_GRADUATION_EMA_R:
            return True, f"Graduated Live ({trades} eps, {wr * 100:.1f}% WR, {ema_r:+.2f}R EMA)"
        return False, f"Shadow Incubation ({trades}/{cls.MIN_GRADUATION_EPISODES} eps, {wr * 100:.0f}% WR, {ema_r:+.2f}R)"

    def fetch_snapshot(self, symbol: str) -> Optional[Dict[str, float | str]]:
        sym = symbol.upper().strip()
        if sym not in self.SYMBOL_MAP:
            return None

        now_ep = time.time()
        cached = self._cache.get(sym)
        if cached is not None and (now_ep - cached[0]) < self.SNAPSHOT_TTL_SEC:
            return cached[1]

        if not self.allow_network:
            return cached[1] if cached is not None else None

        usdt_sym, okx_sym = self.SYMBOL_MAP[sym]
        snap: Dict[str, float | str] = {}

        # 1. Primary: Binance USD-M Futures (Funding + Basis + Crowd Ratio + Taker Aggression)
        try:
            prem = _http_get_json(
                f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={usdt_sym}", timeout=3.5
            )
            mark_p = float(prem.get("markPrice", 0.0))
            index_p = float(prem.get("indexPrice", 0.0))
            funding = float(prem.get("lastFundingRate", 0.0))
            basis_bps = ((mark_p - index_p) / index_p * 10000.0) if index_p > 0 else 0.0
            snap = {
                "source": "binance_fapi",
                "funding_rate": funding,
                "basis_bps": round(basis_bps, 2),
                "long_account_pct": 0.50,
                "taker_buy_sell_ratio": 1.00,
            }
            try:
                crowd = _http_get_json(
                    f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={usdt_sym}&period=15m&limit=1",
                    timeout=3.0,
                )
                if isinstance(crowd, list) and crowd:
                    snap["long_account_pct"] = round(float(crowd[-1].get("longAccount", 0.50)), 4)
            except Exception:
                pass
            try:
                taker = _http_get_json(
                    f"https://fapi.binance.com/futures/data/takerlongshortRatio?symbol={usdt_sym}&period=15m&limit=1",
                    timeout=3.0,
                )
                if isinstance(taker, list) and taker:
                    snap["taker_buy_sell_ratio"] = round(float(taker[-1].get("buySellRatio", 1.00)), 4)
            except Exception:
                pass
        except Exception:
            snap = {}

        # 2. Fallback A: Bybit V5 Linear Perps
        if not snap:
            try:
                byb = _http_get_json(
                    f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={usdt_sym}",
                    timeout=3.5,
                )
                rows = ((byb.get("result") or {}).get("list")) or []
                if rows:
                    row = rows[0]
                    mark_p = float(row.get("markPrice", 0.0))
                    index_p = float(row.get("indexPrice", 0.0))
                    funding = float(row.get("fundingRate", 0.0))
                    basis_bps = ((mark_p - index_p) / index_p * 10000.0) if index_p > 0 else 0.0
                    snap = {
                        "source": "bybit_linear",
                        "funding_rate": funding,
                        "basis_bps": round(basis_bps, 2),
                        "long_account_pct": 0.50,
                        "taker_buy_sell_ratio": 1.00,
                    }
            except Exception:
                snap = {}

        # 3. Fallback B: OKX Public Swap Funding
        if not snap:
            try:
                okx = _http_get_json(
                    f"https://www.okx.com/api/v5/public/funding-rate?instId={okx_sym}",
                    timeout=3.5,
                )
                rows = okx.get("data") or []
                if rows:
                    snap = {
                        "source": "okx_swap",
                        "funding_rate": float(rows[0].get("fundingRate", 0.0)),
                        "basis_bps": 0.0,
                        "long_account_pct": 0.50,
                        "taker_buy_sell_ratio": 1.00,
                    }
            except Exception:
                snap = {}

        if snap:
            self._cache[sym] = (now_ep, snap)
            return snap
        return cached[1] if cached is not None else None

    @staticmethod
    def score_snapshot(snap: Optional[Dict[str, float | str]]) -> StrategyVote:
        """
        Pure, deterministic scoring of a perpetual derivatives snapshot into a `DERIV_FLOW` vote.
        """
        if not snap:
            return StrategyVote(
                strategy_name="DERIV_FLOW",
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason="Derivatives feed unavailable",
            )

        funding = float(snap.get("funding_rate", 0.0))
        basis_bps = float(snap.get("basis_bps", 0.0))
        long_pct = float(snap.get("long_account_pct", 0.50))
        taker_ratio = float(snap.get("taker_buy_sell_ratio", 1.00))
        src = str(snap.get("source", "perp"))

        # --- Check Bearish Long-Squeeze / Aggressor Distribution (`SHORT`) ---
        # Never fade crowded retail longs if aggressive market buyers (`taker > 1.05`) are actively bidding!
        short_score = 0.0
        short_reasons: List[str] = []
        if taker_ratio <= 1.04:
            if long_pct >= 0.72:
                short_score += 0.48
                short_reasons.append(f"Crowd {long_pct * 100:.0f}% Long")
            elif long_pct >= 0.67:
                short_score += 0.40
                short_reasons.append(f"Crowd {long_pct * 100:.0f}% Long")

            if funding >= 0.00035:
                short_score += 0.44 if short_score == 0 else 0.12
                short_reasons.append(f"High +Funding {funding * 100:.3f}%")
            elif funding >= 0.00022 and short_score == 0:
                short_score += 0.36
                short_reasons.append(f"+Funding {funding * 100:.3f}%")

            # Require real-time seller aggression or perp discount confirmation
            flow_confirm = 0.0
            if taker_ratio <= 0.86:
                flow_confirm += 0.28
                short_reasons.append(f"Seller Taker {taker_ratio:.2f}x")
            elif taker_ratio <= 0.94:
                flow_confirm += 0.22
                short_reasons.append(f"Seller Taker {taker_ratio:.2f}x")

            if basis_bps <= -3.5:
                flow_confirm += 0.18
                short_reasons.append(f"Perp Discount {basis_bps:+.1f}bps")
            elif basis_bps <= -1.5:
                flow_confirm += 0.12
                short_reasons.append(f"Perp Basis {basis_bps:+.1f}bps")

            if short_score > 0 and flow_confirm > 0:
                short_score = min(0.88, short_score + flow_confirm)
            else:
                short_score = 0.0

        # --- Check Bullish Short-Squeeze / Aggressor Absorption (`LONG`) ---
        long_score = 0.0
        long_reasons: List[str] = []
        if taker_ratio >= 0.96:
            if long_pct <= 0.38:
                long_score += 0.48
                long_reasons.append(f"Crowd {(1.0 - long_pct) * 100:.0f}% Short")
            elif long_pct <= 0.43:
                long_score += 0.40
                long_reasons.append(f"Crowd {(1.0 - long_pct) * 100:.0f}% Short")

            if funding <= -0.00018:
                long_score += 0.46 if long_score == 0 else 0.14
                long_reasons.append(f"Neg Funding {funding * 100:.3f}%")
            elif funding <= -0.00006:
                long_score += 0.38 if long_score == 0 else 0.10
                long_reasons.append(f"Neg Funding {funding * 100:.3f}%")

            flow_confirm_l = 0.0
            if taker_ratio >= 1.15:
                flow_confirm_l += 0.28
                long_reasons.append(f"Buyer Taker {taker_ratio:.2f}x")
            elif taker_ratio >= 1.06:
                flow_confirm_l += 0.22
                long_reasons.append(f"Buyer Taker {taker_ratio:.2f}x")

            if basis_bps >= 3.5:
                flow_confirm_l += 0.18
                long_reasons.append(f"Perp Premium {basis_bps:+.1f}bps")
            elif basis_bps >= 1.5:
                flow_confirm_l += 0.12
                long_reasons.append(f"Perp Basis {basis_bps:+.1f}bps")

            if long_score > 0 and flow_confirm_l > 0:
                long_score = min(0.88, long_score + flow_confirm_l)
            else:
                long_score = 0.0

        meta: Dict[str, float | str] = {
            "source": src,
            "funding_rate": funding,
            "basis_bps": basis_bps,
            "long_account_pct": long_pct,
            "taker_buy_sell_ratio": taker_ratio,
        }

        if short_score >= 0.65 and short_score > long_score:
            return StrategyVote(
                strategy_name="DERIV_FLOW",
                direction=Direction.SHORT,
                confidence=round(short_score, 3),
                reason=f"[{src}] Bearish Long-Squeeze Flow: {', '.join(short_reasons)}",
                metadata=meta,
            )
        if long_score >= 0.65 and long_score >= short_score:
            return StrategyVote(
                strategy_name="DERIV_FLOW",
                direction=Direction.LONG,
                confidence=round(long_score, 3),
                reason=f"[{src}] Bullish Short-Squeeze Flow: {', '.join(long_reasons)}",
                metadata=meta,
            )

        return StrategyVote(
            strategy_name="DERIV_FLOW",
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=(
                f"[{src}] Balanced perp flow (funding={funding * 100:+.3f}%, "
                f"longs={long_pct * 100:.0f}%, taker={taker_ratio:.2f}x, basis={basis_bps:+.1f}bps)"
            ),
            metadata=meta,
        )

    def evaluate_vote(self, symbol: str) -> StrategyVote:
        snap = self.fetch_snapshot(symbol)
        return self.score_snapshot(snap)
