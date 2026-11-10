"""
Cloud MetaTrader 5 HTTPS REST Bridge (MetaApi.cloud Adapter)
=============================================================
Drop-in pure-HTTPS implementation of the MetaTrader5 (`mt5`) Python interface
so the trading engine can run 24/7 on any machine without Windows Administrator
privileges, without `terminal64.exe` installed (e.g. corporate/office laptops),
or inside Linux cloud containers / GitHub Actions.

Configured via `.env`:
  - `METAAPI_TOKEN`      : API token from https://app.metaapi.cloud/token
  - `METAAPI_ACCOUNT_ID` : Cloud MT5 Account UUID from https://app.metaapi.cloud/accounts
  - `METAAPI_REGION`     : Optional region (auto-detected if omitted; e.g. `new-york`, `london`, `singapore`)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class MetaApiMT5Bridge:
    """Drop-in replacement for the `MetaTrader5` module backed by MetaApi HTTPS REST endpoints."""

    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_TIME_GTC = 0
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1

    TIMEFRAME_M5 = "5m"
    TIMEFRAME_M15 = "15m"
    TIMEFRAME_H1 = "1h"
    TIMEFRAME_H4 = "4h"

    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_RETCODE_DONE = 10009

    def __init__(
        self,
        token: Optional[str] = None,
        account_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> None:
        self.token = (token or os.environ.get("METAAPI_TOKEN", "")).strip()
        self.account_id = (account_id or os.environ.get("METAAPI_ACCOUNT_ID", "")).strip()
        self.region = (region or os.environ.get("METAAPI_REGION", "")).strip() or "new-york"
        self._last_error = (0, "OK")
        self._symbols_cache: Optional[List[str]] = None
        self._spec_cache: Dict[str, Any] = {}
        self._leverage: float = 500.0

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.account_id)

    def _client_base_url(self) -> str:
        return f"https://mt-client-api-v1.{self.region}.agiliumtrade.ai"

    def _market_data_base_url(self) -> str:
        return f"https://mt-market-data-client-api-v1.{self.region}.agiliumtrade.ai"

    def _request(
        self,
        url: str,
        method: str = "GET",
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
    ) -> Any:
        headers = {
            "auth-token": self.token,
            "Accept": "application/json",
            "User-Agent": "QuantDataEngine/1.0",
        }
        data_bytes = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                pass
            self._last_error = (exc.code, f"HTTP {exc.code}: {err_body[:200]}")
            return None
        except Exception as exc:
            self._last_error = (-1, str(exc))
            return None

    def initialize(self, **kwargs: Any) -> bool:
        """Auto-detect account region and verify Cloud MT5 connection."""
        if not self.is_configured:
            self._last_error = (-2, "METAAPI_TOKEN or METAAPI_ACCOUNT_ID not set")
            return False

        # 1. Discover region from provisioning API if possible
        prov_url = (
            f"https://mt-provisioning-api-v1.agiliumtrade.agiliumtrade.ai"
            f"/users/current/accounts/{self.account_id}"
        )
        acct_meta = self._request(prov_url, timeout=8.0)
        if isinstance(acct_meta, dict) and acct_meta.get("region"):
            self.region = str(acct_meta["region"])

        # 2. Verify client API across discovered region (or fallback regions)
        candidate_regions = list(
            dict.fromkeys([self.region, "new-york", "london", "singapore"])
        )
        for reg in candidate_regions:
            self.region = reg
            info = self.account_info()
            if info is not None and float(getattr(info, "balance", 0.0)) >= 0:
                return True
        return False

    def last_error(self) -> Any:
        return self._last_error

    def account_info(self) -> Optional[SimpleNamespace]:
        url = (
            f"{self._client_base_url()}/users/current/accounts/"
            f"{self.account_id}/account-information"
        )
        data = self._request(url)
        if not isinstance(data, dict) or "balance" not in data:
            return None
        lev = float(data.get("leverage", 500.0) or 500.0)
        self._leverage = max(lev, 1.0)
        return SimpleNamespace(
            login=int(data.get("login", 0) or 0),
            server=str(data.get("server", "")),
            company=str(data.get("broker", data.get("name", "MetaApi Cloud MT5"))),
            currency=str(data.get("currency", "USD")),
            balance=float(data.get("balance", 0.0) or 0.0),
            equity=float(data.get("equity", data.get("balance", 0.0)) or 0.0),
            margin=float(data.get("margin", 0.0) or 0.0),
            margin_free=float(data.get("freeMargin", data.get("equity", 0.0)) or 0.0),
            leverage=int(self._leverage),
        )

    def symbols_get(self) -> List[SimpleNamespace]:
        if self._symbols_cache is None:
            url = f"{self._client_base_url()}/users/current/accounts/{self.account_id}/symbols"
            res = self._request(url)
            if isinstance(res, list):
                self._symbols_cache = [str(s) for s in res]
            else:
                self._symbols_cache = []
        return [SimpleNamespace(name=s) for s in self._symbols_cache]

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        return True

    def symbol_info(self, symbol: str) -> Optional[SimpleNamespace]:
        if symbol in self._spec_cache:
            return self._spec_cache[symbol]
        enc_sym = urllib.parse.quote(symbol, safe="")
        url = (
            f"{self._client_base_url()}/users/current/accounts/"
            f"{self.account_id}/symbols/{enc_sym}/specification"
        )
        spec = self._request(url)
        if not isinstance(spec, dict) or "symbol" not in spec:
            return None
        digits = int(spec.get("digits", 5) or 5)
        tick_size = float(spec.get("tickSize", 10 ** (-digits)) or (10 ** (-digits)))
        tick_val = float(spec.get("tickValue", 1.0) or 1.0)
        info = SimpleNamespace(
            name=str(spec.get("symbol", symbol)),
            digits=digits,
            volume_min=float(spec.get("minVolume", 0.01) or 0.01),
            volume_max=float(spec.get("maxVolume", 100.0) or 100.0),
            volume_step=float(spec.get("volumeStep", 0.01) or 0.01),
            trade_tick_value=tick_val,
            trade_tick_size=tick_size,
            trade_contract_size=float(spec.get("contractSize", 1.0) or 1.0),
            trade_mode=4,
        )
        self._spec_cache[symbol] = info
        return info

    def symbol_info_tick(self, symbol: str) -> Optional[SimpleNamespace]:
        enc_sym = urllib.parse.quote(symbol, safe="")
        url = (
            f"{self._client_base_url()}/users/current/accounts/"
            f"{self.account_id}/symbols/{enc_sym}/current-price"
        )
        tick = self._request(url)
        if not isinstance(tick, dict) or "bid" not in tick or "ask" not in tick:
            return None
        return SimpleNamespace(
            bid=float(tick.get("bid", 0.0) or 0.0),
            ask=float(tick.get("ask", 0.0) or 0.0),
            time=int(datetime.now(timezone.utc).timestamp()),
        )

    def copy_rates_from_pos(
        self, symbol: str, timeframe: str, start_pos: int, count: int
    ) -> Optional[List[Dict[str, Any]]]:
        enc_sym = urllib.parse.quote(symbol, safe="")
        tf_str = str(timeframe) if isinstance(timeframe, str) else "15m"
        url = (
            f"{self._market_data_base_url()}/users/current/accounts/"
            f"{self.account_id}/historical-market-data/symbols/{enc_sym}"
            f"/timeframes/{tf_str}/candles?limit={int(count)}"
        )
        candles = self._request(url, timeout=12.0)
        if not isinstance(candles, list) or not candles:
            return None
        rows: List[Dict[str, Any]] = []
        for c in candles:
            t_str = str(c.get("time", ""))
            try:
                dt = datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                ts = int(dt.timestamp())
            except Exception:
                ts = int(datetime.now(timezone.utc).timestamp())
            rows.append(
                {
                    "time": ts,
                    "open": float(c.get("open", 0.0)),
                    "high": float(c.get("high", 0.0)),
                    "low": float(c.get("low", 0.0)),
                    "close": float(c.get("close", 0.0)),
                    "tick_volume": float(c.get("tickVolume", c.get("volume", 1.0)) or 1.0),
                }
            )
        rows.sort(key=lambda r: r["time"])
        return rows

    def positions_get(self, symbol: Optional[str] = None) -> List[SimpleNamespace]:
        url = f"{self._client_base_url()}/users/current/accounts/{self.account_id}/positions"
        raw_list = self._request(url)
        if not isinstance(raw_list, list):
            return []
        out: List[SimpleNamespace] = []
        for p in raw_list:
            sym = str(p.get("symbol", ""))
            if symbol and sym.upper() != symbol.upper():
                continue
            p_type_str = str(p.get("type", "POSITION_TYPE_BUY")).upper()
            p_type = self.POSITION_TYPE_BUY if "BUY" in p_type_str else self.POSITION_TYPE_SELL
            try:
                ticket_int = int(p.get("id", 0))
            except Exception:
                ticket_int = 0
            out.append(
                SimpleNamespace(
                    ticket=ticket_int,
                    symbol=sym,
                    type=p_type,
                    price_open=float(p.get("openPrice", 0.0) or 0.0),
                    price_current=float(p.get("currentPrice", p.get("openPrice", 0.0)) or 0.0),
                    sl=float(p.get("stopLoss", 0.0) or 0.0),
                    tp=float(p.get("takeProfit", 0.0) or 0.0),
                    volume=float(p.get("volume", 0.0) or 0.0),
                    profit=float(p.get("profit", 0.0) or 0.0),
                    swap=float(p.get("swap", 0.0) or 0.0),
                    magic=int(p.get("magic", 0) or 0),
                    comment=str(p.get("comment", "")),
                )
            )
        return out

    def order_calc_margin(
        self, order_type: int, symbol: str, volume: float, price: float
    ) -> Optional[float]:
        spec = self.symbol_info(symbol)
        contract_sz = float(getattr(spec, "trade_contract_size", 1.0) or 1.0)
        return round((price * volume * contract_sz) / max(self._leverage, 10.0), 2)

    def order_send(self, request: Dict[str, Any]) -> Optional[SimpleNamespace]:
        action = request.get("action")
        url = f"{self._client_base_url()}/users/current/accounts/{self.account_id}/trade"

        if action == self.TRADE_ACTION_SLTP:
            payload: Dict[str, Any] = {
                "actionType": "POSITION_MODIFY",
                "positionId": str(request.get("position")),
            }
            if float(request.get("sl", 0.0) or 0.0) > 0:
                payload["stopLoss"] = float(request["sl"])
            if float(request.get("tp", 0.0) or 0.0) > 0:
                payload["takeProfit"] = float(request["tp"])
        elif action == self.TRADE_ACTION_DEAL and request.get("position"):
            # Partial or full close of an existing position
            payload = {
                "actionType": "POSITION_PARTIAL",
                "positionId": str(request["position"]),
                "volume": float(request.get("volume", 0.01)),
            }
        else:
            is_buy = int(request.get("type", 0)) == self.ORDER_TYPE_BUY
            payload = {
                "actionType": "ORDER_TYPE_BUY" if is_buy else "ORDER_TYPE_SELL",
                "symbol": str(request.get("symbol", "")),
                "volume": float(request.get("volume", 0.01)),
                "magic": int(request.get("magic", 0) or 0),
                "comment": str(request.get("comment", ""))[:31],
            }
            if float(request.get("sl", 0.0) or 0.0) > 0:
                payload["stopLoss"] = float(request["sl"])
            if float(request.get("tp", 0.0) or 0.0) > 0:
                payload["takeProfit"] = float(request["tp"])

        res = self._request(url, method="POST", payload=payload, timeout=15.0)
        if not isinstance(res, dict):
            return SimpleNamespace(retcode=-1, comment=str(self._last_error))

        num_code = int(res.get("numericCode", 0) or 0)
        str_code = str(res.get("stringCode", ""))
        ok_codes = {
            10009,
            10008,
            0,
        }
        is_ok = num_code in ok_codes and (
            "DONE" in str_code or "PLACED" in str_code or "orderId" in res or "positionId" in res
        )
        retcode = self.TRADE_RETCODE_DONE if is_ok else (num_code or -1)
        order_id = 0
        try:
            order_id = int(res.get("orderId", res.get("positionId", 0)) or 0)
        except Exception:
            order_id = 0
        return SimpleNamespace(
            retcode=retcode,
            order=order_id,
            deal=order_id,
            price=float(request.get("price", 0.0) or 0.0),
            comment=str(res.get("message", str_code)),
        )

    def history_deals_get(
        self, from_dt: Any = None, to_dt: Any = None, position: Optional[int] = None
    ) -> List[SimpleNamespace]:
        if position is None:
            return []
        url = (
            f"{self._client_base_url()}/users/current/accounts/"
            f"{self.account_id}/history-deals/position/{int(position)}"
        )
        raw_deals = self._request(url)
        if not isinstance(raw_deals, list):
            return []
        deals: List[SimpleNamespace] = []
        for d in raw_deals:
            entry_str = str(d.get("entryType", "")).upper()
            entry_code = 1 if ("OUT" in entry_str or "INOUT" in entry_str) else 0
            deals.append(
                SimpleNamespace(
                    position_id=int(position),
                    entry=entry_code,
                    profit=float(d.get("profit", 0.0) or 0.0),
                    swap=float(d.get("swap", 0.0) or 0.0),
                    commission=float(d.get("commission", 0.0) or 0.0),
                    fee=float(d.get("fee", 0.0) or 0.0),
                    price=float(d.get("price", 0.0) or 0.0),
                    volume=float(d.get("volume", 0.0) or 0.0),
                )
            )
        return deals
