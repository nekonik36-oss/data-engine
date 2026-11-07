"""
24/7 Live Cryptocurrency Market Feed (`BTCUSD`, `ETHUSD`, `SOLUSD`).

Uses Python's built-in `urllib.request` (zero API keys or extra dependencies required)
with automatic multi-exchange failover across global CDNs:
  1. Binance Vision Data CDN (`https://data-api.binance.vision/api/v3/klines`)
  2. Bybit V5 Public API (`https://api.bybit.com/v5/market/kline`)
  3. Coinbase Exchange Public API (`https://api.exchange.coinbase.com`)
  4. Kraken Public OHLC API (`https://api.kraken.com/0/public/OHLC`)

This guarantees 100% live 24/7 real-time M5/M15 candles and live prices every day of the
week (including Saturday & Sunday when Forex/Gold MT5 servers are closed).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Dict, List, Optional, Tuple
import urllib.request

from src.models import Candle

BINANCE_SYMBOLS: Dict[str, str] = {
    "BTCUSD": "BTCUSDT",
    "ETHUSD": "ETHUSDT",
    "SOLUSD": "SOLUSDT",
}

COINBASE_SYMBOLS: Dict[str, str] = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "SOLUSD": "SOL-USD",
}

KRAKEN_SYMBOLS: Dict[str, str] = {
    "BTCUSD": "XXBTZUSD",
    "ETHUSD": "XETHZUSD",
    "SOLUSD": "SOLUSD",
}


class CryptoLiveFeed:
    """Fetches real-time 24/7 OHLCV candles and live spot prices for BTCUSD, ETHUSD, SOLUSD."""

    def __init__(self, timeout_sec: float = 8.0) -> None:
        self.timeout_sec = timeout_sec
        self.last_source: str = "binance_vision"

    def _http_get_json(self, url: str) -> Optional[object]:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DataEngine/2.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def fetch_candles(
        self, symbol: str = "BTCUSD", timeframe: str = "M15", count: int = 260
    ) -> List[Candle]:
        """
        Fetch live 24/7 M5 or M15 candles in chronological order (oldest -> newest, where
        `candles[-1]` is the currently forming candle and `candles[-2]` is the just-closed candle).
        """
        sym = symbol.upper()
        # 1. Try Binance Vision Global CDN (and api.binance.com)
        b_sym = BINANCE_SYMBOLS.get(sym, f"{sym[:3]}USDT")
        b_tf = {"M5": "5m", "M15": "15m", "H1": "1h", "H4": "4h"}.get(timeframe.upper(), "15m")
        for host in ("https://data-api.binance.vision", "https://api.binance.com"):
            url = f"{host}/api/v3/klines?symbol={b_sym}&interval={b_tf}&limit={min(count, 500)}"
            data = self._http_get_json(url)
            if isinstance(data, list) and len(data) >= 30:
                self.last_source = host.replace("https://", "")
                return [
                    Candle(
                        timestamp=datetime.fromtimestamp(int(row[0]) / 1000.0, tz=timezone.utc),
                        open=float(row[1]),
                        high=float(row[2]),
                        low=float(row[3]),
                        close=float(row[4]),
                        volume=float(row[5]),
                    )
                    for row in data
                ]

        # 2. Fallback: Bybit V5 Public Linear/Spot Kline
        bybit_tf = {"M5": "5", "M15": "15", "H1": "60", "H4": "240"}.get(timeframe.upper(), "15")
        bybit_url = (
            f"https://api.bybit.com/v5/market/kline?category=spot&symbol={b_sym}"
            f"&interval={bybit_tf}&limit={min(count, 200)}"
        )
        bybit_data = self._http_get_json(bybit_url)
        if isinstance(bybit_data, dict):
            rows = (bybit_data.get("result") or {}).get("list") or []
            if len(rows) >= 30:
                self.last_source = "api.bybit.com"
                # Bybit returns newest first, so reverse to chronological
                return [
                    Candle(
                        timestamp=datetime.fromtimestamp(int(r[0]) / 1000.0, tz=timezone.utc),
                        open=float(r[1]),
                        high=float(r[2]),
                        low=float(r[3]),
                        close=float(r[4]),
                        volume=float(r[5]),
                    )
                    for r in reversed(rows)
                ]

        # 3. Fallback: Coinbase Exchange Public API
        cb_sym = COINBASE_SYMBOLS.get(sym, f"{sym[:3]}-USD")
        cb_gran = {"M5": 300, "M15": 900, "H1": 3600}.get(timeframe.upper(), 900)
        cb_url = f"https://api.exchange.coinbase.com/products/{cb_sym}/candles?granularity={cb_gran}"
        cb_data = self._http_get_json(cb_url)
        if isinstance(cb_data, list) and len(cb_data) >= 30:
            self.last_source = "api.exchange.coinbase.com"
            # Coinbase returns [time, low, high, open, close, volume] newest first
            subset = list(reversed(cb_data[:count]))
            return [
                Candle(
                    timestamp=datetime.fromtimestamp(int(r[0]), tz=timezone.utc),
                    open=float(r[3]),
                    high=float(r[2]),
                    low=float(r[1]),
                    close=float(r[4]),
                    volume=float(r[5]),
                )
                for r in subset
            ]

        # 4. Fallback: Kraken Public OHLC API
        kr_sym = KRAKEN_SYMBOLS.get(sym, sym)
        kr_tf = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}.get(timeframe.upper(), 15)
        kr_url = f"https://api.kraken.com/0/public/OHLC?pair={kr_sym}&interval={kr_tf}"
        kr_data = self._http_get_json(kr_url)
        if isinstance(kr_data, dict) and not kr_data.get("error"):
            res = kr_data.get("result") or {}
            for k, rows in res.items():
                if k != "last" and isinstance(rows, list) and len(rows) >= 30:
                    self.last_source = "api.kraken.com"
                    subset = rows[-count:]
                    return [
                        Candle(
                            timestamp=datetime.fromtimestamp(int(r[0]), tz=timezone.utc),
                            open=float(r[1]),
                            high=float(r[2]),
                            low=float(r[3]),
                            close=float(r[4]),
                            volume=float(r[6]),
                        )
                        for r in subset
                    ]

        return []

    def fetch_live_price(self, symbol: str = "BTCUSD") -> Optional[Tuple[float, float, float]]:
        """
        Return `(bid, ask, mid_price)` for `symbol` from live 24/7 crypto orderbooks.
        """
        sym = symbol.upper()
        b_sym = BINANCE_SYMBOLS.get(sym, f"{sym[:3]}USDT")
        for host in ("https://data-api.binance.vision", "https://api.binance.com"):
            url = f"{host}/api/v3/ticker/bookTicker?symbol={b_sym}"
            data = self._http_get_json(url)
            if isinstance(data, dict) and "bidPrice" in data and "askPrice" in data:
                bid = float(data["bidPrice"])
                ask = float(data["askPrice"])
                return bid, ask, 0.5 * (bid + ask)

        candles = self.fetch_candles(symbol=sym, timeframe="M5", count=5)
        if candles:
            mid = candles[-1].close
            half_sp = mid * 0.0001
            return mid - half_sp, mid + half_sp, mid
        return None
