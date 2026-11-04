"""Strictly causal technical indicators (ATR, EMA, RSI, ADX, Bollinger Bands)."""

from __future__ import annotations

from typing import List, Sequence, Tuple
import math

from src.models import Candle


def compute_ema(values: Sequence[float], period: int) -> List[float]:
    """Compute Exponential Moving Average causally."""
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    ema: List[float] = [values[0]]
    for val in values[1:]:
        ema.append(alpha * val + (1.0 - alpha) * ema[-1])
    return ema


def compute_atr(candles: Sequence[Candle], period: int = 14) -> List[float]:
    """Compute Wilder's Average True Range (ATR) causally."""
    if not candles:
        return []
    trs: List[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr = c.high - c.low
        else:
            prev_c = candles[i - 1].close
            tr = max(c.high - c.low, abs(c.high - prev_c), abs(c.low - prev_c))
        trs.append(max(tr, 1e-9))

    atr: List[float] = [trs[0]]
    for i in range(1, len(trs)):
        if i < period:
            atr.append(sum(trs[: i + 1]) / (i + 1))
        else:
            atr.append(((period - 1) * atr[-1] + trs[i]) / period)
    return atr


def compute_rsi(closes: Sequence[float], period: int = 14) -> List[float]:
    """Compute Wilder's Relative Strength Index (0-100) causally."""
    n = len(closes)
    if n == 0:
        return []
    rsi: List[float] = [50.0]
    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, n):
        delta = closes[i] - closes[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        if i <= period:
            avg_gain = ((i - 1) * avg_gain + gain) / i
            avg_loss = ((i - 1) * avg_loss + loss) / i
        else:
            avg_gain = ((period - 1) * avg_gain + gain) / period
            avg_loss = ((period - 1) * avg_loss + loss) / period

        if avg_loss < 1e-12:
            rsi.append(100.0 if avg_gain > 0 else 50.0)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100.0 - (100.0 / (1.0 + rs)))
    return rsi


def compute_bollinger_bands(
    closes: Sequence[float], period: int = 20, num_std: float = 2.0
) -> Tuple[List[float], List[float], List[float]]:
    """Return (middle_band, upper_band, lower_band) causally."""
    mid: List[float] = []
    upper: List[float] = []
    lower: List[float] = []
    for i in range(len(closes)):
        start = max(0, i - period + 1)
        window = closes[start : i + 1]
        m = sum(window) / len(window)
        var = sum((x - m) ** 2 for x in window) / max(len(window), 1)
        sd = math.sqrt(var)
        mid.append(m)
        upper.append(m + num_std * sd)
        lower.append(m - num_std * sd)
    return mid, upper, lower


def compute_adx(candles: Sequence[Candle], period: int = 14) -> List[float]:
    """Compute Average Directional Index (ADX) causally to detect trend vs range."""
    n = len(candles)
    if n == 0:
        return []
    adx_list: List[float] = [20.0]
    tr_sm = max(candles[0].high - candles[0].low, 1e-9)
    plus_dm_sm = 0.0
    minus_dm_sm = 0.0
    dx_list: List[float] = [20.0]

    for i in range(1, n):
        curr = candles[i]
        prev = candles[i - 1]
        up_move = curr.high - prev.high
        down_move = prev.low - curr.low
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close),
            1e-9,
        )

        tr_sm = ((period - 1) * tr_sm + tr) / period
        plus_dm_sm = ((period - 1) * plus_dm_sm + plus_dm) / period
        minus_dm_sm = ((period - 1) * minus_dm_sm + minus_dm) / period

        plus_di = 100.0 * (plus_dm_sm / max(tr_sm, 1e-9))
        minus_di = 100.0 * (minus_dm_sm / max(tr_sm, 1e-9))
        denom = plus_di + minus_di
        dx = 100.0 * abs(plus_di - minus_di) / denom if denom > 1e-9 else 0.0
        dx_list.append(dx)
        adx_val = ((period - 1) * adx_list[-1] + dx) / period
        adx_list.append(adx_val)

    return adx_list
