"""Reusable stress-scenario builders for backtest simulations."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from ..models import Candle


def latency_spike(candles: list[Candle], *, seconds: int = 5) -> list[Candle]:
    return [replace(c, timestamp=c.timestamp + timedelta(seconds=seconds)) for c in candles]


def gap_bars(candles: list[Candle], *, every_n: int = 20, gap_pct: float = 0.005) -> list[Candle]:
    out: list[Candle] = []
    for idx, c in enumerate(candles):
        if every_n > 0 and idx > 0 and idx % every_n == 0:
            gap_open = c.open * (1.0 + gap_pct)
            out.append(replace(c, open=gap_open))
        else:
            out.append(c)
    return out


def missing_ticks(candles: list[Candle], *, drop_every_n: int = 15) -> list[Candle]:
    if drop_every_n <= 1:
        return candles[:]
    return [c for i, c in enumerate(candles) if i % drop_every_n != 0]


def api_throttle(candles: list[Candle], *, keep_every_n: int = 2) -> list[Candle]:
    if keep_every_n <= 1:
        return candles[:]
    return [c for i, c in enumerate(candles) if i % keep_every_n == 0]
