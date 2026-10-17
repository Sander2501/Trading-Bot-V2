"""Simulated execution for backtesting.

Applies a configurable cost model:
  * half-spread on each side (buy at ask, sell at bid),
  * fixed slippage in the adverse direction,
  * flat per-trade commission.
"""

from __future__ import annotations

from datetime import datetime, time

from ..config import CostConfig, InstrumentSpecs
from ..models import Order, Side
from .base import ExecutionEngine, Fill


class SimulatedExecution(ExecutionEngine):
    def __init__(self, costs: CostConfig, specs: InstrumentSpecs | None = None) -> None:
        self.costs = costs
        self.specs = specs

    def execute(
        self,
        order: Order,
        *,
        reference_price: float,
        timestamp: datetime | None = None,
        volatility_ratio: float | None = None,
        stress_multiplier: float = 1.0,
    ) -> Fill:
        spread = self.specs.spread(order.epic) if self.specs else self.costs.spread_points
        session = _session_name(timestamp.hour) if timestamp else None
        spread_mult = (
            self.costs.session_spread_multipliers.get(session, 1.0)
            if session and self.costs.session_spread_multipliers
            else 1.0
        )
        slip_mult = (
            self.costs.session_slippage_multipliers.get(session, 1.0)
            if session and self.costs.session_slippage_multipliers
            else 1.0
        )
        spread = calculate_dynamic_spread(order.epic, timestamp, spread)
        spread *= spread_mult * max(stress_multiplier, 0.0)
        half_spread = spread / 2.0
        slip = self.costs.slippage_points * slip_mult * max(stress_multiplier, 0.0)
        if volatility_ratio is not None and volatility_ratio > 1.0:
            slip *= 1.0 + ((volatility_ratio - 1.0) * self.costs.volatility_slippage_multiplier)
        # Buys fill higher, sells fill lower (adverse).
        if order.side is Side.BUY:
            price = reference_price + half_spread + slip
        else:
            price = reference_price - half_spread - slip
        return Fill(
            epic=order.epic,
            side=order.side,
            size=order.size,
            price=price,
            commission=self.costs.commission_per_trade,
        )


def _session_name(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 22:
        return "newyork"
    return "overnight"


def calculate_dynamic_spread(
    symbol: str, current_time_utc: datetime | None, base_spread: float
) -> float:
    """Widen FX spreads during the daily rollover and Friday close."""
    del symbol
    if current_time_utc is None:
        return base_spread
    current_time = current_time_utc.time()
    if time(21, 55) <= current_time <= time(22, 15):
        return base_spread * 7.0
    if current_time_utc.weekday() == 4 and current_time_utc.hour >= 21:
        return base_spread * 5.0
    return base_spread
