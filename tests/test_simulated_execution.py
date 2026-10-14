from datetime import datetime, timezone

from forex_bot.config import CostConfig, InstrumentConfig, InstrumentSpecs
from forex_bot.execution.simulated import SimulatedExecution
from forex_bot.models import Order, OrderType, Side


def _order(side=Side.BUY):
    return Order(epic="EURUSD", side=side, size=1.0, order_type=OrderType.MARKET)


def test_session_multipliers_and_volatility_increase_slippage():
    costs = CostConfig(
        spread_points=0.0002,
        slippage_points=0.0001,
        session_spread_multipliers={"london": 2.0},
        session_slippage_multipliers={"london": 1.5},
        volatility_slippage_multiplier=2.0,
    )
    specs = InstrumentSpecs([InstrumentConfig(epic="EURUSD", spread_points=0.0002)])
    ex = SimulatedExecution(costs, specs)
    ts = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)  # london session
    fill = ex.execute(
        _order(Side.BUY),
        reference_price=1.2000,
        timestamp=ts,
        volatility_ratio=2.0,
    )
    # spread: 0.0002*2 => half-spread 0.0002
    # slippage: 0.0001*1.5*(1 + (2-1)*2) = 0.00045
    assert abs(fill.price - 1.20065) < 1e-9


def test_stress_multiplier_applies_to_spread_and_slippage():
    costs = CostConfig(spread_points=0.0002, slippage_points=0.0001)
    ex = SimulatedExecution(costs)
    base = ex.execute(_order(Side.SELL), reference_price=1.0000)
    stressed = ex.execute(_order(Side.SELL), reference_price=1.0000, stress_multiplier=3.0)
    assert stressed.price < base.price
