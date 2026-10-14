from forex_bot.backtest.stress import api_throttle, gap_bars, latency_spike, missing_ticks
from tests.helpers import make_candles


def test_latency_spike_shifts_timestamps():
    candles = make_candles([1.0, 1.1, 1.2], epic="T")
    shifted = latency_spike(candles, seconds=7)
    assert (shifted[0].timestamp - candles[0].timestamp).total_seconds() == 7


def test_gap_bars_injects_periodic_open_gap():
    candles = make_candles([1.0, 1.1, 1.2, 1.3, 1.4], epic="T")
    gapped = gap_bars(candles, every_n=2, gap_pct=0.01)
    assert gapped[2].open != candles[2].open
    assert gapped[1].open == candles[1].open


def test_missing_ticks_and_api_throttle_reduce_samples():
    candles = make_candles([1 + i * 0.01 for i in range(20)], epic="T")
    sparse = missing_ticks(candles, drop_every_n=4)
    throttled = api_throttle(candles, keep_every_n=3)
    assert len(sparse) < len(candles)
    assert len(throttled) < len(candles)
