"""Tests for the time-series momentum (TSMOM) strategy."""

import unittest
from datetime import datetime, timedelta, timezone

from forex_bot.backtest.engine import Backtester
from forex_bot.config import CostConfig, InstrumentConfig, RiskConfig, TradingConfig
from forex_bot.models import Candle, Position, Side, SignalType
from forex_bot.research.walkforward import DEFAULT_GRIDS, param_combinations
from forex_bot.strategy import STRATEGY_REGISTRY, build_strategy
from forex_bot.strategy.base import StrategyContext
from forex_bot.strategy.tsmom import TsmomStrategy

MONDAY = datetime(2024, 1, 1, tzinfo=timezone.utc)  # 2024-01-01 is a Monday


def weekday_stamps(n, start=MONDAY):
    """``n`` consecutive weekday timestamps (no Saturdays/Sundays), like FX dailies."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def daily_candles(closes, epic="TEST", timeframe="DAY", start=MONDAY, spread=0.0005):
    candles, prev = [], closes[0]
    for ts, c in zip(weekday_stamps(len(closes), start), closes, strict=True):
        o = prev
        candles.append(Candle(
            epic=epic, timeframe=timeframe, timestamp=ts, open=o,
            high=max(o, c) + spread, low=min(o, c) - spread, close=c, volume=1.0,
        ))
        prev = c
    return candles


def rising(n=400, start=1.0, step=0.001):
    return [start + step * i for i in range(n)]


def falling(n=400, start=2.0, step=0.001):
    return [start - step * i for i in range(n)]


def ctx(history, position=None):
    return StrategyContext(history=history, position=position, equity=10_000.0)


class TestTsmomSignals(unittest.TestCase):
    def test_needs_warmup(self):
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(rising(252))  # one short of max(lookback)+1 = 253
        self.assertIsNone(strat.on_candle(hist[-1], ctx(hist)))
        hist = daily_candles(rising(253))
        self.assertIsNotNone(strat.on_candle(hist[-1], ctx(hist)))

    def test_uptrend_goes_long_with_full_agreement(self):
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(rising())
        sig = strat.on_candle(hist[-1], ctx(hist))
        self.assertEqual(sig.type, SignalType.ENTER_LONG)
        self.assertEqual(sig.meta["votes"], [1, 1, 1])
        self.assertAlmostEqual(sig.meta["score"], 1.0)
        self.assertGreater(sig.meta["atr"], 0)  # needed by vol_target sizing

    def test_downtrend_goes_short(self):
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(falling())
        sig = strat.on_candle(hist[-1], ctx(hist))
        self.assertEqual(sig.type, SignalType.ENTER_SHORT)
        self.assertEqual(sig.meta["votes"], [-1, -1, -1])

    def test_majority_vote_overrides_the_short_horizon(self):
        # 12M and 3M up, 1M down -> 2 of 3 say long.
        closes = [100.0] * 300
        closes[-22] = 105.0   # 21 bars ago is higher than now -> -1
        closes[-64] = 90.0    # 63 bars ago is lower than now  -> +1
        closes[-253] = 80.0   # 252 bars ago is lower than now -> +1
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(closes)
        sig = strat.on_candle(hist[-1], ctx(hist))
        self.assertEqual(sig.type, SignalType.ENTER_LONG)
        self.assertEqual(sig.meta["votes"], [-1, 1, 1])
        self.assertAlmostEqual(sig.meta["score"], 1 / 3)

    def test_tied_vote_changes_nothing(self):
        closes = [100.0] * 100
        closes[-22] = 105.0   # -1
        closes[-64] = 90.0    # +1
        strat = TsmomStrategy(lookbacks=(21, 63), rebalance="daily")
        hist = daily_candles(closes)
        self.assertIsNone(strat.on_candle(hist[-1], ctx(hist)))

    def test_exactly_equal_price_is_a_zero_vote_not_a_short_vote(self):
        hist = daily_candles([100.0] * 300)  # perfectly flat
        strat = TsmomStrategy(rebalance="daily")
        self.assertIsNone(strat.on_candle(hist[-1], ctx(hist)))

    def test_no_repeat_signal_when_already_positioned_correctly(self):
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(rising())
        pos = Position(epic="TEST", side=Side.BUY, size=1.0, entry_price=1.0)
        self.assertIsNone(strat.on_candle(hist[-1], ctx(hist, pos)))

    def test_reverses_when_position_is_on_the_wrong_side(self):
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(rising())
        pos = Position(epic="TEST", side=Side.SELL, size=1.0, entry_price=1.0)
        sig = strat.on_candle(hist[-1], ctx(hist, pos))
        self.assertEqual(sig.type, SignalType.ENTER_LONG)

    def test_default_has_no_stop(self):
        strat = TsmomStrategy(rebalance="daily")
        hist = daily_candles(rising())
        self.assertIsNone(strat.on_candle(hist[-1], ctx(hist)).stop_loss)

    def test_optional_atr_stop_is_on_the_protective_side(self):
        strat = TsmomStrategy(rebalance="daily", atr_stop_mult=3.0)
        up = daily_candles(rising())
        sig = strat.on_candle(up[-1], ctx(up))
        self.assertAlmostEqual(sig.stop_loss, up[-1].close - 3.0 * sig.meta["atr"])
        down = daily_candles(falling())
        sig = strat.on_candle(down[-1], ctx(down))
        self.assertAlmostEqual(sig.stop_loss, down[-1].close + 3.0 * sig.meta["atr"])


class TestTsmomRebalance(unittest.TestCase):
    def _signal_stamps(self, rebalance, n=520):
        strat = TsmomStrategy(rebalance=rebalance)
        hist = daily_candles(rising(n))
        stamps = []
        for i in range(strat.warmup, len(hist) + 1):
            window = hist[:i]
            if strat.on_candle(window[-1], ctx(window)) is not None:
                stamps.append(window[-1].timestamp)
        return stamps, hist, strat

    def test_weekly_fires_exactly_once_per_new_week(self):
        stamps, hist, strat = self._signal_stamps("weekly")
        expected = [
            hist[i].timestamp for i in range(strat.warmup - 1, len(hist))
            if hist[i].timestamp.isocalendar()[:2] != hist[i - 1].timestamp.isocalendar()[:2]
        ]
        self.assertEqual(stamps, expected)
        self.assertGreater(len(stamps), 40)
        # one per ISO week, none repeated
        weeks = [s.isocalendar()[:2] for s in stamps]
        self.assertEqual(len(weeks), len(set(weeks)))

    def test_monthly_fires_once_per_new_month(self):
        stamps, _, _ = self._signal_stamps("monthly")
        months = [(s.year, s.month) for s in stamps]
        self.assertEqual(len(months), len(set(months)))
        self.assertTrue(all(s.day <= 7 for s in stamps))  # first trading days of a month

    def test_daily_fires_every_bar(self):
        stamps, hist, strat = self._signal_stamps("daily", n=300)
        self.assertEqual(len(stamps), len(hist) - strat.warmup + 1)

    def test_a_missing_monday_still_rebalances_once(self):
        # Drop the first Monday of a week from the data: the Tuesday is then the
        # first bar of the new week, and it must still trigger the rebalance.
        hist = daily_candles(rising(400))
        monday = next(c for c in hist[300:] if c.timestamp.weekday() == 0)
        trimmed = [c for c in hist if c is not monday]
        idx = next(i for i, c in enumerate(trimmed) if c.timestamp > monday.timestamp)
        strat = TsmomStrategy(rebalance="weekly")
        sig = strat.on_candle(trimmed[idx], ctx(trimmed[: idx + 1]))
        self.assertIsNotNone(sig)  # the Tuesday opened the new ISO week


class TestTsmomGuardsAndRegistry(unittest.TestCase):
    def test_rejects_intraday_bars(self):
        strat = TsmomStrategy()
        hist = daily_candles(rising(), timeframe="MINUTE_15")
        with self.assertRaises(ValueError):
            strat.on_candle(hist[-1], ctx(hist))

    def test_timeframe_guard_can_be_disabled(self):
        strat = TsmomStrategy(rebalance="daily", required_timeframe=None)
        hist = daily_candles(rising(), timeframe="MINUTE_15")
        self.assertIsNotNone(strat.on_candle(hist[-1], ctx(hist)))

    def test_rejects_bad_params(self):
        for kwargs in (
            {"lookbacks": ()},
            {"lookbacks": (0, 5)},
            {"rebalance": "hourly"},
            {"atr_period": 0},
            {"atr_stop_mult": 0.0},
            {"atr_stop_mult": -1.0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TsmomStrategy(**kwargs)

    def test_accepts_list_lookbacks_from_yaml(self):
        strat = build_strategy("tsmom", {"lookbacks": [21, 63, 252]})
        self.assertEqual(strat.lookbacks, (21, 63, 252))
        self.assertEqual(strat.warmup, 253)

    def test_registered_with_a_single_point_grid(self):
        self.assertIs(STRATEGY_REGISTRY["tsmom"], TsmomStrategy)
        self.assertEqual(len(param_combinations(DEFAULT_GRIDS["tsmom"])), 1)


class TestTsmomBacktest(unittest.TestCase):
    def _config(self, epics, **risk):
        risk_kwargs = dict(max_position_pct=0.2, max_open_positions=len(epics),
                           max_daily_loss_pct=1.0, sizing_mode="vol_target",
                           vol_target_pct=0.01)
        risk_kwargs.update(risk)
        return TradingConfig(
            starting_equity=10_000.0,
            instruments=[InstrumentConfig(e, timeframe="DAY") for e in epics],
            risk=RiskConfig(**risk_kwargs),
            costs=CostConfig(spread_points=0.0),
            strategy="tsmom",
        )

    def test_takes_each_pairs_own_direction_through_the_real_engine(self):
        candles = {
            "UP": daily_candles(rising(), epic="UP"),
            "DOWN": daily_candles(falling(), epic="DOWN"),
        }
        result = Backtester(build_strategy("tsmom"), self._config(candles)).run(candles)
        sides = {t.epic: t.side for t in result.trades}
        self.assertEqual(sides["UP"], Side.BUY)
        self.assertEqual(sides["DOWN"], Side.SELL)
        # Trend-following a clean trend: each pair is entered once and held.
        self.assertEqual(sorted(t.epic for t in result.trades), ["DOWN", "UP"])
        self.assertTrue(all(t.pnl > 0 for t in result.trades))

    def test_vol_target_sizes_by_atr(self):
        calm = [1.0 + 0.001 * i for i in range(400)]
        wild = [1.0 + 0.001 * i + (0.02 if i % 2 else -0.02) for i in range(400)]
        candles = {"CALM": daily_candles(calm, epic="CALM"),
                   "WILD": daily_candles(wild, epic="WILD")}
        result = Backtester(build_strategy("tsmom"), self._config(candles)).run(candles)
        size = {t.epic: t.size for t in result.trades}
        self.assertGreater(size["CALM"], size["WILD"])  # quieter pair gets more units

    def test_open_position_cap_below_pair_count_leaves_pairs_untraded(self):
        # Documents the trap in the module docstring: cap < basket size means the
        # alphabetically-last pairs never get a position.
        candles = {e: daily_candles(rising(), epic=e) for e in ("A", "B", "C")}
        cfg = self._config(candles, max_open_positions=2)
        result = Backtester(build_strategy("tsmom"), cfg).run(candles)
        self.assertEqual(sorted({t.epic for t in result.trades}), ["A", "B"])


if __name__ == "__main__":
    unittest.main()