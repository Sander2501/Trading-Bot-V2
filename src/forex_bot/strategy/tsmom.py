"""Time-series momentum (TSMOM) on daily bars.

Each instrument is judged on its *own* trailing return — "has this pair gone up
or down over the last 1/3/12 months?" — and the bot takes that direction.
This differs from ``xsec_momentum`` and ``carry_trend_basket``, which *rank*
instruments against each other and hold a long/short sleeve. TSMOM has no
ranking: every pair can be long, or every pair short, at the same time.

Rule (no fitted parameters)
---------------------------
For each lookback in ``lookbacks`` (default 21/63/252 trading days, roughly
1/3/12 months) the vote is ``+1`` if the close is above the close that many bars
ago, ``-1`` if below, ``0`` if exactly equal. The position follows the sign of
the vote total: majority long -> long, majority short -> short. With the default
three lookbacks the total is never zero except on exact ties, so the strategy
is essentially always in the market and reverses when the majority flips.

The ensemble of lookbacks is a design choice inspired by Moskowitz, Ooi and
Pedersen (2012), who study individual lookback/holding combinations; the
specific equal-weight 1/3/12-month sign vote is *not* taken from the paper, so
treat it as an untested-by-default choice and let the holdout judge it.

Stateless rebalancing
---------------------
Decisions are only made on the first bar of a new ISO week (``rebalance="weekly"``,
the default), a new month, or every bar (``"daily"``). The first bar of a period
is detected by comparing this candle's timestamp with the previous candle's in
``context.history`` — no counters or stored state — so the same code runs
identically in backtest and live and survives restarts and missing bars. Because
daily FX candles on some feeds open on Sunday evening, "first bar of the week"
may be the bar labelled Monday or the one labelled Tuesday; what matters is that
it is exactly one decision per week.

Risk-layer requirements (read before backtesting)
-------------------------------------------------
* This strategy emits no stop by default (``atr_stop_mult=None``): a classic
  trend system exits on signal reversal, not on a tight stop. Without a stop,
  ``fixed_fractional`` sizing falls back to the ``max_position_pct`` notional
  cap and R-multiples are undefined (the report skips the avg-R rule). Use
  ``risk.sizing_mode: vol_target`` — the signal carries ``meta["atr"]`` so each
  pair contributes similar risk — or set ``atr_stop_mult`` for a wide
  catastrophic stop (and note a stopped-out pair re-enters only at the next
  rebalance).
* Being always in the market across a basket, it needs ``risk.max_open_positions``
  >= the number of pairs. A smaller cap silently trades only the pairs that
  sort first alphabetically. Correlation/currency caps will also veto USD-stacked
  entries; that is intended risk control, but it changes the strategy under test,
  so keep it identical across the runs you compare.
* Daily bars only (``required_timeframe="DAY"``). Lookbacks are measured in
  bars, so on an intraday feed "252" would mean a few days, not a year.
* The first signal needs ``max(lookbacks) + 1`` bars of history; in walk-forward
  keep ``is_bars`` and ``warmup_bars`` comfortably above that (e.g. warmup_bars
  >= 260) or early OOS bars are idle.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..indicators import atr
from ..models import Candle, Signal, SignalType
from .base import StrategyBase, StrategyContext

REBALANCE_MODES = ("daily", "weekly", "monthly")

#: Default lookbacks in trading days: ~1 month, ~3 months, ~12 months.
DEFAULT_LOOKBACKS: tuple[int, ...] = (21, 63, 252)

# ATR is computed on a fixed trailing window rather than the whole buffer, so the
# value does not depend on how many bars the engine happened to keep.
_ATR_WINDOW_MULT = 10


class TsmomStrategy(StrategyBase):
    def __init__(
        self,
        lookbacks: Sequence[int] = DEFAULT_LOOKBACKS,
        rebalance: str = "weekly",
        atr_period: int = 14,
        atr_stop_mult: float | None = None,
        required_timeframe: str | None = "DAY",
    ) -> None:
        lookbacks = tuple(int(lb) for lb in lookbacks)
        if not lookbacks:
            raise ValueError("lookbacks must not be empty")
        if any(lb < 1 for lb in lookbacks):
            raise ValueError("every lookback must be >= 1")
        if rebalance not in REBALANCE_MODES:
            raise ValueError(f"rebalance must be one of {list(REBALANCE_MODES)}")
        if atr_period < 1:
            raise ValueError("atr_period must be >= 1")
        if atr_stop_mult is not None and atr_stop_mult <= 0:
            raise ValueError("atr_stop_mult must be > 0 (or None for no stop)")

        self.lookbacks = lookbacks
        self.rebalance = rebalance
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.required_timeframe = required_timeframe
        self.warmup = max(max(lookbacks) + 1, atr_period + 1)

    # ------------------------------------------------------------------ #
    def on_candle(self, candle: Candle, context: StrategyContext) -> Signal | None:
        if self.required_timeframe is not None and candle.timeframe != self.required_timeframe:
            raise ValueError(
                f"tsmom requires {self.required_timeframe} bars (lookbacks are in bars), "
                f"got {candle.timeframe} for {candle.epic}"
            )

        history = context.history
        if len(history) < self.warmup:
            return None
        if not self._is_rebalance_bar(history):
            return None

        price = history[-1].close
        votes = [_vote(price, history[-1 - lb].close) for lb in self.lookbacks]
        total = sum(votes)
        if total == 0:
            return None  # tied vote: keep whatever we hold
        direction = 1 if total > 0 else -1

        position = context.position
        if position is not None and position.side.sign == direction:
            return None  # already positioned the right way; nothing to do

        atr_now = self._atr(history)
        stop = None
        if self.atr_stop_mult is not None and atr_now:
            stop = price - direction * self.atr_stop_mult * atr_now

        return Signal(
            epic=candle.epic,
            type=SignalType.ENTER_LONG if direction > 0 else SignalType.ENTER_SHORT,
            timestamp=candle.timestamp,
            stop_loss=stop,
            meta={
                "atr": atr_now,
                "votes": votes,
                "score": total / len(votes),
            },
        )

    # ------------------------------------------------------------------ #
    def _is_rebalance_bar(self, history: Sequence[Candle]) -> bool:
        """True on the first bar of a new period, judged only from timestamps."""
        if self.rebalance == "daily":
            return True
        if len(history) < 2:
            return False
        cur, prev = history[-1].timestamp, history[-2].timestamp
        if self.rebalance == "weekly":
            return cur.isocalendar()[:2] != prev.isocalendar()[:2]
        return (cur.year, cur.month) != (prev.year, prev.month)

    def _atr(self, history: Sequence[Candle]) -> float | None:
        window = history[-(self.atr_period * _ATR_WINDOW_MULT) :]
        series = atr(
            [c.high for c in window],
            [c.low for c in window],
            [c.close for c in window],
            self.atr_period,
        )
        return series[-1] if series and series[-1] is not None else None


def _vote(price: float, past_price: float) -> int:
    if price > past_price:
        return 1
    if price < past_price:
        return -1
    return 0