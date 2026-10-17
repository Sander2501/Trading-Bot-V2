"""Stationary bootstrap confidence intervals for a return series (pure stdlib).

Resampling single returns assumes they are independent, which daily strategy
returns are not (volatility clusters, and a trend position makes consecutive
days related). The *stationary bootstrap* (Politis & Romano, 1994) resamples
**blocks** of consecutive returns whose lengths are random (geometric, with mean
``mean_block``) and wraps around the end of the series, so the resampled series
keeps short-range dependence while remaining stationary.

Use it on **periodic strategy returns** (``backtest.metrics.daily_returns``), not
per-trade P&L.

A typical use is a go/no-go on the lower bound::

    r = daily_returns(result.equity_curve)
    ci = bootstrap_sharpe_ci(r)            # annualized, 95%, seeded
    if ci.lower <= 0: ...                  # edge not distinguishable from zero

Caveats worth knowing:

* It quantifies sampling uncertainty *within the history you have*. It cannot
  tell you about regimes absent from that history, and it does not correct for
  how many strategies you tried — pair it with ``deflated_sharpe_ratio``.
* ``mean_block`` matters. The default (``n ** (1/3)``) is a common rule of thumb;
  check that the conclusion survives a few values (e.g. 10, 20, 40) before
  trusting it.
* Percentile intervals are somewhat optimistic on short samples (< ~250 points).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .deflated_sharpe import sharpe_ratio

Statistic = Callable[[Sequence[float]], float]


@dataclass(frozen=True)
class BootstrapCI:
    statistic: float      # value on the original series
    lower: float
    upper: float
    confidence: float
    n_resamples: int
    mean_block: float

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0 or self.upper < 0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def default_mean_block(n: int) -> float:
    """Rule-of-thumb mean block length: ``n ** (1/3)``, at least 2."""
    return max(2.0, round(n ** (1.0 / 3.0)))


def stationary_resample(
    values: Sequence[float], mean_block: float, rng: random.Random
) -> list[float]:
    """One stationary-bootstrap resample of the same length as ``values``.

    Starts at a random index and, at each step, jumps to a fresh random index
    with probability ``1/mean_block`` or otherwise continues to the next index
    (wrapping around the end).
    """
    n = len(values)
    p_jump = 1.0 / mean_block
    out: list[float] = []
    i = rng.randrange(n)
    while len(out) < n:
        out.append(values[i])
        i = rng.randrange(n) if rng.random() < p_jump else (i + 1) % n
    return out


def bootstrap_ci(
    values: Sequence[float],
    statistic: Statistic = mean,
    *,
    mean_block: float | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapCI:
    """Percentile confidence interval for ``statistic`` via stationary bootstrap.

    ``seed`` makes the result reproducible (default 0); pass ``None`` for a
    nondeterministic run.
    """
    n = len(values)
    if n < 10:
        raise ValueError("need at least 10 values to bootstrap")
    if n_resamples < 100:
        raise ValueError("n_resamples must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    block = default_mean_block(n) if mean_block is None else float(mean_block)
    if block < 1:
        raise ValueError("mean_block must be >= 1")

    rng = random.Random(seed)
    stats = sorted(
        statistic(stationary_resample(values, block, rng)) for _ in range(n_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    lo = stats[max(0, int(math.floor(tail * n_resamples)))]
    hi = stats[min(n_resamples - 1, int(math.ceil((1.0 - tail) * n_resamples)) - 1)]
    return BootstrapCI(
        statistic=statistic(values),
        lower=lo,
        upper=hi,
        confidence=confidence,
        n_resamples=n_resamples,
        mean_block=block,
    )


def bootstrap_sharpe_ci(
    returns: Sequence[float],
    *,
    periods_per_year: float = 252.0,
    mean_block: float | None = None,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapCI:
    """Stationary-bootstrap CI for the **annualized** Sharpe ratio."""

    def annualized_sharpe(sample: Sequence[float]) -> float:
        try:
            return sharpe_ratio(sample, periods_per_year=periods_per_year)
        except ValueError:  # a resample with zero variance (e.g. all-flat)
            return 0.0

    return bootstrap_ci(
        returns,
        annualized_sharpe,
        mean_block=mean_block,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )