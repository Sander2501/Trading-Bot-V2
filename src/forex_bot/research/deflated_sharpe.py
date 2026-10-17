"""Probabilistic / deflated Sharpe ratio and sample-size (power) helpers.

Pure stdlib. These answer three questions a backtest Sharpe cannot answer alone:

1. **Is it distinguishable from luck?** ``probabilistic_sharpe_ratio`` (PSR) gives
   the probability the true Sharpe exceeds a benchmark, accounting for sample
   length and the skew/fat tails of the returns (Bailey & López de Prado, 2012).
2. **Is it still distinguishable after I tried many things?**
   ``deflated_sharpe_ratio`` (DSR) raises the benchmark to the Sharpe you would
   expect from the *best of N trials of pure noise* (Bailey & López de Prado,
   2014), so the more configurations you tried, the more a result must clear.
3. **Could this much data ever tell me?** ``min_observations`` /
   ``min_detectable_sharpe`` give the sample needed to detect a given Sharpe.

Units
-----
Everything inside the PSR/DSR maths is **per-period** (e.g. per-day) Sharpe.
Annualized figures are only produced by the helpers that say so. Feed these
functions the strategy's *periodic* returns (see
``forex_bot.backtest.metrics.daily_returns``), not per-trade P&L: the formulas
assume an evenly-spaced return series.

Sample size is ``T`` (days of the portfolio return series)
----------------------------------------------------------
The standard error of a Sharpe ratio estimated from one portfolio return series
is governed by that series' length ``T``. Trading eight pairs does **not** give
8*T observations: the portfolio's variance already contains the pairs'
correlations. Diversification raises the Sharpe a basket can *have*
(``basket_sharpe``); it does not shorten the time needed to *measure* it.

Counting trials honestly
------------------------
``n_trials`` must include every configuration, grid point, and strategy you
evaluated on this data on the way to the result you are testing — including ones
you rejected. The research registry stores a status per (strategy, setup), not a
count of evaluations, so this number has to be tracked by you. Trials that are
near-copies of each other overstate N; passing ``trial_sharpes`` (the Sharpe of
each trial) lets the measured dispersion do part of that correction.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

_NORMAL = NormalDist()
EULER_MASCHERONI = 0.5772156649015329


# --------------------------------------------------------------------------- #
# basic statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReturnMoments:
    n: int
    mean: float
    std: float        # sample standard deviation (ddof=1)
    skew: float       # third standardized moment
    kurtosis: float   # fourth standardized moment (a normal distribution is 3.0)

    @property
    def sharpe(self) -> float:
        """Per-period Sharpe ratio (zero risk-free rate)."""
        return self.mean / self.std


def return_moments(returns: Sequence[float]) -> ReturnMoments:
    n = len(returns)
    if n < 4:
        raise ValueError("need at least 4 returns")
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 <= 0:
        raise ValueError("returns have zero variance; Sharpe ratio is undefined")
    m3 = sum((r - mean) ** 3 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    return ReturnMoments(
        n=n,
        mean=mean,
        std=math.sqrt(m2 * n / (n - 1)),
        skew=m3 / m2 ** 1.5,
        kurtosis=m4 / m2 ** 2,
    )


def sharpe_ratio(returns: Sequence[float], *, periods_per_year: float = 1.0) -> float:
    """Sharpe ratio of ``returns``, annualized by ``sqrt(periods_per_year)``.

    The default ``periods_per_year=1`` returns the per-period Sharpe.
    """
    return return_moments(returns).sharpe * math.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# PSR / DSR
# --------------------------------------------------------------------------- #
def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sharpe: float = 0.0,
) -> float:
    """P(true Sharpe > ``benchmark_sharpe``) given one observed Sharpe.

    All Sharpe inputs are **per-period**. ``kurtosis`` is the raw (non-excess)
    kurtosis, 3.0 for a normal distribution. Returns ``nan`` in the degenerate
    case where the variance term is not positive.
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    variance_term = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if variance_term <= 0:
        return float("nan")
    z = (sharpe - benchmark_sharpe) * math.sqrt(n - 1) / math.sqrt(variance_term)
    return _NORMAL.cdf(z)


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` skill-less trials.

    ``trial_sharpe_variance`` is the variance of the per-period Sharpe ratios
    across trials. Uses the Bailey/López de Prado extreme-value approximation
    ``sqrt(V) * ((1-g) Z^-1(1-1/N) + g Z^-1(1-1/(N e)))`` with ``g`` the
    Euler-Mascheroni constant. For a single trial the expected maximum of a
    skill-less strategy is 0.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if trial_sharpe_variance < 0:
        raise ValueError("trial_sharpe_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    g = EULER_MASCHERONI
    z = (1.0 - g) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials) + g * _NORMAL.inv_cdf(
        1.0 - 1.0 / (n_trials * math.e)
    )
    return math.sqrt(trial_sharpe_variance) * z


@dataclass(frozen=True)
class DeflatedSharpe:
    n: int
    sharpe_period: float         # observed, per period
    sharpe_annual: float
    skew: float
    kurtosis: float
    n_trials: int
    benchmark_period: float      # expected max Sharpe of n_trials noise trials
    benchmark_annual: float
    psr: float                   # P(true SR > 0)
    dsr: float                   # P(true SR > benchmark): the deflated probability

    def is_significant(self, level: float = 0.95) -> bool:
        return self.dsr >= level


def deflated_sharpe_ratio(
    returns: Sequence[float],
    n_trials: int | None = None,
    *,
    trial_sharpes: Sequence[float] | None = None,
    trial_sharpe_variance: float | None = None,
    periods_per_year: float = 252.0,
) -> DeflatedSharpe:
    """Deflated Sharpe ratio of a return series after ``n_trials`` attempts.

    The trial Sharpe dispersion, in order of preference:

    * ``trial_sharpes`` — per-period Sharpe of every trial; its sample variance
      is used (and ``n_trials`` defaults to ``len(trial_sharpes)``).
    * ``trial_sharpe_variance`` — supply the variance directly.
    * neither — assume the trials are skill-less, whose Sharpe estimates have
      variance ``1/(T-1)``. This is an assumption, not a measurement.

    ``dsr`` is the probability the true Sharpe beats the best-of-N-noise
    benchmark; treat ``>= 0.95`` as the usual bar. ``psr`` (benchmark 0) is
    reported alongside so you can see how much of the haircut is multiple
    testing versus plain sampling error.
    """
    if trial_sharpes is not None:
        if len(trial_sharpes) < 2:
            raise ValueError("trial_sharpes needs at least 2 values")
        if n_trials is None:
            n_trials = len(trial_sharpes)
        if trial_sharpe_variance is None:
            trial_sharpe_variance = statistics.variance(trial_sharpes)
    if n_trials is None:
        raise ValueError("give n_trials (or trial_sharpes)")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    m = return_moments(returns)
    sr = m.sharpe
    if trial_sharpe_variance is None:
        trial_sharpe_variance = 1.0 / (m.n - 1)
    benchmark = expected_max_sharpe(n_trials, trial_sharpe_variance)
    root = math.sqrt(periods_per_year)
    return DeflatedSharpe(
        n=m.n,
        sharpe_period=sr,
        sharpe_annual=sr * root,
        skew=m.skew,
        kurtosis=m.kurtosis,
        n_trials=n_trials,
        benchmark_period=benchmark,
        benchmark_annual=benchmark * root,
        psr=probabilistic_sharpe_ratio(sr, m.n, m.skew, m.kurtosis, 0.0),
        dsr=probabilistic_sharpe_ratio(sr, m.n, m.skew, m.kurtosis, benchmark),
    )


# --------------------------------------------------------------------------- #
# sample size / power
# --------------------------------------------------------------------------- #
def _z_sum(alpha: float, power: float) -> float:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    return _NORMAL.inv_cdf(1.0 - alpha) + _NORMAL.inv_cdf(power)


def min_observations(
    annual_sharpe: float,
    *,
    periods_per_year: float = 252.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Periods of data needed to detect a true ``annual_sharpe`` > 0.

    One-sided test at level ``alpha`` with the given ``power``, normal returns,
    independent periods: ``N = ((z_alpha + z_power) / SR_period)^2``. Fat tails
    and autocorrelation make the true requirement larger.
    """
    if annual_sharpe <= 0:
        raise ValueError("annual_sharpe must be > 0")
    sr_period = annual_sharpe / math.sqrt(periods_per_year)
    return math.ceil((_z_sum(alpha, power) / sr_period) ** 2)


def min_detectable_sharpe(
    n_obs: int,
    *,
    periods_per_year: float = 252.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Smallest true annualized Sharpe detectable with ``n_obs`` periods."""
    if n_obs < 1:
        raise ValueError("n_obs must be >= 1")
    return _z_sum(alpha, power) / math.sqrt(n_obs) * math.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# what can a basket plausibly deliver?
# --------------------------------------------------------------------------- #
def mean_pairwise_correlation(series: Mapping[str, Sequence[float]]) -> float:
    """Average Pearson correlation over all pairs of equal-length return series."""
    names = sorted(series)
    if len(names) < 2:
        raise ValueError("need at least 2 series")
    length = len(series[names[0]])
    if any(len(series[n]) != length for n in names):
        raise ValueError("series must have equal length (align them first)")
    pairs = [
        statistics.correlation(series[a], series[b])
        for i, a in enumerate(names)
        for b in names[i + 1:]
    ]
    return sum(pairs) / len(pairs)


def effective_independent_bets(n_assets: int, mean_correlation: float) -> float:
    """Number of independent bets equivalent to ``n_assets`` with equal pairwise
    correlation ``mean_correlation``: ``N / (1 + (N-1) * rho)``, capped at ``N``.

    An approximation that treats all pairs as equally correlated.
    """
    if n_assets < 1:
        raise ValueError("n_assets must be >= 1")
    if n_assets == 1:
        return 1.0
    denom = 1.0 + (n_assets - 1) * mean_correlation
    if denom <= 0:
        return float(n_assets)
    return min(float(n_assets), n_assets / denom)


def basket_sharpe(asset_sharpe: float, n_assets: int, mean_correlation: float) -> float:
    """Sharpe of an equal-risk basket of ``n_assets`` that each have
    ``asset_sharpe`` and share pairwise correlation ``mean_correlation``.

    Equals ``asset_sharpe * sqrt(effective_independent_bets)``. Use it to ask
    whether the edge you hope for is large enough for ``T`` days of data to see.
    """
    return asset_sharpe * math.sqrt(effective_independent_bets(n_assets, mean_correlation))