"""Tests for the probabilistic / deflated Sharpe ratio and sample-size helpers.

Reference numbers marked "scipy" were computed independently with
``scipy.stats.norm`` from the published formulas, not with this module.
"""

import math
import random
import unittest
from datetime import datetime, timezone

from forex_bot.backtest.metrics import daily_returns
from forex_bot.research.deflated_sharpe import (
    basket_sharpe,
    deflated_sharpe_ratio,
    effective_independent_bets,
    expected_max_sharpe,
    mean_pairwise_correlation,
    min_detectable_sharpe,
    min_observations,
    probabilistic_sharpe_ratio,
    return_moments,
    sharpe_ratio,
)


def noise(n, seed, mu=0.0, sd=0.01):
    rng = random.Random(seed)
    return [mu + rng.gauss(0.0, sd) for _ in range(n)]


class TestMoments(unittest.TestCase):
    def test_hand_checkable_symmetric_sample(self):
        m = return_moments([-2, -1, 0, 1, 2])
        self.assertAlmostEqual(m.mean, 0.0)
        self.assertAlmostEqual(m.std, math.sqrt(10 / 4))   # sample std
        self.assertAlmostEqual(m.skew, 0.0)
        self.assertAlmostEqual(m.kurtosis, 1.7)            # m4/m2^2 = 6.8/4
        self.assertAlmostEqual(m.sharpe, 0.0)

    def test_rejects_degenerate_input(self):
        with self.assertRaises(ValueError):
            return_moments([0.01, 0.02, 0.03])         # too short
        with self.assertRaises(ValueError):
            return_moments([0.01] * 20)                # zero variance

    def test_annualization(self):
        r = noise(500, 1, mu=0.0005)
        self.assertAlmostEqual(
            sharpe_ratio(r, periods_per_year=252), sharpe_ratio(r) * math.sqrt(252)
        )


class TestPsrAndExpectedMax(unittest.TestCase):
    def test_psr_matches_published_formula(self):
        # scipy: norm.cdf((sr-b)*sqrt(n-1)/sqrt(1-g3*sr+(g4-1)/4*sr^2))
        got = probabilistic_sharpe_ratio(0.05, 1250, skew=-0.4, kurtosis=6.0,
                                         benchmark_sharpe=0.02)
        self.assertAlmostEqual(got, 0.8527234128181264, places=10)

    def test_psr_is_one_half_at_the_benchmark(self):
        self.assertAlmostEqual(probabilistic_sharpe_ratio(0.03, 500, benchmark_sharpe=0.03), 0.5)

    def test_psr_rises_with_sample_length_and_sharpe(self):
        base = probabilistic_sharpe_ratio(0.04, 250)
        self.assertGreater(probabilistic_sharpe_ratio(0.04, 1000), base)
        self.assertGreater(probabilistic_sharpe_ratio(0.08, 250), base)

    def test_fat_tails_and_negative_skew_lower_psr(self):
        normal = probabilistic_sharpe_ratio(0.06, 500, skew=0.0, kurtosis=3.0)
        ugly = probabilistic_sharpe_ratio(0.06, 500, skew=-1.0, kurtosis=9.0)
        self.assertLess(ugly, normal)

    def test_expected_max_sharpe_matches_scipy_reference(self):
        self.assertAlmostEqual(expected_max_sharpe(2, 0.01), 0.051976, places=6)
        self.assertAlmostEqual(expected_max_sharpe(10, 0.01), 0.157460, places=6)
        self.assertAlmostEqual(expected_max_sharpe(1000, 0.004), 0.205872, places=6)

    def test_expected_max_sharpe_properties(self):
        self.assertEqual(expected_max_sharpe(1, 0.01), 0.0)
        values = [expected_max_sharpe(n, 0.004) for n in (2, 5, 20, 100, 1000)]
        self.assertEqual(values, sorted(values))                       # more trials -> higher bar
        self.assertGreater(expected_max_sharpe(50, 0.01), expected_max_sharpe(50, 0.004))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            expected_max_sharpe(0, 0.01)
        with self.assertRaises(ValueError):
            expected_max_sharpe(5, -0.1)
        with self.assertRaises(ValueError):
            probabilistic_sharpe_ratio(0.1, 1)


class TestDeflatedSharpe(unittest.TestCase):
    def setUp(self):
        self.r = noise(750, seed=11, mu=0.0004)

    def test_single_trial_equals_psr(self):
        res = deflated_sharpe_ratio(self.r, n_trials=1)
        self.assertEqual(res.benchmark_period, 0.0)
        self.assertAlmostEqual(res.dsr, res.psr)

    def test_more_trials_means_lower_dsr(self):
        scores = [deflated_sharpe_ratio(self.r, n_trials=n).dsr for n in (1, 5, 50, 500)]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertLess(scores[-1], scores[0])

    def test_default_variance_is_the_null_sampling_variance(self):
        res = deflated_sharpe_ratio(self.r, n_trials=20)
        self.assertAlmostEqual(
            res.benchmark_period, expected_max_sharpe(20, 1.0 / (len(self.r) - 1))
        )

    def test_measured_trial_dispersion_is_used(self):
        tight = [0.050, 0.051, 0.049, 0.050, 0.052]    # near-identical trials
        loose = [0.01, 0.09, -0.04, 0.12, 0.00]        # dispersed trials
        res_tight = deflated_sharpe_ratio(self.r, trial_sharpes=tight)
        res_loose = deflated_sharpe_ratio(self.r, trial_sharpes=loose)
        self.assertEqual(res_tight.n_trials, 5)        # defaults to len(trial_sharpes)
        self.assertLess(res_tight.benchmark_period, res_loose.benchmark_period)
        self.assertGreater(res_tight.dsr, res_loose.dsr)

    def test_annual_figures_are_period_times_sqrt_ppy(self):
        res = deflated_sharpe_ratio(self.r, n_trials=10, periods_per_year=252)
        self.assertAlmostEqual(res.sharpe_annual, res.sharpe_period * math.sqrt(252))
        self.assertAlmostEqual(res.benchmark_annual, res.benchmark_period * math.sqrt(252))

    def test_pure_noise_does_not_clear_the_bar_after_many_trials(self):
        res = deflated_sharpe_ratio(noise(750, seed=5), n_trials=50)
        self.assertFalse(res.is_significant(0.95))

    def test_strong_edge_survives_deflation(self):
        strong = noise(1250, seed=8, mu=0.0012, sd=0.008)   # Sharpe ~ 2.4 annualized
        res = deflated_sharpe_ratio(strong, n_trials=50)
        self.assertTrue(res.is_significant(0.95))

    def test_requires_trial_count(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(self.r)
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(self.r, n_trials=0)
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(self.r, trial_sharpes=[0.05])


class TestSampleSize(unittest.TestCase):
    def test_minimum_observations_for_sharpe_0_6(self):
        days = min_observations(0.6)
        self.assertEqual(days, 4328)
        self.assertAlmostEqual(days / 252, 17.2, places=1)

    def test_minimum_detectable_sharpe(self):
        self.assertAlmostEqual(min_detectable_sharpe(1000), 1.248, places=3)
        self.assertAlmostEqual(min_detectable_sharpe(1250), 1.116, places=3)

    def test_the_two_are_inverses(self):
        sr = 0.9
        n = min_observations(sr)
        self.assertLessEqual(min_detectable_sharpe(n), sr + 1e-9)
        self.assertGreater(min_detectable_sharpe(n - 1), sr - 1e-6)

    def test_more_power_or_stricter_alpha_needs_more_data(self):
        base = min_observations(0.8)
        self.assertGreater(min_observations(0.8, power=0.95), base)
        self.assertGreater(min_observations(0.8, alpha=0.01), base)

    def test_validation(self):
        with self.assertRaises(ValueError):
            min_observations(0.0)
        with self.assertRaises(ValueError):
            min_detectable_sharpe(0)
        with self.assertRaises(ValueError):
            min_observations(0.5, alpha=1.5)


class TestBasket(unittest.TestCase):
    def test_effective_bets_limits(self):
        self.assertEqual(effective_independent_bets(1, 0.9), 1.0)
        self.assertAlmostEqual(effective_independent_bets(8, 0.0), 8.0)
        self.assertAlmostEqual(effective_independent_bets(8, 1.0), 1.0)
        self.assertAlmostEqual(effective_independent_bets(8, -0.5), 8.0)  # capped at N
        self.assertAlmostEqual(effective_independent_bets(8, 0.5), 8 / 4.5)

    def test_basket_sharpe(self):
        self.assertAlmostEqual(basket_sharpe(0.3, 8, 0.5), 0.3 * math.sqrt(8 / 4.5))
        self.assertAlmostEqual(basket_sharpe(0.3, 1, 0.5), 0.3)

    def test_mean_pairwise_correlation(self):
        x = noise(100, 2)
        self.assertAlmostEqual(mean_pairwise_correlation({"a": x, "b": x}), 1.0)
        neg = [-v for v in x]
        self.assertAlmostEqual(mean_pairwise_correlation({"a": x, "b": neg}), -1.0)
        # pairs: (x,x)=+1, (x,-x)=-1, (x,-x)=-1  ->  -1/3
        self.assertAlmostEqual(
            mean_pairwise_correlation({"a": x, "b": x, "c": neg}), -1 / 3
        )

    def test_mean_pairwise_correlation_validation(self):
        with self.assertRaises(ValueError):
            mean_pairwise_correlation({"a": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError):
            mean_pairwise_correlation({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0]})


class TestDailyReturns(unittest.TestCase):
    def test_collapses_intraday_samples_to_last_value_per_day(self):
        utc = timezone.utc
        curve = [
            (datetime(2024, 1, 1, 10, tzinfo=utc), 100.0),
            (datetime(2024, 1, 1, 16, tzinfo=utc), 101.0),   # day-1 close
            (datetime(2024, 1, 2, 12, tzinfo=utc), 102.01),  # day-2 close
            (datetime(2024, 1, 3, 9, tzinfo=utc), 99.0),
            (datetime(2024, 1, 3, 17, tzinfo=utc), 100.0),   # day-3 close
        ]
        got = daily_returns(curve)
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(got[0], 102.01 / 101.0 - 1.0)
        self.assertAlmostEqual(got[1], 100.0 / 102.01 - 1.0)


if __name__ == "__main__":
    unittest.main()