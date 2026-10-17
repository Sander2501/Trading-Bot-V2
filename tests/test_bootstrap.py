"""Tests for the stationary bootstrap confidence intervals."""

import random
import unittest

from forex_bot.research.bootstrap import (
    bootstrap_ci,
    bootstrap_sharpe_ci,
    default_mean_block,
    stationary_resample,
)
from forex_bot.research.deflated_sharpe import sharpe_ratio


def noise(n, seed, mu=0.0, sd=0.01):
    rng = random.Random(seed)
    return [mu + rng.gauss(0.0, sd) for _ in range(n)]


class TestResample(unittest.TestCase):
    def test_same_length_and_only_original_values(self):
        values = list(range(100))
        out = stationary_resample(values, 5.0, random.Random(1))
        self.assertEqual(len(out), 100)
        self.assertTrue(set(out) <= set(values))

    def test_blocks_keep_consecutive_runs_and_block_one_destroys_them(self):
        values = list(range(1000))
        n = len(values)

        def consecutive_share(mean_block):
            out = stationary_resample(values, mean_block, random.Random(4))
            steps = sum(1 for a, b in zip(out, out[1:], strict=False) if b == (a + 1) % n)
            return steps / (len(out) - 1)

        self.assertGreater(consecutive_share(20.0), 0.85)   # ~1 - 1/20
        self.assertLess(consecutive_share(1.0), 0.05)       # always jumps: iid resampling

    def test_default_block_length_rule(self):
        self.assertEqual(default_mean_block(1000), 10)
        self.assertEqual(default_mean_block(8), 2.0)        # floor of 2


class TestBootstrapCi(unittest.TestCase):
    def test_seeded_runs_are_identical_and_seeds_differ(self):
        x = noise(400, 3, mu=0.0005)
        self.assertEqual(bootstrap_ci(x, seed=5), bootstrap_ci(x, seed=5))
        self.assertNotEqual(bootstrap_ci(x, seed=5), bootstrap_ci(x, seed=6))

    def test_interval_brackets_the_point_estimate(self):
        x = noise(600, 9, mu=0.0004)
        ci = bootstrap_ci(x)
        self.assertLess(ci.lower, ci.statistic)
        self.assertLess(ci.statistic, ci.upper)
        self.assertAlmostEqual(ci.statistic, sum(x) / len(x))

    def test_wider_confidence_gives_wider_interval(self):
        x = noise(600, 9, mu=0.0004)
        narrow = bootstrap_ci(x, confidence=0.80)
        wide = bootstrap_ci(x, confidence=0.99)
        self.assertLess(wide.lower, narrow.lower)
        self.assertGreater(wide.upper, narrow.upper)

    def test_more_data_gives_a_tighter_interval(self):
        short = bootstrap_ci(noise(200, 2, mu=0.0004))
        long_ = bootstrap_ci(noise(2000, 2, mu=0.0004))
        self.assertLess(long_.upper - long_.lower, (short.upper - short.lower) * 0.5)

    def test_detects_a_real_drift_but_not_pure_noise(self):
        drifting = bootstrap_ci(noise(750, 21, mu=0.003))          # ~8 standard errors
        self.assertTrue(drifting.excludes_zero)
        self.assertGreater(drifting.lower, 0.0)
        flat = bootstrap_ci(noise(750, 21, mu=0.0))
        self.assertFalse(flat.excludes_zero)

    def test_validation(self):
        with self.assertRaises(ValueError):
            bootstrap_ci([0.01] * 5)
        x = noise(100, 1)
        with self.assertRaises(ValueError):
            bootstrap_ci(x, n_resamples=10)
        with self.assertRaises(ValueError):
            bootstrap_ci(x, confidence=1.0)
        with self.assertRaises(ValueError):
            bootstrap_ci(x, mean_block=0.5)


class TestBootstrapSharpe(unittest.TestCase):
    def test_reports_annualized_sharpe_with_a_bracketing_interval(self):
        x = noise(750, 14, mu=0.0006)
        ci = bootstrap_sharpe_ci(x)
        self.assertAlmostEqual(ci.statistic, sharpe_ratio(x, periods_per_year=252))
        self.assertLess(ci.lower, ci.statistic)
        self.assertLess(ci.statistic, ci.upper)

    def test_survives_flat_resamples(self):
        # Mostly-flat series: some resamples are entirely zero-variance. That must
        # be handled, not raise.
        x = [0.0] * 40 + [0.01, -0.005] + [0.0] * 40
        ci = bootstrap_sharpe_ci(x, mean_block=2, n_resamples=200)
        self.assertEqual(ci.n_resamples, 200)


if __name__ == "__main__":
    unittest.main()