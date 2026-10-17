"""Research tooling: walk-forward optimization, OOS evaluation, pair screening,
and statistical-significance checks (deflated Sharpe, bootstrap)."""

from .bootstrap import BootstrapCI, bootstrap_ci, bootstrap_sharpe_ci
from .deflated_sharpe import (
    DeflatedSharpe,
    basket_sharpe,
    deflated_sharpe_ratio,
    effective_independent_bets,
    expected_max_sharpe,
    mean_pairwise_correlation,
    min_detectable_sharpe,
    min_observations,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)
from .registry import (
    FROZEN_STATUSES,
    STATUSES,
    ResearchRegistry,
    SetupRecord,
    Status,
    setup_key,
)
from .strategy_report import (
    METRIC_ORDER,
    RESEARCH_METRICS,
    StrategyReportRow,
    Verdict,
    generate_strategy_report,
)
from .walkforward import (
    DEFAULT_GRIDS,
    OptimizationResult,
    SliceResult,
    param_combinations,
    walk_forward_optimize,
)

__all__ = [
    "walk_forward_optimize",
    "param_combinations",
    "DEFAULT_GRIDS",
    "OptimizationResult",
    "SliceResult",
    "generate_strategy_report",
    "StrategyReportRow",
    "Verdict",
    "METRIC_ORDER",
    "RESEARCH_METRICS",
    "ResearchRegistry",
    "SetupRecord",
    "Status",
    "setup_key",
    "STATUSES",
    "FROZEN_STATUSES",
    "BootstrapCI",
    "bootstrap_ci",
    "bootstrap_sharpe_ci",
    "DeflatedSharpe",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "sharpe_ratio",
    "min_observations",
    "min_detectable_sharpe",
    "mean_pairwise_correlation",
    "effective_independent_bets",
    "basket_sharpe",
]