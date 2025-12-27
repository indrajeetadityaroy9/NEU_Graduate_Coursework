"""
Analysis Tools for Bandit Experiments

This package provides metrics and statistical tests
for analyzing bandit algorithm performance.

Modules:
    - metrics: Regret computation, adaptation metrics
    - statistics: Confidence intervals, significance tests
"""

from .metrics import (
    compute_regret,
    compute_cumulative_regret,
    compute_adaptation_regret,
    compute_detection_delay,
    compute_optimal_action_percentage,
    MetricsTracker,
)

from .statistics import (
    compute_confidence_interval,
    bootstrap_ci,
    paired_ttest,
    cohens_d,
    aggregate_runs,
)

__all__ = [
    # Metrics
    'compute_regret',
    'compute_cumulative_regret',
    'compute_adaptation_regret',
    'compute_detection_delay',
    'compute_optimal_action_percentage',
    'MetricsTracker',
    # Statistics
    'compute_confidence_interval',
    'bootstrap_ci',
    'paired_ttest',
    'cohens_d',
    'aggregate_runs',
]
