"""
Analysis Tools for Bandit Experiments

This package provides metrics, statistical tests, and visualization
tools for analyzing bandit algorithm performance.

Modules:
    - metrics: Regret computation, adaptation metrics
    - statistics: Confidence intervals, significance tests
    - visualizations: Publication-quality plotting functions
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

from .visualizations import (
    plot_regret_curves,
    plot_cumulative_regret,
    plot_arm_selection,
    plot_algorithm_comparison,
    plot_change_point_analysis,
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
    # Visualizations
    'plot_regret_curves',
    'plot_cumulative_regret',
    'plot_arm_selection',
    'plot_algorithm_comparison',
    'plot_change_point_analysis',
]
