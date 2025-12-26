#!/usr/bin/env python3
"""
Run All Experiments for Non-Stationary Bandit Study

This script runs all configured experiments and generates results.

Usage:
    python run_all_experiments.py [--quick] [--experiment NAME]

Options:
    --quick         Run with fewer runs for quick testing
    --experiment    Run only a specific experiment
    --save-dir      Directory to save results (default: experiments/results)
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from multi_armed_bandit.algorithms import (
    EpsilonGreedy, EpsilonGreedyConstant,
    UCB1, DiscountedUCB, SlidingWindowUCB,
    ThompsonSampling, DiscountedThompsonSampling,
    GradientBandit,
    EXP3, Rexp3,
)
from multi_armed_bandit.environments import (
    StationaryBandit,
    AbruptChangeBandit,
    GradualDriftBandit,
)
from multi_armed_bandit.experiments import (
    ExperimentConfig,
    run_experiment_suite,
)
from multi_armed_bandit.analysis import (
    aggregate_runs,
    compare_algorithms,
    plot_cumulative_regret,
    plot_algorithm_comparison,
    plot_change_point_analysis,
)


def create_abrupt_study_config(n_runs: int = 50, change_interval: int = 200) -> ExperimentConfig:
    """Create configuration for abrupt change study."""
    algorithms = [
        # Stationary baselines
        {'name': 'ε-Greedy', 'class': EpsilonGreedy, 'kwargs': {'epsilon': 0.1}},
        {'name': 'UCB1', 'class': UCB1, 'kwargs': {}},
        {'name': 'Thompson Sampling', 'class': ThompsonSampling, 'kwargs': {}},

        # Non-stationary variants
        {'name': 'ε-Greedy (α=0.1)', 'class': EpsilonGreedyConstant,
         'kwargs': {'epsilon': 0.1, 'alpha': 0.1}},
        {'name': 'D-UCB (γ=0.99)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.99}},
        {'name': 'D-UCB (γ=0.95)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.95}},
        {'name': 'SW-UCB (τ=100)', 'class': SlidingWindowUCB, 'kwargs': {'window_size': 100}},
        {'name': 'Discounted-TS', 'class': DiscountedThompsonSampling, 'kwargs': {'gamma': 0.99}},
        {'name': 'EXP3', 'class': EXP3, 'kwargs': {'gamma': 0.1}},
        {'name': 'Rexp3', 'class': Rexp3, 'kwargs': {'restart_interval': 100}},
    ]

    return ExperimentConfig(
        name=f"Abrupt Change Study (interval={change_interval})",
        horizon=10000,
        n_runs=n_runs,
        n_arms=5,
        seed=42,
        env_class=AbruptChangeBandit,
        env_kwargs={'change_interval': change_interval, 'gap': 1.0},
        algorithms=algorithms,
    )


def create_drift_study_config(n_runs: int = 50) -> ExperimentConfig:
    """Create configuration for gradual drift study."""
    algorithms = [
        {'name': 'ε-Greedy', 'class': EpsilonGreedy, 'kwargs': {'epsilon': 0.1}},
        {'name': 'UCB1', 'class': UCB1, 'kwargs': {}},
        {'name': 'Thompson Sampling', 'class': ThompsonSampling, 'kwargs': {}},
        {'name': 'ε-Greedy (α=0.1)', 'class': EpsilonGreedyConstant,
         'kwargs': {'epsilon': 0.1, 'alpha': 0.1}},
        {'name': 'D-UCB (γ=0.99)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.99}},
        {'name': 'SW-UCB (τ=200)', 'class': SlidingWindowUCB, 'kwargs': {'window_size': 200}},
        {'name': 'Discounted-TS', 'class': DiscountedThompsonSampling, 'kwargs': {'gamma': 0.99}},
    ]

    return ExperimentConfig(
        name="Gradual Drift Study",
        horizon=10000,
        n_runs=n_runs,
        n_arms=5,
        seed=42,
        env_class=GradualDriftBandit,
        env_kwargs={'drift_type': 'random_walk', 'drift_rate': 0.05},
        algorithms=algorithms,
    )


def create_stationary_baseline_config(n_runs: int = 50) -> ExperimentConfig:
    """Create configuration for stationary baseline."""
    algorithms = [
        {'name': 'ε-Greedy', 'class': EpsilonGreedy, 'kwargs': {'epsilon': 0.1}},
        {'name': 'UCB1', 'class': UCB1, 'kwargs': {}},
        {'name': 'Thompson Sampling', 'class': ThompsonSampling, 'kwargs': {}},
        {'name': 'Gradient Bandit', 'class': GradientBandit, 'kwargs': {'alpha': 0.1}},
        {'name': 'D-UCB (γ=0.99)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.99}},
        {'name': 'Discounted-TS', 'class': DiscountedThompsonSampling, 'kwargs': {'gamma': 0.99}},
    ]

    return ExperimentConfig(
        name="Stationary Baseline",
        horizon=10000,
        n_runs=n_runs,
        n_arms=5,
        seed=42,
        env_class=StationaryBandit,
        env_kwargs={},
        algorithms=algorithms,
    )


def run_and_save_experiment(
    config: ExperimentConfig,
    save_dir: Path,
    generate_plots: bool = True
) -> None:
    """Run experiment and save results."""
    print(f"\n{'='*60}")
    print(f"Running: {config.name}")
    print(f"{'='*60}")

    # Run experiment
    results = run_experiment_suite(config, progress_bar=True, verbose=True)

    # Create save directory
    exp_dir = save_dir / config.name.replace(' ', '_').replace('=', '').lower()
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results.save(str(exp_dir / 'results.json'))

    # Generate summary
    summary = results.get_summary()
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print("\nResults Summary:")
    print("-" * 60)
    print(f"{'Algorithm':<25} {'Regret':>12} {'Optimal %':>10}")
    print("-" * 60)
    for algo, stats in sorted(summary.items(), key=lambda x: x[1]['mean_regret']):
        print(f"{algo:<25} {stats['mean_regret']:>12.1f} {stats['mean_optimal_pct']*100:>9.1f}%")

    # Generate plots
    if generate_plots:
        try:
            # Regret curves
            fig, ax = plot_cumulative_regret(
                results.cumulative_regret,
                change_points=results.change_points,
                title=f"Cumulative Regret: {config.name}"
            )
            fig.savefig(exp_dir / 'regret_curves.png', dpi=150, bbox_inches='tight')

            # Algorithm comparison bar chart
            metrics_for_plot = {
                name: {
                    'mean': stats['mean_regret'],
                    'ci_lower': stats['mean_regret'] - 1.96 * stats['std_regret'] / np.sqrt(config.n_runs),
                    'ci_upper': stats['mean_regret'] + 1.96 * stats['std_regret'] / np.sqrt(config.n_runs),
                }
                for name, stats in summary.items()
            }
            fig, ax = plot_algorithm_comparison(
                metrics_for_plot,
                metric_name="Final Cumulative Regret",
                title=f"Algorithm Comparison: {config.name}"
            )
            fig.savefig(exp_dir / 'comparison.png', dpi=150, bbox_inches='tight')

            print(f"\nPlots saved to: {exp_dir}")

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run Non-Stationary Bandit Experiments")
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer iterations')
    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--save-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    # Set number of runs
    n_runs = 10 if args.quick else 50

    # Set save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = Path(__file__).parent.parent / 'experiments' / 'results'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("Non-Stationary Bandit Study")
    print("="*60)
    print(f"Number of runs per experiment: {n_runs}")
    print(f"Results directory: {save_dir}")
    print(f"Timestamp: {timestamp}")

    # Define experiments
    experiments = {
        'stationary': lambda: create_stationary_baseline_config(n_runs),
        'abrupt_100': lambda: create_abrupt_study_config(n_runs, change_interval=100),
        'abrupt_200': lambda: create_abrupt_study_config(n_runs, change_interval=200),
        'abrupt_500': lambda: create_abrupt_study_config(n_runs, change_interval=500),
        'drift': lambda: create_drift_study_config(n_runs),
    }

    # Run experiments
    if args.experiment:
        if args.experiment in experiments:
            config = experiments[args.experiment]()
            run_and_save_experiment(config, save_dir, not args.no_plots)
        else:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(experiments.keys())}")
            sys.exit(1)
    else:
        for exp_name, config_fn in experiments.items():
            config = config_fn()
            run_and_save_experiment(config, save_dir, not args.no_plots)

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
