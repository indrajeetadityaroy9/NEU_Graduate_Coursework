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

from multi_armed_bandit.experiments import (
    ExperimentConfig,
    run_experiment_suite,
)
from multi_armed_bandit.config import (
    N_RUNS, N_RUNS_QUICK, HORIZON, N_ARMS, GAP, SEED,
    get_algorithm_configs,
    get_environment_config,
)


def create_experiment_config(
    name: str,
    env_name: str,
    n_runs: int,
    **env_overrides
) -> ExperimentConfig:
    """Create an experiment configuration using shared configs."""
    env_config = get_environment_config(env_name, **env_overrides)
    algorithms = get_algorithm_configs(include_stationary=True, include_nonstationary=True)

    return ExperimentConfig(
        name=name,
        horizon=HORIZON,
        n_runs=n_runs,
        n_arms=N_ARMS,
        seed=SEED,
        env_class=env_config['class'],
        env_kwargs=env_config['kwargs'],
        algorithms=algorithms,
    )


def run_and_save_experiment(
    config: ExperimentConfig,
    save_dir: Path,
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

    print(f"\nResults saved to: {exp_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Non-Stationary Bandit Experiments")
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer iterations')
    parser.add_argument('--experiment', type=str, help='Run specific experiment')
    parser.add_argument('--save-dir', type=str, default=None, help='Results directory')
    args = parser.parse_args()

    # Set number of runs
    n_runs = N_RUNS_QUICK if args.quick else N_RUNS

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

    # Define experiments using shared config
    experiments = {
        'stationary': lambda: create_experiment_config(
            "Stationary Baseline", 'stationary', n_runs),
        'abrupt_100': lambda: create_experiment_config(
            "Abrupt Change Study (interval=100)", 'abrupt_100', n_runs),
        'abrupt_200': lambda: create_experiment_config(
            "Abrupt Change Study (interval=200)", 'abrupt_200', n_runs),
        'abrupt_500': lambda: create_experiment_config(
            "Abrupt Change Study (interval=500)", 'abrupt_500', n_runs),
        'drift': lambda: create_experiment_config(
            "Gradual Drift Study", 'drift', n_runs),
    }

    # Run experiments
    if args.experiment:
        if args.experiment in experiments:
            config = experiments[args.experiment]()
            run_and_save_experiment(config, save_dir)
        else:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available: {list(experiments.keys())}")
            sys.exit(1)
    else:
        for exp_name, config_fn in experiments.items():
            config = config_fn()
            run_and_save_experiment(config, save_dir)

    print("\n" + "="*60)
    print("All experiments completed!")
    print("="*60)


if __name__ == '__main__':
    main()
