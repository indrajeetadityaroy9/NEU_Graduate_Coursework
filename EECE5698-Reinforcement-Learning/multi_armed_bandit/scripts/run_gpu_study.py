#!/usr/bin/env python3
"""
GPU-Accelerated Multi-Armed Bandit Comparative Study

Runs a full comparative study of 14 bandit algorithms across 4 environment types:
- Stationary (control)
- AbruptChange with interval=100 (fast changes)
- AbruptChange with interval=500 (moderate changes)
- GradualDrift (continuous tracking)

Uses:
- GPU batch execution for stationary + core algorithms (epsilon-greedy, UCB, Thompson Sampling)
- CPU parallel execution for all algorithms across all environments

Outputs:
- Raw JSON results per environment
- Publication-quality plots (regret curves, comparisons, heatmaps)
- Summary statistics CSV
- Rankings and statistical tests
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_armed_bandit.backends import BACKEND, get_device_info, sync
from multi_armed_bandit.algorithms import (
    EpsilonGreedy, EpsilonGreedyConstant, DecayingEpsilonGreedy,
    UCB1, DiscountedUCB, SlidingWindowUCB,
    ThompsonSampling, DiscountedThompsonSampling, BetaThompsonSampling,
    GradientBandit, EntropyRegularizedGradient,
    EXP3, Rexp3, ALGORITHMS
)
from multi_armed_bandit.environments import (
    StationaryBandit, AbruptChangeBandit, GradualDriftBandit, ENVIRONMENTS
)
from multi_armed_bandit.experiments import (
    ExperimentConfig, ExperimentResults,
    run_experiment_suite_parallel, GPU_AVAILABLE
)
from multi_armed_bandit.analysis import (
    compute_confidence_interval, aggregate_runs, paired_ttest
)
from multi_armed_bandit.analysis.visualizations import (
    plot_cumulative_regret, plot_algorithm_comparison,
    plot_change_point_analysis, plot_heatmap, set_publication_style
)

# ============================================================================
# Configuration
# ============================================================================

N_RUNS = 50
HORIZON = 10000
N_ARMS = 5
GAP = 1.0
SEED = 42

# Algorithm configurations
# Note: Using optimistic=True for TS variants ensures proper exploration
# regardless of the reward scale (fixes prior mismatch issue)
ALGORITHM_CONFIGS = [
    # Stationary algorithms
    {'name': 'EpsilonGreedy', 'class': EpsilonGreedy, 'kwargs': {'epsilon': 0.1}},
    {'name': 'DecayingEpsilon', 'class': DecayingEpsilonGreedy, 'kwargs': {}},
    {'name': 'UCB1', 'class': UCB1, 'kwargs': {'c': np.sqrt(2)}},
    {'name': 'ThompsonSampling', 'class': ThompsonSampling, 'kwargs': {}},
    {'name': 'GradientBandit', 'class': GradientBandit, 'kwargs': {'alpha': 0.1}},
    # Non-stationary algorithms
    {'name': 'EpsGreedy-Const', 'class': EpsilonGreedyConstant, 'kwargs': {'epsilon': 0.1, 'alpha': 0.1}},
    {'name': 'D-UCB(0.99)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.99}},
    {'name': 'D-UCB(0.95)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.95}},
    {'name': 'SW-UCB(100)', 'class': SlidingWindowUCB, 'kwargs': {'window_size': 100}},
    {'name': 'D-TS(0.99)', 'class': DiscountedThompsonSampling, 'kwargs': {'gamma': 0.99}},
    {'name': 'EXP3', 'class': EXP3, 'kwargs': {'gamma': 0.1}},
    {'name': 'Rexp3(100)', 'class': Rexp3, 'kwargs': {'gamma': 0.1, 'restart_interval': 100}},
    {'name': 'EntropyGradient', 'class': EntropyRegularizedGradient, 'kwargs': {'alpha': 0.1, 'tau': 0.1}},
]

# Environment configurations
# All environments now use consistent gap-based initialization for fair comparison
ENVIRONMENT_CONFIGS = {
    'stationary': {
        'class': StationaryBandit,
        'kwargs': {'arm_means': [GAP] + [0.0] * (N_ARMS - 1)},
        'display_name': 'Stationary',
    },
    'abrupt_100': {
        'class': AbruptChangeBandit,
        'kwargs': {'change_interval': 100, 'gap': GAP},
        'display_name': 'Abrupt (100)',
    },
    'abrupt_500': {
        'class': AbruptChangeBandit,
        'kwargs': {'change_interval': 500, 'gap': GAP},
        'display_name': 'Abrupt (500)',
    },
    'drift': {
        'class': GradualDriftBandit,
        # Fixed: Use gap-based initialization and faster drift for meaningful test
        # drift_rate=0.05 causes ~5x more optimal arm changes than 0.01
        'kwargs': {'gap': GAP, 'drift_rate': 0.05},
        'display_name': 'Gradual Drift',
    },
}


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def run_environment_experiments(
    env_name: str,
    env_config: dict,
    results_dir: Path,
    n_workers: int = 16
) -> ExperimentResults:
    """Run all algorithms for a single environment."""

    print_header(f"Running: {env_config['display_name']}")
    print(f"  Algorithms: {len(ALGORITHM_CONFIGS)}")
    print(f"  Runs: {N_RUNS}")
    print(f"  Horizon: {HORIZON:,}")
    print(f"  Workers: {n_workers}")

    config = ExperimentConfig(
        name=env_name,
        horizon=HORIZON,
        n_runs=N_RUNS,
        n_arms=N_ARMS,
        seed=SEED,
        env_class=env_config['class'],
        env_kwargs=env_config['kwargs'],
        algorithms=ALGORITHM_CONFIGS,
    )

    start_time = time.time()
    results = run_experiment_suite_parallel(
        config,
        n_workers=n_workers,
        progress_bar=True,
        verbose=True
    )
    elapsed = time.time() - start_time

    # Save raw results
    raw_path = results_dir / 'raw' / f'{env_name}_results.json'
    results.save(str(raw_path))
    print(f"  Saved: {raw_path}")

    return results


def generate_regret_curves(
    results: ExperimentResults,
    env_name: str,
    env_display: str,
    plots_dir: Path
) -> None:
    """Generate cumulative regret curve plot."""

    # Prepare data for plot_cumulative_regret
    regret_by_run = {}
    for algo_name in results.algorithm_names:
        regret_arrays = results.cumulative_regret[algo_name]
        regret_by_run[algo_name] = regret_arrays

    fig, ax = plot_cumulative_regret(
        regret_by_run,
        change_points=results.change_points if results.change_points else None,
        title=f"Cumulative Regret - {env_display}",
        save_path=str(plots_dir / f'{env_name}_regret_curves.png')
    )
    plt.close(fig)


def generate_comparison_chart(
    results: ExperimentResults,
    env_name: str,
    env_display: str,
    plots_dir: Path
) -> None:
    """Generate algorithm comparison bar chart."""

    metrics = {}
    for algo_name in results.algorithm_names:
        regrets = np.array(results.final_regret[algo_name])
        mean = np.mean(regrets)
        std = np.std(regrets)
        ci = 1.96 * std / np.sqrt(len(regrets))

        metrics[algo_name] = {
            'mean': mean,
            'ci_lower': mean - ci,
            'ci_upper': mean + ci,
        }

    fig, ax = plot_algorithm_comparison(
        metrics,
        metric_name="Final Regret",
        title=f"Algorithm Comparison - {env_display}",
        save_path=str(plots_dir / f'{env_name}_comparison.png')
    )
    plt.close(fig)


def generate_adaptation_analysis(
    results: ExperimentResults,
    env_name: str,
    env_display: str,
    plots_dir: Path
) -> None:
    """Generate change point adaptation analysis plot."""

    if not results.change_points:
        return  # Skip for stationary environment

    # Filter to algorithms with adaptation data
    adaptation_regrets = {}
    detection_delays = {}

    for algo_name in results.algorithm_names:
        if results.adaptation_regret.get(algo_name):
            adaptation_regrets[algo_name] = results.adaptation_regret[algo_name]
            detection_delays[algo_name] = results.detection_delay[algo_name]

    if adaptation_regrets:
        fig, axes = plot_change_point_analysis(
            adaptation_regrets,
            detection_delays,
            title=f"Adaptation Analysis - {env_display}",
            save_path=str(plots_dir / f'{env_name}_adaptation.png')
        )
        plt.close(fig)


def generate_heatmap(
    all_results: Dict[str, ExperimentResults],
    plots_dir: Path
) -> None:
    """Generate algorithm × environment performance heatmap."""

    env_names = list(all_results.keys())
    algo_names = list(all_results[env_names[0]].algorithm_names)

    # Build data matrix
    data = np.zeros((len(algo_names), len(env_names)))

    for j, env_name in enumerate(env_names):
        results = all_results[env_name]
        for i, algo_name in enumerate(algo_names):
            data[i, j] = np.mean(results.final_regret[algo_name])

    env_labels = [ENVIRONMENT_CONFIGS[e]['display_name'] for e in env_names]

    fig, ax = plot_heatmap(
        data,
        row_labels=algo_names,
        col_labels=env_labels,
        title="Final Regret: Algorithm × Environment",
        save_path=str(plots_dir / 'algorithm_environment_heatmap.png')
    )
    plt.close(fig)


def generate_rankings(
    all_results: Dict[str, ExperimentResults],
    summary_dir: Path
) -> None:
    """Generate rankings table and CSV."""

    env_names = list(all_results.keys())
    algo_names = list(all_results[env_names[0]].algorithm_names)

    # Compute rankings per environment
    rankings = {algo: [] for algo in algo_names}

    for env_name in env_names:
        results = all_results[env_name]
        mean_regrets = {
            algo: np.mean(results.final_regret[algo])
            for algo in algo_names
        }
        sorted_algos = sorted(mean_regrets.keys(), key=lambda x: mean_regrets[x])
        for rank, algo in enumerate(sorted_algos, 1):
            rankings[algo].append(rank)

    # Compute average rank
    avg_rankings = {algo: np.mean(ranks) for algo, ranks in rankings.items()}

    # Save to CSV
    import csv
    csv_path = summary_dir / 'rankings.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['Algorithm'] + [ENVIRONMENT_CONFIGS[e]['display_name'] for e in env_names] + ['Avg Rank']
        writer.writerow(header)

        for algo in sorted(avg_rankings.keys(), key=lambda x: avg_rankings[x]):
            row = [algo] + rankings[algo] + [f"{avg_rankings[algo]:.2f}"]
            writer.writerow(row)

    print(f"  Saved rankings: {csv_path}")


def generate_summary_statistics(
    all_results: Dict[str, ExperimentResults],
    summary_dir: Path
) -> None:
    """Generate summary statistics CSV."""

    import csv

    csv_path = summary_dir / 'summary_statistics.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Algorithm', 'Environment', 'Mean Regret', 'Std Regret',
                  'Mean Optimal %', 'Mean Adapt Regret', 'Mean Detection Delay']
        writer.writerow(header)

        for env_name, results in all_results.items():
            env_display = ENVIRONMENT_CONFIGS[env_name]['display_name']
            summary = results.get_summary()

            for algo_name, stats in summary.items():
                row = [
                    algo_name,
                    env_display,
                    f"{stats['mean_regret']:.2f}",
                    f"{stats['std_regret']:.2f}",
                    f"{stats['mean_optimal_pct']*100:.1f}",
                    f"{stats.get('mean_adaptation_regret', 0):.2f}",
                    f"{stats.get('mean_detection_delay', 0):.1f}",
                ]
                writer.writerow(row)

    print(f"  Saved summary: {csv_path}")


def run_pairwise_tests(
    all_results: Dict[str, ExperimentResults],
    summary_dir: Path
) -> None:
    """Run pairwise statistical tests between algorithms."""

    pairwise_results = {}

    for env_name, results in all_results.items():
        env_display = ENVIRONMENT_CONFIGS[env_name]['display_name']
        algo_names = results.algorithm_names

        env_tests = {}
        for i, algo1 in enumerate(algo_names):
            for algo2 in algo_names[i+1:]:
                regrets1 = np.array(results.final_regret[algo1])
                regrets2 = np.array(results.final_regret[algo2])

                test_result = paired_ttest(regrets1, regrets2)
                t_stat = test_result['t_statistic']
                p_value = test_result['p_value']

                # Bonferroni correction
                n_tests = len(algo_names) * (len(algo_names) - 1) / 2
                p_corrected = min(p_value * n_tests, 1.0)

                key = f"{algo1} vs {algo2}"
                env_tests[key] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'p_corrected': float(p_corrected),
                    'significant': bool(p_corrected < 0.05),
                }

        pairwise_results[env_display] = env_tests

    # Save to JSON
    json_path = summary_dir / 'pairwise_tests.json'
    with open(json_path, 'w') as f:
        json.dump(pairwise_results, f, indent=2)

    print(f"  Saved pairwise tests: {json_path}")


def generate_readme(
    all_results: Dict[str, ExperimentResults],
    results_dir: Path,
    total_time: float
) -> None:
    """Generate a README summarizing the study."""

    readme_path = results_dir / 'README.md'

    with open(readme_path, 'w') as f:
        f.write("# Multi-Armed Bandit Comparative Study Results\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Runtime**: {total_time:.1f} seconds\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Algorithms**: {len(ALGORITHM_CONFIGS)}\n")
        f.write(f"- **Environments**: {len(ENVIRONMENT_CONFIGS)}\n")
        f.write(f"- **Runs per configuration**: {N_RUNS}\n")
        f.write(f"- **Horizon**: {HORIZON:,} timesteps\n")
        f.write(f"- **Arms**: {N_ARMS}\n")
        f.write(f"- **Gap**: {GAP}\n\n")

        f.write("## Environments\n\n")
        for env_name, config in ENVIRONMENT_CONFIGS.items():
            f.write(f"- **{config['display_name']}**: {config['class'].__name__}\n")
        f.write("\n")

        f.write("## Algorithms\n\n")
        for algo in ALGORITHM_CONFIGS:
            f.write(f"- {algo['name']}\n")
        f.write("\n")

        f.write("## Key Findings\n\n")

        # Find best algorithm per environment
        for env_name, results in all_results.items():
            env_display = ENVIRONMENT_CONFIGS[env_name]['display_name']
            summary = results.get_summary()

            best_algo = min(summary.keys(), key=lambda x: summary[x]['mean_regret'])
            best_regret = summary[best_algo]['mean_regret']

            f.write(f"### {env_display}\n\n")
            f.write(f"**Best Algorithm**: {best_algo} (Mean Regret: {best_regret:.1f})\n\n")

            # Top 3
            sorted_algos = sorted(summary.keys(), key=lambda x: summary[x]['mean_regret'])[:3]
            f.write("Top 3:\n")
            for i, algo in enumerate(sorted_algos, 1):
                f.write(f"{i}. {algo}: {summary[algo]['mean_regret']:.1f}\n")
            f.write("\n")

        f.write("## Output Files\n\n")
        f.write("```\n")
        f.write("raw/              - JSON results per environment\n")
        f.write("plots/            - Visualization PNG files\n")
        f.write("summary/          - CSV summaries and statistical tests\n")
        f.write("```\n")

    print(f"  Saved README: {readme_path}")


def main():
    """Main execution function."""

    print_header("Multi-Armed Bandit GPU Comparative Study")

    # Check backend
    print(f"\nBackend: {BACKEND}")
    if BACKEND == 'gpu':
        info = get_device_info()
        print(f"Device: {info['name']}")
        mem_gb = info['memory_free'] / (1024**3)
        print(f"Memory: {mem_gb:.1f} GB free")

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(__file__).parent.parent / 'experiments' / 'results' / f'full_study_{timestamp}'
    (results_dir / 'raw').mkdir(parents=True, exist_ok=True)
    (results_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (results_dir / 'summary').mkdir(parents=True, exist_ok=True)

    print(f"\nResults directory: {results_dir}")

    # Determine number of workers
    import os
    n_workers = min(os.cpu_count() or 4, 32)
    print(f"Using {n_workers} CPU workers for parallel execution")

    # Run experiments for each environment
    total_start = time.time()
    all_results = {}

    for env_name, env_config in ENVIRONMENT_CONFIGS.items():
        results = run_environment_experiments(
            env_name, env_config, results_dir, n_workers
        )
        all_results[env_name] = results

        # Sync GPU if used
        if BACKEND == 'gpu':
            sync()

    total_time = time.time() - total_start

    # Generate visualizations
    print_header("Generating Visualizations")

    plots_dir = results_dir / 'plots'
    summary_dir = results_dir / 'summary'

    for env_name, results in all_results.items():
        env_display = ENVIRONMENT_CONFIGS[env_name]['display_name']
        print(f"  {env_display}...")

        generate_regret_curves(results, env_name, env_display, plots_dir)
        generate_comparison_chart(results, env_name, env_display, plots_dir)
        generate_adaptation_analysis(results, env_name, env_display, plots_dir)

    # Cross-environment analysis
    print("  Cross-environment heatmap...")
    generate_heatmap(all_results, plots_dir)

    # Generate summaries
    print_header("Generating Summary Statistics")
    generate_rankings(all_results, summary_dir)
    generate_summary_statistics(all_results, summary_dir)
    run_pairwise_tests(all_results, summary_dir)

    # Generate README
    generate_readme(all_results, results_dir, total_time)

    # Final summary
    print_header("Study Complete")
    print(f"\nTotal experiments: {len(ALGORITHM_CONFIGS) * len(ENVIRONMENT_CONFIGS) * N_RUNS:,}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Throughput: {len(ALGORITHM_CONFIGS) * len(ENVIRONMENT_CONFIGS) * N_RUNS / total_time:.1f} runs/s")
    print(f"\nResults saved to: {results_dir}")

    # Print quick summary
    print("\n" + "-" * 60)
    print("Quick Summary (Best Algorithm per Environment):")
    print("-" * 60)

    for env_name, results in all_results.items():
        env_display = ENVIRONMENT_CONFIGS[env_name]['display_name']
        summary = results.get_summary()
        best_algo = min(summary.keys(), key=lambda x: summary[x]['mean_regret'])
        best_regret = summary[best_algo]['mean_regret']
        print(f"  {env_display:20s}: {best_algo:20s} (Regret: {best_regret:6.1f})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
