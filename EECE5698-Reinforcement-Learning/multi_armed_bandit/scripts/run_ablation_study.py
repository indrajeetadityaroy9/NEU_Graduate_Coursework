#!/usr/bin/env python3
"""
Ablation Study: Window Size and Discount Factor Sensitivity

Implements the strategic recommendations for research-grade analysis:
1. Regret-over-time plots showing dynamics at change points
2. Window size ablation for SW-UCB [50, 100, 200, 500]
3. Discount factor ablation for D-UCB [0.9, 0.95, 0.99, 0.999]

Usage:
    python -m multi_armed_bandit.scripts.run_ablation_study
"""

import sys
from pathlib import Path
from datetime import datetime
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_armed_bandit.config import N_ARMS, GAP, SEED, create_algorithm
from multi_armed_bandit.algorithms import (
    UCB1,
    ThompsonSampling,
    SlidingWindowUCB,
    DiscountedUCB,
    Rexp3,
)
from multi_armed_bandit.benchmarks.supervised_to_bandit import SyntheticDriftBandit


def run_single_experiment(algo, env, horizon: int) -> dict:
    """Run single experiment and track regret over time."""
    algo.reset()
    env.reset()

    cumulative_regret = 0.0
    regret_history = []
    optimal_history = []
    action_history = []

    for t in range(horizon):
        action = algo.select_action()
        reward, info = env.step(action)
        algo.update(action, reward)

        # Compute instantaneous regret
        arm_means = info['arm_means']
        instant_regret = arm_means.max() - arm_means[action]
        cumulative_regret += instant_regret

        regret_history.append(cumulative_regret)
        optimal_history.append(1 if action == info['optimal_arm'] else 0)
        action_history.append(action)

    return {
        'final_regret': cumulative_regret,
        'regret_history': regret_history,
        'optimal_rate': np.mean(optimal_history),
        'optimal_history': optimal_history,
    }


def run_regret_dynamics_study(
    horizon: int = 5000,
    drift_interval: int = 500,
    n_runs: int = 10,
    seed: int = SEED,
):
    """
    Generate regret-over-time plots showing dynamics at change points.
    """
    print("\n" + "=" * 60)
    print("STUDY 1: Regret Dynamics at Change Points")
    print("=" * 60)

    n_arms = N_ARMS
    algorithms = {
        'ThompsonSampling': lambda s: ThompsonSampling(n_arms=n_arms, seed=s),
        'UCB1': lambda s: UCB1(n_arms=n_arms, c=np.sqrt(2), seed=s),
        'SW-UCB(100)': lambda s: SlidingWindowUCB(n_arms=n_arms, window_size=100, seed=s),
        'D-UCB(0.99)': lambda s: DiscountedUCB(n_arms=n_arms, gamma=0.99, seed=s),
        'Rexp3(100)': lambda s: Rexp3(n_arms=n_arms, restart_interval=100, seed=s),
    }

    results = {name: {'regret_runs': [], 'optimal_runs': []} for name in algorithms}

    for run in range(n_runs):
        run_seed = seed + run

        for name, algo_fn in algorithms.items():
            algo = algo_fn(run_seed)

            # Sudden drift environment
            env = SyntheticDriftBandit(
                n_arms=n_arms,
                gap=GAP,
                drift_type='sudden',
                drift_interval=drift_interval,
                noise_std=1.0,
                seed=run_seed,
            )

            result = run_single_experiment(algo, env, horizon)
            results[name]['regret_runs'].append(result['regret_history'])
            results[name]['optimal_runs'].append(result['optimal_history'])

        if (run + 1) % 5 == 0:
            print(f"  Completed {run + 1}/{n_runs} runs")

    # Compute mean and std
    for name in results:
        regret_array = np.array(results[name]['regret_runs'])
        results[name]['mean_regret'] = regret_array.mean(axis=0)
        results[name]['std_regret'] = regret_array.std(axis=0)

        optimal_array = np.array(results[name]['optimal_runs'])
        # Rolling window for optimal rate
        window = 50
        rolling_optimal = np.apply_along_axis(
            lambda x: np.convolve(x, np.ones(window)/window, mode='valid'),
            axis=1, arr=optimal_array
        )
        results[name]['mean_optimal'] = rolling_optimal.mean(axis=0)

    return results, drift_interval


def run_window_ablation(
    horizon: int = 10000,
    drift_interval: int = 500,
    n_runs: int = 20,
    seed: int = SEED,
):
    """
    Window size ablation for SW-UCB: [50, 100, 200, 500]
    """
    print("\n" + "=" * 60)
    print("STUDY 2: Window Size Ablation (SW-UCB)")
    print("=" * 60)

    n_arms = N_ARMS
    window_sizes = [50, 100, 200, 500]

    results = {
        'gradual': {ws: [] for ws in window_sizes},
        'sudden': {ws: [] for ws in window_sizes},
    }

    for drift_type in ['gradual', 'sudden']:
        print(f"\n  {drift_type.upper()} DRIFT:")

        for ws in window_sizes:
            optimal_rates = []

            for run in range(n_runs):
                run_seed = seed + run

                algo = SlidingWindowUCB(n_arms=n_arms, window_size=ws, seed=run_seed)
                env = SyntheticDriftBandit(
                    n_arms=n_arms,
                    gap=GAP,
                    drift_type=drift_type,
                    drift_interval=drift_interval,
                    noise_std=1.0,
                    seed=run_seed,
                )

                result = run_single_experiment(algo, env, horizon)
                optimal_rates.append(result['optimal_rate'])

            mean_rate = np.mean(optimal_rates)
            std_rate = np.std(optimal_rates)
            results[drift_type][ws] = {'mean': mean_rate, 'std': std_rate, 'runs': optimal_rates}
            print(f"    Window={ws:3d}: {mean_rate:.1%} ± {std_rate:.1%}")

    return results


def run_discount_ablation(
    horizon: int = 10000,
    drift_interval: int = 500,
    n_runs: int = 20,
    seed: int = SEED,
):
    """
    Discount factor ablation for D-UCB: [0.9, 0.95, 0.99, 0.999]
    """
    print("\n" + "=" * 60)
    print("STUDY 3: Discount Factor Ablation (D-UCB)")
    print("=" * 60)

    n_arms = N_ARMS
    gammas = [0.9, 0.95, 0.99, 0.999]

    # Compute effective memory for each gamma
    print("\n  Effective Memory (1/(1-γ)):")
    for g in gammas:
        print(f"    γ={g}: ~{1/(1-g):.0f} steps")

    results = {
        'gradual': {g: [] for g in gammas},
        'sudden': {g: [] for g in gammas},
    }

    for drift_type in ['gradual', 'sudden']:
        print(f"\n  {drift_type.upper()} DRIFT:")

        for gamma in gammas:
            optimal_rates = []

            for run in range(n_runs):
                run_seed = seed + run

                algo = DiscountedUCB(n_arms=n_arms, gamma=gamma, seed=run_seed)
                env = SyntheticDriftBandit(
                    n_arms=n_arms,
                    gap=GAP,
                    drift_type=drift_type,
                    drift_interval=drift_interval,
                    noise_std=1.0,
                    seed=run_seed,
                )

                result = run_single_experiment(algo, env, horizon)
                optimal_rates.append(result['optimal_rate'])

            mean_rate = np.mean(optimal_rates)
            std_rate = np.std(optimal_rates)
            results[drift_type][gamma] = {'mean': mean_rate, 'std': std_rate}
            eff_mem = 1 / (1 - gamma)
            print(f"    γ={gamma} (mem≈{eff_mem:.0f}): {mean_rate:.1%} ± {std_rate:.1%}")

    return results


def main():
    print("=" * 60)
    print("Ablation Study: Hyperparameter Sensitivity Analysis")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create results directory
    results_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'ablation'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Study 1: Regret Dynamics
    regret_results, drift_interval = run_regret_dynamics_study(
        horizon=5000,
        drift_interval=500,
        n_runs=10,
    )

    # Study 2: Window Size Ablation
    window_results = run_window_ablation(
        horizon=10000,
        drift_interval=500,
        n_runs=20,
    )

    # Study 3: Discount Factor Ablation
    discount_results = run_discount_ablation(
        horizon=10000,
        drift_interval=500,
        n_runs=20,
    )

    # Summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    print("\n1. WINDOW SIZE (SW-UCB):")
    print("   Optimal window depends on drift type:")
    best_gradual_ws = max(window_results['gradual'].items(), key=lambda x: x[1]['mean'])
    best_sudden_ws = max(window_results['sudden'].items(), key=lambda x: x[1]['mean'])
    print(f"   - Gradual: τ={best_gradual_ws[0]} ({best_gradual_ws[1]['mean']:.1%})")
    print(f"   - Sudden: τ={best_sudden_ws[0]} ({best_sudden_ws[1]['mean']:.1%})")

    print("\n2. DISCOUNT FACTOR (D-UCB):")
    print("   Effective memory should match change interval:")
    best_gradual_g = max(discount_results['gradual'].items(), key=lambda x: x[1]['mean'])
    best_sudden_g = max(discount_results['sudden'].items(), key=lambda x: x[1]['mean'])
    print(f"   - Gradual: γ={best_gradual_g[0]} ({best_gradual_g[1]['mean']:.1%})")
    print(f"   - Sudden: γ={best_sudden_g[0]} ({best_sudden_g[1]['mean']:.1%})")

    print("\n3. STABILITY-PLASTICITY TRADEOFF:")
    print("   - Too small window/discount → over-forgetting (high variance)")
    print("   - Too large window/discount → under-forgetting (slow adaptation)")

    # Save JSON results
    all_results = {
        'window_ablation': {
            drift: {str(k): v for k, v in results.items()}
            for drift, results in window_results.items()
        },
        'discount_ablation': {
            drift: {str(k): v for k, v in results.items()}
            for drift, results in discount_results.items()
        },
    }

    # Convert numpy to native Python for JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(results_dir / 'ablation_results.json', 'w') as f:
        json.dump(convert(all_results), f, indent=2)

    print(f"\n\nResults saved to: {results_dir}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
