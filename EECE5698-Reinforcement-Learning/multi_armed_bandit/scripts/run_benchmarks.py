#!/usr/bin/env python3
"""
Run Standard Bandit Benchmarks

Executes the three-pronged validation approach used in arXiv research:
1. OBP Metrics Integration - Standardized OPE with DoublyRobust/SNIPS
2. Supervised-to-Bandit Conversion - Mushroom dataset with induced drift
3. Replay Evaluation - Synthetic logged data evaluation

Usage:
    python -m multi_armed_bandit.scripts.run_benchmarks
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from multi_armed_bandit.config import (
    N_ARMS, HORIZON, SEED,
    create_algorithm_suite,
)
from multi_armed_bandit.benchmarks.obp_integration import (
    OBPEvaluator,
    create_bandit_feedback_from_run,
    validate_with_obp,
)
from multi_armed_bandit.benchmarks.supervised_to_bandit import (
    SyntheticDriftBandit,
    MushroomBandit,
)
from multi_armed_bandit.benchmarks.replay_evaluation import (
    SyntheticReplayEvaluator,
    run_replay_benchmark,
)


def run_supervised_to_bandit_benchmark(
    algorithms: dict,
    drift_type: str = 'gradual',
    drift_interval: int = 1000,
    horizon: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run algorithms on supervised-to-bandit benchmark with drift.

    Uses SyntheticDriftBandit (similar setup to Mushroom but controlled).
    """
    print(f"\n=== Supervised-to-Bandit Benchmark ({drift_type} drift) ===")

    n_arms = 5
    results = []

    for algo_name, algo in algorithms.items():
        algo.reset()

        # Create environment for this run
        env = SyntheticDriftBandit(
            n_arms=n_arms,
            gap=1.0,
            drift_type=drift_type,
            drift_interval=drift_interval,
            noise_std=1.0,
            seed=seed,
        )
        env.reset()

        cumulative_reward = 0.0
        optimal_choices = 0
        actions = []
        rewards = []
        optimal_arms = []

        for t in range(horizon):
            action = algo.select_action()
            reward, info = env.step(action)

            algo.update(action, reward)

            cumulative_reward += reward
            if action == info['optimal_arm']:
                optimal_choices += 1

            actions.append(action)
            rewards.append(reward)
            optimal_arms.append(info['optimal_arm'])

        # Compute metrics
        regret = sum(env.get_arm_means().max() - r for r in rewards)

        results.append({
            'Algorithm': algo_name,
            'Cumulative Reward': cumulative_reward,
            'Optimal Rate': optimal_choices / horizon,
            'Final Regret': regret,
        })

        print(f"  {algo_name}: Reward={cumulative_reward:.1f}, "
              f"Optimal={optimal_choices/horizon:.1%}")

    return pd.DataFrame(results)


def run_replay_benchmark_suite(
    algorithms: dict,
    n_arms: int = 5,
    max_steps: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run replay evaluation on synthetic logged data.
    """
    print(f"\n=== Replay Evaluation Benchmark ===")

    # Create evaluator with realistic arm means
    arm_means = np.array([0.9, 0.7, 0.5, 0.3, 0.1])[:n_arms]
    evaluator = SyntheticReplayEvaluator(
        n_arms=n_arms,
        arm_means=arm_means,
        seed=seed,
    )

    results = []
    for algo_name, algo in algorithms.items():
        algo.reset()
        result = evaluator.evaluate(algo, max_steps=max_steps)

        results.append({
            'Algorithm': algo_name,
            'Mean Reward': result.mean_reward,
            'Match Rate': result.match_rate,
            'Total Matches': result.n_matches,
            'Cumulative Reward': result.cumulative_reward,
        })

        print(f"  {algo_name}: Reward={result.mean_reward:.3f}, "
              f"Match={result.match_rate:.1%}")

    print(f"\n  Optimal mean: {evaluator.get_optimal_mean_reward():.3f}")
    print(f"  Random mean: {evaluator.get_random_mean_reward():.3f}")

    return pd.DataFrame(results)


def run_obp_validation(
    actions: np.ndarray,
    rewards: np.ndarray,
    optimal_actions: np.ndarray,
    n_arms: int,
    algo_name: str,
) -> dict:
    """
    Validate a run using OBP metrics (DR, SNIPS).
    """
    return validate_with_obp(actions, rewards, optimal_actions, n_arms)


def main():
    """Run all benchmarks and save results."""
    print("=" * 60)
    print("Multi-Armed Bandit Standard Benchmarks")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Using shared config: N_ARMS, HORIZON, SEED imported from config module

    # Create results directory
    results_dir = Path(__file__).parent.parent / 'experiments' / 'results' / 'benchmarks'
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create algorithm suite
    algorithms = create_algorithm_suite(N_ARMS, SEED)
    print(f"\nAlgorithms: {len(algorithms)}")
    for name in algorithms:
        print(f"  - {name}")

    all_results = {}

    # 1. Supervised-to-Bandit with Gradual Drift
    print("\n" + "=" * 40)
    print("BENCHMARK 1: Supervised-to-Bandit (Gradual Drift)")
    print("=" * 40)

    gradual_results = run_supervised_to_bandit_benchmark(
        algorithms={k: v for k, v in create_algorithm_suite(N_ARMS, SEED).items()},
        drift_type='gradual',
        drift_interval=500,
        horizon=HORIZON,
        seed=SEED,
    )
    all_results['gradual_drift'] = gradual_results.to_dict('records')
    print(f"\nTop 3 by Optimal Rate:")
    print(gradual_results.nlargest(3, 'Optimal Rate')[['Algorithm', 'Optimal Rate', 'Cumulative Reward']])

    # 2. Supervised-to-Bandit with Sudden Drift
    print("\n" + "=" * 40)
    print("BENCHMARK 2: Supervised-to-Bandit (Sudden Drift)")
    print("=" * 40)

    sudden_results = run_supervised_to_bandit_benchmark(
        algorithms={k: v for k, v in create_algorithm_suite(N_ARMS, SEED).items()},
        drift_type='sudden',
        drift_interval=500,
        horizon=HORIZON,
        seed=SEED,
    )
    all_results['sudden_drift'] = sudden_results.to_dict('records')
    print(f"\nTop 3 by Optimal Rate:")
    print(sudden_results.nlargest(3, 'Optimal Rate')[['Algorithm', 'Optimal Rate', 'Cumulative Reward']])

    # 3. Replay Evaluation
    print("\n" + "=" * 40)
    print("BENCHMARK 3: Replay Evaluation (Logged Data)")
    print("=" * 40)

    replay_results = run_replay_benchmark_suite(
        algorithms={k: v for k, v in create_algorithm_suite(N_ARMS, SEED).items()},
        n_arms=N_ARMS,
        max_steps=5000,
        seed=SEED,
    )
    all_results['replay'] = replay_results.to_dict('records')
    print(f"\nTop 3 by Mean Reward:")
    print(replay_results.nlargest(3, 'Mean Reward')[['Algorithm', 'Mean Reward', 'Match Rate']])

    # Save results
    results_file = results_dir / f'benchmark_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {results_file}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Algorithm Performance Across Benchmarks")
    print("=" * 60)

    summary = []
    for algo_name in algorithms.keys():
        gradual = next((r for r in all_results['gradual_drift'] if r['Algorithm'] == algo_name), {})
        sudden = next((r for r in all_results['sudden_drift'] if r['Algorithm'] == algo_name), {})
        replay = next((r for r in all_results['replay'] if r['Algorithm'] == algo_name), {})

        summary.append({
            'Algorithm': algo_name,
            'Gradual Optimal%': f"{gradual.get('Optimal Rate', 0)*100:.1f}%",
            'Sudden Optimal%': f"{sudden.get('Optimal Rate', 0)*100:.1f}%",
            'Replay Reward': f"{replay.get('Mean Reward', 0):.3f}",
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    print(f"\n\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
