"""
GPU Batch Runner for Multi-Armed Bandit Experiments.

Provides GPU-accelerated experiment execution for running multiple
independent experiments in parallel on NVIDIA GPUs.

This module requires CuPy and is designed for the H100 GPUs available
in the Lambda Cloud environment.
"""

from typing import Dict, List, Optional, Tuple, Type, Any
from dataclasses import dataclass
import numpy as np
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class GPUBatchConfig:
    """Configuration for GPU batch experiments."""
    n_runs: int
    n_arms: int
    horizon: int
    device: int = 0
    seed: int = 42


class GPUBatchRunner:
    """
    Run multiple bandit experiments in parallel on GPU.

    This runner executes multiple independent runs simultaneously,
    leveraging GPU parallelism for significant speedup.

    Parameters
    ----------
    config : GPUBatchConfig
        Configuration specifying n_runs, n_arms, horizon, etc.

    Examples
    --------
    >>> config = GPUBatchConfig(n_runs=100, n_arms=5, horizon=10000)
    >>> runner = GPUBatchRunner(config)
    >>> results = runner.run_epsilon_greedy(epsilon=0.1, alpha=0.1, arm_means=[0,1,0,0,0])
    """

    def __init__(self, config: GPUBatchConfig):
        if not GPU_AVAILABLE:
            raise RuntimeError("CuPy not available. Install with: pip install cupy-cuda12x")

        self.config = config
        self.device = config.device

        with cp.cuda.Device(self.device):
            # Pre-allocate GPU arrays for batch execution
            self._q_values = cp.zeros(
                (config.n_runs, config.n_arms),
                dtype=cp.float32
            )
            self._counts = cp.zeros(
                (config.n_runs, config.n_arms),
                dtype=cp.int32
            )
            self._rewards = cp.zeros(
                (config.n_runs, config.horizon),
                dtype=cp.float32
            )
            self._actions = cp.zeros(
                (config.n_runs, config.horizon),
                dtype=cp.int32
            )
            self._optimal_arms = cp.zeros(
                (config.n_runs, config.horizon),
                dtype=cp.int32
            )

            # Initialize RNG
            cp.random.seed(config.seed)

    def run_epsilon_greedy(
        self,
        epsilon: float,
        alpha: float,
        arm_means: np.ndarray,
        arm_stds: Optional[np.ndarray] = None,
        initial_value: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Run epsilon-greedy algorithm for all runs in parallel.

        Parameters
        ----------
        epsilon : float
            Exploration probability
        alpha : float
            Learning rate (constant step size)
        arm_means : np.ndarray
            True mean reward for each arm
        arm_stds : np.ndarray, optional
            Standard deviation of rewards (default: 1.0)
        initial_value : float
            Initial Q-value estimate

        Returns
        -------
        dict
            Dictionary with 'rewards', 'actions', 'cumulative_regret'
        """
        n_runs = self.config.n_runs
        n_arms = self.config.n_arms
        horizon = self.config.horizon

        with cp.cuda.Device(self.device):
            # Copy arm parameters to GPU
            arm_means_gpu = cp.asarray(arm_means, dtype=cp.float32)
            if arm_stds is None:
                arm_stds_gpu = cp.ones(n_arms, dtype=cp.float32)
            else:
                arm_stds_gpu = cp.asarray(arm_stds, dtype=cp.float32)

            # Initialize Q-values
            self._q_values.fill(initial_value)
            self._counts.fill(0)

            # Optimal arm (highest mean)
            optimal_arm = int(cp.argmax(arm_means_gpu))
            optimal_value = float(arm_means_gpu[optimal_arm])

            for t in range(horizon):
                # Batch action selection
                explore_mask = cp.random.random(n_runs) < epsilon

                # Greedy actions (argmax of Q-values)
                greedy_actions = cp.argmax(self._q_values, axis=1)

                # Random actions for exploration
                random_actions = cp.random.randint(0, n_arms, size=n_runs)

                # Combine: explore where mask is True, exploit otherwise
                actions = cp.where(explore_mask, random_actions, greedy_actions)

                # Batch reward sampling
                # Get means and stds for selected actions
                selected_means = arm_means_gpu[actions]
                selected_stds = arm_stds_gpu[actions]
                rewards = cp.random.normal(selected_means, selected_stds)

                # Batch Q-value update
                # Q[run, action] += alpha * (reward - Q[run, action])
                run_indices = cp.arange(n_runs)
                current_q = self._q_values[run_indices, actions]
                self._q_values[run_indices, actions] = (
                    current_q + alpha * (rewards - current_q)
                )
                self._counts[run_indices, actions] += 1

                # Record
                self._actions[:, t] = actions
                self._rewards[:, t] = rewards
                self._optimal_arms[:, t] = optimal_arm

            # Compute cumulative regret
            regret = optimal_value - self._rewards
            cumulative_regret = cp.cumsum(regret, axis=1)

            # Transfer results to CPU
            return {
                'rewards': cp.asnumpy(self._rewards),
                'actions': cp.asnumpy(self._actions),
                'cumulative_regret': cp.asnumpy(cumulative_regret),
                'final_regret': cp.asnumpy(cumulative_regret[:, -1]),
                'optimal_percentage': cp.asnumpy(
                    cp.mean(self._actions == optimal_arm, axis=1)
                ),
            }

    def run_ucb(
        self,
        c: float,
        arm_means: np.ndarray,
        arm_stds: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run UCB1 algorithm for all runs in parallel.

        Parameters
        ----------
        c : float
            Exploration coefficient (typically sqrt(2))
        arm_means : np.ndarray
            True mean reward for each arm
        arm_stds : np.ndarray, optional
            Standard deviation of rewards

        Returns
        -------
        dict
            Dictionary with experiment results
        """
        n_runs = self.config.n_runs
        n_arms = self.config.n_arms
        horizon = self.config.horizon

        with cp.cuda.Device(self.device):
            arm_means_gpu = cp.asarray(arm_means, dtype=cp.float32)
            if arm_stds is None:
                arm_stds_gpu = cp.ones(n_arms, dtype=cp.float32)
            else:
                arm_stds_gpu = cp.asarray(arm_stds, dtype=cp.float32)

            # Initialize
            self._q_values.fill(0)
            self._counts.fill(0)

            optimal_arm = int(cp.argmax(arm_means_gpu))
            optimal_value = float(arm_means_gpu[optimal_arm])

            for t in range(horizon):
                # UCB values: Q + c * sqrt(ln(t) / N)
                # Handle unvisited arms (count=0) by giving them high UCB
                t_val = t + 1
                log_t = cp.log(cp.float32(t_val))

                # Avoid division by zero
                safe_counts = cp.maximum(self._counts, 1)
                exploration_bonus = c * cp.sqrt(log_t / safe_counts)

                ucb_values = self._q_values + exploration_bonus

                # Force selection of unvisited arms first
                unvisited_mask = self._counts == 0
                ucb_values = cp.where(unvisited_mask, cp.float32(1e10), ucb_values)

                # Select actions with highest UCB
                actions = cp.argmax(ucb_values, axis=1)

                # Sample rewards
                selected_means = arm_means_gpu[actions]
                selected_stds = arm_stds_gpu[actions]
                rewards = cp.random.normal(selected_means, selected_stds)

                # Update Q-values (sample average)
                run_indices = cp.arange(n_runs)
                self._counts[run_indices, actions] += 1
                n = self._counts[run_indices, actions]
                current_q = self._q_values[run_indices, actions]
                self._q_values[run_indices, actions] = (
                    current_q + (rewards - current_q) / n
                )

                # Record
                self._actions[:, t] = actions
                self._rewards[:, t] = rewards
                self._optimal_arms[:, t] = optimal_arm

            # Compute metrics
            regret = optimal_value - self._rewards
            cumulative_regret = cp.cumsum(regret, axis=1)

            return {
                'rewards': cp.asnumpy(self._rewards),
                'actions': cp.asnumpy(self._actions),
                'cumulative_regret': cp.asnumpy(cumulative_regret),
                'final_regret': cp.asnumpy(cumulative_regret[:, -1]),
                'optimal_percentage': cp.asnumpy(
                    cp.mean(self._actions == optimal_arm, axis=1)
                ),
            }

    def run_thompson_sampling(
        self,
        arm_means: np.ndarray,
        arm_stds: Optional[np.ndarray] = None,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        reward_std: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Run Thompson Sampling for all runs in parallel.

        Parameters
        ----------
        arm_means : np.ndarray
            True mean reward for each arm
        arm_stds : np.ndarray, optional
            Standard deviation of rewards
        prior_mean : float
            Prior mean for all arms
        prior_std : float
            Prior standard deviation
        reward_std : float
            Assumed reward standard deviation

        Returns
        -------
        dict
            Dictionary with experiment results
        """
        n_runs = self.config.n_runs
        n_arms = self.config.n_arms
        horizon = self.config.horizon

        with cp.cuda.Device(self.device):
            arm_means_gpu = cp.asarray(arm_means, dtype=cp.float32)
            if arm_stds is None:
                arm_stds_gpu = cp.ones(n_arms, dtype=cp.float32)
            else:
                arm_stds_gpu = cp.asarray(arm_stds, dtype=cp.float32)

            # Posterior sufficient statistics
            sum_rewards = cp.zeros((n_runs, n_arms), dtype=cp.float32)
            counts = cp.zeros((n_runs, n_arms), dtype=cp.float32)

            prior_precision = 1.0 / (prior_std ** 2)
            reward_precision = 1.0 / (reward_std ** 2)

            optimal_arm = int(cp.argmax(arm_means_gpu))
            optimal_value = float(arm_means_gpu[optimal_arm])

            for t in range(horizon):
                # Compute posterior parameters for all arms
                # Posterior precision = prior_precision + n * reward_precision
                posterior_precision = prior_precision + counts * reward_precision

                # Posterior mean
                sample_means = cp.where(
                    counts > 0,
                    sum_rewards / cp.maximum(counts, 1e-10),
                    0.0
                )
                posterior_means = cp.where(
                    counts > 0,
                    (prior_precision * prior_mean +
                     counts * reward_precision * sample_means) / posterior_precision,
                    prior_mean
                )
                posterior_stds = 1.0 / cp.sqrt(posterior_precision)

                # Sample from posteriors: shape (n_runs, n_arms)
                samples = cp.random.normal(posterior_means, posterior_stds)

                # Select actions with highest sample
                actions = cp.argmax(samples, axis=1)

                # Sample rewards
                selected_means = arm_means_gpu[actions]
                selected_stds = arm_stds_gpu[actions]
                rewards = cp.random.normal(selected_means, selected_stds)

                # Update sufficient statistics
                run_indices = cp.arange(n_runs)
                counts[run_indices, actions] += 1
                sum_rewards[run_indices, actions] += rewards

                # Record
                self._actions[:, t] = actions
                self._rewards[:, t] = rewards
                self._optimal_arms[:, t] = optimal_arm

            # Compute metrics
            regret = optimal_value - self._rewards
            cumulative_regret = cp.cumsum(regret, axis=1)

            return {
                'rewards': cp.asnumpy(self._rewards),
                'actions': cp.asnumpy(self._actions),
                'cumulative_regret': cp.asnumpy(cumulative_regret),
                'final_regret': cp.asnumpy(cumulative_regret[:, -1]),
                'optimal_percentage': cp.asnumpy(
                    cp.mean(self._actions == optimal_arm, axis=1)
                ),
            }


def run_gpu_experiment_suite(
    algorithms: List[Dict[str, Any]],
    arm_means: np.ndarray,
    arm_stds: Optional[np.ndarray] = None,
    n_runs: int = 50,
    horizon: int = 10000,
    device: int = 0,
    seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run multiple algorithms on GPU and return results.

    Parameters
    ----------
    algorithms : list
        List of algorithm configs with 'name' and 'type' keys
    arm_means : np.ndarray
        True arm means
    arm_stds : np.ndarray, optional
        Arm standard deviations
    n_runs : int
        Number of independent runs per algorithm
    horizon : int
        Number of timesteps per run
    device : int
        GPU device ID
    seed : int
        Random seed

    Returns
    -------
    dict
        Results keyed by algorithm name
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    n_arms = len(arm_means)
    config = GPUBatchConfig(
        n_runs=n_runs,
        n_arms=n_arms,
        horizon=horizon,
        device=device,
        seed=seed
    )

    runner = GPUBatchRunner(config)
    results = {}

    for algo in algorithms:
        name = algo['name']
        algo_type = algo.get('type', 'epsilon_greedy')

        start_time = time.time()

        if algo_type == 'epsilon_greedy':
            results[name] = runner.run_epsilon_greedy(
                epsilon=algo.get('epsilon', 0.1),
                alpha=algo.get('alpha', 0.1),
                arm_means=arm_means,
                arm_stds=arm_stds
            )
        elif algo_type == 'ucb':
            results[name] = runner.run_ucb(
                c=algo.get('c', np.sqrt(2)),
                arm_means=arm_means,
                arm_stds=arm_stds
            )
        elif algo_type == 'thompson_sampling':
            results[name] = runner.run_thompson_sampling(
                arm_means=arm_means,
                arm_stds=arm_stds,
                prior_mean=algo.get('prior_mean', 0.0),
                prior_std=algo.get('prior_std', 1.0),
                reward_std=algo.get('reward_std', 1.0)
            )

        elapsed = time.time() - start_time
        results[name]['wall_time'] = elapsed

    return results
