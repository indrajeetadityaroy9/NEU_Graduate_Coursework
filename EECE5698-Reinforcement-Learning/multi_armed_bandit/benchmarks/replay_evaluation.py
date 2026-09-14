"""
Replay Evaluation for Open Bandit Dataset

Implements the unbiased Replay Method for evaluating bandit algorithms
on logged data from a random behavior policy.

The key insight: When the logging policy is uniform random, we can
"replay" our algorithm's decisions against the log. If our algorithm
would have chosen the same action as the log, we observe that reward.
Otherwise, we skip that log entry.

This provides an unbiased estimate of online performance without
needing complex OPE estimators.

Reference:
- Li et al. (2011) "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation"
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import os

try:
    from obp.dataset import OpenBanditDataset
    OBD_AVAILABLE = True
except ImportError:
    OBD_AVAILABLE = False


@dataclass
class ReplayResult:
    """Results from replay evaluation."""
    n_matches: int              # Number of times algorithm matched log
    n_total: int               # Total log entries processed
    cumulative_reward: float   # Sum of rewards when matched
    mean_reward: float         # Average reward per match
    match_rate: float          # Fraction of matches
    rewards_over_time: List[float]  # Cumulative reward trajectory


class ReplayEvaluator:
    """
    Replay evaluation on Open Bandit Dataset.

    Uses the random policy subset for unbiased evaluation.

    Example:
        evaluator = ReplayEvaluator()

        # Your algorithm must implement select_arm(context) -> int
        result = evaluator.evaluate(
            algorithm=my_algorithm,
            max_steps=10000,
        )

        print(f"Mean reward: {result.mean_reward:.3f}")
    """

    def __init__(
        self,
        campaign: str = "all",
        data_path: Optional[str] = None,
    ):
        """
        Args:
            campaign: OBD campaign to use ('all', 'men', 'women')
            data_path: Path to OBD data (downloads if not found)
        """
        if not OBD_AVAILABLE:
            raise ImportError("OBP not installed. Run: pip install obp")

        self.campaign = campaign
        self.data_path = data_path or os.path.expanduser("~/.obp/obd")

        # Check if data exists
        if not os.path.exists(self.data_path):
            print(f"OBD data not found at {self.data_path}")
            print("Download from: https://research.zozo.com/data.html")
            print("Or use SyntheticReplayEvaluator for testing")
            self._data_available = False
        else:
            self._data_available = True
            self._load_data()

    def _load_data(self) -> None:
        """Load OBD random policy data."""
        try:
            self.dataset = OpenBanditDataset(
                behavior_policy="random",
                campaign=self.campaign,
                data_path=self.data_path,
            )
            self.bandit_feedback = self.dataset.obtain_batch_bandit_feedback()
            self.n_actions = self.bandit_feedback['n_actions']
            self.n_rounds = self.bandit_feedback['n_rounds']
        except Exception as e:
            print(f"Error loading OBD: {e}")
            self._data_available = False

    def evaluate(
        self,
        algorithm,  # Must have select_arm(context) -> int method
        max_steps: int = 10000,
        context_processor: Optional[Callable] = None,
    ) -> ReplayResult:
        """
        Evaluate algorithm using replay method.

        Args:
            algorithm: Bandit algorithm with select_arm(context) method
            max_steps: Maximum number of matched steps to collect
            context_processor: Optional function to preprocess context

        Returns:
            ReplayResult with evaluation metrics
        """
        if not self._data_available:
            raise RuntimeError("OBD data not available")

        actions = self.bandit_feedback['action']
        rewards = self.bandit_feedback['reward']
        contexts = self.bandit_feedback['context']

        n_matches = 0
        n_total = 0
        cumulative_reward = 0.0
        rewards_over_time = []

        for i in range(self.n_rounds):
            if n_matches >= max_steps:
                break

            n_total += 1

            # Get context and preprocess if needed
            context = contexts[i] if contexts is not None else None
            if context_processor is not None and context is not None:
                context = context_processor(context)

            # Get algorithm's action (support both select_arm and select_action)
            if hasattr(algorithm, 'select_action'):
                algo_action = algorithm.select_action()
            elif hasattr(algorithm, 'select_arm'):
                if context is not None:
                    algo_action = algorithm.select_arm(context)
                else:
                    algo_action = algorithm.select_arm()
            else:
                raise ValueError("Algorithm must have select_action() or select_arm() method")

            # Check if algorithm matches log
            if algo_action == actions[i]:
                n_matches += 1
                reward = rewards[i]
                cumulative_reward += reward

                # Update algorithm with observed reward
                if hasattr(algorithm, 'update'):
                    algorithm.update(algo_action, reward)

                rewards_over_time.append(cumulative_reward)

        match_rate = n_matches / n_total if n_total > 0 else 0
        mean_reward = cumulative_reward / n_matches if n_matches > 0 else 0

        return ReplayResult(
            n_matches=n_matches,
            n_total=n_total,
            cumulative_reward=cumulative_reward,
            mean_reward=mean_reward,
            match_rate=match_rate,
            rewards_over_time=rewards_over_time,
        )


class SyntheticReplayEvaluator:
    """
    Synthetic replay evaluator for testing without OBD download.

    Simulates a logged bandit dataset with known ground truth for validation.
    """

    def __init__(
        self,
        n_arms: int = 10,
        n_rounds: int = 100000,
        arm_means: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        self.n_arms = n_arms
        self.n_rounds = n_rounds
        self.rng = np.random.default_rng(seed)

        if arm_means is None:
            # Create arm means with clear winner
            self.arm_means = np.linspace(0.1, 0.9, n_arms)
        else:
            self.arm_means = np.array(arm_means)

        self._generate_log()

    def _generate_log(self) -> None:
        """Generate logged data from uniform random policy."""
        # Random policy: uniform over arms
        self.actions = self.rng.integers(0, self.n_arms, size=self.n_rounds)

        # Generate rewards based on arm means (Bernoulli)
        self.rewards = np.zeros(self.n_rounds)
        for i, a in enumerate(self.actions):
            self.rewards[i] = float(self.rng.random() < self.arm_means[a])

        # No context for classical MAB
        self.contexts = None

    def evaluate(
        self,
        algorithm,
        max_steps: int = 10000,
    ) -> ReplayResult:
        """
        Evaluate algorithm using replay method.

        Args:
            algorithm: Bandit algorithm with select_arm() method
            max_steps: Maximum number of matched steps

        Returns:
            ReplayResult with evaluation metrics
        """
        n_matches = 0
        n_total = 0
        cumulative_reward = 0.0
        rewards_over_time = []

        for i in range(self.n_rounds):
            if n_matches >= max_steps:
                break

            n_total += 1

            # Get algorithm's action (support both select_arm and select_action)
            if hasattr(algorithm, 'select_action'):
                algo_action = algorithm.select_action()
            elif hasattr(algorithm, 'select_arm'):
                algo_action = algorithm.select_arm()
            else:
                raise ValueError("Algorithm must have select_action() or select_arm() method")

            # Check if algorithm matches log
            if algo_action == self.actions[i]:
                n_matches += 1
                reward = self.rewards[i]
                cumulative_reward += reward

                # Update algorithm with observed reward
                if hasattr(algorithm, 'update'):
                    algorithm.update(algo_action, reward)

                rewards_over_time.append(cumulative_reward)

        match_rate = n_matches / n_total if n_total > 0 else 0
        mean_reward = cumulative_reward / n_matches if n_matches > 0 else 0

        return ReplayResult(
            n_matches=n_matches,
            n_total=n_total,
            cumulative_reward=cumulative_reward,
            mean_reward=mean_reward,
            match_rate=match_rate,
            rewards_over_time=rewards_over_time,
        )

    def get_optimal_mean_reward(self) -> float:
        """Return the mean reward of the optimal arm (upper bound)."""
        return np.max(self.arm_means)

    def get_random_mean_reward(self) -> float:
        """Return the mean reward of random policy (baseline)."""
        return np.mean(self.arm_means)


def run_replay_benchmark(
    algorithms: Dict[str, Any],
    n_arms: int = 10,
    max_steps: int = 5000,
    seed: int = 42,
) -> Dict[str, ReplayResult]:
    """
    Run replay benchmark on multiple algorithms.

    Args:
        algorithms: Dict mapping name to algorithm instance
        n_arms: Number of arms for synthetic log
        max_steps: Steps to evaluate
        seed: Random seed

    Returns:
        Dict mapping algorithm name to ReplayResult
    """
    # Create synthetic evaluator
    arm_means = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9])[:n_arms]
    evaluator = SyntheticReplayEvaluator(
        n_arms=n_arms,
        arm_means=arm_means,
        seed=seed,
    )

    results = {}
    for name, algo in algorithms.items():
        # Reset algorithm
        if hasattr(algo, 'reset'):
            algo.reset()

        result = evaluator.evaluate(algo, max_steps=max_steps)
        results[name] = result

        print(f"{name}: Mean Reward = {result.mean_reward:.3f}, "
              f"Match Rate = {result.match_rate:.1%}, "
              f"Matches = {result.n_matches}")

    # Print comparison
    print(f"\nOptimal mean reward: {evaluator.get_optimal_mean_reward():.3f}")
    print(f"Random mean reward: {evaluator.get_random_mean_reward():.3f}")

    return results
