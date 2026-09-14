"""
Open Bandit Pipeline (OBP) Integration

Provides standardized Off-Policy Evaluation (OPE) using industry-standard estimators:
- Doubly Robust (DR): Low variance, robust to model misspecification
- Self-Normalized IPW (SNIPS): Stable propensity-weighted estimation

Reference: https://github.com/st-tech/zr-obp
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

try:
    from obp.ope import (
        OffPolicyEvaluation,
        DoublyRobust,
        SelfNormalizedInverseProbabilityWeighting as SNIPS,
        InverseProbabilityWeighting as IPW,
        DirectMethod,
    )
    from obp.utils import check_bandit_feedback_inputs
    OBP_AVAILABLE = True
except ImportError:
    OBP_AVAILABLE = False


@dataclass
class BanditFeedback:
    """OBP-compatible bandit feedback format."""
    n_rounds: int
    n_actions: int
    action: np.ndarray          # shape (n_rounds,) - actions taken
    reward: np.ndarray          # shape (n_rounds,) - observed rewards
    pscore: np.ndarray          # shape (n_rounds,) - propensity scores
    context: Optional[np.ndarray] = None  # shape (n_rounds, dim_context)
    action_context: Optional[np.ndarray] = None  # shape (n_actions, dim_action_context)
    position: Optional[np.ndarray] = None  # for slate/ranking problems

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OBP dictionary format."""
        return {
            'n_rounds': self.n_rounds,
            'n_actions': self.n_actions,
            'action': self.action,
            'reward': self.reward,
            'pscore': self.pscore,
            'context': self.context,
            'action_context': self.action_context,
            'position': self.position if self.position is not None else np.zeros(self.n_rounds, dtype=int),
        }


def create_bandit_feedback_from_run(
    actions: np.ndarray,
    rewards: np.ndarray,
    n_arms: int,
    algorithm_name: str = "unknown",
    epsilon: float = 0.1,
) -> BanditFeedback:
    """
    Convert a bandit run to OBP feedback format.

    For non-contextual bandits, we estimate propensity scores based on
    the algorithm type (epsilon-greedy assumes uniform exploration).

    Args:
        actions: Array of actions taken, shape (T,)
        rewards: Array of rewards received, shape (T,)
        n_arms: Number of arms
        algorithm_name: Name of algorithm (for propensity estimation)
        epsilon: Exploration rate (for epsilon-greedy algorithms)

    Returns:
        BanditFeedback object compatible with OBP
    """
    n_rounds = len(actions)

    # Estimate propensity scores based on algorithm type
    # For epsilon-greedy: P(a) = epsilon/K + (1-epsilon) * I(a = greedy)
    # We use a conservative estimate assuming exploration
    if 'epsilon' in algorithm_name.lower() or 'greedy' in algorithm_name.lower():
        # Lower bound: at least epsilon/K probability for any action
        pscore = np.full(n_rounds, epsilon / n_arms + (1 - epsilon) * 0.5)
    elif 'ucb' in algorithm_name.lower():
        # UCB is deterministic given history, but we use uniform as conservative bound
        pscore = np.full(n_rounds, 1.0 / n_arms)
    elif 'thompson' in algorithm_name.lower():
        # Thompson Sampling has stochastic exploration
        pscore = np.full(n_rounds, 1.0 / n_arms)
    elif 'exp3' in algorithm_name.lower():
        # EXP3 maintains explicit probabilities, use uniform as approximation
        pscore = np.full(n_rounds, 1.0 / n_arms)
    else:
        # Default: uniform random
        pscore = np.full(n_rounds, 1.0 / n_arms)

    return BanditFeedback(
        n_rounds=n_rounds,
        n_actions=n_arms,
        action=actions.astype(int),
        reward=rewards.astype(float),
        pscore=pscore,
    )


class OBPEvaluator:
    """
    Standardized evaluation using Open Bandit Pipeline estimators.

    Uses DoublyRobust and SNIPS for robust off-policy evaluation.

    Example:
        evaluator = OBPEvaluator(n_arms=5)

        # From your experiment run
        feedback = create_bandit_feedback_from_run(actions, rewards, n_arms=5)

        # Evaluate a target policy against logged data
        results = evaluator.evaluate(
            bandit_feedback=feedback,
            action_dist=target_policy_action_dist,  # shape (n_rounds, n_actions)
        )
    """

    def __init__(self, n_arms: int):
        if not OBP_AVAILABLE:
            raise ImportError("OBP not installed. Run: pip install obp")

        self.n_arms = n_arms
        self.estimators = [
            DoublyRobust(),
            SNIPS(),
            IPW(),  # Include for comparison, but prefer DR/SNIPS
        ]

    def evaluate(
        self,
        bandit_feedback: BanditFeedback,
        action_dist: np.ndarray,
        estimated_rewards_by_reg_model: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate a target policy using OPE estimators.

        Args:
            bandit_feedback: Logged data from behavior policy
            action_dist: Target policy action distribution, shape (n_rounds, n_actions)
                        Will be converted to 3D (n_rounds, n_actions, 1) for OBP
            estimated_rewards_by_reg_model: Reward model predictions for DR
                                            shape (n_rounds, n_actions)

        Returns:
            Dictionary with policy value estimates from each estimator
        """
        feedback_dict = bandit_feedback.to_dict()

        # OBP expects 3D action_dist: (n_rounds, n_actions, len_list)
        # For simple bandits, len_list = 1
        if action_dist.ndim == 2:
            action_dist = action_dist[:, :, np.newaxis]

        # If no reward model provided, use simple mean estimator for DR
        if estimated_rewards_by_reg_model is None:
            # Compute empirical mean rewards per action
            action_counts = np.zeros(self.n_arms)
            action_sums = np.zeros(self.n_arms)

            for t in range(bandit_feedback.n_rounds):
                a = bandit_feedback.action[t]
                action_counts[a] += 1
                action_sums[a] += bandit_feedback.reward[t]

            # Use running mean as reward estimate
            mean_rewards = np.divide(
                action_sums,
                action_counts,
                out=np.zeros_like(action_sums),
                where=action_counts > 0
            )
            # OBP expects 3D: (n_rounds, n_actions, 1)
            estimated_rewards_by_reg_model = np.tile(
                mean_rewards, (bandit_feedback.n_rounds, 1)
            )[:, :, np.newaxis]

        elif estimated_rewards_by_reg_model.ndim == 2:
            estimated_rewards_by_reg_model = estimated_rewards_by_reg_model[:, :, np.newaxis]

        ope = OffPolicyEvaluation(
            bandit_feedback=feedback_dict,
            ope_estimators=self.estimators,
        )

        results = ope.estimate_policy_values(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_rewards_by_reg_model,
        )

        return results

    def compare_algorithms(
        self,
        algorithm_runs: Dict[str, Tuple[np.ndarray, np.ndarray]],
        baseline_name: str = "random",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple algorithms using OPE.

        Args:
            algorithm_runs: Dict mapping algorithm name to (actions, rewards) tuple
            baseline_name: Name of baseline algorithm to use as behavior policy

        Returns:
            Dictionary with relative performance metrics
        """
        results = {}

        # Use baseline as behavior policy
        if baseline_name not in algorithm_runs:
            raise ValueError(f"Baseline '{baseline_name}' not in algorithm_runs")

        baseline_actions, baseline_rewards = algorithm_runs[baseline_name]
        n_rounds = len(baseline_actions)

        # Create feedback from baseline (behavior policy)
        baseline_feedback = create_bandit_feedback_from_run(
            baseline_actions, baseline_rewards, self.n_arms, baseline_name
        )

        for algo_name, (actions, rewards) in algorithm_runs.items():
            if algo_name == baseline_name:
                continue

            # Create action distribution from algorithm's choices
            # This is a deterministic policy: P(a|t) = 1 if algorithm chose a, else 0
            action_dist = np.zeros((n_rounds, self.n_arms))
            for t, a in enumerate(actions):
                action_dist[t, int(a)] = 1.0

            try:
                algo_results = self.evaluate(baseline_feedback, action_dist)
                results[algo_name] = algo_results
            except Exception as e:
                results[algo_name] = {'error': str(e)}

        return results


def validate_with_obp(
    actions: np.ndarray,
    rewards: np.ndarray,
    optimal_actions: np.ndarray,
    n_arms: int,
) -> Dict[str, float]:
    """
    Quick validation of a bandit run using OBP metrics.

    Args:
        actions: Actions taken by algorithm
        rewards: Rewards received
        optimal_actions: Ground truth optimal actions at each step
        n_arms: Number of arms

    Returns:
        Dictionary with validation metrics
    """
    if not OBP_AVAILABLE:
        return {'error': 'OBP not installed', 'optimal_rate': np.mean(actions == optimal_actions)}

    n_rounds = len(actions)

    # Compute standard metrics
    optimal_rate = np.mean(actions == optimal_actions)

    # Create feedback
    feedback = create_bandit_feedback_from_run(actions, rewards, n_arms)

    # Compute what an optimal policy would have gotten (upper bound)
    # Shape: (n_rounds, n_arms) - evaluator will convert to 3D
    optimal_dist = np.zeros((n_rounds, n_arms))
    for t, a in enumerate(optimal_actions):
        optimal_dist[t, int(a)] = 1.0

    # Compute what a random policy would have gotten (lower bound)
    random_dist = np.full((n_rounds, n_arms), 1.0 / n_arms)

    evaluator = OBPEvaluator(n_arms)

    try:
        optimal_value = evaluator.evaluate(feedback, optimal_dist)
        random_value = evaluator.evaluate(feedback, random_dist)

        return {
            'optimal_rate': float(optimal_rate),
            'optimal_value_dr': float(optimal_value.get('dr', np.nan)),
            'random_value_dr': float(random_value.get('dr', np.nan)),
            'optimal_value_snips': float(optimal_value.get('snipw', np.nan)),
            'random_value_snips': float(random_value.get('snipw', np.nan)),
        }
    except Exception as e:
        return {
            'optimal_rate': float(optimal_rate),
            'error': str(e),
        }
