"""
Gradient Bandit Algorithm

Policy gradient approach using softmax action selection and
stochastic gradient ascent on expected reward.
"""

from typing import Optional
import numpy as np
from .base import BanditAlgorithm


def _softmax(preferences: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities with numerical stability.

    Parameters
    ----------
    preferences : np.ndarray
        Preference values H(a) for each action

    Returns
    -------
    np.ndarray
        Probability distribution over actions
    """
    exp_prefs = np.exp(preferences - np.max(preferences))
    return exp_prefs / np.sum(exp_prefs)


class GradientBandit(BanditAlgorithm):
    """
    Gradient bandit with softmax policy.

    Maintains preference values H(a) for each action and uses
    softmax to convert to probabilities: π(a) = exp(H(a)) / Σ exp(H(a')).

    Updates preferences using gradient ascent on expected reward:
        H(a) ← H(a) + α * (r - r̄) * (𝟙[a=A_t] - π(a))

    Parameters
    ----------
    n_arms : int
        Number of arms
    alpha : float
        Step size for preference updates
    use_baseline : bool
        Whether to use reward baseline (average reward)
    seed : Optional[int]
        Random seed

    References
    ----------
    Sutton, R. S., & Barto, A. G. (2018).
    Reinforcement Learning: An Introduction.
    Chapter 2.8: Gradient Bandit Algorithms.
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float = 0.1,
        use_baseline: bool = True,
        seed: Optional[int] = None
    ):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.use_baseline = use_baseline
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize preferences and baseline."""
        self._preferences = np.zeros(self.n_arms)
        self._average_reward = 0.0
        self._counts = np.zeros(self.n_arms, dtype=int)

    def _softmax(self) -> np.ndarray:
        """Compute softmax probabilities from preferences."""
        return _softmax(self._preferences)

    def select_action(self) -> int:
        """Select action according to softmax policy."""
        probs = self._softmax()
        return self.rng.choice(self.n_arms, p=probs)

    def update(self, action: int, reward: float) -> None:
        """Update preferences using gradient ascent."""
        self._counts[action] += 1
        self.t += 1

        # Update baseline (average reward)
        if self.use_baseline:
            self._average_reward += (reward - self._average_reward) / self.t
            baseline = self._average_reward
        else:
            baseline = 0.0

        # Compute current policy
        probs = self._softmax()

        # Vectorized gradient update for all actions
        # For selected action: H += α * (r - baseline) * (1 - π(a))
        # For other actions: H += α * (r - baseline) * (0 - π(a))
        advantage = reward - baseline
        indicator = np.zeros(self.n_arms)
        indicator[action] = 1.0
        self._preferences += self.alpha * advantage * (indicator - probs)

    def get_action_values(self) -> np.ndarray:
        """Return preferences (not true values)."""
        return self._preferences.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def get_action_probabilities(self) -> np.ndarray:
        """Get current action probabilities."""
        return self._softmax()

    @property
    def name(self) -> str:
        baseline_str = "+baseline" if self.use_baseline else ""
        return f"GradientBandit(α={self.alpha}{baseline_str})"


class EntropyRegularizedGradient(BanditAlgorithm):
    """
    Gradient bandit with entropy regularization.

    Adds entropy bonus to encourage exploration:
        J(π) = E[r] + τ * H(π)

    where H(π) = -Σ π(a) log π(a) is the entropy.

    Parameters
    ----------
    n_arms : int
        Number of arms
    alpha : float
        Step size for preference updates
    tau : float
        Temperature/entropy coefficient. Higher = more exploration.
    use_baseline : bool
        Whether to use reward baseline
    seed : Optional[int]
        Random seed

    Notes
    -----
    The gradient of entropy w.r.t. preferences is:
        ∂H/∂H(a) = -π(a) * (1 + log π(a)) + Σ_a' π(a')² * (1 + log π(a'))
    """

    def __init__(
        self,
        n_arms: int,
        alpha: float = 0.1,
        tau: float = 0.1,
        use_baseline: bool = True,
        seed: Optional[int] = None
    ):
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if tau < 0:
            raise ValueError(f"tau must be non-negative, got {tau}")
        self.alpha = alpha
        self.tau = tau
        self.use_baseline = use_baseline
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize preferences and baseline."""
        self._preferences = np.zeros(self.n_arms)
        self._average_reward = 0.0
        self._counts = np.zeros(self.n_arms, dtype=int)

    def _softmax(self) -> np.ndarray:
        """Compute softmax probabilities."""
        return _softmax(self._preferences)

    def _entropy(self, probs: np.ndarray) -> float:
        """Compute entropy of distribution."""
        # Avoid log(0)
        log_probs = np.log(probs + 1e-10)
        return -np.sum(probs * log_probs)

    def select_action(self) -> int:
        """Select action according to softmax policy."""
        probs = self._softmax()
        return self.rng.choice(self.n_arms, p=probs)

    def update(self, action: int, reward: float) -> None:
        """Update with entropy-regularized gradient."""
        self._counts[action] += 1
        self.t += 1

        if self.use_baseline:
            self._average_reward += (reward - self._average_reward) / self.t
            baseline = self._average_reward
        else:
            baseline = 0.0

        probs = self._softmax()
        advantage = reward - baseline

        # Vectorized standard gradient + entropy gradient
        indicator = np.zeros(self.n_arms)
        indicator[action] = 1.0

        # Reward gradient: (indicator - probs) * advantage
        reward_grad = advantage * (indicator - probs)

        # Entropy gradient (vectorized - O(n) instead of O(n²))
        # Pre-compute the entropy correction term once
        log_probs = np.log(probs + 1e-10)
        entropy_correction = np.sum(probs * (log_probs + 1))
        entropy_grad = -probs * (log_probs + 1) + probs * entropy_correction

        # Combined update
        self._preferences += self.alpha * (reward_grad + self.tau * entropy_grad)

    def get_action_values(self) -> np.ndarray:
        return self._preferences.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"EntropyGradient(α={self.alpha}, τ={self.tau})"
