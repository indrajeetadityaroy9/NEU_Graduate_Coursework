"""
Epsilon-Greedy Algorithms

Provides both decaying and constant step-size variants for
stationary and non-stationary environments respectively.
"""

from typing import Optional
import numpy as np
from .base import BanditAlgorithm


def _epsilon_greedy_select(rng: np.random.Generator, q_values: np.ndarray,
                           epsilon: float, n_arms: int) -> int:
    """
    Shared epsilon-greedy action selection.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    q_values : np.ndarray
        Current action value estimates
    epsilon : float
        Exploration probability
    n_arms : int
        Number of arms

    Returns
    -------
    int
        Selected action
    """
    if rng.random() < epsilon:
        # Explore: random action
        return rng.integers(0, n_arms)
    else:
        # Exploit: greedy action (break ties randomly)
        max_value = np.max(q_values)
        best_actions = np.where(q_values == max_value)[0]
        return rng.choice(best_actions)


class EpsilonGreedy(BanditAlgorithm):
    """
    ε-greedy with sample-average updates (1/n step size).

    Standard ε-greedy algorithm suitable for stationary environments.
    Explores with probability ε, exploits with probability 1-ε.
    Uses sample averages for value estimation.

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Exploration probability (default: 0.1)
    initial_value : float
        Initial action value estimate (optimistic initialization)
    seed : Optional[int]
        Random seed

    Examples
    --------
    >>> algo = EpsilonGreedy(n_arms=10, epsilon=0.1)
    >>> for t in range(1000):
    ...     action = algo.select_action()
    ...     reward = env.pull(action)
    ...     algo.update(action, reward)
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        initial_value: float = 0.0,
        seed: Optional[int] = None
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = epsilon
        self.initial_value = initial_value
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize value estimates and counts."""
        self._q_values = np.full(self.n_arms, self.initial_value, dtype=float)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def select_action(self) -> int:
        """Select action using ε-greedy policy."""
        return _epsilon_greedy_select(self.rng, self._q_values, self.epsilon, self.n_arms)

    def update(self, action: int, reward: float) -> None:
        """Update value estimate using sample average."""
        self._counts[action] += 1
        self.t += 1

        # Sample average update: Q = Q + (1/n)(r - Q)
        alpha = 1.0 / self._counts[action]
        self._q_values[action] += alpha * (reward - self._q_values[action])

    def get_action_values(self) -> np.ndarray:
        return self._q_values.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"ε-Greedy(ε={self.epsilon})"


class EpsilonGreedyConstant(BanditAlgorithm):
    """
    ε-greedy with constant step size α.

    Suitable for non-stationary environments because it gives
    more weight to recent observations. With constant α, the
    algorithm "forgets" old observations exponentially.

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon : float
        Exploration probability
    alpha : float
        Constant step size (learning rate)
    initial_value : float
        Initial action value estimate
    seed : Optional[int]
        Random seed

    Notes
    -----
    The effective sample size is approximately 1/α due to
    exponential recency-weighting. Larger α adapts faster
    but has higher variance.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        initial_value: float = 0.0,
        seed: Optional[int] = None
    ):
        if not 0 <= epsilon <= 1:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.epsilon = epsilon
        self.alpha = alpha
        self.initial_value = initial_value
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize value estimates and counts."""
        self._q_values = np.full(self.n_arms, self.initial_value, dtype=float)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def select_action(self) -> int:
        """Select action using ε-greedy policy."""
        return _epsilon_greedy_select(self.rng, self._q_values, self.epsilon, self.n_arms)

    def update(self, action: int, reward: float) -> None:
        """Update value estimate with constant step size."""
        self._counts[action] += 1
        self.t += 1

        # Constant step size update: Q = Q + α(r - Q)
        self._q_values[action] += self.alpha * (reward - self._q_values[action])

    def get_action_values(self) -> np.ndarray:
        return self._q_values.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"ε-Greedy(ε={self.epsilon}, α={self.alpha})"


class DecayingEpsilonGreedy(BanditAlgorithm):
    """
    ε-greedy with decaying exploration rate.

    Exploration probability decays over time: ε_t = ε_0 / (1 + decay_rate * t).
    Useful when you want to explore early and exploit more later.

    Parameters
    ----------
    n_arms : int
        Number of arms
    epsilon_init : float
        Initial exploration probability
    decay_rate : float
        Rate of epsilon decay
    epsilon_min : float
        Minimum exploration probability
    alpha : float
        Step size (None for sample average)
    seed : Optional[int]
        Random seed
    """

    def __init__(
        self,
        n_arms: int,
        epsilon_init: float = 1.0,
        decay_rate: float = 0.01,
        epsilon_min: float = 0.01,
        alpha: Optional[float] = None,
        seed: Optional[int] = None
    ):
        if not 0 <= epsilon_init <= 1:
            raise ValueError(f"epsilon_init must be in [0, 1], got {epsilon_init}")
        if decay_rate < 0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}")
        if not 0 <= epsilon_min <= 1:
            raise ValueError(f"epsilon_min must be in [0, 1], got {epsilon_min}")
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.epsilon_init = epsilon_init
        self.decay_rate = decay_rate
        self.epsilon_min = epsilon_min
        self.alpha = alpha
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize value estimates and counts."""
        self._q_values = np.zeros(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)

    @property
    def epsilon(self) -> float:
        """Current exploration probability."""
        return max(
            self.epsilon_min,
            self.epsilon_init / (1 + self.decay_rate * self.t)
        )

    def select_action(self) -> int:
        """Select action using decaying ε-greedy policy."""
        return _epsilon_greedy_select(self.rng, self._q_values, self.epsilon, self.n_arms)

    def update(self, action: int, reward: float) -> None:
        """Update value estimate."""
        self._counts[action] += 1
        self.t += 1

        if self.alpha is None:
            # Sample average
            step = 1.0 / self._counts[action]
        else:
            step = self.alpha

        self._q_values[action] += step * (reward - self._q_values[action])

    def get_action_values(self) -> np.ndarray:
        return self._q_values.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"DecayingEpsilon(ε₀={self.epsilon_init})"
