"""
Upper Confidence Bound (UCB) Algorithms

Provides UCB1 for stationary environments and variants for non-stationarity:
- Discounted UCB (D-UCB): Exponential discounting of old observations
- Sliding Window UCB (SW-UCB): Only uses recent observations within a window
"""

from typing import Optional, Deque
from collections import deque
import numpy as np
from .base import BanditAlgorithm
from .utils import validate_positive, validate_discount_factor, validate_positive_int


class UCB1(BanditAlgorithm):
    """
    UCB1 algorithm for stationary bandits.

    Selects the arm with highest Upper Confidence Bound, balancing
    exploitation (high mean) with exploration (high uncertainty).

    Mathematical Formulation
    ------------------------
    Action selection: A_t = argmax_a [Q̂(a) + c√(ln(t)/N(a))]
    Value update: Q̂(a) = (1/N(a)) Σ r_s  (sample average)

    where:
        Q̂(a) = estimated mean reward for arm a
        N(a) = number of times arm a has been pulled
        c = exploration parameter (√2 is theoretically optimal)
        t = current timestep

    Parameters
    ----------
    n_arms : int
        Number of arms (K in literature)
    c : float
        Exploration parameter c > 0. Default √2 achieves O(√(KT ln T)) regret.
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
    Finite-time analysis of the multiarmed bandit problem.
    Machine learning, 47(2), 235-256.
    """

    def __init__(
        self,
        n_arms: int,
        c: float = np.sqrt(2),
        seed: Optional[int] = None
    ):
        validate_positive(c, "c")
        self.c = c
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize value estimates and counts."""
        self._q_values = np.zeros(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def select_action(self) -> int:
        """Select action with highest UCB value."""
        # First, try each arm once
        for arm in range(self.n_arms):
            if self._counts[arm] == 0:
                return arm

        # Compute UCB values
        total = np.sum(self._counts)
        exploration_bonus = self.c * np.sqrt(np.log(total) / self._counts)
        ucb_values = self._q_values + exploration_bonus

        # Break ties randomly
        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        return self.rng.choice(best_actions)

    def update(self, action: int, reward: float) -> None:
        """Update value estimate using sample average."""
        self._counts[action] += 1
        self.t += 1

        # Sample average update
        alpha = 1.0 / self._counts[action]
        self._q_values[action] += alpha * (reward - self._q_values[action])

    def get_action_values(self) -> np.ndarray:
        return self._q_values.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def get_ucb_values(self) -> np.ndarray:
        """Get current UCB values including exploration bonus."""
        if np.any(self._counts == 0):
            ucb = np.full(self.n_arms, np.inf)
            mask = self._counts > 0
            total = np.sum(self._counts)
            ucb[mask] = self._q_values[mask] + self.c * np.sqrt(
                np.log(total) / self._counts[mask]
            )
            return ucb

        total = np.sum(self._counts)
        exploration = self.c * np.sqrt(np.log(total) / self._counts)
        return self._q_values + exploration

    @property
    def name(self) -> str:
        return f"UCB1(c={self.c:.2f})"


class DiscountedUCB(BanditAlgorithm):
    """
    Discounted UCB (D-UCB) for non-stationary bandits.

    Uses exponential discounting to give more weight to recent observations.
    Effective for tracking slowly drifting reward distributions.

    Mathematical Formulation
    ------------------------
    Discounted count: N_γ(a) = Σ_{s: a_s=a} γ^(t-s)
    Discounted mean: Q̂_γ(a) = [Σ_{s: a_s=a} γ^(t-s) r_s] / N_γ(a)
    Action selection: A_t = argmax_a [Q̂_γ(a) + c√(ξ·ln(N_γ)/N_γ(a))]

    where:
        γ = discount factor ∈ (0, 1), effective memory ≈ 1/(1-γ)
        N_γ = Σ_a N_γ(a) = total discounted count
        c, ξ = exploration parameters

    Note: The parameter `gamma` here is a discount factor, distinct from
    `exploration_rate` in EXP3 which controls uniform mixing.

    Parameters
    ----------
    n_arms : int
        Number of arms (K in literature)
    gamma : float
        Discount factor γ ∈ (0, 1). Effective memory ≈ 1/(1-γ) steps.
        γ=0.99 → ~100 steps memory. Smaller γ adapts faster but has higher variance.
    c : float
        Exploration bonus coefficient c > 0
    xi : float
        Additional exploration parameter ξ > 0 for non-stationarity
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    Garivier, A., & Moulines, E. (2011).
    On upper-confidence bound policies for switching bandit problems.
    ALT 2011.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.99,
        c: float = 1.0,
        xi: float = 0.5,
        seed: Optional[int] = None
    ):
        validate_discount_factor(gamma)
        validate_positive(c, "c")
        validate_positive(xi, "xi")
        self.gamma = gamma
        self.c = c
        self.xi = xi
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize discounted statistics."""
        self._discounted_rewards = np.zeros(self.n_arms)
        self._discounted_counts = np.zeros(self.n_arms)

    def _apply_discount(self) -> None:
        """Apply discount to all statistics."""
        self._discounted_rewards *= self.gamma
        self._discounted_counts *= self.gamma

    def select_action(self) -> int:
        """Select action with highest discounted UCB value."""
        # Initialize unsampled arms
        for arm in range(self.n_arms):
            if self._discounted_counts[arm] < 1e-6:
                return arm

        # Compute discounted UCB values
        total_count = np.sum(self._discounted_counts)
        exploration = self.c * np.sqrt(
            self.xi * np.log(total_count) / self._discounted_counts
        )
        q_values = self._discounted_rewards / self._discounted_counts
        ucb_values = q_values + exploration

        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        return self.rng.choice(best_actions)

    def update(self, action: int, reward: float) -> None:
        """Update discounted statistics."""
        self._apply_discount()
        self._discounted_counts[action] += 1
        self._discounted_rewards[action] += reward
        self.t += 1

    def get_action_values(self) -> np.ndarray:
        """Get current discounted value estimates."""
        values = np.zeros(self.n_arms)
        mask = self._discounted_counts > 1e-6
        values[mask] = (
            self._discounted_rewards[mask] / self._discounted_counts[mask]
        )
        return values

    def get_action_counts(self) -> np.ndarray:
        """Get discounted counts (not integer)."""
        return self._discounted_counts.copy()

    @property
    def name(self) -> str:
        return f"D-UCB(γ={self.gamma})"


class SlidingWindowUCB(BanditAlgorithm):
    """
    Sliding Window UCB for non-stationary bandits.

    Only uses observations from the last τ timesteps.
    Effective when distribution changes are not too frequent
    (need enough samples per window to estimate means).

    Parameters
    ----------
    n_arms : int
        Number of arms
    window_size : int
        Number of recent observations to use (τ)
    c : float
        Exploration parameter
    xi : float
        Additional exploration parameter
    seed : Optional[int]
        Random seed

    References
    ----------
    Garivier, A., & Moulines, E. (2011).
    On upper-confidence bound policies for switching bandit problems.
    ALT 2011.
    """

    def __init__(
        self,
        n_arms: int,
        window_size: int = 100,
        c: float = 1.0,
        xi: float = 0.5,
        seed: Optional[int] = None
    ):
        validate_positive_int(window_size, "window_size")
        validate_positive(c, "c")
        validate_positive(xi, "xi")
        self.window_size = window_size
        self.c = c
        self.xi = xi
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize sliding window buffers."""
        # Store (action, reward) tuples in a deque
        self._history: Deque = deque(maxlen=self.window_size)

    def _get_window_stats(self) -> tuple:
        """Compute statistics from current window."""
        counts = np.zeros(self.n_arms)
        sums = np.zeros(self.n_arms)

        for action, reward in self._history:
            counts[action] += 1
            sums[action] += reward

        means = np.zeros(self.n_arms)
        mask = counts > 0
        means[mask] = sums[mask] / counts[mask]

        return means, counts

    def select_action(self) -> int:
        """Select action with highest window-based UCB value."""
        means, counts = self._get_window_stats()

        # Try unsampled arms first
        for arm in range(self.n_arms):
            if counts[arm] == 0:
                return arm

        # Compute UCB values
        total = min(len(self._history), self.window_size)
        exploration = self.c * np.sqrt(self.xi * np.log(total) / counts)
        ucb_values = means + exploration

        max_ucb = np.max(ucb_values)
        best_actions = np.where(ucb_values == max_ucb)[0]
        return self.rng.choice(best_actions)

    def update(self, action: int, reward: float) -> None:
        """Add observation to sliding window."""
        self._history.append((action, reward))
        self.t += 1

    def get_action_values(self) -> np.ndarray:
        """Get value estimates from current window."""
        means, _ = self._get_window_stats()
        return means

    def get_action_counts(self) -> np.ndarray:
        """Get counts from current window."""
        _, counts = self._get_window_stats()
        return counts

    @property
    def name(self) -> str:
        return f"SW-UCB(τ={self.window_size})"
