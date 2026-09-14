"""
Abrupt Change Bandit Environment

A non-stationary environment where the optimal arm changes suddenly
at fixed intervals. This is the primary test environment for studying
algorithm adaptation to distribution shift.
"""

from typing import Optional, List
import numpy as np
from .base import BanditEnvironment


class AbruptChangeBandit(BanditEnvironment):
    """
    Non-stationary bandit with abrupt optimal arm changes.

    At each change point, a new arm is randomly selected to become optimal.
    The means are shuffled to create a clear gap between the optimal arm
    and suboptimal arms.

    Parameters
    ----------
    n_arms : int
        Number of arms (must be at least 2 for arm switching)
    change_interval : int
        Number of steps between optimal arm changes
    gap : float
        Gap between optimal and suboptimal arm means
    seed : Optional[int]
        Random seed for reproducibility

    Examples
    --------
    >>> env = AbruptChangeBandit(n_arms=5, change_interval=100, gap=1.0)
    >>> for t in range(500):
    ...     reward = env.pull(0)
    ...     env.step()  # May trigger change at t=100, 200, etc.
    >>> print(env.change_points)  # [100, 200, 300, 400]
    """

    def __init__(
        self,
        n_arms: int,
        change_interval: int = 100,
        gap: float = 1.0,
        seed: Optional[int] = None
    ):
        if n_arms < 2:
            raise ValueError(f"AbruptChangeBandit requires at least 2 arms, got {n_arms}")
        if change_interval < 1:
            raise ValueError(f"change_interval must be positive, got {change_interval}")
        self.change_interval = change_interval
        self.gap = gap
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize arm distributions with one optimal arm."""
        # All suboptimal arms have mean 0, optimal arm has mean = gap
        self._means = np.zeros(self.n_arms)
        self._stds = np.ones(self.n_arms)

        # Randomly select initial optimal arm
        self._optimal_arm = self.rng.integers(0, self.n_arms)
        self._means[self._optimal_arm] = self.gap

        # Record initial state
        self._optimal_arm_history = [(0, self._optimal_arm)]
        self._optimal_value_history = [self.gap]

    def _get_arm_mean(self, arm: int) -> float:
        return self._means[arm]

    def _get_arm_std(self, arm: int) -> float:
        return self._stds[arm]

    def _check_for_change(self) -> None:
        """Check if it's time for an abrupt change."""
        if self.t > 0 and self.t % self.change_interval == 0:
            self._trigger_change()

    def _trigger_change(self) -> None:
        """Change the optimal arm to a different arm."""
        old_optimal = self._optimal_arm

        # Select a new optimal arm (different from current)
        candidates = [a for a in range(self.n_arms) if a != old_optimal]
        new_optimal = self.rng.choice(candidates)

        # Update means
        self._means[old_optimal] = 0.0
        self._means[new_optimal] = self.gap
        self._optimal_arm = new_optimal

        # Record change
        self._change_points.append(self.t)
        self._optimal_arm_history.append((self.t, new_optimal))
        self._optimal_value_history.append(self.gap)

    @property
    def name(self) -> str:
        return f"AbruptChange(interval={self.change_interval})"

    def get_info(self) -> dict:
        """Get extended information about the environment."""
        info = super().get_info()
        info.update({
            'change_interval': self.change_interval,
            'gap': self.gap,
            'n_changes': len(self._change_points),
            'optimal_arm_history': self._optimal_arm_history,
        })
        return info


class RotatingBandit(BanditEnvironment):
    """
    Non-stationary bandit where optimal arm rotates in a deterministic cycle.

    Unlike AbruptChangeBandit which selects random new optimal arms,
    RotatingBandit cycles through arms in order: 0 -> 1 -> 2 -> ... -> 0.

    Parameters
    ----------
    n_arms : int
        Number of arms
    change_interval : int
        Number of steps between rotations
    gap : float
        Gap between optimal and suboptimal arms
    seed : Optional[int]
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_arms: int,
        change_interval: int = 100,
        gap: float = 1.0,
        seed: Optional[int] = None
    ):
        if n_arms < 1:
            raise ValueError(f"RotatingBandit requires at least 1 arm, got {n_arms}")
        if change_interval < 1:
            raise ValueError(f"change_interval must be positive, got {change_interval}")
        self.change_interval = change_interval
        self.gap = gap
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize with arm 0 as optimal."""
        self._means = np.zeros(self.n_arms)
        self._stds = np.ones(self.n_arms)
        self._optimal_arm = 0
        self._means[0] = self.gap

        self._optimal_arm_history = [(0, 0)]
        self._optimal_value_history = [self.gap]

    def _get_arm_mean(self, arm: int) -> float:
        return self._means[arm]

    def _get_arm_std(self, arm: int) -> float:
        return self._stds[arm]

    def _check_for_change(self) -> None:
        """Check if it's time to rotate."""
        if self.t > 0 and self.t % self.change_interval == 0:
            self._trigger_rotation()

    def _trigger_rotation(self) -> None:
        """Rotate optimal arm to next in sequence."""
        old_optimal = self._optimal_arm
        new_optimal = (old_optimal + 1) % self.n_arms

        self._means[old_optimal] = 0.0
        self._means[new_optimal] = self.gap
        self._optimal_arm = new_optimal

        self._change_points.append(self.t)
        self._optimal_arm_history.append((self.t, new_optimal))
        self._optimal_value_history.append(self.gap)

    @property
    def name(self) -> str:
        return f"Rotating(interval={self.change_interval})"
