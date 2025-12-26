"""
Gradual Drift Bandit Environment

A non-stationary environment where arm means drift continuously over time.
This tests whether algorithms can track slow changes without explicit
change detection.
"""

from typing import Optional, List, Literal
import numpy as np
from .base import BanditEnvironment


class GradualDriftBandit(BanditEnvironment):
    """
    Non-stationary bandit with gradual mean drift.

    Arm means evolve according to a random walk or linear drift,
    potentially causing the optimal arm to change over time.

    Parameters
    ----------
    n_arms : int
        Number of arms
    drift_type : str
        Type of drift: 'random_walk' or 'linear'
    drift_rate : float
        Rate of drift per timestep
        - For random_walk: standard deviation of per-step noise
        - For linear: magnitude of linear drift
    mean_bounds : tuple
        (min, max) bounds for arm means
    seed : Optional[int]
        Random seed for reproducibility

    Examples
    --------
    >>> env = GradualDriftBandit(n_arms=5, drift_type='random_walk', drift_rate=0.01)
    >>> for t in range(1000):
    ...     reward = env.pull(env.get_optimal_arm())
    ...     env.step()
    >>> print(len(env.change_points))  # Number of times optimal arm changed
    """

    def __init__(
        self,
        n_arms: int,
        drift_type: Literal['random_walk', 'linear'] = 'random_walk',
        drift_rate: float = 0.01,
        mean_bounds: tuple = (0.0, 10.0),
        initial_means: Optional[List[float]] = None,
        gap: float = 1.0,
        seed: Optional[int] = None
    ):
        self.drift_type = drift_type
        self.drift_rate = drift_rate
        self.mean_bounds = mean_bounds
        self._initial_means = initial_means
        self._gap = gap
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize arm distributions.

        If initial_means is provided, uses those values.
        Otherwise, uses gap-based initialization (arm 0 = gap, others = 0)
        for consistency with other environments.
        """
        if self._initial_means is not None:
            # Use explicitly provided means
            self._means = np.array(self._initial_means, dtype=float)
        else:
            # Consistent gap-based initialization (like stationary/abrupt)
            self._means = np.zeros(self.n_arms)
            self._means[0] = self._gap

        self._stds = np.ones(self.n_arms)

        # For linear drift, assign random drift directions
        if self.drift_type == 'linear':
            self._drift_directions = self.rng.choice(
                [-1, 1], size=self.n_arms
            ).astype(float)

        # Record initial state
        self._optimal_arm_history = [(0, self.get_optimal_arm())]
        self._optimal_value_history = [self.get_optimal_value()]
        self._previous_optimal = self.get_optimal_arm()

    def _get_arm_mean(self, arm: int) -> float:
        return self._means[arm]

    def _get_arm_std(self, arm: int) -> float:
        return self._stds[arm]

    def _check_for_change(self) -> None:
        """Apply drift and check if optimal arm changed."""
        self._apply_drift()

        # Check if optimal arm changed
        current_optimal = self.get_optimal_arm()
        if current_optimal != self._previous_optimal:
            self._change_points.append(self.t)
            self._optimal_arm_history.append((self.t, current_optimal))
            self._optimal_value_history.append(self.get_optimal_value())
            self._previous_optimal = current_optimal

    def _apply_drift(self) -> None:
        """
        Apply drift to arm means and enforce boundary constraints.

        Boundary Behavior
        -----------------
        - **Random walk**: Means are hard-clamped to mean_bounds.
          Values can "stick" at boundaries until noise pulls them back.
        - **Linear drift**: Means are clamped AND drift direction reverses,
          creating oscillation between boundaries.

        This means random walk may have reduced variance near boundaries,
        while linear drift maintains consistent coverage of the range.
        """
        if self.drift_type == 'random_walk':
            # Gaussian random walk
            noise = self.rng.normal(0, self.drift_rate, size=self.n_arms)
            self._means += noise
        elif self.drift_type == 'linear':
            # Linear drift with occasional direction reversal
            self._means += self._drift_directions * self.drift_rate

        # Reflect at boundaries
        for i in range(self.n_arms):
            if self._means[i] < self.mean_bounds[0]:
                self._means[i] = self.mean_bounds[0]
                if self.drift_type == 'linear':
                    self._drift_directions[i] = 1
            elif self._means[i] > self.mean_bounds[1]:
                self._means[i] = self.mean_bounds[1]
                if self.drift_type == 'linear':
                    self._drift_directions[i] = -1

    @property
    def name(self) -> str:
        return f"GradualDrift({self.drift_type}, rate={self.drift_rate})"

    def get_info(self) -> dict:
        """Get extended information about the environment."""
        info = super().get_info()
        info.update({
            'drift_type': self.drift_type,
            'drift_rate': self.drift_rate,
            'mean_bounds': self.mean_bounds,
            'n_optimal_changes': len(self._change_points),
        })
        return info


class RandomWalkBandit(GradualDriftBandit):
    """
    Convenience class for random walk drift bandit.

    Parameters
    ----------
    n_arms : int
        Number of arms
    sigma : float
        Standard deviation of random walk noise per step
    seed : Optional[int]
        Random seed
    """

    def __init__(
        self,
        n_arms: int,
        sigma: float = 0.01,
        seed: Optional[int] = None
    ):
        super().__init__(
            n_arms=n_arms,
            drift_type='random_walk',
            drift_rate=sigma,
            seed=seed
        )

    @property
    def name(self) -> str:
        return f"RandomWalk(σ={self.drift_rate})"


class LinearDriftBandit(GradualDriftBandit):
    """
    Convenience class for linear drift bandit.

    Parameters
    ----------
    n_arms : int
        Number of arms
    drift_rate : float
        Linear drift magnitude per step
    seed : Optional[int]
        Random seed
    """

    def __init__(
        self,
        n_arms: int,
        drift_rate: float = 0.01,
        seed: Optional[int] = None
    ):
        super().__init__(
            n_arms=n_arms,
            drift_type='linear',
            drift_rate=drift_rate,
            seed=seed
        )

    @property
    def name(self) -> str:
        return f"LinearDrift(rate={self.drift_rate})"
