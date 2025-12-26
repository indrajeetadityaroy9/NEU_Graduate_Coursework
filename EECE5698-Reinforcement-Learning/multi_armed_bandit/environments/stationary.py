"""
Stationary Bandit Environment

A baseline environment where arm reward distributions never change.
Used as a control condition to measure algorithm performance degradation
under non-stationarity.
"""

from typing import Optional, List
import numpy as np
from .base import BanditEnvironment


class StationaryBandit(BanditEnvironment):
    """
    Stationary multi-armed bandit with fixed Gaussian reward distributions.

    Parameters
    ----------
    n_arms : int
        Number of arms
    arm_means : Optional[List[float]]
        Mean reward for each arm. If None, randomly generated.
    arm_stds : Optional[List[float]]
        Standard deviation for each arm. If None, all set to 1.0.
    seed : Optional[int]
        Random seed for reproducibility

    Examples
    --------
    >>> env = StationaryBandit(n_arms=5, arm_means=[1, 2, 3, 4, 5])
    >>> reward = env.pull(4)  # Pull best arm
    >>> env.step()  # Advance time
    """

    def __init__(
        self,
        n_arms: int,
        arm_means: Optional[List[float]] = None,
        arm_stds: Optional[List[float]] = None,
        seed: Optional[int] = None
    ):
        self._arm_means_init = arm_means
        self._arm_stds_init = arm_stds
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize arm distributions."""
        if self._arm_means_init is not None:
            self._means = np.array(self._arm_means_init, dtype=float)
        else:
            # Random means uniformly in [0, 10]
            self._means = self.rng.uniform(0, 10, size=self.n_arms)

        if self._arm_stds_init is not None:
            self._stds = np.array(self._arm_stds_init, dtype=float)
        else:
            self._stds = np.ones(self.n_arms)

        # Record initial optimal arm
        self._optimal_arm_history = [(0, self.get_optimal_arm())]
        self._optimal_value_history = [self.get_optimal_value()]

    def _get_arm_mean(self, arm: int) -> float:
        return self._means[arm]

    def _get_arm_std(self, arm: int) -> float:
        return self._stds[arm]

    @property
    def name(self) -> str:
        return "Stationary"


def create_standard_bandit(n_arms: int = 10, gap: float = 1.0, seed: Optional[int] = None) -> StationaryBandit:
    """
    Create a standard test bandit with controlled gap.

    Parameters
    ----------
    n_arms : int
        Number of arms
    gap : float
        Gap between best and second-best arm
    seed : Optional[int]
        Random seed

    Returns
    -------
    StationaryBandit
        Configured bandit environment
    """
    # Arm 0 is optimal with mean = gap, others have mean = 0
    means = [0.0] * n_arms
    means[0] = gap
    return StationaryBandit(n_arms=n_arms, arm_means=means, seed=seed)
