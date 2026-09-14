"""
Abstract Base Class for Bandit Environments

All bandit environments should inherit from BanditEnvironment and implement
the required abstract methods for reward generation and state management.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import numpy as np


class BanditEnvironment(ABC):
    """
    Abstract base class for multi-armed bandit environments.

    Supports both stationary and non-stationary reward distributions.
    The environment tracks time internally and can change reward
    distributions based on the timestep.

    Parameters
    ----------
    n_arms : int
        Number of arms in the bandit
    seed : Optional[int]
        Random seed for reproducibility

    Attributes
    ----------
    n_arms : int
        Number of arms
    t : int
        Current timestep
    rng : np.random.Generator
        Random number generator
    _change_points : List[int]
        Timesteps where distribution changes occurred
    _optimal_arm_history : List[Tuple[int, int]]
        List of (timestep, optimal_arm) pairs
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None):
        if n_arms < 1:
            raise ValueError(f"n_arms must be positive, got {n_arms}")
        self.n_arms = n_arms
        self._seed = seed  # Store seed for deterministic reset
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._change_points: List[int] = []
        self._optimal_arm_history: List[Tuple[int, int]] = []
        self._optimal_value_history: List[float] = []
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize environment-specific state."""
        pass

    @abstractmethod
    def _get_arm_mean(self, arm: int) -> float:
        """
        Get the current mean reward for an arm.

        Parameters
        ----------
        arm : int
            Arm index

        Returns
        -------
        float
            Current mean reward for the arm
        """
        pass

    @abstractmethod
    def _get_arm_std(self, arm: int) -> float:
        """
        Get the current standard deviation for an arm.

        Parameters
        ----------
        arm : int
            Arm index

        Returns
        -------
        float
            Current standard deviation for the arm
        """
        pass

    def pull(self, arm: int) -> float:
        """
        Pull an arm and receive a reward.

        Parameters
        ----------
        arm : int
            Arm index to pull (0 to n_arms-1)

        Returns
        -------
        float
            Sampled reward from the arm's distribution
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm {arm}. Must be in [0, {self.n_arms-1}]")

        mean = self._get_arm_mean(arm)
        std = self._get_arm_std(arm)
        reward = self.rng.normal(mean, std)
        return reward

    def step(self) -> None:
        """
        Advance time by one step.

        This may trigger changes in the reward distribution for
        non-stationary environments.
        """
        self.t += 1
        self._check_for_change()

    def _check_for_change(self) -> None:
        """
        Check if a distribution change should occur.

        Override in non-stationary environments.
        """
        pass

    def get_optimal_arm(self) -> int:
        """
        Get the current optimal arm (arm with highest mean).

        Returns
        -------
        int
            Index of the optimal arm
        """
        means = np.array([self._get_arm_mean(a) for a in range(self.n_arms)])
        return int(np.argmax(means))

    def get_optimal_value(self) -> float:
        """
        Get the current optimal value (highest mean).

        Returns
        -------
        float
            Mean of the optimal arm
        """
        return self._get_arm_mean(self.get_optimal_arm())

    def get_arm_means(self) -> np.ndarray:
        """
        Get current means of all arms.

        Returns
        -------
        np.ndarray
            Array of means for each arm
        """
        return np.array([self._get_arm_mean(a) for a in range(self.n_arms)])

    def get_arm_gaps(self) -> np.ndarray:
        """
        Get the gap between each arm and the optimal arm.

        Returns
        -------
        np.ndarray
            Gap for each arm (0 for optimal arm)
        """
        means = self.get_arm_means()
        optimal = np.max(means)
        return optimal - means

    @property
    def change_points(self) -> List[int]:
        """
        Get list of timesteps where distribution changes occurred.

        Returns
        -------
        List[int]
            List of change point timesteps
        """
        return self._change_points.copy()

    def reset(self) -> None:
        """Reset the environment to its initial state."""
        self.t = 0
        self.rng = np.random.default_rng(self._seed)  # Reset RNG for determinism
        self._change_points = []
        self._optimal_arm_history = []
        self._optimal_value_history = []
        self._initialize()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the environment."""
        pass

    def get_info(self) -> dict:
        """
        Get information about the current environment state.

        Returns
        -------
        dict
            Environment information
        """
        return {
            'name': self.name,
            'n_arms': self.n_arms,
            't': self.t,
            'optimal_arm': self.get_optimal_arm(),
            'optimal_value': self.get_optimal_value(),
            'arm_means': self.get_arm_means().tolist(),
            'change_points': self.change_points,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"
