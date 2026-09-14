"""
Abstract Base Class for Bandit Algorithms

All bandit algorithms should inherit from BanditAlgorithm and implement
the required abstract methods for action selection and updating.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np


class BanditAlgorithm(ABC):
    """
    Abstract base class for multi-armed bandit algorithms.

    Parameters
    ----------
    n_arms : int
        Number of arms in the bandit problem
    seed : Optional[int]
        Random seed for reproducibility

    Attributes
    ----------
    n_arms : int
        Number of arms
    t : int
        Current timestep (0-indexed)
    rng : np.random.Generator
        Random number generator
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None):
        self.n_arms = n_arms
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self._initialize()

    @abstractmethod
    def _initialize(self) -> None:
        """Initialize algorithm-specific state. Called by __init__ and reset."""
        pass

    @abstractmethod
    def select_action(self) -> int:
        """
        Select an action (arm) to pull.

        Returns
        -------
        int
            Index of the selected arm (0 to n_arms-1)
        """
        pass

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """
        Update the algorithm's state after observing a reward.

        Parameters
        ----------
        action : int
            The action that was taken
        reward : float
            The reward received
        """
        pass

    def reset(self) -> None:
        """Reset the algorithm to its initial state."""
        self.t = 0
        self._initialize()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable name for the algorithm."""
        pass

    def get_action_values(self) -> np.ndarray:
        """
        Get current action values maintained by the algorithm.

        Returns
        -------
        np.ndarray
            Values for each arm. Semantics vary by algorithm type:

            - **Value-based** (ε-Greedy, UCB, Thompson Sampling):
              Returns estimated mean rewards (Q-values)
            - **Preference-based** (Gradient Bandit):
              Returns preference values H(a), NOT probabilities.
              Use get_action_probabilities() for π(a).
            - **Adversarial** (EXP3):
              Returns cumulative rewards (not normalized estimates)

        Notes
        -----
        Default returns zeros. See subclass implementations for details.
        """
        return np.zeros(self.n_arms)

    def get_action_counts(self) -> np.ndarray:
        """
        Get the number of times each action has been selected.

        Returns
        -------
        np.ndarray
            Count for each arm

        Notes
        -----
        Default implementation returns zeros. Override in subclasses.
        """
        return np.zeros(self.n_arms)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current internal state of the algorithm.

        Useful for debugging and analysis.

        Returns
        -------
        dict
            Dictionary containing internal state
        """
        return {
            'name': self.name,
            'n_arms': self.n_arms,
            't': self.t,
            'action_values': self.get_action_values().tolist(),
            'action_counts': self.get_action_counts().tolist(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_arms={self.n_arms})"
