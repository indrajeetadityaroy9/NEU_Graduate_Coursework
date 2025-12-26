"""
EXP3 (Exponential-weight algorithm for Exploration and Exploitation)

Adversarial bandit algorithms that work under worst-case assumptions
about how rewards are generated. Useful as a robust baseline for
non-stationary environments.
"""

from typing import Optional
import numpy as np
from .base import BanditAlgorithm


class EXP3(BanditAlgorithm):
    """
    EXP3 algorithm for adversarial bandits.

    Maintains weights for each arm and samples proportionally.
    Uses importance-weighted reward estimates to handle partial feedback.

    Parameters
    ----------
    n_arms : int
        Number of arms
    gamma : float
        Exploration parameter (mixing with uniform). Should be in (0, 1].
        Higher gamma = more exploration.
    eta : float
        Learning rate. If None, uses optimal rate sqrt(ln(K)/(K*T)).
    seed : Optional[int]
        Random seed

    References
    ----------
    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem.
    SIAM Journal on Computing, 32(1), 48-77.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.1,
        eta: Optional[float] = None,
        seed: Optional[int] = None
    ):
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if eta is not None and eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        self.gamma = gamma
        self.eta = eta
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize weights."""
        self._weights = np.ones(self.n_arms)
        self._cumulative_rewards = np.zeros(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)

        # Compute learning rate if not specified
        if self.eta is None:
            # This assumes horizon T is unknown; use conservative rate
            self._eta = np.sqrt(np.log(self.n_arms) / self.n_arms)
        else:
            self._eta = self.eta

    def _get_probabilities(self) -> np.ndarray:
        """Compute sampling probabilities."""
        # Mix weights with uniform distribution
        weight_probs = self._weights / np.sum(self._weights)
        uniform = np.ones(self.n_arms) / self.n_arms
        return (1 - self.gamma) * weight_probs + self.gamma * uniform

    def select_action(self) -> int:
        """Select action by sampling from probability distribution."""
        probs = self._get_probabilities()
        return self.rng.choice(self.n_arms, p=probs)

    def update(self, action: int, reward: float) -> None:
        """Update weights using importance-weighted reward."""
        self._counts[action] += 1
        self.t += 1

        probs = self._get_probabilities()

        # Importance-weighted reward estimate
        # Only update for selected action (others get 0 estimate)
        reward_estimate = reward / probs[action]

        # Accumulate for tracking
        self._cumulative_rewards[action] += reward

        # Update weight using exponential update
        self._weights[action] *= np.exp(self._eta * reward_estimate / self.n_arms)

        # Renormalize to prevent numerical issues
        if np.max(self._weights) > 1e10:
            self._weights /= np.max(self._weights)

    def get_action_values(self) -> np.ndarray:
        """Return cumulative rewards (not estimates)."""
        return self._cumulative_rewards.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def get_probabilities(self) -> np.ndarray:
        """Get current sampling probabilities."""
        return self._get_probabilities()

    @property
    def name(self) -> str:
        return f"EXP3(γ={self.gamma})"


class Rexp3(BanditAlgorithm):
    """
    Restarting EXP3 for non-stationary adversarial bandits.

    Periodically resets the weights to adapt to distribution changes.
    Useful when changes are approximately known or can be detected.

    Parameters
    ----------
    n_arms : int
        Number of arms
    gamma : float
        Exploration parameter
    restart_interval : int
        Number of steps between restarts
    eta : float
        Learning rate (if None, uses default)
    seed : Optional[int]
        Random seed

    References
    ----------
    Besbes, O., Gur, Y., & Zeevi, A. (2014).
    Stochastic multi-armed-bandit problem with non-stationary rewards.
    NeurIPS 2014.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.1,
        restart_interval: int = 100,
        eta: Optional[float] = None,
        seed: Optional[int] = None
    ):
        if not 0 < gamma <= 1:
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if restart_interval < 1:
            raise ValueError(f"restart_interval must be positive, got {restart_interval}")
        if eta is not None and eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        self.gamma = gamma
        self.restart_interval = restart_interval
        self.eta = eta
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize EXP3 state."""
        self._weights = np.ones(self.n_arms)
        self._cumulative_rewards = np.zeros(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)
        self._steps_since_restart = 0

        if self.eta is None:
            self._eta = np.sqrt(np.log(self.n_arms) / self.n_arms)
        else:
            self._eta = self.eta

    def _restart(self) -> None:
        """Reset weights to uniform."""
        self._weights = np.ones(self.n_arms)
        self._steps_since_restart = 0

    def _get_probabilities(self) -> np.ndarray:
        """Compute sampling probabilities."""
        weight_probs = self._weights / np.sum(self._weights)
        uniform = np.ones(self.n_arms) / self.n_arms
        return (1 - self.gamma) * weight_probs + self.gamma * uniform

    def select_action(self) -> int:
        """Select action by sampling."""
        probs = self._get_probabilities()
        return self.rng.choice(self.n_arms, p=probs)

    def update(self, action: int, reward: float) -> None:
        """Update weights and check for restart."""
        self._counts[action] += 1
        self.t += 1
        self._steps_since_restart += 1

        probs = self._get_probabilities()

        # Importance-weighted update
        reward_estimate = reward / probs[action]
        self._cumulative_rewards[action] += reward
        self._weights[action] *= np.exp(self._eta * reward_estimate / self.n_arms)

        if np.max(self._weights) > 1e10:
            self._weights /= np.max(self._weights)

        # Check for restart
        if self._steps_since_restart >= self.restart_interval:
            self._restart()

    def get_action_values(self) -> np.ndarray:
        return self._cumulative_rewards.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"Rexp3(interval={self.restart_interval})"


class EXP3IX(BanditAlgorithm):
    """
    EXP3-IX: EXP3 with Implicit eXploration.

    A variant that handles the exploration implicitly through
    a modified loss estimator, avoiding the explicit mixing parameter.

    Parameters
    ----------
    n_arms : int
        Number of arms
    eta : float
        Learning rate
    gamma : float
        Implicit exploration parameter (added to probability denominator)
    seed : Optional[int]
        Random seed

    References
    ----------
    Neu, G. (2015).
    Explore no more: Improved high-probability regret bounds
    for non-stochastic bandits.
    NeurIPS 2015.
    """

    def __init__(
        self,
        n_arms: int,
        eta: float = 0.1,
        gamma: float = 0.01,
        seed: Optional[int] = None
    ):
        if eta <= 0:
            raise ValueError(f"eta must be positive, got {eta}")
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.eta = eta
        self.gamma = gamma
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize weights."""
        self._weights = np.ones(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def _get_probabilities(self) -> np.ndarray:
        """Compute sampling probabilities (just softmax of weights)."""
        probs = self._weights / np.sum(self._weights)
        return probs

    def select_action(self) -> int:
        """Select action by sampling."""
        probs = self._get_probabilities()
        return self.rng.choice(self.n_arms, p=probs)

    def update(self, action: int, reward: float) -> None:
        """Update using implicit exploration estimator."""
        self._counts[action] += 1
        self.t += 1

        probs = self._get_probabilities()

        # Implicit exploration: add gamma to denominator
        reward_estimate = reward / (probs[action] + self.gamma)

        # Update weight
        self._weights[action] *= np.exp(self.eta * reward_estimate)

        # Renormalize
        if np.max(self._weights) > 1e10:
            self._weights /= np.max(self._weights)

    def get_action_values(self) -> np.ndarray:
        return self._weights.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"EXP3-IX(η={self.eta})"
