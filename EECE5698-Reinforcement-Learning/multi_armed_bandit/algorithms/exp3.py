"""
EXP3 (Exponential-weight algorithm for Exploration and Exploitation)

Adversarial bandit algorithms that work under worst-case assumptions
about how rewards are generated. Useful as a robust baseline for
non-stationary environments.

Note on Parameter Naming:
    - `exploration_rate` (γ_exp): Mixing parameter for uniform exploration in EXP3/Rexp3
    - `implicit_exploration` (γ_ix): Implicit exploration bias in EXP3-IX
    These are distinct from `gamma` (γ) used as discount factor in D-UCB/Discounted-TS.
"""

from typing import Optional
import numpy as np
from .base import BanditAlgorithm
from .utils import mixed_probability, validate_exploration_rate, validate_positive, validate_positive_int


class EXP3(BanditAlgorithm):
    """
    EXP3 algorithm for adversarial bandits.

    Maintains weights for each arm and samples proportionally.
    Uses importance-weighted reward estimates to handle partial feedback.

    Mathematical Formulation
    ------------------------
    Probability: p_t(a) = (1 - γ) * w_t(a)/Σw_t + γ/K
    Weight update: w_{t+1}(a) = w_t(a) * exp(η * r̂_t(a) / K)
    Importance-weighted estimate: r̂_t(a) = r_t / p_t(a) if a = A_t, else 0

    Parameters
    ----------
    n_arms : int
        Number of arms (K in literature)
    exploration_rate : float
        Mixing parameter γ ∈ (0, 1] for uniform exploration.
        p(a) = (1-γ)*w(a)/Σw + γ/K. Higher = more exploration.
        Note: This is distinct from discount factor γ in D-UCB.
    eta : float, optional
        Learning rate η > 0. If None, uses optimal rate √(ln(K)/K).
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002).
    The nonstochastic multiarmed bandit problem.
    SIAM Journal on Computing, 32(1), 48-77.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_rate: float = 0.1,
        eta: Optional[float] = None,
        seed: Optional[int] = None
    ):
        validate_exploration_rate(exploration_rate)
        if eta is not None:
            validate_positive(eta, "eta")
        self.exploration_rate = exploration_rate
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
        """Compute sampling probabilities: p(a) = (1-γ)*w(a)/Σw + γ/K."""
        return mixed_probability(self._weights, self.exploration_rate)

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
        return f"EXP3(γ={self.exploration_rate})"


class Rexp3(BanditAlgorithm):
    """
    Restarting EXP3 for non-stationary adversarial bandits.

    Periodically resets the weights to adapt to distribution changes.
    Useful when changes are approximately known or can be detected.

    Mathematical Formulation
    ------------------------
    Same as EXP3, but weights reset to uniform every τ steps:
    If t mod τ = 0: w(a) ← 1 for all a

    Parameters
    ----------
    n_arms : int
        Number of arms (K in literature)
    exploration_rate : float
        Mixing parameter γ ∈ (0, 1] for uniform exploration.
        Note: This is distinct from discount factor γ in D-UCB.
    restart_interval : int
        Number of steps τ between weight resets
    eta : float, optional
        Learning rate η > 0. If None, uses default.
    seed : int, optional
        Random seed for reproducibility

    References
    ----------
    Besbes, O., Gur, Y., & Zeevi, A. (2014).
    Stochastic multi-armed-bandit problem with non-stationary rewards.
    NeurIPS 2014.
    """

    def __init__(
        self,
        n_arms: int,
        exploration_rate: float = 0.1,
        restart_interval: int = 100,
        eta: Optional[float] = None,
        seed: Optional[int] = None
    ):
        validate_exploration_rate(exploration_rate)
        validate_positive_int(restart_interval, "restart_interval")
        if eta is not None:
            validate_positive(eta, "eta")
        self.exploration_rate = exploration_rate
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
        """Compute sampling probabilities: p(a) = (1-γ)*w(a)/Σw + γ/K."""
        return mixed_probability(self._weights, self.exploration_rate)

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

    Mathematical Formulation
    ------------------------
    Probability: p_t(a) = w_t(a) / Σw_t  (no explicit mixing)
    Implicit exploration estimator: r̂_t(a) = r_t / (p_t(a) + γ_ix)
    Weight update: w_{t+1}(a) = w_t(a) * exp(η * r̂_t(a))

    Parameters
    ----------
    n_arms : int
        Number of arms (K in literature)
    eta : float
        Learning rate η > 0
    implicit_exploration : float
        Implicit exploration bias γ_ix > 0 added to probability denominator.
        Controls stability of importance weights. Smaller = less bias, more variance.
        Note: This is distinct from both exploration_rate in EXP3 and
        discount factor γ in D-UCB.
    seed : int, optional
        Random seed for reproducibility

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
        implicit_exploration: float = 0.01,
        seed: Optional[int] = None
    ):
        validate_positive(eta, "eta")
        validate_positive(implicit_exploration, "implicit_exploration")
        self.eta = eta
        self.implicit_exploration = implicit_exploration
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
        """Update using implicit exploration estimator: r̂ = r / (p(a) + γ_ix)."""
        self._counts[action] += 1
        self.t += 1

        probs = self._get_probabilities()

        # Implicit exploration: add γ_ix to denominator for stability
        reward_estimate = reward / (probs[action] + self.implicit_exploration)

        # Update weight: w(a) ← w(a) * exp(η * r̂)
        self._weights[action] *= np.exp(self.eta * reward_estimate)

        # Renormalize to prevent numerical overflow
        if np.max(self._weights) > 1e10:
            self._weights /= np.max(self._weights)

    def get_action_values(self) -> np.ndarray:
        """Return current weights (not value estimates)."""
        return self._weights.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return f"EXP3-IX(η={self.eta})"
