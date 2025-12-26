"""
Thompson Sampling Algorithms

Bayesian approach to exploration that maintains posterior distributions
over arm means and samples from them to select actions.

Includes:
- ThompsonSampling: Standard TS for stationary environments (Gaussian)
- DiscountedThompsonSampling: TS with posterior forgetting for non-stationarity
"""

from typing import Optional
import numpy as np
from .base import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling with Gaussian reward model.

    Maintains a Gaussian posterior over each arm's mean reward.
    At each timestep, samples from posteriors and selects the
    arm with highest sampled value.

    Assumes known reward variance σ² and uses conjugate Gaussian prior.
    Posterior after n observations: N(μ_n, σ²/n) where μ_n is sample mean.

    Parameters
    ----------
    n_arms : int
        Number of arms
    prior_mean : float
        Prior mean for each arm. Use higher values for optimistic exploration.
    prior_std : float
        Prior standard deviation (uncertainty before any observations).
        Higher values encourage more exploration.
    reward_std : float
        Known/assumed standard deviation of rewards
    optimistic : bool
        If True, uses optimistic prior (prior_mean=5.0, prior_std=3.0)
        which encourages exploration and works better when reward
        range is unknown. Overrides prior_mean/prior_std if set.
    seed : Optional[int]
        Random seed

    References
    ----------
    Thompson, W. R. (1933).
    On the likelihood that one unknown probability exceeds another
    in view of the evidence of two samples.
    Biometrika, 25(3/4), 285-294.
    """

    def __init__(
        self,
        n_arms: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        reward_std: float = 1.0,
        optimistic: bool = False,
        seed: Optional[int] = None
    ):
        if prior_std <= 0:
            raise ValueError(f"prior_std must be positive, got {prior_std}")
        if reward_std <= 0:
            raise ValueError(f"reward_std must be positive, got {reward_std}")

        # Optimistic initialization for better exploration
        if optimistic:
            self.prior_mean = 5.0
            self.prior_std = 3.0
        else:
            self.prior_mean = prior_mean
            self.prior_std = prior_std

        self.reward_std = reward_std
        self.optimistic = optimistic
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize posterior parameters."""
        # Posterior mean and precision (1/variance)
        # Using precision makes updates simpler
        self._prior_precision = 1.0 / (self.prior_std ** 2)
        self._reward_precision = 1.0 / (self.reward_std ** 2)

        # Track sufficient statistics
        self._sum_rewards = np.zeros(self.n_arms)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def _get_posterior_params(self, arm: int) -> tuple:
        """
        Get posterior mean and std for an arm.

        Returns
        -------
        tuple
            (posterior_mean, posterior_std)
        """
        n = self._counts[arm]
        if n == 0:
            return self.prior_mean, self.prior_std

        # Posterior precision = prior_precision + n * reward_precision
        posterior_precision = self._prior_precision + n * self._reward_precision

        # Posterior mean = weighted combination of prior and data
        sample_mean = self._sum_rewards[arm] / n
        posterior_mean = (
            self._prior_precision * self.prior_mean +
            n * self._reward_precision * sample_mean
        ) / posterior_precision

        posterior_std = 1.0 / np.sqrt(posterior_precision)

        return posterior_mean, posterior_std

    def _get_all_posterior_params(self) -> tuple:
        """
        Get posterior means and stds for all arms (vectorized).

        Returns
        -------
        tuple
            (posterior_means, posterior_stds) as numpy arrays
        """
        n = self._counts.astype(float)
        mask = n > 0

        # Posterior precision for all arms
        posterior_precision = np.where(
            mask,
            self._prior_precision + n * self._reward_precision,
            self._prior_precision
        )

        # Sample means (avoid division by zero)
        sample_means = np.where(mask, self._sum_rewards / np.maximum(n, 1), 0.0)

        # Posterior means
        posterior_means = np.where(
            mask,
            (self._prior_precision * self.prior_mean +
             n * self._reward_precision * sample_means) / posterior_precision,
            self.prior_mean
        )

        # Posterior stds
        posterior_stds = 1.0 / np.sqrt(posterior_precision)

        return posterior_means, posterior_stds

    def select_action(self) -> int:
        """Select action by sampling from posteriors (vectorized)."""
        means, stds = self._get_all_posterior_params()
        samples = self.rng.normal(means, stds)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        """Update posterior with new observation."""
        self._counts[action] += 1
        self._sum_rewards[action] += reward
        self.t += 1

    def get_action_values(self) -> np.ndarray:
        """Get posterior means (vectorized)."""
        means, _ = self._get_all_posterior_params()
        return means.copy()

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    def get_posterior_stds(self) -> np.ndarray:
        """Get posterior standard deviations (vectorized)."""
        _, stds = self._get_all_posterior_params()
        return stds.copy()

    @property
    def name(self) -> str:
        if self.optimistic:
            return "Thompson Sampling (Optimistic)"
        return "Thompson Sampling"


class DiscountedThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling with discounted posterior for non-stationarity.

    Applies exponential discounting to past observations, making the
    posterior "forget" old data. This allows tracking changing distributions.

    Parameters
    ----------
    n_arms : int
        Number of arms
    gamma : float
        Discount factor (0 < γ < 1). Smaller values adapt faster.
    prior_mean : float
        Prior mean for each arm
    prior_std : float
        Prior standard deviation
    reward_std : float
        Assumed standard deviation of rewards
    optimistic : bool
        If True, uses optimistic prior (prior_mean=5.0, prior_std=3.0)
    seed : Optional[int]
        Random seed

    Notes
    -----
    The effective sample size is approximately 1/(1-γ).
    For γ=0.99, this is about 100 samples.
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float = 0.99,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        reward_std: float = 1.0,
        optimistic: bool = False,
        seed: Optional[int] = None
    ):
        if not 0 < gamma < 1:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")
        if prior_std <= 0:
            raise ValueError(f"prior_std must be positive, got {prior_std}")
        if reward_std <= 0:
            raise ValueError(f"reward_std must be positive, got {reward_std}")

        self.gamma = gamma
        self.optimistic = optimistic

        if optimistic:
            self.prior_mean = 5.0
            self.prior_std = 3.0
        else:
            self.prior_mean = prior_mean
            self.prior_std = prior_std

        self.reward_std = reward_std
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize discounted statistics."""
        self._prior_precision = 1.0 / (self.prior_std ** 2)
        self._reward_precision = 1.0 / (self.reward_std ** 2)

        # Discounted sufficient statistics
        self._discounted_sum = np.zeros(self.n_arms)
        self._discounted_count = np.zeros(self.n_arms)

    def _apply_discount(self) -> None:
        """Apply discount to statistics."""
        self._discounted_sum *= self.gamma
        self._discounted_count *= self.gamma

    def _get_posterior_params(self, arm: int) -> tuple:
        """Get discounted posterior parameters."""
        n = self._discounted_count[arm]

        if n < 1e-6:
            return self.prior_mean, self.prior_std

        # Discounted posterior
        posterior_precision = self._prior_precision + n * self._reward_precision
        sample_mean = self._discounted_sum[arm] / n
        posterior_mean = (
            self._prior_precision * self.prior_mean +
            n * self._reward_precision * sample_mean
        ) / posterior_precision

        posterior_std = 1.0 / np.sqrt(posterior_precision)

        return posterior_mean, posterior_std

    def _get_all_posterior_params(self) -> tuple:
        """
        Get discounted posterior means and stds for all arms (vectorized).

        Returns
        -------
        tuple
            (posterior_means, posterior_stds) as numpy arrays
        """
        n = self._discounted_count
        mask = n > 1e-6

        # Posterior precision for all arms
        posterior_precision = np.where(
            mask,
            self._prior_precision + n * self._reward_precision,
            self._prior_precision
        )

        # Sample means (avoid division by zero)
        sample_means = np.where(mask, self._discounted_sum / np.maximum(n, 1e-10), 0.0)

        # Posterior means
        posterior_means = np.where(
            mask,
            (self._prior_precision * self.prior_mean +
             n * self._reward_precision * sample_means) / posterior_precision,
            self.prior_mean
        )

        # Posterior stds
        posterior_stds = 1.0 / np.sqrt(posterior_precision)

        return posterior_means, posterior_stds

    def select_action(self) -> int:
        """Select action by sampling from discounted posteriors (vectorized)."""
        means, stds = self._get_all_posterior_params()
        samples = self.rng.normal(means, stds)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        """Update discounted posterior."""
        self._apply_discount()
        self._discounted_count[action] += 1
        self._discounted_sum[action] += reward
        self.t += 1

    def get_action_values(self) -> np.ndarray:
        """Get discounted posterior means (vectorized)."""
        means, _ = self._get_all_posterior_params()
        return means.copy()

    def get_action_counts(self) -> np.ndarray:
        """Get discounted counts."""
        return self._discounted_count.copy()

    @property
    def name(self) -> str:
        base = f"Discounted-TS(γ={self.gamma})"
        if self.optimistic:
            return f"{base} (Optimistic)"
        return base


class BetaThompsonSampling(BanditAlgorithm):
    """
    Thompson Sampling with Beta-Bernoulli model.

    For binary (0/1) rewards. Maintains Beta posterior over success probability.

    Parameters
    ----------
    n_arms : int
        Number of arms
    alpha_prior : float
        Prior alpha parameter (pseudo-successes)
    beta_prior : float
        Prior beta parameter (pseudo-failures)
    seed : Optional[int]
        Random seed

    Notes
    -----
    alpha_prior = beta_prior = 1 gives uniform prior.
    """

    def __init__(
        self,
        n_arms: int,
        alpha_prior: float = 1.0,
        beta_prior: float = 1.0,
        seed: Optional[int] = None
    ):
        if alpha_prior <= 0:
            raise ValueError(f"alpha_prior must be positive, got {alpha_prior}")
        if beta_prior <= 0:
            raise ValueError(f"beta_prior must be positive, got {beta_prior}")
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        super().__init__(n_arms=n_arms, seed=seed)

    def _initialize(self) -> None:
        """Initialize Beta parameters."""
        self._alphas = np.full(self.n_arms, self.alpha_prior)
        self._betas = np.full(self.n_arms, self.beta_prior)
        self._counts = np.zeros(self.n_arms, dtype=int)

    def select_action(self) -> int:
        """Select action by sampling from Beta posteriors."""
        samples = self.rng.beta(self._alphas, self._betas)
        return int(np.argmax(samples))

    def update(self, action: int, reward: float) -> None:
        """Update Beta posterior (reward should be 0 or 1)."""
        if not 0 <= reward <= 1:
            raise ValueError(
                f"BetaThompsonSampling requires reward in [0, 1], got {reward}. "
                "Use ThompsonSampling for non-binary rewards."
            )
        self._counts[action] += 1
        # Treat reward as Bernoulli: add to alpha (success) or beta (failure)
        self._alphas[action] += reward
        self._betas[action] += (1 - reward)
        self.t += 1

    def get_action_values(self) -> np.ndarray:
        """Get posterior means."""
        return self._alphas / (self._alphas + self._betas)

    def get_action_counts(self) -> np.ndarray:
        return self._counts.copy()

    @property
    def name(self) -> str:
        return "Beta-TS"
