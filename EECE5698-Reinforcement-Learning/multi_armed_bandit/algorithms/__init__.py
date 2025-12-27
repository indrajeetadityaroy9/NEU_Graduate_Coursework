"""
Bandit Algorithms Package

This package provides implementations of various multi-armed bandit algorithms,
including both stationary and non-stationary variants.

Stationary Algorithms:
    - EpsilonGreedy: ε-greedy with sample average updates
    - DecayingEpsilonGreedy: ε-greedy with decaying exploration
    - UCB1: Upper Confidence Bound
    - ThompsonSampling: Bayesian posterior sampling (Gaussian)
    - BetaThompsonSampling: Thompson Sampling for Bernoulli rewards
    - GradientBandit: Softmax policy gradient

Non-Stationary Variants:
    - EpsilonGreedyConstant: ε-greedy with constant α (for non-stationarity)
    - DiscountedUCB: UCB with exponential discounting
    - SlidingWindowUCB: UCB using only recent observations
    - DiscountedThompsonSampling: TS with posterior forgetting
    - EXP3: Adversarial bandit algorithm
    - Rexp3: Restarting EXP3
    - EXP3IX: EXP3 with implicit exploration
    - EntropyRegularizedGradient: Maximum entropy exploration
"""

from .base import BanditAlgorithm
from .epsilon_greedy import EpsilonGreedy, EpsilonGreedyConstant, DecayingEpsilonGreedy
from .ucb import UCB1, DiscountedUCB, SlidingWindowUCB
from .thompson_sampling import ThompsonSampling, DiscountedThompsonSampling, BetaThompsonSampling
from .gradient_bandit import GradientBandit, EntropyRegularizedGradient
from .exp3 import EXP3, Rexp3, EXP3IX

# Algorithm registry for easy access
ALGORITHMS = {
    # Stationary
    'epsilon_greedy': EpsilonGreedy,
    'epsilon_greedy_decaying': DecayingEpsilonGreedy,
    'ucb1': UCB1,
    'thompson_sampling': ThompsonSampling,
    'beta_ts': BetaThompsonSampling,
    'gradient_bandit': GradientBandit,

    # Non-stationary
    'epsilon_greedy_constant': EpsilonGreedyConstant,
    'discounted_ucb': DiscountedUCB,
    'sliding_window_ucb': SlidingWindowUCB,
    'discounted_ts': DiscountedThompsonSampling,
    'exp3': EXP3,
    'rexp3': Rexp3,
    'exp3ix': EXP3IX,
    'entropy_gradient': EntropyRegularizedGradient,
}

# Group algorithms by category
STATIONARY_ALGORITHMS = [
    'epsilon_greedy', 'epsilon_greedy_decaying', 'ucb1',
    'thompson_sampling', 'beta_ts', 'gradient_bandit',
]

NONSTATIONARY_ALGORITHMS = [
    'epsilon_greedy_constant', 'discounted_ucb', 'sliding_window_ucb',
    'discounted_ts', 'entropy_gradient',
]

ADVERSARIAL_ALGORITHMS = ['exp3', 'rexp3', 'exp3ix']

__all__ = [
    # Base class
    'BanditAlgorithm',
    # Epsilon-Greedy variants
    'EpsilonGreedy', 'EpsilonGreedyConstant', 'DecayingEpsilonGreedy',
    # UCB variants
    'UCB1', 'DiscountedUCB', 'SlidingWindowUCB',
    # Thompson Sampling variants
    'ThompsonSampling', 'DiscountedThompsonSampling', 'BetaThompsonSampling',
    # Gradient-based
    'GradientBandit', 'EntropyRegularizedGradient',
    # Adversarial
    'EXP3', 'Rexp3', 'EXP3IX',
    # Registries
    'ALGORITHMS', 'STATIONARY_ALGORITHMS', 'NONSTATIONARY_ALGORITHMS', 'ADVERSARIAL_ALGORITHMS',
]
