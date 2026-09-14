"""
Non-Stationary Bandit Environments Package

This package provides various bandit environments with different
types of non-stationarity for testing algorithm robustness.

Environments:
    - StationaryBandit: Baseline with fixed reward distributions
    - AbruptChangeBandit: Optimal arm changes suddenly at fixed intervals
    - RotatingBandit: Optimal arm cycles through arms deterministically
    - GradualDriftBandit: Arm means drift continuously over time
    - RandomWalkBandit: Arm means follow random walks
    - LinearDriftBandit: Arm means drift linearly
"""

from .base import BanditEnvironment
from .stationary import StationaryBandit, create_standard_bandit
from .abrupt_change import AbruptChangeBandit, RotatingBandit
from .gradual_drift import GradualDriftBandit, RandomWalkBandit, LinearDriftBandit

ENVIRONMENTS = {
    'stationary': StationaryBandit,
    'abrupt': AbruptChangeBandit,
    'rotating': RotatingBandit,
    'drift': GradualDriftBandit,
    'random_walk': RandomWalkBandit,
    'linear_drift': LinearDriftBandit,
}

__all__ = [
    'BanditEnvironment',
    'StationaryBandit',
    'create_standard_bandit',
    'AbruptChangeBandit',
    'RotatingBandit',
    'GradualDriftBandit',
    'RandomWalkBandit',
    'LinearDriftBandit',
    'ENVIRONMENTS',
]
