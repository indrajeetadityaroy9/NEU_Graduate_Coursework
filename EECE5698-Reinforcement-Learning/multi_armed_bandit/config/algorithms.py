"""
Algorithm configurations and factory functions.

Uses the existing ALGORITHMS registry from algorithms/__init__.py.
"""
from typing import Dict, List, Any, Optional

import numpy as np

from ..algorithms import (
    EpsilonGreedy, EpsilonGreedyConstant, DecayingEpsilonGreedy,
    UCB1, DiscountedUCB, SlidingWindowUCB,
    ThompsonSampling, DiscountedThompsonSampling, BetaThompsonSampling,
    GradientBandit, EntropyRegularizedGradient,
    EXP3, Rexp3, EXP3IX,
    ALGORITHMS,
)


# =============================================================================
# Standard Algorithm Configurations
# =============================================================================

# Each config: {'name': str, 'class': Type, 'kwargs': dict}

STATIONARY_CONFIGS = [
    {'name': 'EpsilonGreedy', 'class': EpsilonGreedy, 'kwargs': {'epsilon': 0.1}},
    {'name': 'DecayingEpsilon', 'class': DecayingEpsilonGreedy, 'kwargs': {}},
    {'name': 'UCB1', 'class': UCB1, 'kwargs': {'c': np.sqrt(2)}},
    {'name': 'ThompsonSampling', 'class': ThompsonSampling, 'kwargs': {}},
    {'name': 'GradientBandit', 'class': GradientBandit, 'kwargs': {'alpha': 0.1}},
]

NONSTATIONARY_CONFIGS = [
    {'name': 'EpsGreedy-Const', 'class': EpsilonGreedyConstant,
     'kwargs': {'epsilon': 0.1, 'alpha': 0.1}},
    # D-UCB uses gamma as discount factor (0 < γ < 1)
    {'name': 'D-UCB(0.99)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.99}},
    {'name': 'D-UCB(0.95)', 'class': DiscountedUCB, 'kwargs': {'gamma': 0.95}},
    {'name': 'SW-UCB(100)', 'class': SlidingWindowUCB, 'kwargs': {'window_size': 100}},
    # D-TS uses gamma as discount factor (0 < γ < 1)
    {'name': 'D-TS(0.99)', 'class': DiscountedThompsonSampling, 'kwargs': {'gamma': 0.99}},
    # EXP3 uses exploration_rate for uniform mixing (0 < γ ≤ 1)
    {'name': 'EXP3', 'class': EXP3, 'kwargs': {'exploration_rate': 0.1}},
    {'name': 'Rexp3(100)', 'class': Rexp3, 'kwargs': {'exploration_rate': 0.1, 'restart_interval': 100}},
    {'name': 'EntropyGradient', 'class': EntropyRegularizedGradient,
     'kwargs': {'alpha': 0.1, 'tau': 0.1}},
]

ADVERSARIAL_CONFIGS = [
    # EXP3/Rexp3 use exploration_rate for uniform mixing
    {'name': 'EXP3', 'class': EXP3, 'kwargs': {'exploration_rate': 0.1}},
    {'name': 'Rexp3(100)', 'class': Rexp3, 'kwargs': {'exploration_rate': 0.1, 'restart_interval': 100}},
    # EXP3IX uses implicit_exploration for stability bias
    {'name': 'EXP3IX', 'class': EXP3IX, 'kwargs': {'implicit_exploration': 0.01}},
]

# Combined standard suite (for comprehensive studies)
# Avoid duplicates by using a set for names
_seen_names = set()
FULL_ALGORITHM_CONFIGS = []
for config in STATIONARY_CONFIGS + NONSTATIONARY_CONFIGS:
    if config['name'] not in _seen_names:
        FULL_ALGORITHM_CONFIGS.append(config)
        _seen_names.add(config['name'])


# =============================================================================
# Factory Functions
# =============================================================================

def create_algorithm(
    name: str,
    n_arms: int,
    seed: int = None,
    **override_kwargs
):
    """
    Create a single algorithm instance by name.

    Args:
        name: Algorithm name (config name like 'D-UCB(0.99)' or registry key like 'discounted_ucb')
        n_arms: Number of arms
        seed: Random seed
        **override_kwargs: Override default kwargs

    Returns:
        Configured algorithm instance
    """
    # Look up in registry first (by registry key)
    if name in ALGORITHMS:
        algo_class = ALGORITHMS[name]
        return algo_class(n_arms=n_arms, seed=seed, **override_kwargs)

    # Otherwise find in configs (by display name)
    for config in FULL_ALGORITHM_CONFIGS + ADVERSARIAL_CONFIGS:
        if config['name'] == name:
            kwargs = {**config['kwargs'], **override_kwargs}
            return config['class'](n_arms=n_arms, seed=seed, **kwargs)

    raise ValueError(f"Unknown algorithm: {name}. "
                     f"Available config names: {[c['name'] for c in FULL_ALGORITHM_CONFIGS]}")


def create_algorithm_suite(
    n_arms: int,
    seed: int = 42,
    include_stationary: bool = True,
    include_nonstationary: bool = True,
    include_adversarial: bool = False,
) -> Dict[str, Any]:
    """
    Create a dictionary of algorithm instances for benchmarking.

    Args:
        n_arms: Number of arms
        seed: Random seed for all algorithms
        include_stationary: Include stationary algorithms
        include_nonstationary: Include non-stationary algorithms
        include_adversarial: Include adversarial algorithms

    Returns:
        Dict mapping algorithm name to instance
    """
    algorithms = {}

    configs = []
    if include_stationary:
        configs.extend(STATIONARY_CONFIGS)
    if include_nonstationary:
        configs.extend(NONSTATIONARY_CONFIGS)
    if include_adversarial:
        configs.extend([c for c in ADVERSARIAL_CONFIGS if c['name'] not in
                        [cfg['name'] for cfg in configs]])

    for config in configs:
        name = config['name']
        if name not in algorithms:  # Avoid duplicates
            algorithms[name] = config['class'](
                n_arms=n_arms,
                seed=seed,
                **config['kwargs']
            )

    return algorithms


def get_algorithm_configs(
    include_stationary: bool = True,
    include_nonstationary: bool = True,
    include_adversarial: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get algorithm config dicts for ExperimentConfig.

    Returns list of dicts with 'name', 'class', 'kwargs' keys.
    """
    configs = []
    seen_names = set()

    if include_stationary:
        for c in STATIONARY_CONFIGS:
            if c['name'] not in seen_names:
                configs.append(c)
                seen_names.add(c['name'])

    if include_nonstationary:
        for c in NONSTATIONARY_CONFIGS:
            if c['name'] not in seen_names:
                configs.append(c)
                seen_names.add(c['name'])

    if include_adversarial:
        for c in ADVERSARIAL_CONFIGS:
            if c['name'] not in seen_names:
                configs.append(c)
                seen_names.add(c['name'])

    return configs
