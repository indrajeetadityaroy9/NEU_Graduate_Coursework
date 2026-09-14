"""
Environment configurations for different experiment types.
"""
from typing import Dict, Any

from ..environments import (
    StationaryBandit,
    AbruptChangeBandit,
    RotatingBandit,
    GradualDriftBandit,
    RandomWalkBandit,
    LinearDriftBandit,
    ENVIRONMENTS,
)
from .defaults import N_ARMS, GAP


# =============================================================================
# Standard Environment Configurations
# =============================================================================

def _get_environment_configs() -> Dict[str, Dict[str, Any]]:
    """
    Generate environment configs with current defaults.

    Returns dict dynamically so it uses current N_ARMS/GAP values.
    """
    return {
        'stationary': {
            'class': StationaryBandit,
            'kwargs': {'arm_means': [GAP] + [0.0] * (N_ARMS - 1)},
            'display_name': 'Stationary',
        },
        'abrupt_100': {
            'class': AbruptChangeBandit,
            'kwargs': {'change_interval': 100, 'gap': GAP},
            'display_name': 'Abrupt (100)',
        },
        'abrupt_200': {
            'class': AbruptChangeBandit,
            'kwargs': {'change_interval': 200, 'gap': GAP},
            'display_name': 'Abrupt (200)',
        },
        'abrupt_500': {
            'class': AbruptChangeBandit,
            'kwargs': {'change_interval': 500, 'gap': GAP},
            'display_name': 'Abrupt (500)',
        },
        'drift': {
            'class': GradualDriftBandit,
            'kwargs': {'gap': GAP, 'drift_rate': 0.05},
            'display_name': 'Gradual Drift',
        },
        'random_walk': {
            'class': RandomWalkBandit,
            'kwargs': {'step_size': 0.01},
            'display_name': 'Random Walk',
        },
    }


# Default configs (static for backwards compatibility)
ENVIRONMENT_CONFIGS = _get_environment_configs()


def get_environment_config(name: str, **override_kwargs) -> Dict[str, Any]:
    """
    Get environment config with optional overrides.

    Args:
        name: Environment name from ENVIRONMENT_CONFIGS
        **override_kwargs: Override default kwargs

    Returns:
        Config dict with 'class', 'kwargs', 'display_name'
    """
    configs = _get_environment_configs()

    if name not in configs:
        raise ValueError(f"Unknown environment: {name}. "
                         f"Available: {list(configs.keys())}")

    config = configs[name].copy()
    config['kwargs'] = {**config['kwargs'], **override_kwargs}
    return config


def create_environment(
    name: str,
    n_arms: int = N_ARMS,
    seed: int = None,
    **override_kwargs
):
    """
    Create an environment instance.

    Args:
        name: Environment name
        n_arms: Number of arms
        seed: Random seed
        **override_kwargs: Override default kwargs

    Returns:
        Configured environment instance
    """
    config = get_environment_config(name, **override_kwargs)
    kwargs = {'n_arms': n_arms, 'seed': seed, **config['kwargs']}
    return config['class'](**kwargs)
