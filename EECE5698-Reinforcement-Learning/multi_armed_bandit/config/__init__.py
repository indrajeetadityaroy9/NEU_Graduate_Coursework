"""
Shared configuration module for multi-armed bandit experiments.

Provides centralized constants, algorithm configs, and environment configs
to avoid duplication across scripts.

Usage:
    from multi_armed_bandit.config import (
        # Constants
        N_RUNS, HORIZON, N_ARMS, GAP, SEED,

        # Algorithm configs
        FULL_ALGORITHM_CONFIGS,
        STATIONARY_CONFIGS,
        NONSTATIONARY_CONFIGS,
        create_algorithm_suite,
        get_algorithm_configs,

        # Environment configs
        ENVIRONMENT_CONFIGS,
        get_environment_config,
        create_environment,
    )
"""

from .defaults import (
    N_RUNS,
    N_RUNS_QUICK,
    HORIZON,
    N_ARMS,
    GAP,
    SEED,
    MAX_WORKERS,
)

from .algorithms import (
    STATIONARY_CONFIGS,
    NONSTATIONARY_CONFIGS,
    ADVERSARIAL_CONFIGS,
    FULL_ALGORITHM_CONFIGS,
    create_algorithm,
    create_algorithm_suite,
    get_algorithm_configs,
)

from .environments import (
    ENVIRONMENT_CONFIGS,
    get_environment_config,
    create_environment,
)

__all__ = [
    # Constants
    'N_RUNS', 'N_RUNS_QUICK', 'HORIZON', 'N_ARMS', 'GAP', 'SEED', 'MAX_WORKERS',
    # Algorithm configs
    'STATIONARY_CONFIGS', 'NONSTATIONARY_CONFIGS', 'ADVERSARIAL_CONFIGS',
    'FULL_ALGORITHM_CONFIGS',
    'create_algorithm', 'create_algorithm_suite', 'get_algorithm_configs',
    # Environment configs
    'ENVIRONMENT_CONFIGS', 'get_environment_config', 'create_environment',
]
