"""
Shared utilities for bandit algorithms.

Provides common mathematical functions and validation utilities
used across multiple algorithm implementations.
"""

import numpy as np
from typing import Optional


# ============================================================================
# Mathematical Functions
# ============================================================================

def softmax(values: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities with numerical stability.

    Uses the max-subtraction trick to prevent overflow.

    Parameters
    ----------
    values : np.ndarray
        Input values (e.g., preferences, logits)

    Returns
    -------
    np.ndarray
        Probability distribution summing to 1
    """
    exp_values = np.exp(values - np.max(values))
    return exp_values / np.sum(exp_values)


def mixed_probability(
    weights: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Compute EXP3-style mixed probability distribution.

    Mixes weight-proportional probabilities with uniform distribution
    for exploration.

    Parameters
    ----------
    weights : np.ndarray
        Non-negative weights for each action
    gamma : float
        Mixing parameter in (0, 1]. Higher gamma = more exploration.

    Returns
    -------
    np.ndarray
        Probability distribution: (1 - gamma) * normalized_weights + gamma * uniform
    """
    n = len(weights)
    weight_probs = weights / np.sum(weights)
    uniform = np.ones(n) / n
    return (1 - gamma) * weight_probs + gamma * uniform


# ============================================================================
# Validation Functions
# ============================================================================

def validate_epsilon(epsilon: float, param_name: str = "epsilon") -> None:
    """
    Validate exploration probability is in [0, 1].

    Parameters
    ----------
    epsilon : float
        Exploration probability
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If epsilon is not in [0, 1]
    """
    if not 0 <= epsilon <= 1:
        raise ValueError(f"{param_name} must be in [0, 1], got {epsilon}")


def validate_discount_factor(gamma: float, param_name: str = "gamma") -> None:
    """
    Validate discount factor is in (0, 1).

    Used for D-UCB, Discounted Thompson Sampling, etc.

    Parameters
    ----------
    gamma : float
        Discount factor
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If gamma is not in (0, 1)
    """
    if not 0 < gamma < 1:
        raise ValueError(f"{param_name} must be in (0, 1), got {gamma}")


def validate_exploration_rate(rate: float, param_name: str = "exploration_rate") -> None:
    """
    Validate exploration mixing rate is in (0, 1].

    Used for EXP3-style algorithms where the exploration rate controls
    mixing with uniform distribution: p(a) = (1-γ)*w(a)/Σw + γ/K.

    Note: This is distinct from discount factor γ in D-UCB/Discounted-TS.

    Parameters
    ----------
    rate : float
        Exploration rate γ ∈ (0, 1]
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If rate is not in (0, 1]
    """
    if not 0 < rate <= 1:
        raise ValueError(f"{param_name} must be in (0, 1], got {rate}")


def validate_positive(value: float, param_name: str) -> None:
    """
    Validate parameter is strictly positive.

    Parameters
    ----------
    value : float
        Value to check
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If value <= 0
    """
    if value <= 0:
        raise ValueError(f"{param_name} must be positive, got {value}")


def validate_non_negative(value: float, param_name: str) -> None:
    """
    Validate parameter is non-negative.

    Parameters
    ----------
    value : float
        Value to check
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If value < 0
    """
    if value < 0:
        raise ValueError(f"{param_name} must be non-negative, got {value}")


def validate_step_size(alpha: float, param_name: str = "alpha") -> None:
    """
    Validate step size / learning rate is in (0, 1].

    Parameters
    ----------
    alpha : float
        Step size
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If alpha is not in (0, 1]
    """
    if not 0 < alpha <= 1:
        raise ValueError(f"{param_name} must be in (0, 1], got {alpha}")


def validate_positive_int(value: int, param_name: str) -> None:
    """
    Validate parameter is a positive integer.

    Parameters
    ----------
    value : int
        Value to check
    param_name : str
        Name for error message

    Raises
    ------
    ValueError
        If value < 1
    """
    if value < 1:
        raise ValueError(f"{param_name} must be positive, got {value}")
