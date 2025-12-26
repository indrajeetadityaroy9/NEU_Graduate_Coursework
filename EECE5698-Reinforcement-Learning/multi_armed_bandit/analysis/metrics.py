"""
Metrics for Bandit Algorithm Evaluation

Provides regret computation, adaptation metrics, and tracking utilities
for comprehensive algorithm analysis.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np


def compute_regret(
    rewards: np.ndarray,
    optimal_values: np.ndarray
) -> np.ndarray:
    """
    Compute per-step regret.

    Parameters
    ----------
    rewards : np.ndarray
        Rewards received at each timestep
    optimal_values : np.ndarray
        Optimal (best arm) mean at each timestep

    Returns
    -------
    np.ndarray
        Per-step regret: optimal_value - reward

    Raises
    ------
    ValueError
        If rewards and optimal_values have different lengths
    """
    if len(rewards) != len(optimal_values):
        raise ValueError(
            f"rewards and optimal_values must have same length: "
            f"{len(rewards)} vs {len(optimal_values)}"
        )
    return optimal_values - rewards


def compute_cumulative_regret(
    rewards: np.ndarray,
    optimal_values: np.ndarray
) -> np.ndarray:
    """
    Compute cumulative regret over time.

    Parameters
    ----------
    rewards : np.ndarray
        Rewards received at each timestep
    optimal_values : np.ndarray
        Optimal mean at each timestep

    Returns
    -------
    np.ndarray
        Cumulative regret at each timestep

    Raises
    ------
    ValueError
        If rewards and optimal_values have different lengths
    """
    # Validation happens in compute_regret
    regret = compute_regret(rewards, optimal_values)
    return np.cumsum(regret)


def compute_adaptation_regret(
    rewards: np.ndarray,
    optimal_values: np.ndarray,
    change_points: List[int],
    window: int = 50
) -> Dict[str, Any]:
    """
    Compute regret in windows after change points.

    This metric measures how quickly an algorithm adapts after
    the distribution changes.

    Parameters
    ----------
    rewards : np.ndarray
        Rewards received at each timestep
    optimal_values : np.ndarray
        Optimal mean at each timestep
    change_points : List[int]
        Timesteps where changes occurred
    window : int
        Number of steps after change to consider

    Returns
    -------
    dict
        Dictionary containing:
        - 'per_change': List of regret values for each change point
        - 'mean': Average adaptation regret
        - 'std': Standard deviation
        - 'total': Total adaptation regret

    Raises
    ------
    ValueError
        If rewards and optimal_values have different lengths
    """
    if len(rewards) != len(optimal_values):
        raise ValueError(
            f"rewards and optimal_values must have same length: "
            f"{len(rewards)} vs {len(optimal_values)}"
        )
    T = len(rewards)
    adaptation_regrets = []

    for cp in change_points:
        end = min(cp + window, T)
        if end > cp:
            window_regret = np.sum(optimal_values[cp:end] - rewards[cp:end])
            adaptation_regrets.append(window_regret)

    if len(adaptation_regrets) == 0:
        return {
            'per_change': [],
            'mean': 0.0,
            'std': 0.0,
            'total': 0.0,
        }

    return {
        'per_change': adaptation_regrets,
        'mean': np.mean(adaptation_regrets),
        'std': np.std(adaptation_regrets),
        'total': np.sum(adaptation_regrets),
    }


def compute_detection_delay(
    actions: np.ndarray,
    optimal_arms: np.ndarray,
    change_points: List[int],
    max_delay: int = 100
) -> Dict[str, Any]:
    """
    Compute delay until optimal arm is selected after each change.

    Parameters
    ----------
    actions : np.ndarray
        Actions selected at each timestep
    optimal_arms : np.ndarray
        Optimal arm at each timestep
    change_points : List[int]
        Timesteps where optimal arm changed
    max_delay : int
        Maximum delay to consider (if not detected, returns max_delay)

    Returns
    -------
    dict
        Dictionary containing:
        - 'per_change': List of delays for each change point
        - 'mean': Average detection delay
        - 'std': Standard deviation
        - 'detection_rate': Fraction of changes detected within max_delay

    Raises
    ------
    ValueError
        If actions and optimal_arms have different lengths
    """
    if len(actions) != len(optimal_arms):
        raise ValueError(
            f"actions and optimal_arms must have same length: "
            f"{len(actions)} vs {len(optimal_arms)}"
        )
    T = len(actions)
    delays = []

    for cp in change_points:
        detected = False
        for t in range(cp, min(cp + max_delay, T)):
            if actions[t] == optimal_arms[t]:
                delays.append(t - cp)
                detected = True
                break
        if not detected:
            delays.append(max_delay)

    if len(delays) == 0:
        return {
            'per_change': [],
            'mean': 0.0,
            'std': 0.0,
            'detection_rate': 1.0,
        }

    return {
        'per_change': delays,
        'mean': np.mean(delays),
        'std': np.std(delays),
        'detection_rate': np.mean(np.array(delays) < max_delay),
    }


def compute_optimal_action_percentage(
    actions: np.ndarray,
    optimal_arms: np.ndarray,
    window: Optional[int] = None
) -> np.ndarray:
    """
    Compute percentage of optimal actions over time.

    Parameters
    ----------
    actions : np.ndarray
        Actions selected at each timestep
    optimal_arms : np.ndarray
        Optimal arm at each timestep
    window : int, optional
        If provided, compute rolling average over this window

    Returns
    -------
    np.ndarray
        Optimal action percentage at each timestep

    Raises
    ------
    ValueError
        If actions and optimal_arms have different lengths
    """
    if len(actions) != len(optimal_arms):
        raise ValueError(
            f"actions and optimal_arms must have same length: "
            f"{len(actions)} vs {len(optimal_arms)}"
        )
    optimal = (actions == optimal_arms).astype(float)

    if window is None:
        # Cumulative percentage
        return np.cumsum(optimal) / np.arange(1, len(optimal) + 1)
    else:
        # Rolling window using O(T) convolution-based approach
        T = len(optimal)
        if T == 0:
            return np.array([])

        # Use cumsum trick for efficient rolling mean
        # cumsum[i] - cumsum[i-window] gives sum of window elements
        cumsum = np.cumsum(optimal)
        cumsum = np.insert(cumsum, 0, 0)  # Prepend 0 for easier indexing

        result = np.zeros(T)

        # Handle the initial boundary (partial windows)
        for t in range(min(window - 1, T)):
            result[t] = cumsum[t + 1] / (t + 1)

        # Full windows using vectorized difference
        if T >= window:
            full_window_start = window - 1
            result[full_window_start:] = (
                cumsum[window:T + 1] - cumsum[:T - window + 1]
            ) / window

        return result


def compute_steady_state_regret(
    rewards: np.ndarray,
    optimal_values: np.ndarray,
    change_points: List[int],
    skip_after_change: int = 50
) -> Dict[str, float]:
    """
    Compute regret during steady-state periods (between changes).

    Excludes the adaptation window after each change point.

    Parameters
    ----------
    rewards : np.ndarray
        Rewards at each timestep
    optimal_values : np.ndarray
        Optimal values at each timestep
    change_points : List[int]
        Change point timesteps
    skip_after_change : int
        Number of steps to skip after each change

    Returns
    -------
    dict
        'mean_regret': Average per-step regret in steady state
        'total_regret': Total steady state regret
        'n_steps': Number of steady state steps

    Raises
    ------
    ValueError
        If rewards and optimal_values have different lengths
    """
    if len(rewards) != len(optimal_values):
        raise ValueError(
            f"rewards and optimal_values must have same length: "
            f"{len(rewards)} vs {len(optimal_values)}"
        )
    T = len(rewards)
    mask = np.ones(T, dtype=bool)

    for cp in change_points:
        mask[cp:min(cp + skip_after_change, T)] = False

    steady_regret = (optimal_values - rewards)[mask]

    return {
        'mean_regret': np.mean(steady_regret) if len(steady_regret) > 0 else 0.0,
        'total_regret': np.sum(steady_regret),
        'n_steps': np.sum(mask),
    }


class MetricsTracker:
    """
    Track metrics during experiment execution.

    Collects rewards, actions, and optimal values for later analysis.
    Uses pre-allocated arrays for better performance.

    Parameters
    ----------
    horizon : int, optional
        Expected number of timesteps. Pre-allocates arrays of this size.
        If exceeded, arrays are automatically expanded.
    use_gpu : bool, optional
        Whether to use GPU arrays (requires backends module with CuPy).
        Default is False for compatibility.

    Examples
    --------
    >>> tracker = MetricsTracker(horizon=10000)
    >>> for t in range(T):
    ...     action = algo.select_action()
    ...     reward = env.pull(action)
    ...     tracker.record(
    ...         action=action,
    ...         reward=reward,
    ...         optimal_arm=env.get_optimal_arm(),
    ...         optimal_value=env.get_optimal_value()
    ...     )
    >>> results = tracker.compute_all_metrics(change_points)
    """

    def __init__(self, horizon: int = 10000, use_gpu: bool = False):
        self._initial_horizon = horizon
        self._use_gpu = use_gpu
        self.reset()

    def reset(self) -> None:
        """Reset all tracked data with pre-allocated arrays."""
        self._horizon = self._initial_horizon
        self._idx = 0

        # Pre-allocate arrays
        self._actions = np.zeros(self._horizon, dtype=np.int32)
        self._rewards = np.zeros(self._horizon, dtype=np.float32)
        self._optimal_arms = np.zeros(self._horizon, dtype=np.int32)
        self._optimal_values = np.zeros(self._horizon, dtype=np.float32)

    def _expand(self) -> None:
        """Double array capacity when needed."""
        new_horizon = self._horizon * 2

        # Expand each array
        self._actions = np.concatenate([
            self._actions,
            np.zeros(self._horizon, dtype=np.int32)
        ])
        self._rewards = np.concatenate([
            self._rewards,
            np.zeros(self._horizon, dtype=np.float32)
        ])
        self._optimal_arms = np.concatenate([
            self._optimal_arms,
            np.zeros(self._horizon, dtype=np.int32)
        ])
        self._optimal_values = np.concatenate([
            self._optimal_values,
            np.zeros(self._horizon, dtype=np.float32)
        ])

        self._horizon = new_horizon

    def record(
        self,
        action: int,
        reward: float,
        optimal_arm: int,
        optimal_value: float
    ) -> None:
        """Record a single timestep."""
        if self._idx >= self._horizon:
            self._expand()

        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._optimal_arms[self._idx] = optimal_arm
        self._optimal_values[self._idx] = optimal_value
        self._idx += 1

    @property
    def actions(self) -> np.ndarray:
        """Return recorded actions (view, not copy)."""
        return self._actions[:self._idx]

    @property
    def rewards(self) -> np.ndarray:
        """Return recorded rewards (view, not copy)."""
        return self._rewards[:self._idx]

    @property
    def optimal_arms(self) -> np.ndarray:
        """Return recorded optimal arms (view, not copy)."""
        return self._optimal_arms[:self._idx]

    @property
    def optimal_values(self) -> np.ndarray:
        """Return recorded optimal values (view, not copy)."""
        return self._optimal_values[:self._idx]

    def __len__(self) -> int:
        """Return number of recorded timesteps."""
        return self._idx

    def compute_all_metrics(
        self,
        change_points: List[int],
        adaptation_window: int = 50
    ) -> Dict[str, Any]:
        """
        Compute all metrics.

        Parameters
        ----------
        change_points : List[int]
            List of change point timesteps
        adaptation_window : int
            Window size for adaptation metrics

        Returns
        -------
        dict
            Dictionary with all computed metrics
        """
        rewards = self.rewards
        optimal_values = self.optimal_values
        actions = self.actions
        optimal_arms = self.optimal_arms

        # Handle empty data case
        if len(rewards) == 0:
            return {
                'cumulative_regret': np.array([]),
                'final_regret': 0.0,
                'adaptation': {'per_change': [], 'mean': 0.0, 'std': 0.0, 'total': 0.0},
                'detection': {'per_change': [], 'mean': 0.0, 'std': 0.0, 'detection_rate': 1.0},
                'optimal_percentage': 0.0,
                'steady_state': {'mean_regret': 0.0, 'total_regret': 0.0, 'n_steps': 0},
            }

        cum_regret = compute_cumulative_regret(rewards, optimal_values)
        opt_pct = compute_optimal_action_percentage(actions, optimal_arms)

        return {
            'cumulative_regret': cum_regret,
            'final_regret': cum_regret[-1] if len(cum_regret) > 0 else 0.0,
            'adaptation': compute_adaptation_regret(
                rewards, optimal_values, change_points, adaptation_window
            ),
            'detection': compute_detection_delay(
                actions, optimal_arms, change_points
            ),
            'optimal_percentage': opt_pct[-1] if len(opt_pct) > 0 else 0.0,
            'steady_state': compute_steady_state_regret(
                rewards, optimal_values, change_points, adaptation_window
            ),
        }
