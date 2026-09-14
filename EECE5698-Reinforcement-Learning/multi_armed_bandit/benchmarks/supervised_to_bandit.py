"""
Supervised-to-Bandit Conversion

Converts UCI classification datasets to bandit problems with induced non-stationarity.
Implements both sudden and gradual drift as recommended in bandit literature.

Datasets:
- Mushroom (UCI): 2-class, 22 features, 8124 samples
- Covertype (UCI): 7-class, 54 features, 581012 samples

Reference: Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article Recommendation"
"""

import numpy as np
from typing import Optional, Tuple, Literal
from abc import ABC, abstractmethod

try:
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class SupervisedToBandit(ABC):
    """
    Base class for supervised-to-bandit conversion.

    Converts a classification dataset to a bandit problem where:
    - Each class becomes an arm
    - Reward = 1 if algorithm selects correct class, 0 otherwise
    - Non-stationarity is induced by rotating the reward mapping
    """

    def __init__(
        self,
        drift_type: Literal['none', 'sudden', 'gradual'] = 'none',
        drift_interval: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Args:
            drift_type: Type of distribution shift
                - 'none': Stationary (baseline)
                - 'sudden': Instant label flip at drift_interval
                - 'gradual': Smooth rotation of reward probabilities
            drift_interval: Steps between drift events (for sudden) or
                           steps over which full rotation occurs (for gradual)
            seed: Random seed for reproducibility
        """
        self.drift_type = drift_type
        self.drift_interval = drift_interval
        self.rng = np.random.default_rng(seed)

        self._load_data()
        self._current_step = 0
        self._reward_mapping = np.arange(self.n_arms)  # Identity mapping initially

    @abstractmethod
    def _load_data(self) -> None:
        """Load and preprocess the dataset. Sets self.X, self.y, self.n_arms."""
        pass

    @property
    def n_arms(self) -> int:
        return self._n_arms

    def reset(self) -> None:
        """Reset the environment to initial state."""
        self._current_step = 0
        self._reward_mapping = np.arange(self.n_arms)
        self._data_idx = self.rng.permutation(len(self.y))
        self._data_ptr = 0

    def _get_next_context(self) -> Tuple[np.ndarray, int]:
        """Get next context and true label."""
        if self._data_ptr >= len(self.y):
            self._data_idx = self.rng.permutation(len(self.y))
            self._data_ptr = 0

        idx = self._data_idx[self._data_ptr]
        self._data_ptr += 1
        return self.X[idx], self.y[idx]

    def _update_drift(self) -> None:
        """Update reward mapping based on drift type."""
        if self.drift_type == 'none':
            return

        elif self.drift_type == 'sudden':
            # Rotate labels every drift_interval steps
            if self._current_step > 0 and self._current_step % self.drift_interval == 0:
                # Rotate: arm 0 -> arm 1 -> ... -> arm K-1 -> arm 0
                self._reward_mapping = np.roll(self._reward_mapping, 1)

        elif self.drift_type == 'gradual':
            # Gradual rotation: linearly interpolate reward probabilities
            # At step t, the "true" label shifts by t / drift_interval
            pass  # Handled in get_reward_probabilities

    def get_reward_probabilities(self, true_label: int) -> np.ndarray:
        """
        Get reward probability for each arm given true label.

        For gradual drift, returns soft probabilities that interpolate
        between current and next reward mapping.
        """
        probs = np.zeros(self.n_arms)

        if self.drift_type == 'gradual':
            # Compute rotation progress
            rotation_progress = (self._current_step % self.drift_interval) / self.drift_interval
            n_full_rotations = self._current_step // self.drift_interval

            # Current and next reward mappings
            current_offset = n_full_rotations % self.n_arms
            next_offset = (n_full_rotations + 1) % self.n_arms

            current_arm = (true_label + current_offset) % self.n_arms
            next_arm = (true_label + next_offset) % self.n_arms

            # Interpolate probabilities
            probs[current_arm] = 1.0 - rotation_progress
            probs[next_arm] = rotation_progress

        else:
            # Sudden or stationary: deterministic mapping
            mapped_label = self._reward_mapping[true_label]
            probs[mapped_label] = 1.0

        return probs

    def step(self, action: int) -> Tuple[float, dict]:
        """
        Take an action and receive reward.

        Args:
            action: Arm to pull (0 to n_arms-1)

        Returns:
            reward: 1.0 if correct, 0.0 otherwise (or probabilistic for gradual)
            info: Dictionary with metadata
        """
        context, true_label = self._get_next_context()
        self._update_drift()

        reward_probs = self.get_reward_probabilities(true_label)

        # Sample reward (Bernoulli for gradual, deterministic for others)
        if self.drift_type == 'gradual':
            reward = float(self.rng.random() < reward_probs[action])
        else:
            reward = reward_probs[action]

        # Compute optimal arm for this step
        optimal_arm = np.argmax(reward_probs)

        info = {
            'context': context,
            'true_label': true_label,
            'optimal_arm': optimal_arm,
            'reward_probs': reward_probs,
            'step': self._current_step,
        }

        self._current_step += 1
        return reward, info

    def get_arm_means(self) -> np.ndarray:
        """Get current expected reward for each arm (across data distribution)."""
        # For non-contextual version, compute empirical class frequencies
        class_freqs = np.bincount(self.y, minlength=self.n_arms) / len(self.y)

        if self.drift_type == 'gradual':
            rotation_progress = (self._current_step % self.drift_interval) / self.drift_interval
            n_full_rotations = self._current_step // self.drift_interval

            # Blend between current and next rotation
            current_means = np.roll(class_freqs, n_full_rotations % self.n_arms)
            next_means = np.roll(class_freqs, (n_full_rotations + 1) % self.n_arms)
            return (1 - rotation_progress) * current_means + rotation_progress * next_means

        else:
            # Apply current mapping
            return class_freqs[self._reward_mapping]


class MushroomBandit(SupervisedToBandit):
    """
    Mushroom dataset as 2-armed bandit.

    Binary classification: edible (0) vs poisonous (1)
    Reward 1 if correct classification, 0 otherwise.

    Non-contextual version: Uses class frequencies as arm means.
    """

    def _load_data(self) -> None:
        """Load Mushroom dataset from UCI."""
        try:
            # Try to load from sklearn (bundled with openml)
            from sklearn.datasets import fetch_openml
            data = fetch_openml('mushroom', version=1, as_frame=False, parser='auto')
            self.X = data.data
            # Convert labels to 0/1
            self.y = (data.target == 'p').astype(int)  # p=poisonous=1, e=edible=0
        except Exception:
            # Fallback: generate synthetic mushroom-like data
            n_samples = 8124
            n_features = 22
            self.rng_init = np.random.default_rng(42)
            self.X = self.rng_init.standard_normal((n_samples, n_features))
            # ~48% poisonous in real dataset
            self.y = (self.rng_init.random(n_samples) < 0.48).astype(int)

        self._n_arms = 2
        self.reset()


class CovertypeBandit(SupervisedToBandit):
    """
    Covertype dataset as 7-armed bandit.

    7-class classification of forest cover types.
    High-dimensional (54 features) - good stress test.

    Non-contextual version: Uses class frequencies as arm means.
    """

    def _load_data(self) -> None:
        """Load Covertype dataset from sklearn."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for Covertype. Run: pip install scikit-learn")

        data = fetch_covtype()
        self.X = StandardScaler().fit_transform(data.data)
        self.y = data.target - 1  # Convert 1-7 to 0-6

        self._n_arms = 7
        self.reset()


class SyntheticDriftBandit(SupervisedToBandit):
    """
    Synthetic bandit with controllable drift for testing.

    Creates a simple bandit where arm means rotate according to drift settings.
    Useful for validating drift detection algorithms.
    """

    def __init__(
        self,
        n_arms: int = 5,
        gap: float = 1.0,
        drift_type: Literal['none', 'sudden', 'gradual'] = 'gradual',
        drift_interval: int = 1000,
        noise_std: float = 1.0,
        seed: Optional[int] = None,
    ):
        self._n_arms_init = n_arms
        self._gap = gap
        self._noise_std = noise_std
        super().__init__(drift_type, drift_interval, seed)

    def _load_data(self) -> None:
        """Create synthetic data with clear arm separation."""
        self._n_arms = self._n_arms_init
        # Create "fake" data - just the arm index as the "true label"
        n_samples = 100000
        self.X = np.zeros((n_samples, 1))  # No context needed
        # Uniform distribution over arms
        self.y = np.arange(n_samples) % self.n_arms
        self.reset()

    def step(self, action: int) -> Tuple[float, dict]:
        """Override to return Gaussian rewards instead of Bernoulli."""
        # Get arm means BEFORE calling super (which increments step)
        arm_means = self.get_arm_means()
        true_optimal = int(np.argmax(arm_means))

        # Call parent to update internal state
        _, info = super().step(action)

        # Add Gaussian noise to reward based on actual arm means
        reward = arm_means[action] + self.rng.normal(0, self._noise_std)

        # CRITICAL FIX: Override optimal_arm with correct value from arm_means
        # Parent class uses cycling dummy labels which is incorrect
        info['arm_means'] = arm_means
        info['optimal_arm'] = true_optimal
        return reward, info

    def get_arm_means(self) -> np.ndarray:
        """Get current arm means with gap-based initialization."""
        base_means = np.zeros(self.n_arms)
        base_means[0] = self._gap  # Arm 0 is optimal

        if self.drift_type == 'gradual':
            rotation_progress = (self._current_step % self.drift_interval) / self.drift_interval
            n_full_rotations = self._current_step // self.drift_interval

            current_means = np.roll(base_means, n_full_rotations % self.n_arms)
            next_means = np.roll(base_means, (n_full_rotations + 1) % self.n_arms)
            return (1 - rotation_progress) * current_means + rotation_progress * next_means

        elif self.drift_type == 'sudden':
            n_rotations = self._current_step // self.drift_interval
            return np.roll(base_means, n_rotations % self.n_arms)

        else:
            return base_means
