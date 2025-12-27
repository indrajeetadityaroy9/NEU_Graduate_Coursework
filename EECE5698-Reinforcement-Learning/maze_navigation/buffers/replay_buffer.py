"""Experience Replay Buffer for off-policy algorithms (DQN)."""

from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


class ReplayBuffer:
    """Efficient replay buffer with numpy storage and torch batch sampling.

    Uses a ring buffer implementation that overwrites oldest entries when full.
    Pre-allocates numpy arrays for memory efficiency.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        seed: Optional[int] = None
    ):
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            obs_dim: Dimension of observations.
            seed: Random seed for reproducible sampling.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._rng = np.random.default_rng(seed)

        # Pre-allocate numpy arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0  # Next write position
        self.size = 0  # Current buffer size

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode terminated.
        """
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary containing batched tensors.
        """
        indices = self._rng.integers(0, self.size, size=batch_size)

        return {
            'observations': torch.from_numpy(self.observations[indices]),
            'actions': torch.from_numpy(self.actions[indices]),
            'rewards': torch.from_numpy(self.rewards[indices]),
            'next_observations': torch.from_numpy(self.next_observations[indices]),
            'dones': torch.from_numpy(self.dones[indices]),
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training.

        Args:
            min_size: Minimum required samples.

        Returns:
            True if buffer has at least min_size samples.
        """
        return self.size >= min_size

    def save(self, path: str) -> None:
        """Save buffer contents to disk.

        Args:
            path: Path to save file (without extension).
        """
        np.savez_compressed(
            path,
            observations=self.observations[:self.size],
            actions=self.actions[:self.size],
            rewards=self.rewards[:self.size],
            next_observations=self.next_observations[:self.size],
            dones=self.dones[:self.size],
            ptr=self.ptr,
            size=self.size,
        )

    def load(self, path: str) -> None:
        """Load buffer contents from disk.

        Args:
            path: Path to load file.
        """
        data = np.load(path)
        size = int(data['size'])

        self.observations[:size] = data['observations']
        self.actions[:size] = data['actions']
        self.rewards[:size] = data['rewards']
        self.next_observations[:size] = data['next_observations']
        self.dones[:size] = data['dones']
        self.ptr = int(data['ptr'])
        self.size = size


class NStepBuffer:
    """Buffer for computing n-step returns.

    Accumulates transitions and computes n-step discounted returns.
    Handles episode boundaries by flushing with variable-step lookaheads.

    Reference:
        Sutton & Barto, "Reinforcement Learning: An Introduction", Chapter 7
    """

    def __init__(self, n_step: int, gamma: float):
        """Initialize n-step buffer.

        Args:
            n_step: Number of steps for n-step returns.
            gamma: Discount factor.
        """
        self.n_step = n_step
        self.gamma = gamma
        self._buffer: deque = deque(maxlen=n_step)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> Optional[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Add a transition and return n-step transition if ready.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.

        Returns:
            N-step transition tuple if buffer has n_step entries, else None.
        """
        self._buffer.append((obs, action, reward, next_obs, done))

        if len(self._buffer) < self.n_step:
            return None

        # Compute n-step discounted return
        n_reward = sum(
            (self.gamma ** i) * self._buffer[i][2]
            for i in range(self.n_step)
        )

        # Get first observation/action and last next_observation/done
        first_obs = self._buffer[0][0]
        first_action = self._buffer[0][1]
        last_next_obs = self._buffer[-1][3]
        last_done = self._buffer[-1][4]

        return (first_obs, first_action, n_reward, last_next_obs, last_done)

    def flush(self) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Flush remaining transitions at episode end.

        Computes variable-length lookaheads (1-step, 2-step, etc.)
        for remaining buffer items pointing to the terminal state.

        Returns:
            List of k-step transitions from remaining buffer entries.
        """
        transitions = []

        while self._buffer:
            k = len(self._buffer)  # Variable k-step (1 to n-1)

            # Compute k-step discounted return
            k_reward = sum(
                (self.gamma ** i) * self._buffer[i][2]
                for i in range(k)
            )

            transitions.append((
                self._buffer[0][0],   # First obs
                self._buffer[0][1],   # First action
                k_reward,             # k-step return
                self._buffer[-1][3],  # Terminal next_obs
                True                  # Episode done
            ))
            self._buffer.popleft()

        return transitions

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
