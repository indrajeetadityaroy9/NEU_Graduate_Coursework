"""Rollout Buffer for on-policy algorithms (PPO)."""

from typing import Generator, Dict
import numpy as np
import torch


class RolloutBuffer:
    """Rollout buffer for on-policy algorithms.

    Stores complete trajectories and computes GAE (Generalized Advantage Estimation)
    advantages and discounted returns.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        device: torch.device = None
    ):
        """Initialize the rollout buffer.

        Args:
            capacity: Maximum number of steps per rollout.
            obs_dim: Dimension of observations.
            device: Torch device for tensor creation.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.device = device or torch.device('cpu')

        # Storage arrays
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self.ptr = 0
        self.path_start_idx = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Add a step to the buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            value: Value estimate.
            log_prob: Log probability of action.
            done: Whether episode ended.
        """
        assert self.ptr < self.capacity, "Buffer overflow"

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = float(done)

        self.ptr += 1

    def compute_returns_and_advantages(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Uses Generalized Advantage Estimation (GAE) for lower variance
        advantage estimates.

        Args:
            last_value: Value estimate of the final state.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter (trade-off bias vs variance).
        """
        gae = 0.0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]

            # TD error
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )

            # GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        # Returns = advantages + values
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(
        self,
        batch_size: int
    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Generate shuffled mini-batches.

        Args:
            batch_size: Size of each mini-batch.

        Yields:
            Dictionary containing batch tensors.
        """
        indices = np.random.permutation(self.ptr)

        for start in range(0, self.ptr, batch_size):
            end = min(start + batch_size, self.ptr)
            batch_indices = indices[start:end]

            yield {
                'observations': torch.from_numpy(
                    self.observations[batch_indices]
                ).to(self.device),
                'actions': torch.from_numpy(
                    self.actions[batch_indices]
                ).to(self.device),
                'old_log_probs': torch.from_numpy(
                    self.log_probs[batch_indices]
                ).to(self.device),
                'advantages': torch.from_numpy(
                    self.advantages[batch_indices]
                ).to(self.device),
                'returns': torch.from_numpy(
                    self.returns[batch_indices]
                ).to(self.device),
                'values': torch.from_numpy(
                    self.values[batch_indices]
                ).to(self.device),
            }

    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all stored data as tensors.

        Returns:
            Dictionary containing all stored data.
        """
        return {
            'observations': torch.from_numpy(
                self.observations[:self.ptr]
            ).to(self.device),
            'actions': torch.from_numpy(
                self.actions[:self.ptr]
            ).to(self.device),
            'old_log_probs': torch.from_numpy(
                self.log_probs[:self.ptr]
            ).to(self.device),
            'advantages': torch.from_numpy(
                self.advantages[:self.ptr]
            ).to(self.device),
            'returns': torch.from_numpy(
                self.returns[:self.ptr]
            ).to(self.device),
            'values': torch.from_numpy(
                self.values[:self.ptr]
            ).to(self.device),
        }

    def reset(self) -> None:
        """Reset the buffer for a new rollout."""
        self.ptr = 0
        self.path_start_idx = 0

    def is_full(self) -> bool:
        """Check if buffer is full.

        Returns:
            True if buffer has reached capacity.
        """
        return self.ptr >= self.capacity

    def __len__(self) -> int:
        """Return current number of stored steps."""
        return self.ptr
