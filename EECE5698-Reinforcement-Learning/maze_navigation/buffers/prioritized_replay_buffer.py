"""Prioritized Experience Replay Buffer for improved sample efficiency.

Implements PER using a sum tree for O(log n) sampling based on TD-error priorities.

Reference:
    Schaul et al., "Prioritized Experience Replay"
    https://arxiv.org/abs/1511.05952
"""

from typing import Dict, Optional, Tuple
import numpy as np
import torch


class SumTree:
    """Sum tree data structure for efficient priority-based sampling.

    A binary tree where each parent node is the sum of its children.
    Leaf nodes store priorities; internal nodes store sums.
    Enables O(log n) sampling and priority updates.
    """

    def __init__(self, capacity: int):
        """Initialize the sum tree.

        Args:
            capacity: Maximum number of leaf nodes (experiences).
        """
        self.capacity = capacity
        # Tree has 2*capacity - 1 nodes total
        # First capacity-1 nodes are internal, last capacity are leaves
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    def add(self, priority: float) -> int:
        """Add a new priority and return its index.

        Args:
            priority: Priority value for the new experience.

        Returns:
            Data index (0 to capacity-1) where the experience is stored.
        """
        # Calculate leaf index in tree
        tree_idx = self.data_pointer + self.capacity - 1
        self.update(tree_idx, priority)

        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        return data_idx

    def update(self, tree_idx: int, priority: float) -> None:
        """Update priority at a tree index and propagate change upward.

        Args:
            tree_idx: Index in the tree array.
            priority: New priority value.
        """
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value: float) -> Tuple[int, float, int]:
        """Find a leaf based on a cumulative priority value.

        Args:
            value: Random value in [0, total_priority).

        Returns:
            Tuple of (tree_idx, priority, data_idx).
        """
        parent_idx = 0

        while True:
            left_child = 2 * parent_idx + 1
            right_child = left_child + 1

            if left_child >= len(self.tree):
                # Reached leaf
                leaf_idx = parent_idx
                break

            if value <= self.tree[left_child]:
                parent_idx = left_child
            else:
                value -= self.tree[left_child]
                parent_idx = right_child

        data_idx = leaf_idx - self.capacity + 1

        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self) -> float:
        """Get total priority (root value)."""
        return self.tree[0]

    @property
    def max_priority(self) -> float:
        """Get maximum priority in leaves."""
        return np.max(self.tree[self.capacity - 1:])

    @property
    def min_priority(self) -> float:
        """Get minimum non-zero priority in leaves."""
        leaves = self.tree[self.capacity - 1:]
        non_zero = leaves[leaves > 0]
        if len(non_zero) == 0:
            return 1.0
        return np.min(non_zero)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer.

    Uses a sum tree for efficient O(log n) sampling based on TD-error priorities.
    Implements importance sampling weights to correct for non-uniform sampling.

    Features:
    - Proportional prioritization with configurable alpha
    - Importance sampling weights with annealing beta
    - Efficient sum tree for priority-based sampling
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100000,
        epsilon: float = 1e-6,
        seed: Optional[int] = None,
    ):
        """Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
            obs_dim: Dimension of observations.
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization).
            beta_start: Initial importance sampling weight.
            beta_end: Final importance sampling weight.
            beta_frames: Number of frames to anneal beta.
            epsilon: Small constant to ensure non-zero priorities.
            seed: Random seed for reproducible sampling.
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self._rng = np.random.default_rng(seed)

        # Sum tree for priority-based sampling
        self.tree = SumTree(capacity)

        # Pre-allocate numpy arrays for experience storage
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self.size = 0
        self.frame_count = 0
        self.max_priority = 1.0

    def _get_beta(self) -> float:
        """Get current beta value with linear annealing."""
        fraction = min(self.frame_count / self.beta_frames, 1.0)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition with optional priority.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode terminated.
            priority: Priority value (uses max_priority if None).
        """
        # Use max priority for new experiences
        if priority is None:
            priority = self.max_priority

        # Add to sum tree and get data index
        data_idx = self.tree.add((priority + self.epsilon) ** self.alpha)

        # Store experience
        self.observations[data_idx] = obs
        self.actions[data_idx] = action
        self.rewards[data_idx] = reward
        self.next_observations[data_idx] = next_obs
        self.dones[data_idx] = float(done)

        self.size = min(self.size + 1, self.capacity)
        self.frame_count += 1

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample a batch based on priorities.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (batch dict, importance weights, tree indices).
        """
        indices = np.zeros(batch_size, dtype=np.int32)
        tree_indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)

        # Divide priority range into segments for stratified sampling
        total_priority = self.tree.total_priority
        segment_size = total_priority / batch_size

        beta = self._get_beta()

        for i in range(batch_size):
            # Sample uniformly within segment
            low = segment_size * i
            high = segment_size * (i + 1)
            value = self._rng.uniform(low, high)

            tree_idx, priority, data_idx = self.tree.get_leaf(value)
            tree_indices[i] = tree_idx
            indices[i] = data_idx
            priorities[i] = priority

        # Compute importance sampling weights
        # w_i = (1/N * 1/P(i))^beta / max_w
        sampling_probs = priorities / total_priority
        weights = (self.size * sampling_probs) ** (-beta)
        weights = weights / weights.max()  # Normalize for stability

        batch = {
            'observations': torch.from_numpy(self.observations[indices]),
            'actions': torch.from_numpy(self.actions[indices]),
            'rewards': torch.from_numpy(self.rewards[indices]),
            'next_observations': torch.from_numpy(self.next_observations[indices]),
            'dones': torch.from_numpy(self.dones[indices]),
        }

        return batch, weights.astype(np.float32), tree_indices

    def update_priorities(
        self,
        tree_indices: np.ndarray,
        td_errors: np.ndarray,
    ) -> None:
        """Update priorities based on TD errors.

        Args:
            tree_indices: Indices in the sum tree.
            td_errors: Absolute TD errors for priority calculation.
        """
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha

        for tree_idx, priority in zip(tree_indices, priorities):
            self.tree.update(tree_idx, priority)

        # Track max priority for new experiences
        self.max_priority = max(self.max_priority, np.max(np.abs(td_errors)))

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples.

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
            tree=self.tree.tree,
            tree_pointer=self.tree.data_pointer,
            size=self.size,
            frame_count=self.frame_count,
            max_priority=self.max_priority,
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
        self.tree.tree = data['tree']
        self.tree.data_pointer = int(data['tree_pointer'])
        self.size = size
        self.frame_count = int(data['frame_count'])
        self.max_priority = float(data['max_priority'])
