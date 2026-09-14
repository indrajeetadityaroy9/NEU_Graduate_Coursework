"""DQN, Double DQN, and Dueling DQN agent implementations with Rainbow support.

Supports Core Rainbow features via configuration flags:
- Double DQN (use_double): Reduces overestimation bias
- Dueling architecture (use_dueling): Separates V and A streams
- Prioritized Experience Replay (use_per): TD-error based sampling
- N-step returns (n_step): Multi-step TD targets
"""

import warnings
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from networks.mlp import QNetwork
from networks.dueling_network import DuelingNetwork
from buffers.replay_buffer import ReplayBuffer, NStepBuffer


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with Rainbow extensions.

    Features:
    - Epsilon-greedy exploration with linear decay
    - Experience replay buffer (standard or prioritized)
    - Target network with periodic hard or soft updates
    - Gradient clipping for stability

    Rainbow extensions (configurable via flags):
    - Double DQN: Reduces overestimation bias
    - Dueling architecture: Separates V and A streams
    - Prioritized Experience Replay: TD-error based sampling
    - N-step returns: Multi-step TD targets
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10000,
        target_update_freq: int = 1000,
        tau: float = 1.0,
        buffer_size: int = 100000,
        batch_size: int = 64,
        min_buffer_size: int = 1000,
        device: str = 'auto',
        seed: Optional[int] = None,
        mixed_precision: bool = False,
        compile_mode: Optional[str] = None,
        # Rainbow feature flags
        use_double: bool = False,
        use_dueling: bool = False,
        use_per: bool = False,
        n_step: int = 1,
        # PER parameters (only used if use_per=True)
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_frames: int = 100000,
    ):
        """Initialize the DQN agent.

        Args:
            obs_dim: Observation dimension.
            n_actions: Number of actions.
            hidden_dims: Hidden layer sizes.
            learning_rate: Optimizer learning rate.
            gamma: Discount factor.
            epsilon_start: Initial exploration rate.
            epsilon_end: Final exploration rate.
            epsilon_decay_steps: Steps over which to decay epsilon.
            target_update_freq: Steps between target network updates.
            tau: Soft update coefficient (1.0 = hard update).
            buffer_size: Replay buffer capacity.
            batch_size: Training batch size.
            min_buffer_size: Minimum buffer size before training.
            device: Compute device.
            seed: Random seed.
            mixed_precision: Enable AMP (automatic mixed precision).
            compile_mode: torch.compile mode ('reduce-overhead', 'max-autotune', None).
            use_double: Enable Double DQN (reduces overestimation).
            use_dueling: Enable Dueling architecture (separate V/A streams).
            use_per: Enable Prioritized Experience Replay.
            n_step: Number of steps for n-step returns (1 = standard TD).
            per_alpha: PER prioritization exponent (0=uniform, 1=full prioritization).
            per_beta_start: Initial importance sampling correction.
            per_beta_end: Final importance sampling correction.
            per_beta_frames: Frames over which to anneal beta.
        """
        super().__init__(obs_dim, n_actions, device, seed, mixed_precision, compile_mode)

        # Core hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_freq = target_update_freq
        self.tau = tau

        # Rainbow feature flags (MUST be set before _build_network is called)
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.use_per = use_per
        self.n_step = n_step

        # PER parameters
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_end = per_beta_end
        self.per_beta_frames = per_beta_frames

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Networks (with optional torch.compile)
        self.q_network = self._maybe_compile(
            self._build_network(obs_dim, n_actions, hidden_dims)
        )
        self.target_network = self._maybe_compile(
            self._build_network(obs_dim, n_actions, hidden_dims)
        )
        # Copy weights after potential compilation
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=learning_rate
        )

        # Replay buffer (standard or prioritized)
        if self.use_per:
            from buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
            self.buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                obs_dim=obs_dim,
                alpha=per_alpha,
                beta_start=per_beta_start,
                beta_end=per_beta_end,
                beta_frames=per_beta_frames,
                seed=seed,
            )
        else:
            self.buffer = ReplayBuffer(buffer_size, obs_dim, seed=seed)

        # N-step buffer for transition accumulation
        if self.n_step > 1:
            self._n_step_buffer = NStepBuffer(self.n_step, self.gamma)
        else:
            self._n_step_buffer = None

        # Training state
        self.update_counter = 0
        self._rng = np.random.default_rng(seed)

    def _build_network(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...]
    ) -> nn.Module:
        """Build the Q-network. Uses Dueling architecture if enabled.

        Args:
            obs_dim: Observation dimension.
            n_actions: Number of actions.
            hidden_dims: Hidden layer sizes.

        Returns:
            Q-network module (standard or dueling).
        """
        if self.use_dueling:
            return DuelingNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        return QNetwork(obs_dim, n_actions, hidden_dims).to(self.device)

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            observation: Current observation.
            deterministic: If True, always select greedy action.

        Returns:
            Selected action index.
        """
        if not deterministic and self._rng.random() < self.epsilon:
            return self._rng.integers(0, self.n_actions)

        with torch.no_grad():
            obs_tensor = self.to_tensor(observation).unsqueeze(0)
            q_values = self.q_network(obs_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the replay buffer.

        Routes through NStepBuffer if n_step > 1 to compute n-step returns.
        Flushes remaining transitions at episode end.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode ended.
        """
        if self._n_step_buffer is not None:
            # Route through n-step buffer
            n_step_transition = self._n_step_buffer.add(obs, action, reward, next_obs, done)
            if n_step_transition is not None:
                self.buffer.add(*n_step_transition)

            # Flush remaining transitions at episode end
            if done:
                for transition in self._n_step_buffer.flush():
                    self.buffer.add(*transition)
        else:
            # Standard 1-step transition
            self.buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> Optional[Dict[str, float]]:
        """Perform one gradient update step with AMP support.

        Handles both standard and prioritized experience replay.
        For PER, samples with importance weights and updates priorities.

        Returns:
            Dictionary of training metrics, or None if buffer not ready.
        """
        if not self.buffer.is_ready(self.min_buffer_size):
            return None

        # Sample batch (with importance weights for PER)
        if self.use_per:
            batch, is_weights, tree_indices = self.buffer.sample(self.batch_size)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            is_weights = torch.from_numpy(is_weights).float().to(self.device)
        else:
            batch = self.buffer.sample(self.batch_size)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            is_weights = None
            tree_indices = None

        # Compute loss with automatic mixed precision
        with self.get_autocast_context():
            loss, td_errors, q_mean = self._compute_loss(batch, is_weights)

        # Gradient update with GradScaler for AMP
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update PER priorities after backward pass
        if self.use_per and tree_indices is not None:
            priorities = td_errors.detach().cpu().numpy() + 1e-6  # Small epsilon for stability
            self.buffer.update_priorities(tree_indices, priorities)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self._update_target_network()

        return {
            'loss': loss.item(),
            'q_mean': q_mean,
            'epsilon': self.epsilon,
        }

    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        is_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Compute the DQN loss with Rainbow extensions.

        Supports Double DQN, n-step returns, and importance sampling weights.

        Args:
            batch: Dictionary of batched tensors.
            is_weights: Importance sampling weights for PER (None for uniform).

        Returns:
            Tuple of (loss tensor, TD errors for PER, mean Q-value for logging).
        """
        # Current Q-values
        current_q = self.q_network(batch['observations'])
        current_q = current_q.gather(1, batch['actions'].unsqueeze(1).long()).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: online network selects actions, target network evaluates
                next_q_online = self.q_network(batch['next_observations'])
                best_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.target_network(batch['next_observations'])
                next_q_max = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                # Standard DQN: target network selects and evaluates
                next_q = self.target_network(batch['next_observations'])
                next_q_max = next_q.max(dim=1)[0]

            # Use gamma^n_step for n-step returns
            effective_gamma = self.gamma ** self.n_step
            target_q = batch['rewards'] + effective_gamma * next_q_max * (1 - batch['dones'])

        # Compute TD errors (for PER priority updates)
        td_errors = (current_q - target_q).abs()

        # Compute loss with importance sampling weights
        if is_weights is not None:
            # PER: weighted Huber loss (smooth_l1) for stability
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (is_weights * elementwise_loss).mean()
        else:
            # Standard MSE loss
            loss = F.mse_loss(current_q, target_q)

        return loss, td_errors, current_q.mean().item()

    def _update_target_network(self) -> None:
        """Update target network with current network weights.

        Handles DDP-wrapped modules by accessing the underlying module
        to avoid state_dict key mismatch ('module.' prefix issue).
        """
        # Handle DDP-wrapped modules by accessing underlying module
        q_net = self.q_network
        if hasattr(q_net, 'module'):
            q_net = q_net.module

        if self.tau == 1.0:
            # Hard update
            self.target_network.load_state_dict(q_net.state_dict())
        else:
            # Soft update
            for target_param, param in zip(
                self.target_network.parameters(),
                q_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1.0 - self.tau) * target_param.data
                )

    def save(self, path: str) -> None:
        """Save agent state to disk.

        Args:
            path: Save path.
        """
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epsilon': self.epsilon,
            'update_counter': self.update_counter,
            # Rainbow feature flags
            'use_double': self.use_double,
            'use_dueling': self.use_dueling,
            'use_per': self.use_per,
            'n_step': self.n_step,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: Load path.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
        self.epsilon = checkpoint['epsilon']
        self.update_counter = checkpoint['update_counter']
        # Verify Rainbow flags match (warn if mismatched)
        for flag in ['use_double', 'use_dueling', 'use_per', 'n_step']:
            if flag in checkpoint and getattr(self, flag) != checkpoint[flag]:
                warnings.warn(
                    f"Checkpoint {flag}={checkpoint[flag]} differs from agent {flag}={getattr(self, flag)}. "
                    "Using agent's current setting.",
                    UserWarning
                )


class DoubleDQNAgent(DQNAgent):
    """Double DQN agent (DEPRECATED).

    DEPRECATED: Use DQNAgent(use_double=True) instead.

    This class is maintained for backward compatibility only.
    New code should use DQNAgent with the use_double flag.

    Reference:
        van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning"
        https://arxiv.org/abs/1509.06461
    """

    def __init__(self, *args, **kwargs):
        """Initialize with Double DQN enabled."""
        warnings.warn(
            "DoubleDQNAgent is deprecated. Use DQNAgent(use_double=True) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        kwargs['use_double'] = True
        super().__init__(*args, **kwargs)


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN agent (DEPRECATED).

    DEPRECATED: Use DQNAgent(use_dueling=True) instead.

    This class is maintained for backward compatibility only.
    New code should use DQNAgent with the use_dueling flag.

    Reference:
        Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"
        https://arxiv.org/abs/1511.06581
    """

    def __init__(self, *args, **kwargs):
        """Initialize with Dueling architecture enabled."""
        warnings.warn(
            "DuelingDQNAgent is deprecated. Use DQNAgent(use_dueling=True) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        kwargs['use_dueling'] = True
        super().__init__(*args, **kwargs)
