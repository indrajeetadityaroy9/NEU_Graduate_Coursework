"""Dueling Network Architecture for Dueling DQN."""

from typing import Tuple
import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """Dueling Network Architecture.

    Separates state value estimation from advantage estimation,
    then combines them: Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Reference:
        Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning"
        https://arxiv.org/abs/1511.06581
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize the Dueling Network.

        Args:
            obs_dim: Dimension of the observation space.
            n_actions: Number of discrete actions.
            hidden_dims: Tuple of hidden layer sizes for shared layers.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Shared feature extraction layers
        shared_layers = []
        prev_dim = obs_dim

        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()

        # If no shared layers, use obs_dim as input to streams
        stream_input_dim = hidden_dims[-2] if len(hidden_dims) > 1 else obs_dim
        final_hidden = hidden_dims[-1]

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(stream_input_dim, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(stream_input_dim, final_hidden),
            nn.ReLU(),
            nn.Linear(final_hidden, n_actions),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass computing Q-values.

        Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Q-values tensor of shape (batch_size, n_actions).
        """
        # Shared feature extraction
        features = self.shared_network(obs)

        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

    def get_action(self, obs: torch.Tensor) -> int:
        """Get the greedy action for a single observation.

        Args:
            obs: Single observation tensor.

        Returns:
            Action index with highest Q-value.
        """
        with torch.no_grad():
            q_values = self.forward(obs.unsqueeze(0))
            return q_values.argmax(dim=1).item()
