"""MLP Q-Network for DQN agents."""

from typing import Tuple
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Multi-layer perceptron Q-Network.

    Standard feedforward network that outputs Q-values for each action.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize the Q-Network.

        Args:
            obs_dim: Dimension of the observation space.
            n_actions: Number of discrete actions.
            hidden_dims: Tuple of hidden layer sizes.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, n_actions))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Q-values tensor of shape (batch_size, n_actions).
        """
        return self.network(obs)

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
