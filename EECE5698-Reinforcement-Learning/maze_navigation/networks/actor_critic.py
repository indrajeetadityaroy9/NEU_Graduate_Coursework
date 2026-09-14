"""Actor-Critic Network for PPO."""

from typing import Tuple
import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network with shared feature extraction.

    Outputs action logits (actor) and state value estimate (critic).
    Uses separate heads on top of shared feature layers.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ):
        """Initialize the Actor-Critic Network.

        Args:
            obs_dim: Dimension of the observation space.
            n_actions: Number of discrete actions.
            hidden_dims: Tuple of hidden layer sizes.
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Shared feature extraction
        shared_layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh(),  # Tanh often works better for policy gradients
            ])
            prev_dim = hidden_dim

        self.shared_network = nn.Sequential(*shared_layers)

        # Actor head: outputs action logits
        self.actor_head = nn.Linear(prev_dim, n_actions)

        # Critic head: outputs state value
        self.critic_head = nn.Linear(prev_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.shared_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(module.bias)

        # Smaller initialization for output heads
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.zeros_(self.actor_head.bias)

        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(
        self,
        obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value.

        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).

        Returns:
            Tuple of:
                - action_logits: Tensor of shape (batch_size, n_actions)
                - values: Tensor of shape (batch_size, 1)
        """
        features = self.shared_network(obs)
        action_logits = self.actor_head(features)
        values = self.critic_head(features)
        return action_logits, values

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value.

        Args:
            obs: Observation tensor.
            action: Optional action to evaluate (for computing log_prob).
            deterministic: If True, return argmax action.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
        action_logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=action_logits)

        if action is None:
            if deterministic:
                action = action_logits.argmax(dim=1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, values.squeeze(-1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get only the value estimate.

        Args:
            obs: Observation tensor.

        Returns:
            Value tensor of shape (batch_size,).
        """
        features = self.shared_network(obs)
        return self.critic_head(features).squeeze(-1)
