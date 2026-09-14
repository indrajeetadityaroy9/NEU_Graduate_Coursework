"""Proximal Policy Optimization (PPO) agent implementation."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_agent import BaseAgent
from networks.actor_critic import ActorCriticNetwork
from buffers.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) agent with clipped objective.

    Features:
    - Clipped surrogate objective for stable updates
    - Generalized Advantage Estimation (GAE)
    - Entropy bonus for exploration
    - Value function clipping (optional)
    - Multiple epochs of minibatch updates per rollout

    Reference:
        Schulman et al., "Proximal Policy Optimization Algorithms"
        https://arxiv.org/abs/1707.06347
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 2048,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = 'auto',
        seed: Optional[int] = None,
        mixed_precision: bool = False,
        compile_mode: Optional[str] = None,
    ):
        """Initialize the PPO agent.

        Args:
            obs_dim: Observation dimension.
            n_actions: Number of actions.
            hidden_dims: Hidden layer sizes.
            learning_rate: Optimizer learning rate.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.
            clip_epsilon: PPO clipping parameter.
            value_coef: Value loss coefficient.
            entropy_coef: Entropy bonus coefficient.
            max_grad_norm: Maximum gradient norm for clipping.
            n_steps: Number of steps per rollout.
            n_epochs: Number of update epochs per rollout.
            batch_size: Minibatch size for updates.
            device: Compute device.
            seed: Random seed.
            mixed_precision: Enable AMP (automatic mixed precision).
            compile_mode: torch.compile mode ('reduce-overhead', 'max-autotune', None).
        """
        super().__init__(obs_dim, n_actions, device, seed, mixed_precision, compile_mode)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Actor-Critic network (with optional torch.compile)
        self.ac_network = self._maybe_compile(
            ActorCriticNetwork(obs_dim, n_actions, hidden_dims).to(self.device)
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.ac_network.parameters(), lr=learning_rate
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(n_steps, obs_dim, device=self.device)

        self._rng = np.random.default_rng(seed)

    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """Select action and return action, log_prob, and value.

        Note: Unlike DQNAgent.select_action() which returns a single int,
        PPO returns a tuple for training efficiency (action, log_prob, value
        are all needed for the PPO update). When using in evaluation code
        that expects int, extract action[0] from the tuple.

        Args:
            observation: Current observation.
            deterministic: If True, select most probable action.

        Returns:
            Tuple of (action, log_prob, value) where:
                - action (int): Selected action index
                - log_prob (float): Log probability of the action
                - value (float): Value estimate for the observation
        """
        with torch.no_grad():
            obs_tensor = self.to_tensor(observation).unsqueeze(0)
            action_logits, value = self.ac_network(obs_tensor)

            dist = torch.distributions.Categorical(logits=action_logits)

            if deterministic:
                action = action_logits.argmax(dim=1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

            return action.item(), log_prob.item(), value.squeeze().item()

    def get_value(self, observation: np.ndarray) -> float:
        """Get value estimate for an observation.

        Args:
            observation: Current observation.

        Returns:
            Value estimate.
        """
        with torch.no_grad():
            obs_tensor = self.to_tensor(observation).unsqueeze(0)
            return self.ac_network.get_value(obs_tensor).item()

    def store_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ) -> None:
        """Store a transition in the rollout buffer.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            value: Value estimate.
            log_prob: Log probability of action.
            done: Whether episode ended.
        """
        self.buffer.add(obs, action, reward, value, log_prob, done)

    def compute_returns_and_advantages(self, last_value: float) -> None:
        """Compute GAE advantages and returns for the current rollout.

        Args:
            last_value: Value estimate of the final state.
        """
        self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )

    def update(self) -> Dict[str, float]:
        """Perform PPO update over the collected rollout with AMP support.

        Returns:
            Dictionary of training metrics.
        """
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
        }

        n_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Forward pass with automatic mixed precision
                with self.get_autocast_context():
                    action_logits, values = self.ac_network(batch['observations'])
                    dist = torch.distributions.Categorical(logits=action_logits)

                    log_probs = dist.log_prob(batch['actions'])
                    entropy = dist.entropy().mean()

                    # Compute ratio
                    ratio = torch.exp(log_probs - batch['old_log_probs'])

                    # Normalize advantages
                    advantages = batch['advantages']
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Clipped surrogate objective
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon
                    ) * advantages

                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    values_pred = values.squeeze(-1)
                    value_loss = F.mse_loss(values_pred, batch['returns'])

                    # Total loss
                    loss = (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                    )

                # Gradient update with GradScaler for AMP
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.ac_network.parameters(), self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = (
                        (ratio - 1.0).abs() > self.clip_epsilon
                    ).float().mean()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['approx_kl'] += approx_kl.item()
                metrics['clip_fraction'] += clip_fraction.item()
                n_updates += 1

        # Average metrics
        if n_updates > 0:
            for key in metrics:
                metrics[key] /= n_updates

        # Clear buffer after update
        self.buffer.reset()

        return metrics

    def is_ready_to_update(self) -> bool:
        """Check if buffer is full and ready for update.

        Returns:
            True if buffer is full.
        """
        return self.buffer.is_full()

    def save(self, path: str) -> None:
        """Save agent state to disk.

        Args:
            path: Save path.
        """
        torch.save({
            'ac_network': self.ac_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: Load path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.ac_network.load_state_dict(checkpoint['ac_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler'])
