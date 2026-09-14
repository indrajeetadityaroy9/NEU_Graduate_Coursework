"""Evaluation script for trained agents."""

import argparse
from pathlib import Path

import numpy as np
import torch

from envs import MazeEnv
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent
from utils import load_config, MazeVisualizer


AGENTS = {
    'dqn': DQNAgent,
    'ddqn': DoubleDQNAgent,
    'dueling_dqn': DuelingDQNAgent,
    'ppo': PPOAgent,
}


def load_agent(algorithm: str, checkpoint_path: str, obs_dim: int, n_actions: int, config, device: str):
    """Load a trained agent from checkpoint.

    Args:
        algorithm: Algorithm name.
        checkpoint_path: Path to checkpoint file.
        obs_dim: Observation dimension.
        n_actions: Number of actions.
        config: Agent configuration.
        device: Compute device.

    Returns:
        Loaded agent.
    """
    agent_cls = AGENTS[algorithm]

    if algorithm in ('dqn', 'ddqn', 'dueling_dqn'):
        agent = agent_cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=config.hidden_dims,
            device=device,
        )
    elif algorithm == 'ppo':
        agent = agent_cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=config.hidden_dims,
            n_steps=config.n_steps,
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    agent.load(checkpoint_path)
    return agent


def evaluate_agent(env: MazeEnv, agent, n_episodes: int, max_steps: int, render: bool = False):
    """Evaluate agent performance.

    Args:
        env: Maze environment.
        agent: Trained agent.
        n_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        render: Whether to render episodes.

    Returns:
        Dictionary of evaluation metrics.
    """
    rewards = []
    lengths = []
    successes = []
    all_paths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        path = [env.current_position]

        while not done and episode_length < max_steps:
            action = agent.select_action(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)
            path.append(env.current_position)
            done = terminated
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()

        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(1.0 if terminated else 0.0)
        all_paths.append(path)

        if terminated:
            print(f"Episode {ep + 1}: SUCCESS | Steps: {episode_length} | Reward: {episode_reward:.1f}")
        else:
            print(f"Episode {ep + 1}: FAILED  | Steps: {episode_length} | Reward: {episode_reward:.1f}")

    metrics = {
        'eval_reward_mean': np.mean(rewards),
        'eval_reward_std': np.std(rewards),
        'eval_length_mean': np.mean(lengths),
        'eval_success_rate': np.mean(successes),
    }

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {metrics['eval_reward_mean']:.2f} +/- {metrics['eval_reward_std']:.2f}")
    print(f"Mean Length: {metrics['eval_length_mean']:.2f}")
    print(f"Success Rate: {metrics['eval_success_rate'] * 100:.1f}%")

    return metrics, all_paths


def extract_policy(env: MazeEnv, agent) -> np.ndarray:
    """Extract the learned policy as a 2D array.

    Args:
        env: Maze environment.
        agent: Trained agent.

    Returns:
        2D array of action indices.
    """
    policy = np.zeros((env.rows, env.cols), dtype=np.int32)

    for i in range(env.rows):
        for j in range(env.cols):
            if (i, j) not in env.wall_positions:
                obs = np.array([i / (env.rows - 1), j / (env.cols - 1)], dtype=np.float32)
                action = agent.select_action(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                policy[i, j] = action

    return policy


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--n_episodes', type=int, default=10,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render', action='store_true',
        help='Render episodes'
    )
    parser.add_argument(
        '--save_policy', type=str, default=None,
        help='Path to save policy visualization'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        help='Device to use'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create environment
    env = MazeEnv(
        rows=config.env.rows,
        cols=config.env.cols,
        stochasticity=config.env.stochasticity,
        render_mode='human' if args.render else None,
        goal_reward=config.rewards.goal,
        oil_reward=config.rewards.oil,
        bump_reward=config.rewards.bump,
        action_reward=config.rewards.action,
    )

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Load agent
    agent_config = config.get_agent_config()
    agent = load_agent(
        config.algorithm,
        args.checkpoint,
        obs_dim,
        n_actions,
        agent_config,
        args.device,
    )

    print(f"Loaded {config.algorithm.upper()} agent from {args.checkpoint}")

    # Evaluate
    metrics, paths = evaluate_agent(
        env, agent, args.n_episodes,
        config.env.max_episode_steps, args.render
    )

    # Extract and visualize policy
    if args.save_policy:
        policy = extract_policy(env, agent)

        visualizer = MazeVisualizer(enabled=True, save_only=True)
        visualizer.plot_maze_with_policy(
            policy=policy,
            wall_positions=env.wall_positions,
            oil_positions=env.oil_positions,
            bump_positions=env.bump_positions,
            start_position=env.start_position,
            goal_position=env.goal_position,
            save_path=args.save_policy,
            title=f"{config.algorithm.upper()} Learned Policy"
        )
        print(f"Policy visualization saved to {args.save_policy}")

    env.close()


if __name__ == "__main__":
    main()
