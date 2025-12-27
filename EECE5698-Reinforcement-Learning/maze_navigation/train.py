"""Main training script for deep RL algorithms on maze environment."""

import argparse
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from envs import MazeEnv
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent
from utils import set_global_seed, load_config, ExperimentLogger, MazeVisualizer


AGENTS = {
    'dqn': DQNAgent,
    'ddqn': DoubleDQNAgent,
    'dueling_dqn': DuelingDQNAgent,
    'rainbow': DQNAgent,  # Rainbow uses DQNAgent with all features enabled
    'ppo': PPOAgent,
}


def create_agent(
    algorithm: str,
    obs_dim: int,
    n_actions: int,
    config: Any,
    device: str,
    seed: int,
    mixed_precision: bool = False,
    compile_mode: str = None,
):
    """Create an agent based on algorithm name.

    Args:
        algorithm: Algorithm name ('dqn', 'ddqn', 'dueling_dqn', 'ppo').
        obs_dim: Observation dimension.
        n_actions: Number of actions.
        config: Algorithm-specific configuration.
        device: Compute device.
        seed: Random seed.
        mixed_precision: Enable AMP (automatic mixed precision).
        compile_mode: torch.compile mode ('reduce-overhead', 'max-autotune', None).

    Returns:
        Agent instance.
    """
    agent_cls = AGENTS[algorithm]

    if algorithm in ('dqn', 'ddqn', 'dueling_dqn', 'rainbow'):
        # For 'rainbow' algorithm, enable all Rainbow features
        use_double = getattr(config, 'use_double', False) or algorithm == 'rainbow'
        use_dueling = getattr(config, 'use_dueling', False) or algorithm == 'rainbow'
        use_per = getattr(config, 'use_per', False) or algorithm == 'rainbow'
        n_step = getattr(config, 'n_step', 1) if algorithm != 'rainbow' else max(getattr(config, 'n_step', 3), 3)

        return agent_cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=config.hidden_dims,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay_steps=config.epsilon_decay_steps,
            target_update_freq=config.target_update_freq,
            tau=config.tau,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            min_buffer_size=config.min_buffer_size,
            device=device,
            seed=seed,
            mixed_precision=mixed_precision,
            compile_mode=compile_mode,
            # Rainbow feature flags
            use_double=use_double,
            use_dueling=use_dueling,
            use_per=use_per,
            n_step=n_step,
            # PER parameters
            per_alpha=getattr(config, 'per_alpha', 0.6),
            per_beta_start=getattr(config, 'per_beta_start', 0.4),
            per_beta_end=getattr(config, 'per_beta_end', 1.0),
            per_beta_frames=getattr(config, 'per_beta_frames', 100000),
        )
    elif algorithm == 'ppo':
        return agent_cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dims=config.hidden_dims,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_epsilon=config.clip_epsilon,
            value_coef=config.value_coef,
            entropy_coef=config.entropy_coef,
            max_grad_norm=config.max_grad_norm,
            n_steps=config.n_steps,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            device=device,
            seed=seed,
            mixed_precision=mixed_precision,
            compile_mode=compile_mode,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_dqn(env: MazeEnv, agent, config, logger: ExperimentLogger, visualizer: MazeVisualizer):
    """Training loop for DQN-family algorithms.

    Args:
        env: Maze environment.
        agent: DQN agent.
        config: Training configuration.
        logger: Experiment logger.
        visualizer: Visualization handler.
    """
    total_timesteps = config.training.total_timesteps
    eval_freq = config.training.eval_freq
    save_freq = config.training.save_freq
    log_freq = config.training.log_freq
    n_eval_episodes = config.training.n_eval_episodes
    max_episode_steps = config.env.max_episode_steps

    obs, info = env.reset(seed=config.seed)
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_rewards = []

    print(f"Starting DQN training for {total_timesteps} timesteps...")

    for step in range(total_timesteps):
        # Select and execute action
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or (episode_length >= max_episode_steps)

        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)

        # Update agent
        update_info = agent.update()

        # Track episode stats
        episode_reward += reward
        episode_length += 1

        # Episode finished
        if done:
            episode_rewards.append(episode_reward)
            logger.log_episode(
                episode=episode_count,
                timestep=step,
                reward=episode_reward,
                length=episode_length,
                metrics=update_info,
            )

            if (episode_count + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode_count + 1} | Step {step + 1} | "
                      f"Reward: {episode_reward:.1f} | Avg(10): {avg_reward:.1f} | "
                      f"Epsilon: {agent.epsilon:.3f}")

            # Reset for next episode
            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
        else:
            obs = next_obs

        # Periodic logging
        if step % log_freq == 0 and update_info:
            logger.log_training_step(step, update_info)

        # Periodic evaluation
        if (step + 1) % eval_freq == 0:
            eval_metrics = evaluate(env, agent, n_eval_episodes, max_episode_steps)
            logger.log_evaluation(step + 1, eval_metrics)
            print(f"Eval @ step {step + 1}: "
                  f"Mean reward: {eval_metrics['eval_reward_mean']:.1f} | "
                  f"Success rate: {eval_metrics['eval_success_rate']:.2f}")

        # Periodic checkpoint
        if (step + 1) % save_freq == 0:
            checkpoint_path = logger.checkpoint_dir / f"checkpoint_{step + 1}.pt"
            agent.save(str(checkpoint_path))

    # Final save
    final_path = logger.checkpoint_dir / "checkpoint_final.pt"
    agent.save(str(final_path))

    # Save learning curve
    visualizer.plot_learning_curve(
        episode_rewards,
        save_path=str(logger.figures_dir / "learning_curve.png"),
        title=f"{config.algorithm.upper()} Learning Curve"
    )

    print(f"Training complete! Final checkpoint saved to {final_path}")


def train_ppo(env: MazeEnv, agent: PPOAgent, config, logger: ExperimentLogger, visualizer: MazeVisualizer):
    """Training loop for PPO algorithm.

    Args:
        env: Maze environment.
        agent: PPO agent.
        config: Training configuration.
        logger: Experiment logger.
        visualizer: Visualization handler.
    """
    total_timesteps = config.training.total_timesteps
    eval_freq = config.training.eval_freq
    save_freq = config.training.save_freq
    n_eval_episodes = config.training.n_eval_episodes
    max_episode_steps = config.env.max_episode_steps
    n_steps = config.ppo.n_steps

    obs, info = env.reset(seed=config.seed)
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    global_step = 0
    episode_rewards = []

    print(f"Starting PPO training for {total_timesteps} timesteps...")

    while global_step < total_timesteps:
        # Collect rollout
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or (episode_length >= max_episode_steps)

            agent.store_transition(obs, action, reward, value, log_prob, done)

            episode_reward += reward
            episode_length += 1
            global_step += 1

            if done:
                episode_rewards.append(episode_reward)
                logger.log_episode(
                    episode=episode_count,
                    timestep=global_step,
                    reward=episode_reward,
                    length=episode_length,
                )

                if (episode_count + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode_count + 1} | Step {global_step} | "
                          f"Reward: {episode_reward:.1f} | Avg(10): {avg_reward:.1f}")

                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
            else:
                obs = next_obs

            if global_step >= total_timesteps:
                break

        # Compute last value for GAE
        last_value = agent.get_value(obs)
        agent.compute_returns_and_advantages(last_value)

        # Update policy
        update_metrics = agent.update()
        logger.log_training_step(global_step, update_metrics)

        # Evaluation
        if global_step % eval_freq < n_steps:
            eval_metrics = evaluate(env, agent, n_eval_episodes, max_episode_steps)
            logger.log_evaluation(global_step, eval_metrics)
            print(f"Eval @ step {global_step}: "
                  f"Mean reward: {eval_metrics['eval_reward_mean']:.1f} | "
                  f"Success rate: {eval_metrics['eval_success_rate']:.2f}")

        # Checkpoint
        if global_step % save_freq < n_steps:
            checkpoint_path = logger.checkpoint_dir / f"checkpoint_{global_step}.pt"
            agent.save(str(checkpoint_path))

    # Final save
    final_path = logger.checkpoint_dir / "checkpoint_final.pt"
    agent.save(str(final_path))

    # Save learning curve
    visualizer.plot_learning_curve(
        episode_rewards,
        save_path=str(logger.figures_dir / "learning_curve.png"),
        title="PPO Learning Curve"
    )

    print(f"Training complete! Final checkpoint saved to {final_path}")


def evaluate(env: MazeEnv, agent, n_episodes: int, max_steps: int) -> Dict[str, float]:
    """Evaluate agent performance.

    Args:
        env: Maze environment.
        agent: Agent to evaluate.
        n_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.

    Returns:
        Dictionary of evaluation metrics.
    """
    rewards = []
    lengths = []
    successes = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            action = agent.select_action(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]  # PPO returns tuple
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(1.0 if terminated else 0.0)

    return {
        'eval_reward_mean': np.mean(rewards),
        'eval_reward_std': np.std(rewards),
        'eval_length_mean': np.mean(lengths),
        'eval_success_rate': np.mean(successes),
    }


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on Maze environment")
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Override seed from config'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Override device (cpu, cuda, auto)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device

    # Set global seed
    set_global_seed(config.seed)

    # Create environment
    env = MazeEnv(
        rows=config.env.rows,
        cols=config.env.cols,
        stochasticity=config.env.stochasticity,
        goal_reward=config.rewards.goal,
        oil_reward=config.rewards.oil,
        bump_reward=config.rewards.bump,
        action_reward=config.rewards.action,
    )

    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Get hardware optimization settings
    mixed_precision = getattr(config.hardware, 'mixed_precision', False) if hasattr(config, 'hardware') else False
    compile_mode = getattr(config.hardware, 'compile_mode', None) if hasattr(config, 'hardware') else None

    # Create agent
    agent_config = config.get_agent_config()
    agent = create_agent(
        config.algorithm, obs_dim, n_actions,
        agent_config, config.device, config.seed,
        mixed_precision=mixed_precision,
        compile_mode=compile_mode,
    )

    # Create logger
    logger = ExperimentLogger(
        log_dir=config.log_dir,
        experiment_name=f"{config.algorithm}_{config.experiment_name or 'maze'}",
        config=config,
    )

    # Create visualizer
    visualizer = MazeVisualizer(
        enabled=config.visualization.enabled,
        save_only=config.visualization.save_only,
    )

    print(f"Algorithm: {config.algorithm}")
    print(f"Device: {agent.device}")
    print(f"Mixed precision: {mixed_precision}")
    print(f"Compile mode: {compile_mode}")
    print(f"Experiment directory: {logger.experiment_dir}")

    # Train
    if config.algorithm in ('dqn', 'ddqn', 'dueling_dqn', 'rainbow'):
        train_dqn(env, agent, config, logger, visualizer)
    elif config.algorithm == 'ppo':
        train_ppo(env, agent, config, logger, visualizer)
    else:
        raise ValueError(f"Unknown algorithm: {config.algorithm}")

    logger.close()
    env.close()


if __name__ == "__main__":
    main()
