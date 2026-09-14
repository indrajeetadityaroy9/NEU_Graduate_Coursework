"""Distributed training script for multi-GPU training with PyTorch DDP.

Usage:
    torchrun --nproc_per_node=2 train_distributed.py --config configs/h100_optimized.yaml

Or with explicit GPU selection:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_distributed.py --config configs/h100_optimized.yaml
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.distributed as dist

from envs import MazeEnv, make_vec_env
from agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, PPOAgent
from utils import set_global_seed, load_config, ExperimentLogger, MazeVisualizer
from utils.distributed import (
    setup_ddp,
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    wrap_ddp,
    sync_dict,
    barrier,
)


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
    """Create an agent with hardware optimizations.

    Args:
        algorithm: Algorithm name.
        obs_dim: Observation dimension.
        n_actions: Number of actions.
        config: Algorithm-specific configuration.
        device: Compute device.
        seed: Random seed.
        mixed_precision: Enable AMP.
        compile_mode: torch.compile mode.

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


def train_dqn_distributed(
    env,
    agent,
    config,
    logger,
    visualizer,
    rank: int,
    world_size: int,
):
    """Distributed training loop for DQN-family algorithms.

    Each GPU runs its own environment and accumulates experience.
    Gradients are synchronized via DDP.

    Args:
        env: Vectorized environment (or single env).
        agent: DQN agent (with DDP-wrapped networks).
        config: Training configuration.
        logger: Experiment logger (only used on main process).
        visualizer: Visualization handler.
        rank: Process rank.
        world_size: Total number of processes.
    """
    total_timesteps = config.training.total_timesteps // world_size  # Split across GPUs
    eval_freq = config.training.eval_freq
    save_freq = config.training.save_freq
    log_freq = config.training.log_freq
    n_eval_episodes = config.training.n_eval_episodes
    max_episode_steps = config.env.max_episode_steps

    obs, info = env.reset(seed=config.seed + rank)
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    episode_rewards = []
    global_step = 0

    if is_main_process():
        print(f"Starting distributed DQN training across {world_size} GPUs...")
        print(f"Each GPU processes {total_timesteps} timesteps")

    for step in range(total_timesteps):
        # Select and execute action
        action = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or (episode_length >= max_episode_steps)

        # Store transition
        agent.store_transition(obs, action, reward, next_obs, done)

        # Update agent (gradients are synced via DDP)
        update_info = agent.update()

        # Track episode stats
        episode_reward += reward
        episode_length += 1
        global_step += 1

        # Episode finished
        if done:
            episode_rewards.append(episode_reward)

            if is_main_process():
                logger.log_episode(
                    episode=episode_count,
                    timestep=global_step * world_size,
                    reward=episode_reward,
                    length=episode_length,
                    metrics=update_info,
                )

                if (episode_count + 1) % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    print(f"Episode {episode_count + 1} | Step {global_step * world_size} | "
                          f"Reward: {episode_reward:.1f} | Avg(10): {avg_reward:.1f} | "
                          f"Epsilon: {agent.epsilon:.3f}")

            obs, info = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_count += 1
        else:
            obs = next_obs

        # Periodic logging (only main process)
        if step % log_freq == 0 and update_info and is_main_process():
            # Sync metrics across processes
            synced_metrics = sync_dict(update_info)
            logger.log_training_step(global_step * world_size, synced_metrics)

        # Periodic evaluation (only main process, but sync first)
        if (step + 1) % eval_freq == 0:
            barrier()  # Sync all processes
            if is_main_process():
                eval_metrics = evaluate(env, agent, n_eval_episodes, max_episode_steps)
                logger.log_evaluation(global_step * world_size, eval_metrics)
                print(f"Eval @ step {global_step * world_size}: "
                      f"Mean reward: {eval_metrics['eval_reward_mean']:.1f} | "
                      f"Success rate: {eval_metrics['eval_success_rate']:.2f}")

        # Periodic checkpoint (only main process)
        if (step + 1) % save_freq == 0:
            barrier()
            if is_main_process():
                checkpoint_path = logger.checkpoint_dir / f"checkpoint_{global_step * world_size}.pt"
                agent.save(str(checkpoint_path))

    # Final save
    barrier()
    if is_main_process():
        final_path = logger.checkpoint_dir / "checkpoint_final.pt"
        agent.save(str(final_path))
        visualizer.plot_learning_curve(
            episode_rewards,
            save_path=str(logger.figures_dir / "learning_curve.png"),
            title=f"{config.algorithm.upper()} Learning Curve (Distributed)"
        )
        print(f"Training complete! Final checkpoint saved to {final_path}")


def train_ppo_distributed(
    env,
    agent,
    config,
    logger,
    visualizer,
    rank: int,
    world_size: int,
):
    """Distributed training loop for PPO algorithm.

    Args:
        env: Vectorized environment.
        agent: PPO agent.
        config: Training configuration.
        logger: Experiment logger.
        visualizer: Visualization handler.
        rank: Process rank.
        world_size: Total number of processes.
    """
    total_timesteps = config.training.total_timesteps // world_size
    eval_freq = config.training.eval_freq
    save_freq = config.training.save_freq
    n_eval_episodes = config.training.n_eval_episodes
    max_episode_steps = config.env.max_episode_steps
    n_steps = config.ppo.n_steps

    obs, info = env.reset(seed=config.seed + rank)
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    global_step = 0
    episode_rewards = []

    if is_main_process():
        print(f"Starting distributed PPO training across {world_size} GPUs...")

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
                if is_main_process():
                    logger.log_episode(
                        episode=episode_count,
                        timestep=global_step * world_size,
                        reward=episode_reward,
                        length=episode_length,
                    )
                    if (episode_count + 1) % 10 == 0:
                        avg_reward = np.mean(episode_rewards[-10:])
                        print(f"Episode {episode_count + 1} | Step {global_step * world_size} | "
                              f"Reward: {episode_reward:.1f} | Avg(10): {avg_reward:.1f}")

                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
            else:
                obs = next_obs

            if global_step >= total_timesteps:
                break

        # Compute returns and update
        last_value = agent.get_value(obs)
        agent.compute_returns_and_advantages(last_value)
        update_metrics = agent.update()

        if is_main_process():
            synced_metrics = sync_dict(update_metrics)
            logger.log_training_step(global_step * world_size, synced_metrics)

        # Evaluation
        if global_step % eval_freq < n_steps:
            barrier()
            if is_main_process():
                eval_metrics = evaluate(env, agent, n_eval_episodes, max_episode_steps)
                logger.log_evaluation(global_step * world_size, eval_metrics)
                print(f"Eval @ step {global_step * world_size}: "
                      f"Mean reward: {eval_metrics['eval_reward_mean']:.1f} | "
                      f"Success rate: {eval_metrics['eval_success_rate']:.2f}")

        # Checkpoint
        if global_step % save_freq < n_steps:
            barrier()
            if is_main_process():
                checkpoint_path = logger.checkpoint_dir / f"checkpoint_{global_step * world_size}.pt"
                agent.save(str(checkpoint_path))

    # Final save
    barrier()
    if is_main_process():
        final_path = logger.checkpoint_dir / "checkpoint_final.pt"
        agent.save(str(final_path))
        visualizer.plot_learning_curve(
            episode_rewards,
            save_path=str(logger.figures_dir / "learning_curve.png"),
            title="PPO Learning Curve (Distributed)"
        )
        print(f"Training complete! Final checkpoint saved to {final_path}")


def evaluate(env, agent, n_episodes: int, max_steps: int) -> Dict[str, float]:
    """Evaluate agent performance.

    Args:
        env: Environment for evaluation.
        agent: Agent to evaluate.
        n_episodes: Number of episodes.
        max_steps: Maximum steps per episode.

    Returns:
        Dictionary of evaluation metrics.
    """
    # Create a fresh env for evaluation to avoid state interference
    eval_env = MazeEnv(
        rows=env.rows if hasattr(env, 'rows') else 20,
        cols=env.cols if hasattr(env, 'cols') else 20,
    )

    rewards = []
    lengths = []
    successes = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done and episode_length < max_steps:
            action = agent.select_action(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated
            episode_reward += reward
            episode_length += 1

        rewards.append(episode_reward)
        lengths.append(episode_length)
        successes.append(1.0 if terminated else 0.0)

    eval_env.close()

    return {
        'eval_reward_mean': np.mean(rewards),
        'eval_reward_std': np.std(rewards),
        'eval_length_mean': np.mean(lengths),
        'eval_success_rate': np.mean(successes),
    }


def main():
    parser = argparse.ArgumentParser(description="Distributed training for maze RL")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--seed', type=int, default=None, help='Override seed')
    args = parser.parse_args()

    # Get distributed info from environment (set by torchrun)
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    # Initialize distributed training
    if world_size > 1:
        setup_ddp(rank, world_size)

    # Load configuration
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed

    # Set seed (different for each rank)
    set_global_seed(config.seed + rank)

    # Set device
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'

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

    # Create agent with hardware optimizations
    agent_config = config.get_agent_config()
    mixed_precision = config.hardware.mixed_precision if hasattr(config, 'hardware') else False
    compile_mode = config.hardware.compile_mode if hasattr(config, 'hardware') else None

    agent = create_agent(
        config.algorithm,
        obs_dim,
        n_actions,
        agent_config,
        device,
        config.seed + rank,
        mixed_precision=mixed_precision,
        compile_mode=compile_mode,
    )

    # Wrap networks with DDP if distributed
    if world_size > 1:
        if hasattr(agent, 'q_network'):
            agent.q_network = wrap_ddp(agent.q_network, rank)
        if hasattr(agent, 'target_network'):
            # Don't wrap target network - it's not trained directly
            pass
        if hasattr(agent, 'ac_network'):
            agent.ac_network = wrap_ddp(agent.ac_network, rank)

    # Create logger (only on main process)
    logger = None
    visualizer = None
    if is_main_process():
        logger = ExperimentLogger(
            log_dir=config.log_dir,
            experiment_name=f"{config.algorithm}_{config.experiment_name or 'maze'}_distributed",
            config=config,
        )
        visualizer = MazeVisualizer(
            enabled=config.visualization.enabled,
            save_only=config.visualization.save_only,
        )
        print(f"Algorithm: {config.algorithm}")
        print(f"World size: {world_size}")
        print(f"Mixed precision: {mixed_precision}")
        print(f"Compile mode: {compile_mode}")
        print(f"Experiment directory: {logger.experiment_dir}")

    # Train
    try:
        if config.algorithm in ('dqn', 'ddqn', 'dueling_dqn', 'rainbow'):
            train_dqn_distributed(env, agent, config, logger, visualizer, rank, world_size)
        elif config.algorithm == 'ppo':
            train_ppo_distributed(env, agent, config, logger, visualizer, rank, world_size)
        else:
            raise ValueError(f"Unknown algorithm: {config.algorithm}")
    finally:
        if is_main_process() and logger:
            logger.close()
        env.close()
        if world_size > 1:
            cleanup_ddp()


if __name__ == "__main__":
    main()
