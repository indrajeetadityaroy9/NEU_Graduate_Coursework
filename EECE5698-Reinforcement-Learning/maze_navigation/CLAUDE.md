# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research-grade deep reinforcement learning implementation for 20x20 grid maze navigation. Implements DQN variants and PPO with PyTorch, featuring H100 GPU optimizations including mixed precision, torch.compile, multi-GPU DDP, vectorized environments, and prioritized experience replay.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Single GPU Training
python train.py --config configs/dqn.yaml
python train.py --config configs/h100_optimized.yaml  # H100-optimized

# Multi-GPU Distributed Training (2x H100)
torchrun --nproc_per_node=2 train_distributed.py --config configs/h100_optimized.yaml

# Evaluate trained model
python evaluate.py --config configs/dqn.yaml --checkpoint experiments/.../checkpoints/checkpoint_final.pt
```

## Dependencies

- torch>=2.0.0
- gymnasium>=0.29.0
- numpy>=1.24.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- pyyaml>=6.0

## Architecture

```
maze_navigation/
├── configs/                     # YAML configuration files
│   ├── dqn.yaml, ddqn.yaml, dueling_dqn.yaml, ppo.yaml
│   └── h100_optimized.yaml      # H100 GPU optimized config
├── envs/                        # Environment implementations
│   ├── maze_env.py              # Gymnasium-compatible MazeEnv
│   └── vec_env.py               # Vectorized environments (Sync/Subproc)
├── agents/                      # RL agent implementations
│   ├── base_agent.py            # Abstract base with AMP support
│   ├── dqn_agent.py             # DQN, DoubleDQN, DuelingDQN
│   └── ppo_agent.py             # PPO with GAE
├── networks/                    # Neural network architectures
│   ├── mlp.py                   # Q-network
│   ├── dueling_network.py       # Dueling architecture
│   └── actor_critic.py          # Actor-Critic network
├── buffers/                     # Experience storage
│   ├── replay_buffer.py         # Standard replay (DQN)
│   ├── rollout_buffer.py        # Rollout storage (PPO)
│   └── prioritized_replay_buffer.py  # PER with sum tree
├── utils/                       # Utilities
│   ├── config.py                # Configuration management
│   ├── logger.py                # CSV/JSON experiment logging
│   ├── seed.py                  # Reproducibility
│   ├── visualization.py         # Conditional matplotlib
│   ├── distributed.py           # DDP utilities
│   └── async_env.py             # Async environment sampling
├── train.py                     # Single-GPU training script
├── train_distributed.py         # Multi-GPU DDP training
└── evaluate.py                  # Evaluation script
```

## Algorithms

| Algorithm | File | Key Features |
|-----------|------|--------------|
| DQN | `agents/dqn_agent.py` | Experience replay, target network, epsilon-greedy |
| Double DQN | `agents/dqn_agent.py` | Reduces overestimation bias |
| Dueling DQN | `agents/dqn_agent.py` | Separate value/advantage streams |
| PPO | `agents/ppo_agent.py` | Clipped objective, GAE, entropy bonus |

## Environment

- **Grid**: 20x20
- **Start**: (15, 4), **Goal**: (3, 13)
- **Actions**: Discrete(4) - Up, Down, Left, Right
- **Observation**: Normalized coordinates `(row/19, col/19)`
- **Stochasticity**: p=0.02 (probability of random action)
- **Rewards**: Goal +200, Oil -5, Bumps -10, Action cost -1

## H100 GPU Optimizations

### Mixed Precision (AMP)
```yaml
hardware:
  mixed_precision: true
```
Enables FP16 training with automatic GradScaler management.

### torch.compile()
```yaml
hardware:
  compile_mode: "reduce-overhead"  # or "max-autotune"
```
PyTorch 2.0 compilation for kernel fusion and optimization.

### Vectorized Environments
```python
from envs import make_vec_env
vec_env = make_vec_env(num_envs=16, use_subproc=True)
```
Parallel environment stepping with `SyncVecEnv` or `SubprocVecEnv`.

### Multi-GPU DDP
```bash
torchrun --nproc_per_node=2 train_distributed.py --config configs/h100_optimized.yaml
```
DistributedDataParallel training across multiple GPUs.

### Prioritized Experience Replay
```yaml
per:
  enabled: true
  alpha: 0.6
  beta_start: 0.4
```
TD-error based sampling with O(log n) sum tree.

## Configuration

All hyperparameters configured via YAML files in `configs/`. Key settings:

```yaml
algorithm: dqn  # dqn, ddqn, dueling_dqn, ppo
seed: 42
device: auto

training:
  total_timesteps: 100000
  eval_freq: 5000

hardware:
  mixed_precision: true
  compile_mode: "reduce-overhead"
  num_envs: 16

distributed:
  enabled: false
  world_size: 2
  backend: "nccl"

per:
  enabled: false
  alpha: 0.6
  beta_start: 0.4

visualization:
  enabled: true
  save_only: true
```

## Experiment Outputs

```
experiments/{timestamp}_{algorithm}/
├── config.json       # Saved hyperparameters
├── episodes.csv      # Per-episode stats
├── evaluations.csv   # Periodic evaluations
├── checkpoints/      # Model checkpoints
└── figures/          # Learning curves, policy plots
```

## Key Implementation Details

- Gymnasium-compatible `MazeEnv` with standard `reset()`/`step()` interface
- Modular agent design with `BaseAgent` abstract class supporting AMP
- `_maybe_compile()` method for optional torch.compile integration
- Vectorized environments for parallel data collection
- Sum tree implementation for O(log n) PER sampling
- DDP utilities in `utils/distributed.py` for multi-GPU training
- Conditional visualization via `enabled` flag (no blocking `plt.show()`)
- CSV/JSON logging for experiment tracking
- Seed management for reproducibility

## Performance Expectations

| Configuration | Throughput |
|--------------|------------|
| Single env, CPU | ~500 steps/s |
| Single env, GPU | ~1K steps/s |
| VecEnv (16), GPU | ~8K steps/s |
| VecEnv + AMP | ~12K steps/s |
| 2x H100 DDP | ~20K steps/s |

## References

- DQN: Mnih et al., 2015
- Double DQN: van Hasselt et al., 2015
- Dueling DQN: Wang et al., 2016
- PPO: Schulman et al., 2017
- PER: Schaul et al., 2015
- GAE: Schulman et al., 2015
