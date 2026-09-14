"""Configuration management with dataclasses and YAML loading."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Any, Dict
from pathlib import Path
import yaml


@dataclass
class EnvConfig:
    """Environment configuration."""
    rows: int = 20
    cols: int = 20
    stochasticity: float = 0.02
    max_episode_steps: int = 1000


@dataclass
class RewardConfig:
    """Reward structure configuration."""
    goal: float = 200.0
    oil: float = -5.0
    bump: float = -10.0
    action: float = -1.0


@dataclass
class DQNConfig:
    """DQN agent configuration with Rainbow extensions.

    Rainbow features (configurable via flags):
    - Double DQN (use_double): Reduces overestimation bias
    - Dueling architecture (use_dueling): Separates V and A streams
    - Prioritized Experience Replay (use_per): TD-error based sampling
    - N-step returns (n_step): Multi-step TD targets
    """
    # Network architecture
    hidden_dims: Tuple[int, ...] = (256, 256)
    learning_rate: float = 1e-4
    gamma: float = 0.99

    # Exploration schedule
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 10000

    # Target network
    target_update_freq: int = 1000
    tau: float = 1.0

    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000

    # Rainbow feature flags
    use_double: bool = False
    use_dueling: bool = False
    use_per: bool = False
    n_step: int = 1

    # PER parameters (only used if use_per=True)
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_beta_frames: int = 100000


@dataclass
class PPOConfig:
    """PPO agent configuration."""
    hidden_dims: Tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_steps: int = 2048
    n_epochs: int = 10
    batch_size: int = 64


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_timesteps: int = 100000
    eval_freq: int = 5000
    save_freq: int = 10000
    log_freq: int = 100
    n_eval_episodes: int = 10


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    enabled: bool = False
    save_only: bool = True
    save_path: str = "figures"


@dataclass
class HardwareConfig:
    """Hardware optimization configuration."""
    mixed_precision: bool = True
    compile_mode: Optional[str] = "reduce-overhead"  # "reduce-overhead", "max-autotune", or None
    num_envs: int = 16
    num_workers: int = 4


@dataclass
class DistributedConfig:
    """Distributed training configuration."""
    enabled: bool = False
    world_size: int = 2
    backend: str = "nccl"


@dataclass
class PERConfig:
    """Prioritized Experience Replay configuration."""
    enabled: bool = False
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_frames: int = 100000


@dataclass
class Config:
    """Main configuration container."""
    algorithm: str = "dqn"
    seed: int = 42
    device: str = "auto"
    experiment_name: Optional[str] = None
    log_dir: str = "experiments"

    env: EnvConfig = field(default_factory=EnvConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    per: PERConfig = field(default_factory=PERConfig)

    dqn: DQNConfig = field(default_factory=DQNConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def get_agent_config(self) -> Any:
        """Get the configuration for the selected algorithm."""
        if self.algorithm in ("dqn", "ddqn", "dueling_dqn"):
            return self.dqn
        elif self.algorithm == "ppo":
            return self.ppo
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


def load_config(config_path: str) -> Config:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Config object with loaded settings.
    """
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    config = Config()

    # Update top-level fields
    for key in ['algorithm', 'seed', 'device', 'experiment_name', 'log_dir']:
        if key in data:
            setattr(config, key, data[key])

    # Update nested configs
    if 'env' in data:
        config.env = EnvConfig(**data['env'])
    if 'rewards' in data:
        config.rewards = RewardConfig(**data['rewards'])
    if 'training' in data:
        config.training = TrainingConfig(**data['training'])
    if 'visualization' in data:
        config.visualization = VisualizationConfig(**data['visualization'])
    if 'hardware' in data:
        config.hardware = HardwareConfig(**data['hardware'])
    if 'distributed' in data:
        config.distributed = DistributedConfig(**data['distributed'])
    if 'per' in data:
        config.per = PERConfig(**data['per'])
    if 'dqn' in data:
        dqn_data = data['dqn'].copy()
        if 'hidden_dims' in dqn_data:
            dqn_data['hidden_dims'] = tuple(dqn_data['hidden_dims'])
        config.dqn = DQNConfig(**dqn_data)
    if 'ppo' in data:
        ppo_data = data['ppo'].copy()
        if 'hidden_dims' in ppo_data:
            ppo_data['hidden_dims'] = tuple(ppo_data['hidden_dims'])
        config.ppo = PPOConfig(**ppo_data)

    return config


def save_config(config: Config, path: str) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Config object to save.
        path: Path to save the YAML file.
    """
    def dataclass_to_dict(obj: Any) -> Any:
        if hasattr(obj, '__dataclass_fields__'):
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    data = dataclass_to_dict(config)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
