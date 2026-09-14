"""Experiment logging with CSV/JSON output."""

from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json
import csv


class ExperimentLogger:
    """Simple CSV/JSON logger for experiment tracking.

    Outputs:
    - episodes.csv: Per-episode statistics
    - evaluations.csv: Periodic evaluation results
    - config.json: Experiment configuration
    - summary.json: Final experiment summary
    """

    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize the logger.

        Args:
            log_dir: Base directory for experiment logs.
            experiment_name: Name for this experiment.
            config: Configuration dict to save.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name is None:
            experiment_name = "experiment"

        self.experiment_dir = Path(log_dir) / f"{timestamp}_{experiment_name}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.figures_dir = self.experiment_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        if config:
            self._save_config(config)

        self._init_csv_files()

        self.best_eval_reward = float('-inf')
        self.total_episodes = 0
        self.total_timesteps = 0
        self._metrics_fp = None
        self._metrics_writer = None

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        self.episodes_file = self.experiment_dir / "episodes.csv"
        with open(self.episodes_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'timestep', 'reward', 'length',
                'loss', 'q_mean', 'epsilon'
            ])

        self.eval_file = self.experiment_dir / "evaluations.csv"
        with open(self.eval_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestep', 'eval_reward_mean', 'eval_reward_std',
                'eval_length_mean', 'eval_success_rate'
            ])

    def _save_config(self, config: Dict) -> None:
        """Save experiment configuration to JSON."""
        config_path = self.experiment_dir / "config.json"

        def convert(obj):
            if hasattr(obj, '__dict__'):
                return {k: convert(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(v) for v in obj]
            return obj

        with open(config_path, 'w') as f:
            json.dump(convert(config), f, indent=2, default=str)

    def log_episode(
        self,
        episode: int,
        timestep: int,
        reward: float,
        length: int,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log episode completion.

        Args:
            episode: Episode number.
            timestep: Current total timestep.
            reward: Total episode reward.
            length: Episode length in steps.
            metrics: Optional training metrics dict.
        """
        metrics = metrics or {}

        with open(self.episodes_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode, timestep, reward, length,
                metrics.get('loss', ''),
                metrics.get('q_mean', ''),
                metrics.get('epsilon', ''),
            ])

        self.total_episodes = episode + 1
        self.total_timesteps = timestep

    def log_training_step(
        self,
        timestep: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log training metrics at given timestep.

        Args:
            timestep: Current timestep.
            metrics: Dictionary of metric name to value.
        """
        metrics_file = self.experiment_dir / "training_metrics.csv"

        if self._metrics_writer is None:
            self._metrics_fp = open(metrics_file, 'w', newline='')
            fieldnames = ['timestep'] + list(metrics.keys())
            self._metrics_writer = csv.DictWriter(
                self._metrics_fp, fieldnames=fieldnames
            )
            self._metrics_writer.writeheader()

        row = {'timestep': timestep, **metrics}
        self._metrics_writer.writerow(row)
        self._metrics_fp.flush()

    def log_evaluation(
        self,
        timestep: int,
        metrics: Dict[str, float]
    ) -> None:
        """Log evaluation results.

        Args:
            timestep: Current timestep.
            metrics: Evaluation metrics dictionary.
        """
        with open(self.eval_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestep,
                metrics.get('eval_reward_mean', ''),
                metrics.get('eval_reward_std', ''),
                metrics.get('eval_length_mean', ''),
                metrics.get('eval_success_rate', ''),
            ])

        if metrics.get('eval_reward_mean', float('-inf')) > self.best_eval_reward:
            self.best_eval_reward = metrics['eval_reward_mean']

    def close(self) -> None:
        """Close files and write summary."""
        if self._metrics_fp:
            self._metrics_fp.close()

        summary = {
            'total_episodes': self.total_episodes,
            'total_timesteps': self.total_timesteps,
            'best_eval_reward': self.best_eval_reward,
            'experiment_dir': str(self.experiment_dir),
        }

        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
