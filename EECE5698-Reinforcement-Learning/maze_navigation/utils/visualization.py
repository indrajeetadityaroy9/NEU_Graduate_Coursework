"""Conditional visualization utilities with flag-controlled rendering."""

from typing import Optional, List, Tuple, Set
import numpy as np


class MazeVisualizer:
    """Conditional visualization with flag-controlled rendering.

    All visualization methods check the enabled flag before executing.
    When save_only=True, uses non-interactive backend to avoid blocking.
    """

    def __init__(self, enabled: bool = False, save_only: bool = True):
        """Initialize the visualizer.

        Args:
            enabled: Whether visualization is enabled.
            save_only: If True, use non-interactive backend (no plt.show()).
        """
        self.enabled = enabled
        self.save_only = save_only
        self.plt = None
        self.sns = None
        self.Rectangle = None

        if enabled:
            import matplotlib
            if save_only:
                matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.patches import Rectangle
            self.plt = plt
            self.sns = sns
            self.Rectangle = Rectangle

    def plot_learning_curve(
        self,
        rewards: List[float],
        window: int = 100,
        save_path: Optional[str] = None,
        title: str = "Learning Curve"
    ) -> None:
        """Plot episode rewards with moving average.

        Args:
            rewards: List of episode rewards.
            window: Window size for moving average.
            save_path: Path to save the figure.
            title: Plot title.
        """
        if not self.enabled:
            return

        fig, ax = self.plt.subplots(figsize=(10, 6))

        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.3, label='Episode Reward')

        if len(rewards) >= window:
            moving_avg = np.convolve(
                rewards, np.ones(window)/window, mode='valid'
            )
            ax.plot(
                episodes[window-1:], moving_avg,
                label=f'{window}-Episode Moving Avg', linewidth=2
            )

        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if not self.save_only:
            self.plt.show()

        self.plt.close(fig)

    def plot_evaluation_curve(
        self,
        timesteps: List[int],
        mean_rewards: List[float],
        std_rewards: Optional[List[float]] = None,
        save_path: Optional[str] = None,
        title: str = "Evaluation Performance"
    ) -> None:
        """Plot evaluation performance over training.

        Args:
            timesteps: List of evaluation timesteps.
            mean_rewards: Mean evaluation rewards.
            std_rewards: Optional standard deviations.
            save_path: Path to save the figure.
            title: Plot title.
        """
        if not self.enabled:
            return

        fig, ax = self.plt.subplots(figsize=(10, 6))

        ax.plot(timesteps, mean_rewards, linewidth=2, label='Mean Reward')

        if std_rewards is not None:
            mean_arr = np.array(mean_rewards)
            std_arr = np.array(std_rewards)
            ax.fill_between(
                timesteps,
                mean_arr - std_arr,
                mean_arr + std_arr,
                alpha=0.3, label='Std Dev'
            )

        ax.set_xlabel('Timestep')
        ax.set_ylabel('Evaluation Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if not self.save_only:
            self.plt.show()

        self.plt.close(fig)

    def plot_maze_with_values(
        self,
        values: np.ndarray,
        wall_positions: Set[Tuple[int, int]],
        oil_positions: Set[Tuple[int, int]],
        bump_positions: Set[Tuple[int, int]],
        start_position: Tuple[int, int],
        goal_position: Tuple[int, int],
        save_path: Optional[str] = None,
        title: str = "State Values"
    ) -> None:
        """Visualize maze with state values as heatmap.

        Args:
            values: 2D array of state values.
            wall_positions: Set of wall positions.
            oil_positions: Set of oil positions.
            bump_positions: Set of bump positions.
            start_position: Start position tuple.
            goal_position: Goal position tuple.
            save_path: Path to save the figure.
            title: Plot title.
        """
        if not self.enabled:
            return

        fig, ax = self.plt.subplots(figsize=(12, 10))

        display_values = values.copy()
        for pos in wall_positions:
            display_values[pos] = np.nan

        heatmap = self.sns.heatmap(
            display_values,
            annot=False,
            fmt=".1f",
            linewidths=0.5,
            linecolor='gray',
            cbar=True,
            cmap='viridis',
            ax=ax
        )

        self._add_maze_patches(
            ax, wall_positions, oil_positions, bump_positions,
            start_position, goal_position
        )

        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if not self.save_only:
            self.plt.show()

        self.plt.close(fig)

    def plot_maze_with_policy(
        self,
        policy: np.ndarray,
        wall_positions: Set[Tuple[int, int]],
        oil_positions: Set[Tuple[int, int]],
        bump_positions: Set[Tuple[int, int]],
        start_position: Tuple[int, int],
        goal_position: Tuple[int, int],
        save_path: Optional[str] = None,
        title: str = "Learned Policy"
    ) -> None:
        """Visualize maze with policy arrows.

        Args:
            policy: 2D array of action indices (0=Up, 1=Down, 2=Left, 3=Right).
            wall_positions: Set of wall positions.
            oil_positions: Set of oil positions.
            bump_positions: Set of bump positions.
            start_position: Start position tuple.
            goal_position: Goal position tuple.
            save_path: Path to save the figure.
            title: Plot title.
        """
        if not self.enabled:
            return

        rows, cols = policy.shape
        fig, ax = self.plt.subplots(figsize=(12, 10))

        background = np.zeros((rows, cols))
        heatmap = self.sns.heatmap(
            background,
            annot=False,
            linewidths=0.5,
            linecolor='gray',
            cbar=False,
            cmap=['white'],
            ax=ax
        )

        self._add_maze_patches(
            ax, wall_positions, oil_positions, bump_positions,
            start_position, goal_position
        )

        action_to_delta = {
            0: (0, -0.3),   # Up
            1: (0, 0.3),    # Down
            2: (-0.3, 0),   # Left
            3: (0.3, 0)     # Right
        }

        for i in range(rows):
            for j in range(cols):
                if (i, j) not in wall_positions:
                    action = int(policy[i, j])
                    dx, dy = action_to_delta.get(action, (0, 0))
                    if dx != 0 or dy != 0:
                        ax.arrow(
                            j + 0.5, i + 0.5, dx, dy,
                            head_width=0.15, head_length=0.1,
                            fc='black', ec='black', width=0.03
                        )

        ax.set_title(title)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if not self.save_only:
            self.plt.show()

        self.plt.close(fig)

    def _add_maze_patches(
        self,
        ax,
        wall_positions: Set[Tuple[int, int]],
        oil_positions: Set[Tuple[int, int]],
        bump_positions: Set[Tuple[int, int]],
        start_position: Tuple[int, int],
        goal_position: Tuple[int, int]
    ) -> None:
        """Add colored patches for maze features."""
        for pos in wall_positions:
            ax.add_patch(self.Rectangle(
                (pos[1], pos[0]), 1, 1,
                fill=True, facecolor='black', edgecolor='black', lw=0.5
            ))

        for pos in oil_positions:
            ax.add_patch(self.Rectangle(
                (pos[1], pos[0]), 1, 1,
                fill=True, facecolor='red', edgecolor='black', lw=0.5
            ))

        for pos in bump_positions:
            ax.add_patch(self.Rectangle(
                (pos[1], pos[0]), 1, 1,
                fill=True, facecolor='bisque', edgecolor='black', lw=0.5
            ))

        ax.add_patch(self.Rectangle(
            (start_position[1], start_position[0]), 1, 1,
            fill=True, facecolor='dodgerblue', edgecolor='black', lw=0.5
        ))

        ax.add_patch(self.Rectangle(
            (goal_position[1], goal_position[0]), 1, 1,
            fill=True, facecolor='yellowgreen', edgecolor='black', lw=0.5
        ))

    def compare_algorithms(
        self,
        results: dict,
        save_path: Optional[str] = None,
        title: str = "Algorithm Comparison"
    ) -> None:
        """Compare learning curves of multiple algorithms.

        Args:
            results: Dict mapping algorithm name to list of rewards.
            save_path: Path to save the figure.
            title: Plot title.
        """
        if not self.enabled:
            return

        colors = ['blue', 'red', 'green', 'orange', 'purple']

        fig, ax = self.plt.subplots(figsize=(12, 8))

        for idx, (name, rewards) in enumerate(results.items()):
            color = colors[idx % len(colors)]
            episodes = np.arange(1, len(rewards) + 1)
            ax.plot(episodes, rewards, linestyle='-', color=color,
                   linewidth=2, label=name)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Accumulated Reward')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        if not self.save_only:
            self.plt.show()

        self.plt.close(fig)
