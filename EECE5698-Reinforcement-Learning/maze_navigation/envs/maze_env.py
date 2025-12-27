"""Gymnasium-compatible 20x20 grid maze environment."""

from typing import Tuple, Dict, Any, Optional, Set
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# Default maze configuration (from original codebase)
DEFAULT_OIL_POSITIONS = [
    (2, 8), (2, 16), (4, 2), (5, 6), (10, 18), (15, 10),
    (16, 10), (17, 14), (17, 17), (18, 7)
]

DEFAULT_BUMP_POSITIONS = [
    (1, 11), (1, 12), (2, 1), (2, 2), (2, 3), (5, 1), (5, 9), (5, 17),
    (6, 17), (7, 17), (8, 17), (7, 10), (7, 11), (7, 2), (12, 11),
    (12, 12), (14, 1), (14, 2), (15, 17), (15, 18), (16, 7)
]


def get_default_wall_positions(nrows: int = 20, ncols: int = 20) -> list:
    """Generate default wall positions including outer boundary and inner maze."""
    # Outer walls
    walls = (
        [(0, i) for i in range(ncols)] +  # Top wall
        [(i, 0) for i in range(nrows)] +  # Left wall
        [(i, ncols - 1) for i in range(nrows)] +  # Right wall
        [(nrows - 1, i) for i in range(ncols)]  # Bottom wall
    )

    # Inner walls
    walls += [
        (2, 5), (3, 5), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8),
        (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16),
        (5, 3), (6, 3), (6, 6), (6, 9), (6, 15), (7, 3), (7, 6), (7, 9),
        (7, 12), (7, 13), (7, 14), (7, 15), (8, 6), (8, 9), (8, 15), (9, 6),
        (9, 9), (9, 15), (10, 1), (10, 2), (10, 3), (10, 4), (10, 6), (10, 9),
        (10, 10), (10, 15), (11, 6), (11, 10), (11, 13), (11, 15), (11, 16),
        (11, 17), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 10),
        (12, 13), (12, 17), (13, 7), (13, 10), (13, 13), (13, 17), (14, 7),
        (14, 10), (14, 13), (15, 7), (15, 13), (15, 14), (15, 15), (15, 16),
        (17, 1), (17, 2), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12)
    ]

    return walls


class MazeEnv(gym.Env):
    """Gymnasium-compatible 20x20 grid maze environment.

    Observation Space: Box(low=0, high=1, shape=(2,), dtype=float32)
        Normalized (row, col) coordinates.

    Action Space: Discrete(4)
        0: Up, 1: Down, 2: Left, 3: Right

    Rewards:
        - Goal reached: +200
        - Oil cell: -5
        - Bump cell: -10
        - Each action: -1
    """

    metadata = {'render_modes': ['human', 'rgb_array', None]}

    # Action mappings
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTION_TO_DELTA = {
        0: (-1, 0),  # Up
        1: (1, 0),   # Down
        2: (0, -1),  # Left
        3: (0, 1),   # Right
    }

    def __init__(
        self,
        rows: int = 20,
        cols: int = 20,
        stochasticity: float = 0.02,
        render_mode: Optional[str] = None,
        goal_reward: float = 200.0,
        oil_reward: float = -5.0,
        bump_reward: float = -10.0,
        action_reward: float = -1.0,
    ):
        """Initialize the maze environment.

        Args:
            rows: Number of rows in the grid.
            cols: Number of columns in the grid.
            stochasticity: Probability of taking a random action instead of intended.
            render_mode: Rendering mode ('human', 'rgb_array', or None).
            goal_reward: Reward for reaching the goal.
            oil_reward: Penalty for stepping on oil.
            bump_reward: Penalty for stepping on bumps.
            action_reward: Cost per action taken.
        """
        super().__init__()

        self.rows = rows
        self.cols = cols
        self.stochasticity = stochasticity
        self.render_mode = render_mode

        # Reward structure
        self.rewards = {
            'goal': goal_reward,
            'oil': oil_reward,
            'bump': bump_reward,
            'action': action_reward,
        }

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        # Maze layout
        self.start_position = (15, 4)
        self.goal_position = (3, 13)
        self.wall_positions: Set[Tuple[int, int]] = set(get_default_wall_positions(rows, cols))
        self.oil_positions: Set[Tuple[int, int]] = set(DEFAULT_OIL_POSITIONS)
        self.bump_positions: Set[Tuple[int, int]] = set(DEFAULT_BUMP_POSITIONS)

        # Current state
        self.current_position: Tuple[int, int] = self.start_position
        self._np_random: Optional[np.random.Generator] = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to the starting state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)

        self.current_position = self.start_position
        observation = self._get_observation()
        info = {
            'position': self.current_position,
            'is_success': False,
        }

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.

        Args:
            action: Action to take (0=Up, 1=Down, 2=Left, 3=Right).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Apply stochasticity
        if self._np_random is not None and self._np_random.random() < self.stochasticity:
            # Take a random different action
            other_actions = [a for a in range(4) if a != action]
            action = self._np_random.choice(other_actions)

        # Compute next position
        next_position = self._compute_next_position(action)

        # Compute reward
        reward = self._compute_reward(next_position)

        # Update state
        self.current_position = next_position

        # Check termination
        terminated = (self.current_position == self.goal_position)
        truncated = False  # Handled by TimeLimit wrapper

        observation = self._get_observation()
        info = {
            'position': self.current_position,
            'action_taken': action,
            'is_success': terminated,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Convert current position to normalized observation.

        Returns:
            Numpy array of shape (2,) with normalized coordinates.
        """
        row, col = self.current_position
        return np.array([
            row / (self.rows - 1),
            col / (self.cols - 1)
        ], dtype=np.float32)

    def _compute_next_position(self, action: int) -> Tuple[int, int]:
        """Compute the next position given an action.

        Args:
            action: The action to take.

        Returns:
            The resulting position after the action.
        """
        dx, dy = self.ACTION_TO_DELTA[action]
        next_row = self.current_position[0] + dx
        next_col = self.current_position[1] + dy

        # Check bounds and walls
        if not self._is_valid_position(next_row, next_col):
            return self.current_position

        return (next_row, next_col)

    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is valid (within bounds and not a wall).

        Args:
            row: Row coordinate.
            col: Column coordinate.

        Returns:
            True if position is valid.
        """
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if (row, col) in self.wall_positions:
            return False
        return True

    def _compute_reward(self, position: Tuple[int, int]) -> float:
        """Compute the reward for moving to a position.

        Args:
            position: The position moved to.

        Returns:
            The reward value.
        """
        reward = self.rewards['action']

        if position in self.bump_positions:
            reward += self.rewards['bump']
        elif position in self.oil_positions:
            reward += self.rewards['oil']
        elif position == self.goal_position:
            reward += self.rewards['goal']

        return reward

    def render(self) -> Optional[np.ndarray]:
        """Render the current state.

        Returns:
            RGB array if render_mode='rgb_array', else None.
        """
        if self.render_mode is None:
            return None

        # Create a simple grid representation
        grid = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)

        # White background for valid cells
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) not in self.wall_positions:
                    grid[i, j] = [255, 255, 255]

        # Walls (black)
        for pos in self.wall_positions:
            grid[pos[0], pos[1]] = [0, 0, 0]

        # Oil (red)
        for pos in self.oil_positions:
            grid[pos[0], pos[1]] = [255, 0, 0]

        # Bumps (tan/bisque)
        for pos in self.bump_positions:
            grid[pos[0], pos[1]] = [255, 228, 196]

        # Start (blue)
        grid[self.start_position[0], self.start_position[1]] = [30, 144, 255]

        # Goal (green)
        grid[self.goal_position[0], self.goal_position[1]] = [154, 205, 50]

        # Current position (orange)
        grid[self.current_position[0], self.current_position[1]] = [255, 165, 0]

        if self.render_mode == 'human':
            try:
                import matplotlib.pyplot as plt
                plt.imshow(grid)
                plt.title(f"Position: {self.current_position}")
                plt.axis('off')
                plt.pause(0.01)
            except ImportError:
                pass

        return grid

    def get_state_count(self) -> int:
        """Get the number of valid (non-wall) states.

        Returns:
            Number of valid states.
        """
        total = self.rows * self.cols
        return total - len(self.wall_positions)

    def close(self) -> None:
        """Clean up resources."""
        pass
