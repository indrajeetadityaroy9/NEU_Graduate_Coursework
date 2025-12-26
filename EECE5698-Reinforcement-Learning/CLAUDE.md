# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic reinforcement learning course project (EECE5698) implementing RL algorithms from first principles using numpy. Three problem domains: multi-armed bandits, maze navigation, and gene regulatory network control.

## Running Scripts

Scripts use relative package imports and must be run from the repository root:

```bash
# From repository root
python -m multi_armed_bandit.part_a
python -m maze_navigation.value_iteration
python -m maze_navigation.td_learning
python -m gene_regulatory_network.dp_control
python -m gene_regulatory_network.td_control

# Or set PYTHONPATH
PYTHONPATH=. python maze_navigation/value_iteration.py
```

Scripts display matplotlib plots interactively (`plt.show()`), requiring a display or X11 forwarding.

## Dependencies

```bash
pip install numpy matplotlib pandas seaborn
```

## Architecture

### Directory Structure (by problem domain)

- **multi_armed_bandit/**: Classic 2-arm bandit problem
  - `part_a.py`: Epsilon-greedy with variable learning rates
  - `part_b.py`: Optimistic initialization effects
  - `part_c.py`: Gradient bandit vs epsilon-greedy comparison

- **maze_navigation/**: 20x20 grid navigation with walls, oil, bumps
  - `maze_dp.py`: Maze class with value/policy iteration (DP methods)
  - `maze_qlearning.py`: Maze2 class with Q-learning/SARSA
  - `maze_actor_critic.py`: MazeAC class with actor-critic and eligibility traces
  - `value_iteration.py`: DP experiments (γ=0.95, θ=0.01)
  - `td_learning.py`: Q-learning, SARSA, actor-critic experiments

- **gene_regulatory_network/**: 4-gene system (ATM, p53, Wip1, MDM2) with 16 states
  - `dp_control.py`: Policy/value iteration for gene control
  - `td_control.py`: Q-learning, SARSA, SARSA-λ, actor-critic for gene control

### Key Patterns

- Maze environments are classes encapsulating state transitions, rewards, and visualization
- Learning algorithms are methods on maze classes or standalone functions
- State representation: (row, col) tuples on grid for maze; binary vectors for genes
- Actions: 'U', 'D', 'L', 'R' for maze; gene activation for network
- Reward structure: goal (+200), oil (-5), bumps (-10), action cost (-1)
- Stochasticity parameter `p=0.02` for probabilistic transitions

### Import Dependencies

- `maze_navigation/value_iteration.py` imports `from maze_navigation.maze_dp import Maze`
- `maze_navigation/td_learning.py` imports `from maze_navigation.maze_actor_critic import MazeAC as Maze` and `from maze_navigation.maze_qlearning import Maze2`
