import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from maze_navigation.maze_dp import Maze

# Maze Setup
# Define maze dimensions, actions, and rewards
nrows, ncols = 20, 20
actions = ['L', 'R', 'U', 'D']  # Left, Right, Up, Down
oil_reward, bump_reward, goal_reward, action_reward, empty_reward = -5, -10, 200, -1, 0
# Goal state the agent aims to reach
goal_position = (3, 13)
# Starting state of the agent
start_position = (15, 4)
# Oil and bump states
oil_positions = [(2, 8), (2, 16), (4, 2), (5, 6), (10, 18), (15, 10), (16, 10), (17, 14), (17, 17), (18, 7)]
bump_positions = [(1, 11), (1, 12), (2, 1), (2, 2), (2, 3), (5, 1), (5, 9), (5, 17), (6, 17), (7, 17), (8, 17), (7, 10),
                  (7, 11), (7, 2), (12, 11), (12, 12), (14, 1), (14, 2), (15, 17), (15, 18), (16, 7)]
# Outer Wall states
wall_positions = ([(0, i) for i in range(ncols)] +  # Top wall
                  [(i, 0) for i in range(nrows)] +  # Left wall
                  [(i, ncols - 1) for i in range(nrows)] +  # Right wall
                  [(nrows - 1, i) for i in range(ncols)])  # Bottom wall

# Inner wall states within the maze not on the boundary
wall_positions += [
    (2, 5), (3, 5), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14), (4, 15), (4, 16),
    (5, 3), (6, 3), (6, 6), (6, 9), (6, 15), (7, 3), (7, 6), (7, 9), (7, 12), (7, 13), (7, 14), (7, 15), (8, 6), (8, 9), (8, 15), (9, 6), (9, 9), (9, 15),
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 6), (10, 9), (10, 10), (10, 15), (11, 6), (11, 10), (11, 13), (11, 15), (11, 16), (11, 17),
    (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 10), (12, 13), (12, 17), (13, 7), (13, 10), (13, 13), (13, 17),
    (14, 7), (14, 10), (14, 13), (15, 7), (15, 13), (15, 14), (15, 15), (15, 16), (17, 1), (17, 2), (17, 7), (17, 8), (17, 9), (17, 10), (17, 11), (17, 12)
]

# Simulation Parameters
theta, gamma, p = 0.01, 0.95, 0.02  # Convergence threshold, discount factor, stochasticity probability

# Initialize the maze environment
maze = Maze(
    nrows, ncols,  # Maze dimensions
    actions,       # Possible actions
    p,          # Stochasticity in action outcomes
    start_position,  # Starting position of the agent
    goal_position,   # Goal position to reach
    oil_positions,   # Positions of oil cells with negative reward
    bump_positions,  # Positions of bump cells with negative reward
    wall_positions,  # Positions of walls where the agent can't go
    oil_reward,      # Reward for oil cells
    bump_reward,     # Reward for bump cells
    goal_reward,     # Reward for reaching the goal
    empty_reward,    # Reward for empty cells
    action_reward    # Cost for taking an action
)
# Perform policy iteration
maze.value_iteration(gamma, theta)
# Copy the state values after policy iteration to a new matrix
Value_Matrix = maze.values.copy()
# Create heatmap to visualize the value matrix
plt.subplots(figsize=(20, 15))
heatmap = sns.heatmap(Value_Matrix, fmt=".2f", linewidths=1, linecolor='black', cbar=False, cmap=['white'])
for i in range(Value_Matrix.shape[0]):
    for j in range(Value_Matrix.shape[1]):
        text_color = "white" if (i, j) in wall_positions else "black"
        text = heatmap.text(j + 0.5, i + 0.5, f"({j},{i})", ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')
# Overlay the maze and special positions (like oil and bumps) on the heatmap
maze.coloring_blocks(heatmap)
plt.show()

# Convergence threshold
theta = 0.01
# Discount factor
gamma = 0.95
# Stochasticity probability for the action outcomes
p = 0.02
# Perform policy iteration with the specified gamma value to find the optimal policy
maze.value_iteration(gamma, theta)

plt.figure(figsize=(10, 7.5))
# Create a heatmap visualization of the maze matrix
heatmap = sns.heatmap(maze.maze, annot=False, fmt=".2f", linewidths=0.25, linecolor='black', cbar=False, cmap=['white'])
# Set background color for non-data cells
heatmap.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap)
# Overlay the optimal path calculated by policy iteration
maze.draw_optimal_path()
plt.show()

# Copy the state value matrix after policy iteration
Value_Matrix = maze.values.copy()
# Set the value of wall positions to NaN for visualization
for k in maze.wall_positions:
    Value_Matrix[k] = np.nan

plt.figure(figsize=(20, 15))
# Create a heatmap of state values
heatmap_values = sns.heatmap(Value_Matrix, annot=np.array(maze.values), fmt=".2f", linewidths=1, linecolor='black',
                             cbar=False, cmap=['white'], annot_kws={'size': 11, 'weight': 'bold', 'color': 'black'})
# Set background color for non-data cells
heatmap_values.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap_values)
plt.show()

plt.figure(figsize=(10, 7.5))
# Create a heatmap for the maze
heatmap_policy = sns.heatmap(maze.maze, annot=False, fmt="", linewidths=1, linecolor='black', cbar=False, cmap=['white'])
# Set background color for non-data cells
heatmap_policy.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap_policy)
# Add arrows to visualize the policy directions at each state
maze.draw_policy_arrows()
plt.show()

# Convergence threshold
theta = 0.01
# Discount factor
gamma = 0.95
# Stochasticity probability for the action outcomes
p = 0.5

# Initialize the maze environment
maze = Maze(
    nrows, ncols,  # Maze dimensions
    actions,       # Possible actions
    p,          # Stochasticity in action outcomes
    start_position,  # Starting position of the agent
    goal_position,   # Goal position to reach
    oil_positions,   # Positions of oil cells with negative reward
    bump_positions,  # Positions of bump cells with negative reward
    wall_positions,  # Positions of walls where the agent can't go
    oil_reward,      # Reward for oil cells
    bump_reward,     # Reward for bump cells
    goal_reward,     # Reward for reaching the goal
    empty_reward,    # Reward for empty cells
    action_reward    # Cost for taking an action
)
# Perform policy iteration
maze.value_iteration(gamma, theta)

plt.figure(figsize=(10, 7.5))
# Create a heatmap visualization of the maze matrix
heatmap = sns.heatmap(maze.maze, annot=False, fmt=".2f", linewidths=0.25, linecolor='black', cbar=False, cmap=['white'])
# Set background color for non-data cells
heatmap.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap)
# Overlay the optimal path calculated by policy iteration
maze.draw_optimal_path()
plt.show()

# Copy the state value matrix after policy iteration
Value_Matrix = maze.values.copy()
# Set the value of wall positions to NaN for visualization
for k in maze.wall_positions:
    Value_Matrix[k] = np.nan

plt.figure(figsize=(20, 15))
# Create a heatmap of state values
heatmap_values = sns.heatmap(Value_Matrix, annot=np.array(maze.values), fmt=".2f", linewidths=1, linecolor='black',
                             cbar=False, cmap=['white'], annot_kws={'size': 11, 'weight': 'bold', 'color': 'black'})
# Set background color for non-data cells
heatmap_values.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap_values)
plt.show()

plt.figure(figsize=(10, 7.5))
# Create a heatmap for the maze
heatmap_policy = sns.heatmap(maze.maze, annot=False, fmt="", linewidths=1, linecolor='black', cbar=False, cmap=['white'])
# Set background color for non-data cells
heatmap_policy.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap_policy)
# Add arrows to visualize the policy directions at each state
maze.draw_policy_arrows()
plt.show()

# Convergence threshold
theta = 0.01
# Discount factor
gamma = 0.55
# Stochasticity probability for the action outcomes
p = 0.02

# Initialize the maze environment
maze = Maze(
    nrows, ncols,  # Maze dimensions
    actions,       # Possible actions
    p,          # Stochasticity in action outcomes
    start_position,  # Starting position of the agent
    goal_position,   # Goal position to reach
    oil_positions,   # Positions of oil cells with negative reward
    bump_positions,  # Positions of bump cells with negative reward
    wall_positions,  # Positions of walls where the agent can't go
    oil_reward,      # Reward for oil cells
    bump_reward,     # Reward for bump cells
    goal_reward,     # Reward for reaching the goal
    empty_reward,    # Reward for empty cells
    action_reward    # Cost for taking an action
)
# Perform policy iteration
maze.value_iteration(gamma, theta)

plt.figure(figsize=(10, 7.5))
# Create a heatmap visualization of the maze matrix
heatmap = sns.heatmap(maze.maze, annot=False, fmt=".2f", linewidths=0.25, linecolor='black', cbar=False, cmap=['white'])
# Set background color for non-data cells
heatmap.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap)
# Overlay the optimal path calculated by policy iteration
maze.draw_optimal_path()
plt.show()

# Copy the state value matrix after policy iteration
Value_Matrix = maze.values.copy()
# Set the value of wall positions to NaN for visualization
for k in maze.wall_positions:
    Value_Matrix[k] = np.nan

plt.figure(figsize=(20, 15))
# Create a heatmap of state values
heatmap_values = sns.heatmap(Value_Matrix, annot=np.array(maze.values), fmt=".2f", linewidths=1, linecolor='black',
                             cbar=False, cmap=['white'], annot_kws={'size': 11, 'weight': 'bold', 'color': 'black'})
# Set background color for non-data cells
heatmap_values.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap_values)
plt.show()

plt.figure(figsize=(10, 7.5))
# Create a heatmap for the maze
heatmap_policy = sns.heatmap(maze.maze, annot=False, fmt="", linewidths=1, linecolor='black', cbar=False, cmap=['white'])
# Set background color for non-data cells
heatmap_policy.set_facecolor('black')
# Add color blocks for special states (goal, oil, etc.)
maze.coloring_blocks(heatmap_policy)
# Add arrows to visualize the policy directions at each state
maze.draw_policy_arrows()
plt.show()
