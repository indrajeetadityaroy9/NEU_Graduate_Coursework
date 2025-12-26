import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from maze_navigation.maze_actor_critic import MazeAC as Maze
from maze_navigation.maze_qlearning import Maze2

# Maze Setup
nrows, ncols = 20, 20
actions = ['U', 'D', 'L', 'R']
start_position = (15, 4)
goal_position = (3, 13)
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
    (2, 5), (3, 5), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14),
    (4, 15), (4, 16),
    (5, 3), (6, 3), (6, 6), (6, 9), (6, 15), (7, 3), (7, 6), (7, 9), (7, 12), (7, 13), (7, 14), (7, 15), (8, 6), (8, 9),
    (8, 15), (9, 6), (9, 9), (9, 15),
    (10, 1), (10, 2), (10, 3), (10, 4), (10, 6), (10, 9), (10, 10), (10, 15), (11, 6), (11, 10), (11, 13), (11, 15),
    (11, 16), (11, 17),
    (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 10), (12, 13), (12, 17), (13, 7), (13, 10), (13, 13), (13, 17),
    (14, 7), (14, 10), (14, 13), (15, 7), (15, 13), (15, 14), (15, 15), (15, 16), (17, 1), (17, 2), (17, 7), (17, 8),
    (17, 9), (17, 10), (17, 11), (17, 12)
]

oil_reward = -5
bump_reward = -10
goal_reward = 200
action_reward = -1
p = 0.02


def run_Q_learning():
    rewards_across_runs = []
    best_policy_score = float('-inf')
    best_maze = None
    path_found_count = 0

    for i in range(10):
        print(f"Starting Q-learning run {i + 1}")
        maze = Maze2(nrows, ncols, actions, p, start_position, goal_position, oil_positions, bump_positions, wall_positions, oil_reward, bump_reward, goal_reward, action_reward)
        Q, path_found, episode_rewards = maze.q_learning(0.1, 0.3, 0.95, 1000, 1000)
        rewards_across_runs.append(episode_rewards)

        if path_found:
            path_found_count += 1
            average_reward = np.mean(episode_rewards) * len(episode_rewards)

            if average_reward > best_policy_score:
                best_policy_score = average_reward
                best_maze = maze

    plot_best_maze(best_maze)
    rewards_across_runs = plot_and_compute_average_accumulated_rewards(rewards_across_runs)
    print(f"In 10 independent runs of Q-Learning navigation, upon the termination of learning, a path from start to goal has been obtained {path_found_count} times.")
    return rewards_across_runs


def run_SARSA_learning():
    rewards_across_runs = []
    best_policy_score = float('-inf')
    best_maze = None
    path_found_count = 0

    for i in range(10):
        print(f"Starting SARSA-learning run {i + 1}")
        maze = Maze2(nrows, ncols, actions, p, start_position, goal_position, oil_positions, bump_positions, wall_positions, oil_reward, bump_reward, goal_reward, action_reward)
        Q, path_found, episode_rewards = maze.sarsa_learning(0.1, 0.3, 0.95, 1000, 1000)
        rewards_across_runs.append(episode_rewards)

        if path_found:
            path_found_count += 1
            average_reward = np.mean(episode_rewards) * len(episode_rewards)

            if average_reward > best_policy_score:
                best_policy_score = average_reward
                best_maze = maze

    plot_best_maze(best_maze)
    rewards_across_runs = plot_and_compute_average_accumulated_rewards(rewards_across_runs)
    print(f"In 10 independent runs of SARSA navigation, upon the termination of learning, a path from start to goal has been obtained {path_found_count} times.")
    return rewards_across_runs


def run_AC_learning():
    rewards_across_runs = []
    best_policy_score = float('-inf')
    best_maze = None
    path_found_count = 0

    for i in range(10):
        print(f"Starting AC-learning run {i + 1}")
        maze = Maze(nrows, ncols, actions, p, start_position, goal_position, oil_positions, bump_positions, wall_positions, oil_reward, bump_reward, goal_reward, action_reward)
        _, path_found, optimal_path, episode_rewards = maze.actor_critic(0.05, 0.3, 0.95, 0.9, 1000, 1000)
        rewards_across_runs.append(episode_rewards)

        if path_found:
            path_found_count += 1
            average_reward = np.mean(episode_rewards) * len(episode_rewards)

            if average_reward > best_policy_score:
                best_policy_score = average_reward
                best_maze = maze

    plot_best_maze(best_maze)
    rewards_across_runs = plot_and_compute_average_accumulated_rewards(rewards_across_runs)
    print(f"In 10 independent runs of Actor-Critic navigation, upon the termination of learning, a path from start to goal has been obtained {path_found_count} times.")
    return rewards_across_runs


def plot_best_maze(best_maze):
    plt.figure(figsize=(10, 7.5))
    heatmap_policy = sns.heatmap(best_maze.maze, annot=False, fmt="", linewidths=1, linecolor='black', cbar=False, cmap=['white'])
    heatmap_policy.set_facecolor('black')
    best_maze.coloring_blocks(heatmap_policy)
    best_maze.draw_policy_arrows()
    plt.show()

    plt.figure(figsize=(10, 7.5))
    heatmap_policy = sns.heatmap(best_maze.maze, annot=False, fmt="", linewidths=1, linecolor='black', cbar=False, cmap=['white'])
    heatmap_policy.set_facecolor('black')
    best_maze.coloring_blocks(heatmap_policy)
    best_maze.draw_optimal_path()
    plt.show()


def plot_comparison_of_algorithms(cumulative_rewards_q, cumulative_rewards_s, cumulative_rewards_ac):
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(cumulative_rewards_q) + 1), cumulative_rewards_q, linestyle='-', color='blue', linewidth=2, label='Q-learning')
    plt.plot(np.arange(1, len(cumulative_rewards_s) + 1), cumulative_rewards_s, linestyle='-', color='red', linewidth=2, label='SARSA')
    plt.plot(np.arange(1, len(cumulative_rewards_ac) + 1), cumulative_rewards_ac, linestyle='-', color='green', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Average accumulated reward')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_and_compute_average_accumulated_rewards(rewards_across_runs):
    # Find the maximum number of episodes among all runs to standardize the length of reward arrays
    max_length = max(len(run) for run in rewards_across_runs)
    # Pad shorter runs with NaNs to make all reward arrays the same length
    padded_rewards_array = np.array(
        [np.pad(np.array(run, dtype=float), (0, max_length - len(run)), 'constant', constant_values=np.nan) for run in
         rewards_across_runs])
    # Compute the mean of the rewards across runs, ignoring NaN values
    averaged_rewards = np.nanmean(padded_rewards_array, axis=0)
    # Determine indices where averaged rewards are not NaN (valid data points)
    valid_indices = ~np.isnan(averaged_rewards)
    # Compute the cumulative average of valid averaged rewards
    cumulative_average_rewards = np.cumsum(averaged_rewards[valid_indices]) / np.arange(1, np.sum(valid_indices) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(cumulative_average_rewards) + 1), cumulative_average_rewards, linestyle='-', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Average accumulated reward')
    plt.grid(True)
    plt.show()

    return cumulative_average_rewards


q_learning_averaged_rewards = run_Q_learning()
sarsa_learning_averaged_rewards = run_SARSA_learning()
ac_learning_averaged_rewards = run_AC_learning()
plot_comparison_of_algorithms(q_learning_averaged_rewards, sarsa_learning_averaged_rewards, ac_learning_averaged_rewards)
