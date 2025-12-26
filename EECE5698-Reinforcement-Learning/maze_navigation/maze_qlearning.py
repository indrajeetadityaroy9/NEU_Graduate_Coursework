import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Maze2:
    def __init__(self, rows, cols, actions, p, start_position, goal_position, oil_positions, bump_positions, wall_positions, oil_reward, bump_reward, goal_reward, action_reward):
        # Set the number of rows and columns for the grid environment
        self.rows, self.cols = rows, cols
        # Set the starting position for the agent
        self.start_position = start_position
        # Set the goal position that the agent aims to reach
        self.goal_position = goal_position
        # Convert the list of wall positions to a set for faster look-up
        self.wall_positions = set(wall_positions)
        # Convert the list of oil spill positions to a set for faster look-up
        self.oil_positions = set(oil_positions)
        # Convert the list of bump positions to a set for faster look-up
        self.bump_positions = set(bump_positions)
        # Set the list of possible actions the agent can take
        self.actions = actions
        # Determine the size of the action space
        self.action_space_size = len(actions)
        # Set the probability p for stochastic transitions
        self.p = p
        # Initialize a dictionary of rewards for various interactions in the environment
        self.rewards = {'oil': oil_reward, 'bump': bump_reward, 'goal': goal_reward, 'action': action_reward}
        # Initialize a 2D array for the maze, filled with zeros
        self.maze = np.zeros((rows, cols))
        # Initialize the default action index, set to the index of 'L' (left)
        self.policy = np.full((rows, cols), 'L', dtype='<U1')
        # Call the method to initialize rewards in the maze based on predefined positions
        self.initialize_rewards()
        # Initialize optimal path variable; to be computed post-learning
        self.optimal_path = None

    def initialize_rewards(self):
        # Set rewards for oil positions on the maze
        for pos in self.oil_positions:
            self.maze[pos] = self.rewards['oil']
        # Set rewards for bump positions on the maze
        for pos in self.bump_positions:
            self.maze[pos] = self.rewards['bump']
        # Set the reward for the goal position on the maze
        self.maze[self.goal_position] = self.rewards['goal']
        # Mark wall positions on the maze as NaN (not a number) to indicate impassable
        for pos in self.wall_positions:
            self.maze[pos] = np.nan

    def compute_reward(self, x1, y1, x2, y2):
        # Check if the new position is out of bounds or a wall
        if not self.bound_check(x2, y2) or self.wall_check(x2, y2):
            return self.rewards['action'], x1, y1
        reward = self.rewards['action']  # Start with the reward for taking an action
        # Check if the new position has a bump and adjust the reward accordingly
        if (x2, y2) in self.bump_positions:
            reward += self.rewards['bump']
        # Check if the new position has oil and adjust the reward accordingly
        elif (x2, y2) in self.oil_positions:
            reward += self.rewards['oil']
        # Check if the new position is the goal and adjust the reward accordingly
        elif (x2, y2) == self.goal_position:
            reward += self.rewards['goal']
        # Return the total reward and the new position
        return reward, x2, y2

    def wall_check(self, x, y):
        # Return True if the specified position is a wall, False otherwise
        return (x, y) in self.wall_positions

    def bound_check(self, x, y):
        # Check if the coordinates are within the grid boundaries
        return 0 <= x < self.rows and 0 <= y < self.cols

    def goal_check(self, x, y):
        # Check if the specified position is the goal
        return (x, y) == self.goal_position

    def epsilon_greedy_step(self, Q_Matrix, cx, cy, epsilon):
        qlist = Q_Matrix[cx, cy, :]
        if np.random.rand() < epsilon:
            # Exploration: if a random number is less than epsilon, choose a random action
            return np.random.randint(0, 4)
        # Exploitation: if the random number is not less than epsilon, choose the best-known action
        # Find the maximum Q-value among available actions for this state
        maxq = np.max(qlist)
        # Find indices of all actions with the maximum Q-value
        max_indices = np.where(np.isclose(qlist, maxq))[0]
        # Randomly select among the best actions to break ties fairly
        index = np.random.choice(max_indices)
        return index

    def create_Q_Matrix(self):
        # Initialize the Q matrix for the entire grid with default values
        Q_Matrix = np.zeros((self.rows, self.cols, self.action_space_size))
        for i in range(self.rows):
            for j in range(self.cols):
                if np.isnan(self.maze[i][j]):
                    # Set Q-values to NaN for states corresponding to walls
                    # This indicates that these states are not valid for choosing actions
                    Q_Matrix[i, j, :] = np.nan
        return Q_Matrix

    def select_stochastic_action(self, action_taken, state_x, state_y):
        limit = 1 - self.p  # Calculate the probability threshold for taking the intended action
        decider = np.random.uniform()  # Generate a random number between 0 and 1
        # Create a list of alternative actions, excluding the action that was originally taken
        action_list = [i for i in range(self.action_space_size) if i != action_taken]
        # Decide whether to perform the main action or select a random one
        # If the random number is below the threshold, perform the intended action
        # Otherwise, choose a random alternative action
        chosen_action = action_taken if decider < limit else np.random.choice(action_list)
        # Define a mapping from action indices to their corresponding movements on the grid
        movement_mapping = {
            0: (-1, 0),  # Up: decrease the x-coordinate
            1: (0, 1),  # Right: increase the y-coordinate
            2: (1, 0),  # Down: increase the x-coordinate
            3: (0, -1)  # Left: decrease the y-coordinate
        }
        # Get the movement values for the chosen action
        dx, dy = movement_mapping[chosen_action]
        # Calculate the new position by adding movement deltas to the current position
        nx, ny = state_x + dx, state_y + dy
        return nx, ny  # Return the new position coordinates

    def q_learning(self, epsilon, alpha, gamma, episodes_limit, steps_limit):
        # Initialize Q-values for each state and action
        Q_Matrix = self.create_Q_Matrix()
        # List to store the average reward per episode
        rewards_in_episodes = []
        for episode in range(episodes_limit):  # Loop over each episode
            # Start each episode at the starting position
            state = self.start_position
            in_episode_total = 0  # Total rewards accumulated in the episode
            episode_step_count = 0  # Count steps taken in the episode
            # Loop until the goal is reached or the step limit is exceeded
            while not self.goal_check(state[0], state[1]) and episode_step_count < steps_limit:
                cx, cy = state  # Current state coordinates
                # Select action using epsilon-greedy strategy
                action = self.epsilon_greedy_step(Q_Matrix, cx, cy, epsilon)
                # Apply the action getting a stochastic result
                nx, ny = self.select_stochastic_action(action, cx, cy)
                # Calculate reward and final state after action
                reward, fx, fy = self.compute_reward(cx, cy, nx, ny)
                # Update total rewards for this episode
                in_episode_total += reward
                # Update Q-value using the Q-learning formula
                # Maximum Q-value for the next state
                max_q = np.max(Q_Matrix[fx, fy, :])
                Q_Matrix[cx, cy, action] += alpha * (reward + gamma * max_q - Q_Matrix[cx, cy, action])
                # Move to the next state
                state = (fx, fy)
                episode_step_count += 1  # Increment step count
            # Calculate and store the average reward per step in this episode
            rewards_in_episodes.append(in_episode_total / episode_step_count)
        # After all episodes, update the policy based on the learned Q-values
        self.update_policy_from_Q(Q_Matrix)
        # Check if a valid path to the goal exists based on the updated policy
        if self.path_exists():
            self.optimal_path = self.compute_path()  # Compute the optimal path if it exists
            path_found = True
        else:
            self.optimal_path = []  # If no path is found, set the optimal path to an empty list
            path_found = False
        return Q_Matrix, path_found, rewards_in_episodes

    def sarsa_learning(self, epsilon, alpha, gamma, episodes_limit, steps_limit):
        # Initialize the Q matrix with zeros for each state-action pair
        Q_Matrix = self.create_Q_Matrix()
        # List to keep track of rewards in each episode
        rewards_in_episodes = []
        # Loop through each episode
        for episode in range(episodes_limit):
            # Start at the beginning position for each episode
            state = self.start_position
            # Total reward accumulated in the episode
            in_episode_total = 0
            # Count of steps taken in the episode
            episode_step_count = 0
            # Execute until goal is reached or step limit is exceeded
            while not self.goal_check(state[0], state[1]) and episode_step_count < steps_limit:
                # Current state coordinates
                cx, cy = state
                # Select an action using epsilon-greedy policy
                action = self.epsilon_greedy_step(Q_Matrix, cx, cy, epsilon)
                # Execute the action and get the new state
                nx, ny = self.select_stochastic_action(action, cx, cy)
                # Compute the reward and final state after action
                reward, fx, fy = self.compute_reward(cx, cy, nx, ny)
                # Accumulate the reward
                in_episode_total += reward
                # If the future state is not the goal
                if not self.goal_check(fx, fy):
                    # Select the next action from the future state
                    next_action = self.epsilon_greedy_step(Q_Matrix, fx, fy, epsilon)
                    # Get the Q-value of the next action at the future state
                    future_q = Q_Matrix[fx, fy, next_action]
                else:
                    # If the future state is the goal, future Q-value is zero (no further rewards)
                    future_q = 0
                    # SARSA Update Formula: Update the Q-value of the current state-action pair
                Q_Matrix[cx, cy, action] += alpha * (reward + gamma * future_q - Q_Matrix[cx, cy, action])
                # Update the state to the next state
                state = (fx, fy)
                # Increment the step counter
                episode_step_count += 1
                # Store the average reward per step for the episode
            rewards_in_episodes.append(in_episode_total / episode_step_count)
            # Update the policy based on the learned Q-values after all episodes are complete
        self.update_policy_from_Q(Q_Matrix)
        # Calculate the optimal path using the updated policy
        if self.path_exists():
            self.optimal_path = self.compute_path()
            path_found = True
        else:
            # Set the optimal path to an empty list if no path is found
            self.optimal_path = []
            path_found = False
        return Q_Matrix, path_found, rewards_in_episodes

    def update_policy_from_Q(self, Q):
        # Mapping from action indices to letters for easier readability
        action_letters = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
        # Iterate through all rows
        for x in range(self.rows):
            # Iterate through all columns
            for y in range(self.cols):
                # Check if the current position is not a wall
                if not self.wall_check(x, y):
                    # Find the index of the maximum Q-value in this state
                    best_action_index = np.argmax(Q[x, y])
                    # Update the policy at position (x, y) with the best action letter
                    self.policy[x, y] = action_letters[best_action_index]

    def path_exists(self):
        # Define movements associated with each action
        action_movements = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
        # Initialize queue with the start position for breadth-first search (BFS)
        queue = [self.start_position]
        # Set to keep track of visited positions to prevent re-processing
        visited = set()
        while queue:
            # Pop the first element from the queue
            curr_position = queue.pop(0)
            # Check if the current position is the goal
            if curr_position == self.goal_position:
                # Return True if the goal is reached
                return True
            # Skip processing if the position has already been visited
            if curr_position in visited:
                continue
            # Mark the current position as visited
            visited.add(curr_position)
            # Iterate through possible actions
            for action in action_movements:
                # Get movement offsets for the action
                dx, dy = action_movements[action]
                # Calculate the next position
                next_position = (curr_position[0] + dx, curr_position[1] + dy)
                # Add the next position to the queue if it's within bounds and not a wall
                if self.bound_check(*next_position) and not self.wall_check(*next_position):
                    queue.append(next_position)
        return False

    def compute_path(self):
        # Return immediately if the start and goal positions are the same
        if self.start_position == self.goal_position:
            return [(self.start_position, 'None')]
        # Define movements for each action
        action_movements = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
        # Initialize the list to hold the path tuples (position, action)
        path = []
        # Start at the initial position
        curr_position = self.start_position
        # Set a limit on iterations to prevent infinite loops
        max_iterations = self.rows * self.cols
        # Counter to track the number of iterations
        iteration = 0
        while curr_position != self.goal_position:
            # Check if the iteration limit is reached
            if iteration >= max_iterations:
                # Exit the loop if too many iterations without reaching the goal
                break
            # Increment the iteration counter
            iteration += 1
            # Retrieve the action from the policy for the current position
            action = self.policy[curr_position[0], curr_position[1]]
            # Get the movement delta for the action
            dx, dy = action_movements[action]
            # Calculate the next position
            next_position = (curr_position[0] + dx, curr_position[1] + dy)
            # Check if the next position is within bounds and not a wall
            if self.bound_check(*next_position) and not self.wall_check(*next_position):
                # Append the current position and action to the path
                path.append((curr_position, action))
                # Update the current position to the next position
                curr_position = next_position
        # Append the final position with a 'None' action since it's the goal
        path.append((curr_position, 'None'))
        # Return the complete path from start to goal
        return path

    def draw_optimal_path(self):
        if not self.optimal_path or len(self.optimal_path) <= 1:
            print("No optimal path.")
            return
        # Iterate over the optimal path, excluding the last position since it's the goal with no action
        for i, (pos, action) in enumerate(self.optimal_path[:-1]):
            r, c = pos
            # Draw an arrow for the action to be taken at the current position
            # The direction and length of the arrow are determined by the action
            if action == 'R':  # If the action is Right
                plt.arrow(c + 0.5, r + 0.5, 0.8, 0, width=0.04, color='black')  # Draw arrow to the right
            elif action == 'L':  # If the action is Left
                plt.arrow(c + 0.5, r + 0.5, -0.8, 0, width=0.04, color='black')  # Draw arrow to the left
            elif action == 'U':  # If the action is Up
                plt.arrow(c + 0.5, r + 0.5, 0, -0.8, width=0.04, color='black')  # Draw arrow upwards
            elif action == 'D':  # If the action is Down
                plt.arrow(c + 0.5, r + 0.5, 0, 0.8, width=0.04, color='black')  # Draw arrow downwards
        plt.show()

    def draw_policy_arrows(self):
        # Convert the policy into a 2D array matching the maze's layout
        policy_matrix = np.array(self.policy).reshape((self.rows, self.cols))
        # Extract the number of rows and columns from the policy matrix
        nrows, ncols = policy_matrix.shape
        # Define the color of the arrows
        arrow_color = 'black'
        # Define the width of the arrows
        arrow_width = 0.04
        # Iterate through each cell in the policy
        for i in range(nrows):
            for j in range(ncols):
                action = policy_matrix[i, j]
                # Map the action to a direction (dx, dy) for drawing arrows
                dx, dy = {'R': (0.3, 0), 'L': (-0.3, 0), 'U': (0, -0.3), 'D': (0, 0.3)}.get(action, (0, 0))
                # If there is a valid direction vector (dx, dy is not (0,0)), draw an arrow
                if dx or dy:
                    # Calculate the starting position of the arrow
                    x, y = j + 0.5, i + 0.5
                    # Draw an arrow on the plot from (x,y) in the direction specified by (dx, dy)
                    plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color,
                              width=arrow_width)

    def coloring_blocks(self, heatmap):
        for state in self.wall_positions:
            heatmap.add_patch(
                Rectangle((state[1], state[0]), 1, 1, fill=True, facecolor='black', edgecolor='black', lw=1))
        for oil_state in self.oil_positions:
            heatmap.add_patch(
                Rectangle((oil_state[1], oil_state[0]), 1, 1, fill=True, facecolor='red', edgecolor='black', lw=1))
        for bump_state in self.bump_positions:
            heatmap.add_patch(
                Rectangle((bump_state[1], bump_state[0]), 1, 1, fill=True, facecolor='bisque', edgecolor='black', lw=1))
        heatmap.add_patch(
            Rectangle((self.start_position[1], self.start_position[0]), 1, 1, fill=True, facecolor='dodgerblue',
                      edgecolor='black', lw=1))
        heatmap.add_patch(
            Rectangle((self.goal_position[1], self.goal_position[0]), 1, 1, fill=True, facecolor='yellowgreen',
                      edgecolor='black', lw=1))