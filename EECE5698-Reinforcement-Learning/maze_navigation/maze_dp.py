import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Maze:
    def __init__(self, rows, cols, actions, p, start_position, goal_position, oil_positions, bump_positions,
                 wall_positions, oil_reward, bump_reward, goal_reward, empty_reward, action_reward):
        self.gamma = None
        self.optimal_path = None # Store the optimal path from start to goal
        self.rows, self.cols = rows, cols # Dimensions of the maze
        self.actions = actions # Possible actions the agent can take
        self.p = p # Probability affecting stochastic movement
        self.start_position = start_position # Starting position of the agent
        self.goal_position = goal_position # Goal position
        # Special state positions
        self.oil_positions = set(oil_positions)
        self.bump_positions = set(bump_positions)
        self.wall_positions = set(wall_positions)
        # Mapping of state types to rewards
        self.rewards = {'oil': oil_reward, 'bump': bump_reward, 'goal': goal_reward,
                        'empty': empty_reward, 'action': action_reward}
        # Initialize the maze with zeros for empty spaces
        self.maze = np.zeros((rows, cols))
        # Values array for storing the value of each state
        self.values = np.zeros((rows, cols))
        # Policy initialized to 'L' (left) action
        self.policy = np.full((rows, cols), 'L', dtype='<U1')
        # Populate maze with special states and walls
        self.initialize_rewards()

    def initialize_rewards(self):
        # Assign oil rewards to oil positions
        if self.oil_positions:
            oil_rows, oil_cols = zip(*self.oil_positions)
            self.maze[oil_rows, oil_cols] = self.rewards['oil']
        # Assign bump rewards to bump positions
        if self.bump_positions:
            bump_rows, bump_cols = zip(*self.bump_positions)
            self.maze[bump_rows, bump_cols] = self.rewards['bump']
        # Set the goal position reward
        if self.goal_position:
            self.maze[self.goal_position] = self.rewards['goal']
        # Mark wall positions as non-navigable spaces
        if self.wall_positions:
            for pos in self.wall_positions:
                self.maze[pos] = np.nan

    def coloring_blocks(self, heatmap):
        # Visualize wall positions on the heatmap
        for state in self.wall_positions:
            heatmap.add_patch(
                Rectangle((state[1], state[0]), 1, 1, fill=True, facecolor='black', edgecolor='black', lw=1))
        # Visualize oil positions on the heatmap
        for oil_state in self.oil_positions:
            heatmap.add_patch(
                Rectangle((oil_state[1], oil_state[0]), 1, 1, fill=True, facecolor='red', edgecolor='black', lw=1))
        # Visualize bump positions on the heatmap
        for bump_state in self.bump_positions:
            heatmap.add_patch(
                Rectangle((bump_state[1], bump_state[0]), 1, 1, fill=True, facecolor='bisque', edgecolor='black', lw=1))
        # Visualize the start position on the heatmap
        heatmap.add_patch(
            Rectangle((self.start_position[1], self.start_position[0]), 1, 1, fill=True, facecolor='dodgerblue',
                      edgecolor='black', lw=1))
        # Visualize the goal position on the heatmap
        heatmap.add_patch(
            Rectangle((self.goal_position[1], self.goal_position[0]), 1, 1, fill=True, facecolor='yellowgreen',
                      edgecolor='black', lw=1))

    def compute_reward(self, x1, y1, x2, y2):
        # Check if the new position is within the maze bounds and not a wall
        if not self.bound_check(x2, y2) or self.wall_check(x2, y2):
            # If out of bounds or a wall, the action leads nowhere; return the action cost and stay in place
            return self.rewards['action'], x1, y1
        # Start with the base cost of taking any action
        reward = self.rewards['action']
        # Check if the new position is a bump and add the bump penalty to the reward
        if (x2, y2) in self.bump_positions:
            reward += self.rewards['bump']
        # Check if the new position is an oil spot and add the oil penalty to the reward
        elif (x2, y2) in self.oil_positions:
            reward += self.rewards['oil']
        # Check if the new position is the goal and add the goal reward
        elif (x2, y2) == self.goal_position:
            reward += self.rewards['goal']
        # Return the total reward for moving to the new position, and the new position itself
        return reward, x2, y2

    def wall_check(self, x, y):
        # Check if the given position (x, y) is a wall within the maze
        # Returns True if the position is a wall, False otherwise
        return (x, y) in self.wall_positions

    def bound_check(self, x, y):
        # Check if the given position (x, y) is within the bounds of the maze
        # The position must be within the maze dimensions from (0, 0) to (rows-1, cols-1)
        # Returns True if the position is within bounds, False otherwise
        return 0 <= x < self.rows and 0 <= y < self.cols

    def compute_expected_utility(self, x, y, action):
        # Define the possible movements in the maze: left, right, up
        directions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
        # Calculate the probability of moving to any of the other three directions (not chosen action)
        other_p = self.p / 3
        # Initialize the total expected value of choosing an action at state (x, y)
        total_value = 0
        # Iterate through all possible actions and their grid movements
        for move_action, (dx, dy) in directions.items():
            # Calculate the next state's coordinates based on the action
            next_x, next_y = x + dx, y + dy
            # If the current iteration's action is the chosen action, use 1-p else use p/3
            p = 1 - self.p if move_action == action else other_p
            # If moving to the next state is not possible due to bounds or walls, stay in the current state
            if not self.bound_check(next_x, next_y) or self.wall_check(next_x, next_y):
                next_x, next_y = x, y
            # Calculate the reward for moving to the next state or staying if movement isn't possible
            reward, _, _ = self.compute_reward(x, y, next_x, next_y)
            # Update the total expected value
            total_value += p * (reward + self.gamma * self.values[next_x, next_y])

        return total_value

    def reset_state_values(self):
        # Reset the values to zero for all states in the maze
        self.values = np.zeros((self.rows, self.cols))

    def policy_evaluation(self, theta=0.01):
        # Loop until the value function changes are smaller than the threshold theta
        while True:
            delta = 0  # Initialize the maximum change
            # Iterate over all states in the maze that are not walls
            for i, j in self.navigable_states():
                if self.wall_check(i, j):
                    continue  # Skip evaluation for wall states
                old_value = self.values[i, j]  # Store the current value for comparison after update
                # Update the state's value based on the expected returns of following the current policy
                action = self.policy[i, j]  # Get the current policy's action for this state
                self.values[i, j] = self.compute_expected_utility(i, j, action)  # Calculate expected return
                # Update delta with the largest change in value function across all states
                delta = max(delta, np.abs(old_value - self.values[i, j]))
            # If the largest change in the value function is below the threshold, stop the loop
            if delta < theta:
                break

    def policy_improvement(self):
        # Initialize a flag to keep track if policy has stabilized
        policy_stable = True
        # Iterate over all states in the maze that are not walls
        for i, j in self.navigable_states():
            # Skip the iteration if the current state is a wall
            if self.wall_check(i, j):
                continue
            # Store the current action for the state to compare later
            old_action = self.policy[i, j]
            # Initialize variables to keep track of the best action and its value found so far
            best_value = float('-inf')
            best_action = None
            # Evaluate each possible action from the current state
            for action in self.actions:
                # Calculate the expected value of taking this action from the current state
                action_value = self.compute_expected_utility(i, j, action)
                # Update the best action if this action's value is higher than the current best
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            # Update the policy with the best action found
            self.policy[i, j] = best_action
            # If the best action is different from the old action, the policy has not stabilized
            if old_action != best_action:
                policy_stable = False
        # Return whether the policy has stabilized
        return policy_stable

    def policy_iteration(self, gamma, theta=0.01):
        # Update the discount factor gamma for the current iteration
        self.gamma = gamma
        # Initialize a counter to track the number of policy improvement iterations
        policy_improvement_iterations = 0
        while True:
            # Evaluate the current policy to update state values
            self.policy_evaluation(theta)
            # Attempt to improve the policy based on the updated state values
            if self.policy_improvement():
                # If the policy is stable (no changes were made), exit the loop
                break
            # Increment the counter if the policy was improved (not yet stable)
            policy_improvement_iterations += 1
        # Once the policy is stable, compute the optimal path based on the stable policy
        self.optimal_path = self.compute_optimal_path()
        # Print the number of iterations it took for the policy to stabilize
        print(f"Policy Iteration completed in {policy_improvement_iterations} iterations.")

    def value_iteration(self, gamma, theta=0.01):
        # Reset the state values to zero before starting value iteration
        self.reset_state_values()
        # Set the discount factor for the value iteration
        self.gamma = gamma
        # Initialize a counter to track the number of iterations
        value_iteration_iterations = 0

        while True:
            # Initialize a variable to track the maximum change in value for any state in this iteration
            delta = 0
            # Iterate over all states using a generator that excludes wall states
            for i, j in self.navigable_states():
                # Store the current value of the state
                old_value = self.values[i, j]
                # Initialize the maximum value found for this state as negative infinity
                max_value = float('-inf')
                # Evaluate each possible action from this state
                for action in self.actions:
                    # Calculate the expected value of taking this action from the current state
                    action_value = self.compute_expected_utility(i, j, action)
                    # Update max_value if this action's value is the highest found so far
                    max_value = max(max_value, action_value)
                # Update the state's value with the highest value found among all actions
                self.values[i, j] = max_value
                # Update delta if the change in value for this state is the largest seen in this iteration
                delta = max(delta, np.abs(old_value - self.values[i, j]))
            # Increment the iteration counter after processing all states
            value_iteration_iterations += 1
            # If the maximum change in value for any state is below the threshold, stop iterating
            if delta < theta:
                break
        # Once the values have converged, perform policy improvement to find the optimal policy
        self.policy_improvement()
        # Compute the optimal path based on the stable policy
        self.optimal_path = self.compute_optimal_path()
        # Print the number of iterations it took for the value function to converge
        print(f"Value Iteration completed in {value_iteration_iterations} iterations.")

    def compute_optimal_path(self):
        # Check if the starting position is the same as the goal position
        if self.start_position == self.goal_position:
            return [(self.start_position, 'None')]  # No movement is needed if the start equals the goal

        path = []  # Initialize the path list to store tuples of position and action
        visited = set()  # Initialize a set to keep track of visited positions to avoid loops
        curr_position = self.start_position  # Start from the initial position

        while curr_position != self.goal_position:
            # Check if the current position has been visited to prevent loops
            if curr_position in visited:
                break  # Exit the loop if a repeated position is encountered
            visited.add(curr_position)  # Mark the current position as visited
            # Retrieve the action recommended by the policy for the current position
            action = self.policy[curr_position]
            # Append the current position and action to the path
            path.append((curr_position, action))
            # Determine movement based on action
            dx, dy = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}[action]
            # Calculate the next position
            next_position = (curr_position[0] + dx, curr_position[1] + dy)
            # Check if the next position is within bounds and not a wall
            if not self.bound_check(next_position[0], next_position[1]) or self.wall_check(next_position[0], next_position[1]):
                print(f"Stopped at {curr_position}, cannot move to {next_position} due to a wall or boundary.")
                break  # Stop if the next position is invalid
            # Move to the next position
            curr_position = next_position
        # Final position has no action associated ('None' action)
        path.append((curr_position, 'None'))
        # Return the computed path
        return path

    def navigable_states(self):
        # Iterate through every cell in the maze
        for x in range(self.rows):
            for y in range(self.cols):
                # Check if the current cell is not a wall, indicating navigable state
                if not self.wall_check(x, y):
                    # If it's navigable, yield its coordinates as a valid state for the agent
                    yield x, y

    def draw_optimal_path(self):
        # Check if there is an optimal path available to draw
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
                    plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color, width=arrow_width)
