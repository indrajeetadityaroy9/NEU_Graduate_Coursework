import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class MazeAC:
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
        # Mapping of actions to their corresponding changes in row and column indices in a grid
        self.ACTION_MAPPINGS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        # Creating a dictionary to map integers to actions; useful for converting indices to action keys
        self.INDEX_TO_ACTION_MAP = {i: act for i, act in enumerate(actions)}
        # Setting the default action index by finding the index of action 'L' in the actions list
        self.default_action_index = self.actions.index('L')
        # Initializing the policy array with the default action index for every cell in a grid of size rows x cols
        self.policy = np.full((rows, cols), self.default_action_index, dtype=int)
        # Calling a method to initialize rewards for the grid, assumed to be defined elsewhere in the class
        self.initialize_rewards()
        # Initialize optimal path variable; to be computed post-learning
        self.optimal_path = None

    def initialize_rewards(self):
        for pos in self.oil_positions:
            self.maze[pos] = self.rewards['oil']
        for pos in self.bump_positions:
            self.maze[pos] = self.rewards['bump']
        self.maze[self.goal_position] = self.rewards['goal']
        for pos in self.wall_positions:
            self.maze[pos] = np.nan  # Use NaN for walls to simplify checks

    def compute_reward(self, x1, y1, x2, y2):
        if not self.bound_check(x2, y2) or self.wall_check(x2, y2):
            return self.rewards['action'], x1, y1
        reward = self.rewards['action']
        if (x2, y2) in self.bump_positions:
            reward += self.rewards['bump']
        elif (x2, y2) in self.oil_positions:
            reward += self.rewards['oil']
        elif (x2, y2) == self.goal_position:
            reward += self.rewards['goal']
        return reward

    def wall_check(self, x, y):
        return (x, y) in self.wall_positions

    def bound_check(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols

    def goal_check(self, x, y):
        return (x, y) == self.goal_position

    def valid_position_check(self, position):
        x, y = position
        return self.bound_check(x, y) and not self.wall_check(x, y)

    def create_V_Matrix(self):
        V_Matrix = np.zeros((self.rows, self.cols, self.action_space_size))
        for i in range(self.rows):
            for j in range(self.cols):
                if np.isnan(self.maze[i][j]):
                    V_Matrix[i, j, :] = np.nan
        return V_Matrix

    def softmax_probability(self, H_matrix, state, temperature):
        H_state = H_matrix[state[0], state[1], :]
        H_shifted = H_state - np.max(H_state)  # Stability trick to avoid overflow
        exp_H = np.exp(H_shifted / temperature)
        policy_probs = exp_H / np.sum(exp_H)
        return policy_probs

    def actor_critic_step(self, H_matrix, state, temperature):
        policy_probs = self.softmax_probability(H_matrix, state, temperature)
        action_index = np.random.choice(self.action_space_size, p=policy_probs)
        return self.actions[action_index], action_index

    def execute_action(self, state, action_index):
        # Convert action index to the corresponding action command
        action = self.INDEX_TO_ACTION_MAP[action_index]
        dx, dy = self.ACTION_MAPPINGS[action]
        nx, ny = state[0] + dx, state[1] + dy
        if not self.valid_position_check((nx, ny)):
            return state, self.rewards['action']  # Stay in place if next state is invalid
        return (nx, ny), self.compute_reward(state[0], state[1], nx, ny)

    def select_stochastic_action(self, action_taken, state):
        limit = 1 - self.p
        decider = np.random.uniform()
        if decider < limit:
            return action_taken
        else:
            action_list = [i for i in range(self.action_space_size) if
                           i != action_taken and self.valid_action_check(i, state)]
            return np.random.choice(action_list) if action_list else action_taken

    def valid_action_check(self, action_index, state):
        action = self.INDEX_TO_ACTION_MAP[action_index]
        dx, dy = self.ACTION_MAPPINGS[action]
        next_position = (state[0] + dx, state[1] + dy)
        return self.valid_position_check(next_position)

    def actor_critic(self, beta, alpha, gamma, lambda_, episodes_limit, steps_limit):
        V_matrix = self.create_V_Matrix()  # State-action value function matrix
        H_matrix = np.zeros((self.rows, self.cols, self.action_space_size))  # Policy matrix
        rewards_in_episodes = []

        temperature = 0.1

        for episode in range(episodes_limit):
            state = self.start_position
            self.trace_dict = {}  # Reset the trace dictionary at the start of each episode

            in_episode_total = 0
            episode_step_count = 0

            for step in range(steps_limit):
                action, action_index = self.actor_critic_step(H_matrix, state, temperature)  # Exploit best action
                chosen_action_index = self.select_stochastic_action(action_index, state)
                next_state, reward = self.execute_action(state, chosen_action_index)
                self.update_policy_with_traces(H_matrix, V_matrix, state, action_index, reward, next_state, alpha, beta, gamma, lambda_)

                state = next_state  # Update state to next state
                in_episode_total += reward
                episode_step_count += 1

                temperature = max(0.1, temperature * 0.99)  # Decay temperature

                if self.goal_check(*state):
                    break

            rewards_in_episodes.append(in_episode_total / episode_step_count)

            self.update_policy_from_H(H_matrix)

        path_exists = self.path_exists()  # Ensure that this method is defined correctly
        self.optimal_path = self.compute_path()
        if path_exists:
            self.optimal_path = self.compute_path()  # Compute the optimal path if it exists
            path_success = True
        else:
            self.optimal_path = []
            path_success = False
        return V_matrix, path_success, self.optimal_path, rewards_in_episodes

    def update_policy_with_traces(self, H_matrix, V_matrix, state, action_index, reward, next_state, alpha, beta, gamma,lambda_):
        x, y = state
        nx, ny = next_state

        delta = reward + gamma * np.max(V_matrix[nx, ny]) - V_matrix[x, y, action_index]
        self.trace_dict.setdefault((x, y, action_index), (1, 1))

        for key, (e, eh) in list(self.trace_dict.items()):
            V_matrix[key[:-1]][key[-1]] += alpha * delta * e
            H_matrix[key[:-1]][key[-1]] += beta * delta * eh
            new_e = gamma * lambda_ * e
            new_eh = gamma * lambda_ * eh
            self.trace_dict[key] = (new_e, new_eh)

    def update_policy_from_H(self, H_matrix):
        left_action_index = self.actions.index('L')

        # Loop through each stored trace in trace_dict
        for (x, y, a), (e, eh) in self.trace_dict.items():
            if not self.wall_check(x, y):
                eh_factor = eh  # Use eh directly from the tuple
                best_action_index = np.argmax(H_matrix[x, y, :] * eh_factor)
                self.policy[x, y] = best_action_index  # Store the index of the best action

        # Handle states not covered by trace_dict or where the default action may not be optimal
        for x in range(self.rows):
            for y in range(self.cols):
                if self.policy[x, y] == left_action_index and not self.wall_check(x, y):
                    best_action_index = np.argmax(H_matrix[x, y, :])
                    self.policy[x, y] = best_action_index

    def path_exists(self):
        if self.start_position == self.goal_position:
            return True

        visited = set()
        queue = [self.start_position]

        while queue:
            curr_position = queue.pop(0)
            visited.add(curr_position)

            if curr_position == self.goal_position:
                return True

            action_index = self.policy[curr_position]
            action = self.INDEX_TO_ACTION_MAP[action_index]
            dx, dy = self.ACTION_MAPPINGS[action]
            next_position = (curr_position[0] + dx, curr_position[1] + dy)

            if self.valid_position_check(next_position) and next_position not in visited:
                queue.append(next_position)
        return False

    def compute_path(self):
        if self.start_position == self.goal_position:
            return [(self.start_position, 'None')]

        path = []
        curr_position = self.start_position

        while curr_position != self.goal_position and len(path) < self.rows * self.cols:
            action_index = self.policy[curr_position]  # This should retrieve the action index
            action = self.INDEX_TO_ACTION_MAP[action_index]  # Convert index to action
            dx, dy = self.ACTION_MAPPINGS[action]
            next_position = (curr_position[0] + dx, curr_position[1] + dy)

            if not self.valid_position_check(next_position):
                path.append((curr_position, 'Blocked'))
                break

            path.append((curr_position, action))
            curr_position = next_position

        if curr_position == self.goal_position:
            path.append((curr_position, 'Goal'))

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
        # Adjusting the mapping from action codes to directions using self.ACTION_MAPPINGS
        # Ensure directions are consistent with ACTION_MAPPINGS
        action_map = {
            'U': (0, -0.3),  # Up
            'R': (0.3, 0),  # Right
            'D': (0, 0.3),  # Down
            'L': (-0.3, 0)  # Left
        }
        # Iterate through each cell in the policy
        for i in range(nrows):
            for j in range(ncols):
                action_index = policy_matrix[i, j]
                action = self.INDEX_TO_ACTION_MAP[action_index]  # Convert index to action using the map
                dx, dy = action_map.get(action, (0, 0))
                # Check if there is a valid direction vector
                if dx or dy:
                    # Calculate the starting position of the arrow
                    x, y = j + 0.5, i + 0.5
                    # Draw an arrow on the plot from (x, y) in the direction specified by (dx, dy)
                    plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.2, fc=arrow_color, ec=arrow_color,
                              width=arrow_width)

        plt.show()

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
