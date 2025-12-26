import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Define the actions where each action corresponds to activating a gene or no action.
actions = [
    np.array([[0], [0], [0], [0]]),  # a1 No action
    np.array([[0], [1], [0], [0]]),  # a2 Activate p53
    np.array([[0], [0], [1], [0]]),  # a3 Activate Wip1
    np.array([[0], [0], [0], [1]])  # a4 Activate MDM2
]

# Generate all possible states representing ON/OFF status of the genes.
# The states are binary indicating the status of each gene (0 for OFF, 1 for ON).
states = [np.array(list(map(int, f"{i:04b}"))).reshape(4, 1) for i in range(16)]

# Define the connectivity matrix which represents the relationships between genes.
C = np.array([
    [0, 0, -1, 0],
    [1, 0, -1, -1],
    [0, 1, 0, 0],
    [-1, 1, 1, 0]
])


def apply_system_dynamics(C, state, action):
    # Multiply the connectivity matrix C with the current state
    connected_state = np.dot(C, state)
    # Convert to binary: 1 if the effect is positive, 0 if zero or negative
    connected_state_mapped = np.where(connected_state > 0, 1, 0)
    # Combine the connected state and the action using XOR
    next_state = np.logical_xor(connected_state_mapped, action).astype(int)
    return next_state


def apply_system_dynamics_with_noise(C, state, action, noise):
    # Apply the original dynamics without noise
    next_state_without_noise = apply_system_dynamics(C, state, action)
    # Apply noise using XOR, ensuring it's converted to binary (0 or 1)
    next_state = np.logical_xor(next_state_without_noise, noise).astype(int)
    return next_state


def compute_reward(s, action, s_prime):
    # Calculate the reward from the state s_prime
    reward_from_state = 5 * (s_prime[0] + s_prime[1] + s_prime[2] + s_prime[3])
    # Define the cost of each action
    action_costs = [0, 1, 1, 0]  # Costs for a1, a2, a3, a4
    # Determine the index of the action in the actions list
    action_index = next((i for i, act in enumerate(actions) if np.array_equal(action, act)), None)
    # Get the cost of the action
    action_cost = action_costs[action_index]
    # Calculate total reward
    total_reward = reward_from_state - action_cost
    return total_reward


def softmax(x):
    # Subtract the maximum value in x from all elements of x.
    # Normalization improves numerical stability by preventing large exponent values.
    e_x = np.exp(x - np.max(x))
    # Divide each element of e_x by the sum of all elements in e_x to normalize
    return e_x / e_x.sum()


def get_state_index(state_vector):
    # Convert the state vector to a string of binary numbers.
    binary_string = ''.join(map(str, state_vector.flatten()))
    # Convert the binary string to an integer using base 2.
    return int(binary_string, 2)


def q_learning(p, gamma, alpha, epsilon, total_episodes, max_steps, num_runs, states, actions, C):
    # Initialize an array of zero Q-tables, one for each run
    Q_tables = [np.zeros((len(states), len(actions))) for _ in range(num_runs)]
    # List to hold the optimal policies determined in each run
    optimal_policies = []
    # List to store rewards from all episodes across all runs
    all_runs_rewards = []
    # Iterate over each run
    for run in range(num_runs):
        Q_table = Q_tables[run]
        rewards_in_episodes = []
        # Iterate over each episode
        for episode in range(total_episodes):
            episode_total_reward = 0
            episode_step_count = 0
            # Randomly select an initial state index
            state_index = random.randint(0, len(states) - 1)
            # Iterate over each step in the episode
            for step in range(max_steps):
                current_state = states[state_index]
                # Epsilon-greedy policy for action selection
                if random.random() < epsilon:
                    # Choose any action except the last action (e.g., a4) randomly
                    action_index = random.randint(0, len(actions) - 2)
                else:
                    # Choose the best action based on Q values, excluding the last action
                    action_index = np.argmax(Q_table[state_index, :-1])
                action = actions[action_index]
                # Compute the next state with noise
                noise = np.random.binomial(1, p, size=(4, 1))
                next_state = apply_system_dynamics_with_noise(C, current_state, action, noise)
                next_state_index = get_state_index(next_state)
                reward = compute_reward(current_state, action, next_state)
                episode_total_reward += reward
                episode_step_count += 1
                # Q-learning update rule
                best_future_q = np.max(Q_table[next_state_index, :-1])  # Only consider a1, a2, a3 for future Q
                Q_table[state_index, action_index] += alpha * (
                        reward + gamma * best_future_q - Q_table[state_index, action_index])
                state_index = next_state_index
            average_reward = episode_total_reward / episode_step_count
            rewards_in_episodes.append(average_reward)
        all_runs_rewards.append(rewards_in_episodes)
        # Compute the optimal policy for this run
        optimal_policy = np.argmax(Q_table[:, :-1], axis=1)
        optimal_policies.append(optimal_policy)
    # Return the array of optimal policies and the rewards from all runs
    return np.array(optimal_policies), np.array(all_runs_rewards)


def sarsa(p, gamma, alpha, epsilon, total_episodes, max_steps, num_runs, states, actions, C):
    # Initialize Q-tables for each run, with dimensions states x actions
    Q_tables = [np.zeros((len(states), len(actions))) for _ in range(num_runs)]
    # List to store the optimal policies from each run
    optimal_policies = []
    # List to track rewards from all episodes across all runs
    all_runs_rewards = []
    # Iterate over each run
    for run in range(num_runs):
        Q_table = Q_tables[run]
        rewards_in_episodes = []
        # Iterate over each episode
        for episode in range(total_episodes):
            episode_total_reward = 0
            episode_step_count = 0
            # Randomly select an initial state index
            state_index = random.randint(0, len(states) - 1)
            # Select initial action using epsilon-greedy method
            if random.random() < epsilon:
                action_index = random.randint(0, len(actions) - 2)
            else:
                action_index = np.argmax(Q_table[state_index, :-1])
            current_state = states[state_index]
            action = actions[action_index]
            # Iterate over each step within the episode
            for step in range(max_steps):
                # Compute the next state with noise
                noise = np.random.binomial(1, p, size=(4, 1))
                next_state = apply_system_dynamics_with_noise(C, current_state, action, noise)
                next_state_index = get_state_index(next_state)
                # Select next action using epsilon-greedy method
                if random.random() < epsilon:
                    next_action_index = random.randint(0, len(actions) - 2)
                else:
                    next_action_index = np.argmax(Q_table[next_state_index, :-1])
                next_action = actions[next_action_index]
                # Calculate the reward for the current action
                reward = compute_reward(current_state, action, next_state)
                episode_total_reward += reward
                episode_step_count += 1
                # Retrieve the Q-value of the next state-action pair
                Q_next = Q_table[next_state_index, next_action_index]
                # SARSA update rule: Update current state-action pair
                Q_table[state_index, action_index] += alpha * (
                        reward + gamma * Q_next - Q_table[state_index, action_index])
                # Transition to the next state and action for the next iteration
                current_state = next_state
                state_index = next_state_index
                action = next_action
                action_index = next_action_index
            # Calculate and store the average reward per step for this episode
            average_reward = episode_total_reward / episode_step_count
            rewards_in_episodes.append(average_reward)
        all_runs_rewards.append(rewards_in_episodes)
        # Compute the optimal policy for each state, excluding the last action
        optimal_policy = np.argmax(Q_table[:, :-1], axis=1)
        optimal_policies.append(optimal_policy)
    # Return arrays containing optimal policies and rewards across all runs
    return np.array(optimal_policies), np.array(all_runs_rewards)


def sarsa_lambda(p, gamma, alpha, epsilon, lambda_, total_episodes, max_steps, num_runs, states, actions, C):
    # Initialize Q-tables and eligibility traces for each run
    Q_tables = [np.zeros((len(states), len(actions))) for _ in range(num_runs)]
    E_tables = [np.zeros((len(states), len(actions))) for _ in range(num_runs)]
    # List to store the optimal policies from each run
    optimal_policies = []
    # List to track rewards from all episodes across all runs
    all_runs_rewards = []
    # Iterate over each run
    for run in range(num_runs):
        Q_table = Q_tables[run]
        E_table = E_tables[run]
        rewards_in_episodes = []
        # Iterate over each episode
        for episode in range(total_episodes):
            # Reset eligibility traces for the new episode
            E_table.fill(0)
            episode_total_reward = 0
            episode_step_count = 0
            # Randomly select an initial state index
            state_index = random.randint(0, len(states) - 1)
            # Select initial action using epsilon-greedy method
            if random.random() < epsilon:
                action_index = random.randint(0, len(actions) - 2)
            else:
                action_index = np.argmax(Q_table[state_index, :-1])
            current_state = states[state_index]
            action = actions[action_index]
            # Iterate over each step within the episode
            for step in range(max_steps):
                # Compute the next state with noise
                noise = np.random.binomial(1, p, size=(4, 1))
                next_state = apply_system_dynamics_with_noise(C, current_state, action, noise)
                next_state_index = get_state_index(next_state)
                # Select next action using epsilon-greedy method
                if random.random() < epsilon:
                    next_action_index = random.randint(0, len(actions) - 2)
                else:
                    next_action_index = np.argmax(Q_table[next_state_index, :-1])
                next_action = actions[next_action_index]
                # Calculate the reward for the current action
                reward = compute_reward(current_state, action, next_state)
                episode_total_reward += reward
                episode_step_count += 1
                # Calculate the TD error
                Q_next = Q_table[next_state_index, next_action_index]
                delta = reward + gamma * Q_next - Q_table[state_index, action_index]
                # Increase eligibility trace for the current state-action pair
                E_table[state_index, action_index] += 1
                # Update Q-values and decay eligibility traces for all state-action pairs
                for s in range(len(states)):
                    for a in range(len(actions)):
                        Q_table[s, a] += alpha * delta * E_table[s, a]
                        E_table[s, a] *= gamma * lambda_  # Apply decay to the eligibility trace
                # Transition to the next state and action for the next iteration
                current_state = next_state
                state_index = next_state_index
                action = next_action
                action_index = next_action_index
            # Calculate and store the average reward per step for this episode
            average_reward = episode_total_reward / episode_step_count
            rewards_in_episodes.append(average_reward)
        all_runs_rewards.append(rewards_in_episodes)
        # Compute the optimal policy for each state, excluding the last action
        optimal_policy = np.argmax(Q_table[:, :-1], axis=1)
        optimal_policies.append(optimal_policy)
    # Return arrays containing optimal policies and rewards across all runs
    return np.array(optimal_policies), np.array(all_runs_rewards)


def actor_critic(p, gamma, alpha, beta, total_episodes, max_steps, num_runs, states, actions, C):
    # Initialize the number of states and actions (excluding the last action a4)
    num_states = len(states)
    num_actions = len(actions) - 1
    # Initialize policy parameters (theta) and value function parameters (w) for each run
    policy_params = [np.random.rand(num_states, num_actions) for _ in range(num_runs)]
    value_params = [np.random.rand(num_states) for _ in range(num_runs)]
    # List to store the optimal policies from each run
    optimal_policies = []
    # List to track rewards from all episodes across all runs
    all_runs_rewards = []
    for run in range(num_runs):
        theta = policy_params[run]
        w = value_params[run]
        rewards_in_episodes = []
        # Iterate over each episode
        for episode in range(total_episodes):
            episode_total_reward = 0
            episode_step_count = 0
            # Randomly select an initial state index
            state_index = random.randint(0, num_states - 1)
            # Iterate over each step within the episode
            for step in range(max_steps):
                current_state = states[state_index]
                # Calculate action probabilities using the softmax function
                probs = softmax(theta[state_index])
                # Choose an action based on the probability distribution
                action_index = np.random.choice(num_actions, p=probs)
                action = actions[action_index]
                # Simulate the environment's response
                noise = np.random.binomial(1, p, size=(4, 1))
                next_state = apply_system_dynamics_with_noise(C, current_state, action, noise)
                next_state_index = get_state_index(next_state)
                # Calculate the reward from taking the action
                reward = compute_reward(current_state, action, next_state)
                episode_total_reward += reward
                episode_step_count += 1
                # Critic update: calculate TD error and update value parameters
                V_current = w[state_index]
                V_next = w[next_state_index]
                td_error = reward + gamma * V_next - V_current
                w[state_index] += alpha * td_error
                # Actor update: adjust policy parameters using the TD error
                for a in range(num_actions):
                    if a == action_index:
                        # Increase the probability for the chosen action
                        theta[state_index, a] += beta * td_error * (1 - probs[a])
                    else:
                        # Decrease the probability for the other actions
                        theta[state_index, a] -= beta * td_error * probs[a]
                # Move to the next state
                state_index = next_state_index
            # Compute the average reward per step for this episode
            average_reward = episode_total_reward / episode_step_count
            rewards_in_episodes.append(average_reward)
        all_runs_rewards.append(rewards_in_episodes)
        # Compute the optimal policy for each state
        optimal_policy = np.argmax(theta, axis=1)
        optimal_policies.append(optimal_policy)
    # Return arrays containing optimal policies and rewards across all runs
    return np.array(optimal_policies), np.array(all_runs_rewards)


def map_actions_to_labels(optimal_policies):
    # Dictionary to map numeric actions to labels
    # Adjust the keys to be 0-based since np.argmax returns 0-based indices
    action_labels = {0: 'a1', 1: 'a2', 2: 'a3'}
    # Initialize an empty list to hold the labeled policies
    labeled_policies = []
    # Iterate over each policy set in the optimal policies
    for policy in optimal_policies:
        # Map each action in the policy using the dictionary
        labeled_policy = [action_labels[action] for action in policy]
        labeled_policies.append(labeled_policy)
    return labeled_policies


def plot_and_compute_average_accumulated_rewards(rewards_across_runs):
    # Find the maximum number of episodes among all runs to standardize the length of reward arrays
    max_length = max(len(run) for run in rewards_across_runs)
    # Pad shorter runs with NaNs to make all reward arrays the same length
    padded_rewards_array = np.array([np.pad(np.array(run, dtype=float), (0, max_length - len(run)), 'constant', constant_values=np.nan) for run in rewards_across_runs])
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


def plot_comparison_of_algorithms(cumulative_rewards_q, cumulative_rewards_s, cumulative_rewards_sl, cumulative_rewards_ac):
    cumulative_rewards_q = np.insert(cumulative_rewards_q, 0, 0)
    cumulative_rewards_s = np.insert(cumulative_rewards_s, 0, 0)
    cumulative_rewards_sl = np.insert(cumulative_rewards_sl, 0, 0)
    cumulative_rewards_ac = np.insert(cumulative_rewards_ac, 0, 0)

    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(1, len(cumulative_rewards_q) + 1), cumulative_rewards_q, linestyle='-', color='blue', linewidth=2, label='Q-learning')
    plt.plot(np.arange(1, len(cumulative_rewards_s) + 1), cumulative_rewards_s, linestyle='-', color='red', linewidth=2, label='SARSA')
    plt.plot(np.arange(1, len(cumulative_rewards_sl) + 1), cumulative_rewards_sl, linestyle='-', color='orange', linewidth=2, label='SARSA-Lambda')
    plt.plot(np.arange(1, len(cumulative_rewards_ac) + 1), cumulative_rewards_ac, linestyle='-', color='green', linewidth=2, label='Actor-Critic')
    plt.xlabel('Episode')
    plt.ylabel('Average accumulated reward')
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.show()


q_optimal_policies, rewards_q_learning = q_learning(0.05, 0.95, 0.2, 0.1, 500, 500, 10, states, actions, C)
q_learning_policy = map_actions_to_labels(q_optimal_policies)
print(q_learning_policy)
cumulative_rewards_q_learning = plot_and_compute_average_accumulated_rewards(rewards_q_learning)

sarsa_optimal_policies, rewards_sarsa_learning = sarsa(0.05, 0.95, 0.2, 0.1, 500, 500, 10, states, actions, C)
sarsa_learning_policy = map_actions_to_labels(sarsa_optimal_policies)
print(sarsa_learning_policy)
cumulative_rewards_sarsa_learning = plot_and_compute_average_accumulated_rewards(rewards_sarsa_learning)

sarsa_lambda_optimal_policies, rewards_sarsa_lambda_learning = sarsa_lambda(0.05, 0.95, 0.2, 0.1, 0.95, 500, 500, 10,
                                                                            states, actions, C)
sarsa_lambda_learning_policy = map_actions_to_labels(sarsa_lambda_optimal_policies)
print(sarsa_lambda_learning_policy)
cumulative_rewards_sarsa_lambda_learning = plot_and_compute_average_accumulated_rewards(rewards_sarsa_lambda_learning)

ac_policies, rewards_actor_critic = actor_critic(0.05, 0.95, 0.2, 0.05, 500, 500, 10, states, actions, C)
ac_policy = map_actions_to_labels(ac_policies)
print(ac_policy)
cumulative_rewards_ac_learning = plot_and_compute_average_accumulated_rewards(rewards_actor_critic)

plot_comparison_of_algorithms(cumulative_rewards_q_learning, cumulative_rewards_sarsa_learning, cumulative_rewards_sarsa_lambda_learning, cumulative_rewards_ac_learning)
