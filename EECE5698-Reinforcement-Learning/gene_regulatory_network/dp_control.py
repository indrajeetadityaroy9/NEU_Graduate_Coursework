import numpy as np
import random
import math

# Define the actions where each action corresponds to activating a gene or no action.
actions = [
    np.array([[0], [0], [0], [0]]),  # No action
    np.array([[1], [0], [0], [0]]),  # Activate ATM
    np.array([[0], [1], [0], [0]]),  # Activate p53
    np.array([[0], [0], [1], [0]]),  # Activate Wip1
    np.array([[0], [0], [0], [1]])  # Activate MDM2
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
    # Convert to binary: 1 if the effect is positive (activation and 0 if not
    connected_state = np.where(connected_state > 0, 1, 0)
    # Apply the action to the connected state using XOR
    next_state = np.logical_xor(connected_state, action).astype(int)
    return next_state


def compute_transition_probability_matrices(p, C, states, actions):
    # The total number of states
    N = len(states)
    # List of transition matrices for each action
    M = []
    # Loop over each action to create its transition matrix
    for action in actions:
        # Initialize a transition matrix for the current action
        M_a = np.zeros((N, N))
        # Loop over each state
        for i, state_i in enumerate(states):
            # Apply system dynamics to get the potential next state
            potential_state = apply_system_dynamics(C, state_i, action)
            # Loop over each state as the possible next state
            for j, state_j in enumerate(states):
                # Initialize probability of transitioning from state i to state j
                prob_transition = 0
                # Consider all possible noise matrices
                for noise in range(16):
                    noise_vector = np.array(list(map(int, f"{noise:04b}"))).reshape(4, 1)
                    # Apply noise via XOR to get the next state
                    next_state = np.logical_xor(potential_state, noise_vector).astype(int)
                    # If the next state with noise matches state j, calculate the probability
                    if np.array_equal(next_state, state_j):
                        # Count the number of genes flipped by noise
                        count = np.sum(np.abs(noise_vector))
                        # Calculate probability of this noise pattern
                        prob_noise = math.pow(p, count) * math.pow(1 - p, 4 - count)
                        # Sum the transition probability from state i to state j
                        prob_transition += prob_noise
                # Store the calculated probability
                M_a[i, j] = prob_transition
        # Normalize the rows of M_a to sum to 1 for valid probability distribution
        row_sums = M_a.sum(axis=1, keepdims=True)
        M_a = np.divide(M_a, row_sums, where=row_sums != 0)
        # Add to the list of matrices
        M.append(M_a)
    return M


# Function to calculate the reward for transitioning from one state to another given an action.
def compute_reward(next_state, action):
    # Calculate the reward based on the next state
    reward_from_state = 5 * sum(next_state)
    # Calculate the cost of action
    cost_of_action = np.sum(np.abs(action))
    # Total reward is the reward from the state minus the cost of action
    total_reward = reward_from_state - cost_of_action
    return total_reward


def compute_R_ss_matrices(states, actions):
    # Number of states in the system
    N = len(states)
    # List to hold the reward matrix for each action
    R_ss = []
    # Iterate over each action to create a corresponding reward matrix
    for action in actions:
        # Initialize the reward matrix for the current action
        R_ss_a = np.zeros((N, N))
        # Calculate reward for transitioning from state i to state j given the action
        for i, state_i in enumerate(states):
            for j, state_j in enumerate(states):
                # Use the reward_function to calculate the reward for transitioning to state_j given the action.
                R_ss_a[i][j] = compute_reward(state_j, action)
        # Append the calculated reward matrix for the current action to the list of matrices
        R_ss.append(R_ss_a)
    return R_ss


def compute_R_s_matrices(M, R_ss_prime, states):
    # List to hold the expected reward vector for each action
    R_s = []
    # Number of states in the system
    N = len(states)
    # Column vector of ones
    one_vector = np.ones((N, 1))
    # Iterate over each action
    for a_index, action in enumerate(actions):
        # Perform element-wise multiplication (Hadamard product) and then matrix multiplication to sum
        R_s_a = np.dot(M[a_index] * R_ss_prime[a_index], one_vector)
        R_s.append(R_s_a.flatten())
    return R_s


def policy_improvement(V, M, R_s, gamma):
    # Total number of states in the system
    num_states = len(V)
    # Total number of actions available
    num_actions = len(M)
    # Initialize the policy with zeros (action 1)
    policy = np.zeros(num_states, dtype=int)
    # Iterate over all states to determine the best action based on current state values
    for s in range(num_states):
        # Calculate the value of each action from this state
        action_values = np.array([R_s[a][s] + gamma * np.dot(M[a][s], V) for a in range(num_actions)])
        # Select the action with the highest value
        policy[s] = np.argmax(action_values)
    return policy


def policy_evaluation(policy, M, R_s, gamma, theta):
    # Number of states
    num_states = len(M[0])
    # Initialize the value function with zeros
    V = np.zeros(num_states)
    # Identity matrix
    I = np.eye(num_states)
    # Repeat the evaluation process until the value function converges
    while True:
        # Initialize matrices for transition probabilities and rewards under the current policy
        M_pi = np.zeros((num_states, num_states))
        R_pi = np.zeros(num_states)
        # For each state, fill in the transition probabilities and rewards based on the current policy
        for s in range(num_states):
            # Transition probabilities for the chosen action in the current policy
            M_pi[s] = M[policy[s]][s]
            # Immediate reward for the chosen action in the current policy
            R_pi[s] = R_s[policy[s]][s]
        # Calculate the new value function
        V_new = np.linalg.inv(I - gamma * M_pi).dot(R_pi)
        # Check if the change in the value function is below the threshold, indicating convergence
        if np.max(np.abs(V - V_new)) < theta:
            break
        # Update the value function for the next iteration
        V = V_new
    return V


def policy_iteration(M, R_s, gamma, theta):
    # Number of states
    num_states = len(M[0])
    # Initialize the policy with zeros
    policy = np.zeros(num_states, dtype=int)
    # Flag to track if the policy has stabilized
    policy_stable = False
    # Counter for the number of iterations performed
    policy_improvement_iterations = 0
    # Continue iterating until the policy stabilizes
    while not policy_stable:
        # Evaluate the current policy to get the value function
        V = policy_evaluation(policy, M, R_s, gamma, theta)
        # Improve the policy based on the current value function
        new_policy = policy_improvement(V, M, R_s, gamma)
        # Increment the iteration counter
        policy_improvement_iterations += 1
        # Check if the policy has changed after the improvement step
        if np.array_equal(policy, new_policy):
            # If the policy hasn't changed, it has stabilized and we can stop iterating
            policy_stable = True
        # Update the current policy to the newly improved policy
        policy = new_policy
    # Return the final value function, optimal stable policy, and the number of iterations
    return V, policy, policy_improvement_iterations


def value_iteration(M, R_s, gamma, theta):
    # Number of states
    num_states = len(M[0])
    # Number of actions
    num_actions = len(M)
    # Initialize the value function with zeros
    V = np.zeros(num_states)
    # Counter for the number of iterations performed
    value_iteration_iterations = 0
    # Continue iterating until the value function converges
    while True:
        # Increment the iteration counter
        value_iteration_iterations += 1
        # Variable to track the maximum change in value function across all states in this iteration
        delta = 0
        # Create a new array to store the updated value function
        V_new = np.zeros(num_states)
        # Iterate over all states to update the value function
        for s in range(num_states):
            # Calculate the value of taking each action from this state and find the maximum value among all actions
            V_new[s] = max([R_s[a][s] + gamma * np.dot(M[a][s], V) for a in range(num_actions)])
        # Determine the maximum change in the value function across all states
        delta = max(np.abs(V_new - V))
        # Update the value function with the newly computed values
        V = V_new
        # If the maximum change is less than the threshold theta, the value function has converged
        if delta < theta:
            break
    # Once the value function has converged, determine the optimal policy based on the final value function
    optimal_policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        # For each state, compute the value of taking each action using the final value function
        action_values = np.array([R_s[a][s] + gamma * np.dot(M[a][s], V) for a in range(num_actions)])
        # The optimal action for this state is the one that maximizes the action value
        optimal_policy[s] = np.argmax(action_values)
    # Return the final value function, optimal stable policy, and the number of iterations
    return V, optimal_policy, value_iteration_iterations



def simulate_episode(M, policy, states):
    # Choose a random initial state index
    current_state_index = random.choice(range(len(states)))
    # Initialize sum
    sum = 0
    # Simulate the system
    for step in range(200):
        # Current state
        current_state = states[current_state_index]
        # Determine the action to take from the policy for the current state
        action_index = policy[current_state_index]
        # Add the number of active genes sum
        sum += np.sum(current_state)
        # Get the transition probabilities to the next states based on the current action
        next_state_p = M[action_index][current_state_index]
        # Choose the next state index based on the transition probabilities
        current_state_index = np.random.choice(range(len(states)), p=next_state_p)
    # Calculate the average activation rate over the number of steps in the episode
    average_activation = sum / 200
    return average_activation


def evaluate_policy(M, pi_star, states):
    # Simulate multiple episodes, each time calculating the activation rate for the given policy
    activation_rates = [simulate_episode(M, pi_star, states) for _ in range(100)]
    # Calculate the average activation rate across all simulated episodes
    avg_activation_rate = np.mean(activation_rates)
    return avg_activation_rate


gamma = 0.95
theta = 0.01
p = 0.05

# Generate M, R_ss, and R_s
M = compute_transition_probability_matrices(p, C, states, actions)
R_ss = compute_R_ss_matrices(states, actions)
R_s = compute_R_s_matrices(M, R_ss, states)

# Policy Iteration
print("Policy Iteration")
V, optimal_policy, num_iterations = policy_iteration(M, R_s, gamma, theta)
print("Optimal Policy:", optimal_policy)
print("Optimal State Values:", V)
print("Number of Policy Iteration Steps:", num_iterations)

# Value Iteration
print("\nValue Iteration")
V, optimal_policy, num_iterations = value_iteration(M, R_s, gamma, theta)
print("Optimal Policy:", optimal_policy)
print("Optimal State Values:", V)
print("Number of Value Iteration Steps:", num_iterations)

# Evaluate Policy
avg_activation_rate = evaluate_policy(M, optimal_policy, states)
print("\nAverage Activation Rate (AvgA) over 100 episodes:", avg_activation_rate)

# No Control Policy
no_control_policy = np.zeros(len(states), dtype=int)
avg_activation_rate_no_control = evaluate_policy(M, no_control_policy, states)
print("Average Activation Rate (AvgA) with no control policy:", avg_activation_rate_no_control)
