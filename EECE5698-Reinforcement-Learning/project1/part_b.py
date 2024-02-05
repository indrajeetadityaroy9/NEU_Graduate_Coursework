import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Arm 1 Gaussian mean and std
mu1, sigma1 = 5, np.sqrt(10)
# Arm 2 Gaussian 1 mean and std
mu21, sigma21 = 10, np.sqrt(15)
# Arm 2 Gaussian 2 mean and std
mu22, sigma22 = 4, np.sqrt(10)
# Fixed epsilon
epsilon = 0.1
# Fixed alpha
learning_rate = 0.1
# Number of agent steps
n_steps = 1000
# Number of agent runs
n_runs = 100
# Initial optimistic Q-values
initial_Q_values = [[0, 0], [5, 7], [20, 20]]


def reward_distribution(lever):
    # If agent selects lever 1, get reward from arm 1 Gaussian
    if lever == 1:
        return np.random.normal(mu1, sigma1)
    else:
        # If agent selects lever 2, get reward from binomial Gaussian with 50% probability from 2 Gaussians
        if np.random.rand() < 0.5:
            return np.random.normal(mu21, sigma21)
        else:
            return np.random.normal(mu22, sigma22)


def epsilon_greedy_selection(Q, epsilon):
    if np.random.rand() < epsilon:
        # Exploration (ε)
        return np.random.choice([0, 1])
    else:
        # Exploitation (1-ε)
        return np.argmax(Q)


def agent_run(epsilon, learning_rate, steps, runs, initial_Q_values):
    # Average reward per step for each set of initial Q-values
    avg_rewards = np.zeros((len(initial_Q_values), steps))
    # Average Q-values over runs for each initial Q-values
    avg_Q_values = np.zeros((len(initial_Q_values), 2))
    # Iterate over each initial Q-values
    for i, initial_Q_value in enumerate(initial_Q_values):
        Q_values = np.zeros(2)
        # Accumulated rewards for each run and step
        accumulated_rewards = np.zeros((runs, steps))
        # Multiple agent runs
        for run in range(runs):
            # Initial Q-values for current run
            Q_values = np.array(initial_Q_value, dtype=float)
            # Total reward for current run
            run_reward = 0
            for step in range(steps):
                # Action selection using epsilon-greedy policy
                action = epsilon_greedy_selection(Q_values, epsilon)
                # Reward based on the selected action
                R = reward_distribution(action + 1)
                # Update  Q-value for selected action
                Q_values[action] += learning_rate * (R - Q_values[action])
                # Reward per step for current run
                run_reward += R
                # Accumulate reward for current step
                accumulated_rewards[run, step] = run_reward / (step + 1)
            # Q-values over runs
            Q_values += Q_values
        # Average reward over all runs for initial Q-values
        avg_rewards[i, :] = np.mean(accumulated_rewards, axis=0)
        avg_Q_values[i] = Q_values / runs
    # Average reward per step and average final Q-values over 100 runs
    return avg_rewards, avg_Q_values


if __name__ == "__main__":
    avg_accumulated_rewards, avg_Q_values = agent_run(epsilon, learning_rate, n_steps, n_runs, initial_Q_values)
    labels = [f"Initial Q-values: {q}" for q in initial_Q_values]

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(avg_accumulated_rewards):
        plt.plot(data, label=labels[i])
    plt.title('Average Accumulated Reward for Different Initial Q-values')
    plt.xlabel('Time (t)')
    plt.ylabel('Average Accumulated Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

    table = {
        'Initial Q values': ['Q = [0 0]', 'Q = [5 7]', 'Q = [20 20]'],
        'Average of action value Q(a1) of 100 runs': avg_Q_values[:, 0],
        'True action value Q*(a1)': [5] * len(initial_Q_values),
        'Average of action value Q(a2) of 100 runs': avg_Q_values[:, 1],
        'True action value Q*(a2)': [7] * len(initial_Q_values),
    }
    df = pd.DataFrame(table)
    print(df.to_string(index=False))
