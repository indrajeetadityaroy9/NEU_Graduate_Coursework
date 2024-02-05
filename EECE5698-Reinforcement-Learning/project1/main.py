import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Arm 1 Gaussian mean and std
mu1, sigma1 = 5, np.sqrt(10)
# Arm 2 Gaussian 1 mean and std
mu21, sigma21 = 10, np.sqrt(15)
# Arm 2 Gaussian 2 mean and std
mu22, sigma22 = 4, np.sqrt(10)
# # Different learning rate types
learning_rate_types = ['constant', 'decay', 'log', 'inverse']
# Different learning rate formulas
learning_rates = {
    'constant': lambda k: 1,
    'decay': lambda k: 0.9 ** k,
    'log': lambda k: 1 / (1 + np.log(1 + k)),
    'inverse': lambda k: 1 / k if k > 0 else 1
}
# Different learning rate formulas
learning_rate_formulas = {
    'constant': r"$\alpha = 1$",
    'decay': r"$\alpha = 0.9^k$",
    'log': r"$\alpha = \frac{1}{1 + \log(1 + k)}$",
    'inverse': r"$\alpha = \frac{1}{k}$"
}
# Different epsilon values
epsilon_values = [0, 0.1, 0.2, 0.5]
# Number of agent steps
n_steps = 1000
# Number of agent runs
n_runs = 100


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


def agent_run(learning_rate, epsilon_values, steps, runs):
    # Cumulative rewards for each combination of epsilon, run, and step
    rewards = np.zeros((len(epsilon_values), runs, steps))
    # Final Q-values for current epsilon value
    epsilon_Q_values = np.zeros((len(epsilon_values), 2))
    # Iterate over each epsilon
    for i, epsilon in enumerate(epsilon_values):
        # Current epsilon Q-values for each run
        run_Q_values = np.zeros((runs, 2))
        # Multiple agent runs
        for run in range(runs):
            # Q-values for the current run
            Q_values = np.zeros(2)
            for step in range(steps):
                # Action selection using epsilon-greedy policy
                action = epsilon_greedy_selection(Q_values, epsilon)
                # Reward based on the selected action
                R = reward_distribution(action + 1)
                # Current learning rate formula output
                alpha = learning_rates[learning_rate](step)
                # Update  Q-value for selected action
                Q_values[action] += alpha * (R - Q_values[action])
                # Accumulate reward for current step
                rewards[i, run, step] = rewards[i, run, step - 1] + R
            # Final Q-values for current run
            run_Q_values[run] = Q_values
        # Average final Q-values over 100 runs for current epsilon value
        epsilon_Q_values[i] = np.mean(run_Q_values, axis=0)
    # Average rewards for epsilon value
    average_rewards = np.mean(rewards, axis=1) / np.arange(1, steps + 1)
    # Average reward per step and average final Q-values over 100 runs
    return average_rewards, epsilon_Q_values


if __name__ == "__main__":
    for learning_rate in learning_rate_types:
        average_accumulated_rewards, epsilon_Q_values = agent_run(learning_rate, epsilon_values, n_steps, n_runs)

        plt.figure(figsize=(12, 8))
        for i, epsilon in enumerate(epsilon_values):
            plt.plot(average_accumulated_rewards[i], label=f'ε = {epsilon}')
        plt.title(f'Average Accumulated Reward for: {learning_rate_formulas[learning_rate]}')
        plt.xlabel('Time (t)')
        plt.ylabel('Average Accumulated Reward')
        plt.legend()
        plt.grid()
        plt.show()

        table = {
            'Epsilon-greedy': [f'ε = {eps}' for eps in epsilon_values],
            'Average of action value Q(a1) of 100 runs': epsilon_Q_values[:, 0],
            'True action value Q*(a1)': [5] * len(epsilon_values),
            'Average of action value Q(a2) of 100 runs': epsilon_Q_values[:, 1],
            'True action value Q*(a2)': [7] * len(epsilon_values),
        }
        df = pd.DataFrame(table)
        print(df.to_string(index=False))
