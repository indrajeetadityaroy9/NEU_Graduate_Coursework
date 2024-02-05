import numpy as np
import matplotlib.pyplot as plt

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


def gradient_selection(H):
    exp_H = np.exp(H - np.max(H))
    return exp_H / np.sum(exp_H)


def update_action_preference(H, pi, reward, avg_reward, learning_rate, action):
    H[action] += learning_rate * (reward - avg_reward) * (1 - pi[action])
    H[1 - action] -= learning_rate * (reward - avg_reward) * pi[1 - action]
    return H


def agent_run_gradient_bandit(steps, learning_rate, runs=100):
    avg_rewards = np.zeros(steps)

    for run in range(runs):
        H = np.zeros(2)
        accumulated_rewards = np.zeros(steps)
        avg_reward = 0.0

        for step in range(steps):
            pi = gradient_selection(H)
            action = np.random.choice([0, 1], p=pi)
            R = reward_distribution(action + 1)
            avg_reward = ((avg_reward * step) + R) / (step + 1)
            H = update_action_preference(H, pi, R, avg_reward, learning_rate, action)
            accumulated_rewards[step] = accumulated_rewards[step - 1] + R if step > 0 else R
        avg_rewards += accumulated_rewards

    avg_rewards /= runs
    avg_accumulated_rewards = avg_rewards / np.arange(1, steps + 1)
    return avg_accumulated_rewards


def agent_run_epsilon_greedy(steps, epsilon, learning_rate, runs=100):
    avg_rewards = np.zeros(steps)
    for run in range(runs):
        Q_values = np.zeros(2)
        accumulated_rewards = np.zeros(steps)

        for step in range(steps):
            action = epsilon_greedy_selection(Q_values, epsilon)
            R = reward_distribution(action + 1)
            Q_values[action] += learning_rate * (R - Q_values[action])
            accumulated_rewards[step] = accumulated_rewards[step - 1] + R if step > 0 else R
        avg_rewards += accumulated_rewards

    avg_rewards /= runs
    avg_accumulated_rewards = avg_rewards / np.arange(1, steps + 1)
    return avg_accumulated_rewards


if __name__ == "__main__":
    gradient_bandit_rewards = agent_run_gradient_bandit(n_steps, learning_rate, n_runs)
    epsilon_greedy_rewards = agent_run_epsilon_greedy(n_steps, epsilon, learning_rate, n_runs)
    gradient_bandit_rewards[0] = 0
    epsilon_greedy_rewards[0] = 0

    plt.figure(figsize=(12, 8))
    plt.plot(gradient_bandit_rewards, label='Gradient Bandit')
    plt.plot(epsilon_greedy_rewards, label='Epsilon-Greedy')
    plt.xlabel('Time (t)')
    plt.ylabel('Average Accumulated Reward')
    plt.title('Comparison of Gradient Bandit and Epsilon-Greedy Policies')
    plt.legend()
    plt.grid(True)
    plt.show()
