import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart, chi2
import matplotlib.patches as patches

def plot_ellipse(axis, covariance_matrix, prob=0.95):
    """
    Plot an ellipse corresponding to a 2x2 covariance matrix.
    
    The ellipse is defined as the contour for which:
      x^T Σ^{-1} x = chi2.ppf(prob, 2)
    """
    scale_factor = np.sqrt(chi2.ppf(prob, df=2))
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[sorted_indices]
    eigen_vectors = eigen_vectors[:, sorted_indices]
    angle_deg = np.degrees(np.arctan2(eigen_vectors[1, 0], eigen_vectors[0, 0]))
    width, height = 2 * scale_factor * np.sqrt(eigen_values)
    ellipse = patches.Ellipse((0, 0), width=width, height=height,
                              angle=angle_deg, edgecolor='blue',
                              facecolor='none', lw=2, alpha=0.5)
    axis.add_patch(ellipse)

def compute_posterior_parameters(prior_degrees_freedom, prior_scale_matrix, data):
    """
    Computes the posterior parameters for the precision matrix given:
      - Prior Degrees of Freedom
      - Prior Scale Matrix
      - Data (each row is an observation)
    
    With S = data^T * data, the posterior parameters become:
      Posterior Degrees of Freedom = prior_degrees_freedom + (number of samples)
      Posterior Scale Matrix = inv( inv(prior_scale_matrix) + S )
    """
    S_matrix = np.dot(data.T, data)
    posterior_degrees_freedom = prior_degrees_freedom + data.shape[0]
    posterior_scale_matrix = np.linalg.inv(np.linalg.inv(prior_scale_matrix) + S_matrix)
    return posterior_degrees_freedom, posterior_scale_matrix

def simulate_data(num_samples, true_covariance):
    """
    Simulate 'num_samples' samples from a 2D Gaussian with zero mean and covariance 'true_covariance'.
    """
    return np.random.multivariate_normal(mean=np.zeros(2), cov=true_covariance, size=num_samples)

def plot_posterior_ellipses(axis, true_covariance, posterior_degrees_freedom, posterior_scale_matrix, num_draws=50):
    """
    Plot the true covariance ellipse (using the 95% contour) and a number of ellipses corresponding
    to samples from the Wishart posterior distribution.
    """
    # Plot the true covariance ellipse (in green for reference)
    plot_ellipse(axis, true_covariance, prob=0.95)
    # Plot ellipses for posterior samples
    for _ in range(num_draws):
        precision_matrix_sample = wishart.rvs(df=posterior_degrees_freedom, scale=posterior_scale_matrix)
        covariance_matrix_sample = np.linalg.inv(precision_matrix_sample)
        plot_ellipse(axis, covariance_matrix_sample, prob=0.95)

# ---------------------------
# Experiment (i): Vary Prior Degrees of Freedom
# ---------------------------
def experiment_vary_degrees():
    np.random.seed(0)
    true_covariance = np.array([[1, 0.5], [0.5, 1]])
    fixed_num_samples = 50  # Fixed number of samples
    data = simulate_data(fixed_num_samples, true_covariance)
    fixed_prior_scale = np.array([[1, 0.5], [0.5, 1]])  # Fixed prior scale matrix

    # Choose 6 different prior degrees of freedom (all valid since they must be >=2 for 2D)
    prior_degrees_list = [3, 5, 7, 10, 15, 20]
    num_draws = 50  # Number of posterior samples to draw

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, prior_deg in enumerate(prior_degrees_list):
        ax = axes[i]
        post_deg, post_scale = compute_posterior_parameters(prior_deg, fixed_prior_scale, data)
        plot_posterior_ellipses(ax, true_covariance, post_deg, post_scale, num_draws)
        ax.set_title(
            f"Prior Degrees: {prior_deg}\n"
            f"Samples: {fixed_num_samples}\n"
            f"Prior Scale: [[1,0.5],[0.5,1]]\n"
            f"Post Degrees: {post_deg}"
        )
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect("equal")
    fig.suptitle("Experiment (i): Varying Prior Degrees of Freedom", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ---------------------------
# Experiment (ii): Vary Number of Samples
# ---------------------------
def experiment_vary_samples():
    np.random.seed(0)
    true_covariance = np.array([[1, 0.5], [0.5, 1]])
    fixed_prior_deg = 5  # Fixed prior degrees of freedom
    fixed_prior_scale = np.array([[1, 0.5], [0.5, 1]])  # Fixed prior scale matrix

    # Choose 6 different sample sizes
    sample_sizes = [5, 10, 25, 50, 100, 300]
    num_draws = 50  # Number of posterior samples to draw

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, n_samples in enumerate(sample_sizes):
        ax = axes[i]
        data = simulate_data(n_samples, true_covariance)
        post_deg, post_scale = compute_posterior_parameters(fixed_prior_deg, fixed_prior_scale, data)
        plot_posterior_ellipses(ax, true_covariance, post_deg, post_scale, num_draws)
        ax.set_title(
            f"Samples: {n_samples}\n"
            f"Prior Degrees: {fixed_prior_deg}\n"
            f"Prior Scale: [[1,0.5],[0.5,1]]\n"
            f"Post Degrees: {post_deg}"
        )
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect("equal")
    fig.suptitle("Experiment (ii): Varying Number of Samples", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ---------------------------
# Experiment (iii): Vary Prior Scale Matrix
# ---------------------------
def experiment_vary_scale():
    np.random.seed(0)
    true_covariance = np.array([[1, 0.5], [0.5, 1]])
    fixed_num_samples = 50  # Fixed number of samples
    data = simulate_data(fixed_num_samples, true_covariance)
    fixed_prior_deg = 5  # Fixed prior degrees of freedom

    # Choose 6 different prior scale matrices of the form [[1, r], [r, 1]] with |r| < 1
    r_values = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    prior_scale_matrices = [np.array([[1, r], [r, 1]]) for r in r_values]
    num_draws = 50  # Number of posterior samples to draw

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, scale_mat in enumerate(prior_scale_matrices):
        ax = axes[i]
        post_deg, post_scale = compute_posterior_parameters(fixed_prior_deg, scale_mat, data)
        plot_posterior_ellipses(ax, true_covariance, post_deg, post_scale, num_draws)
        ax.set_title(
            f"Prior Scale: [[1, {r_values[i]}], [{r_values[i]}, 1]]\n"
            f"Samples: {fixed_num_samples}\n"
            f"Prior Degrees: {fixed_prior_deg}\n"
            f"Post Degrees: {post_deg}"
        )
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_aspect("equal")
    fig.suptitle("Experiment (iii): Varying Prior Scale Matrix", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    experiment_vary_degrees()
    experiment_vary_samples()
    experiment_vary_scale()