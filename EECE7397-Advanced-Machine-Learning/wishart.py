#!/usr/bin/env python3
"""
Posterior Visualization for a Bivariate Gaussian Precision Matrix

We assume a conjugate Wishart prior on the precision matrix Λ:
    p(Λ|W, ν) ∝ |Λ|^((ν - D - 1)/2) exp(-½ tr(W⁻¹ Λ))
After observing N samples (with zero mean) from a bivariate Gaussian,
the posterior becomes
    p(Λ|X) ∝ |Λ|^((ν+N - D - 1)/2) exp(-½ tr((W⁻¹ + S) Λ))
with S = sum(x_i x_i^T) and posterior degrees of freedom ν_post = ν + N.
The posterior scale matrix V_post is defined via
    V_post⁻¹ = W⁻¹ + S.
We then draw samples from this Wishart distribution and, for each draw,
plot the ellipse corresponding to the covariance matrix (i.e. the inverse
of the drawn precision matrix). The ellipse is chosen to represent the 95%
confidence contour (using chi-square quantile with 2 d.o.f).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart
from matplotlib.patches import Ellipse

# ---------------------------------------------------------------
# Helper function to plot an ellipse for a 2x2 covariance matrix.
# ---------------------------------------------------------------
def plot_cov_ellipse(cov, pos, ax, n_std=2.4477, **kwargs):
    """
    Plots an ellipse corresponding to a covariance matrix 'cov' centered at 'pos'.
    
    Parameters:
      cov   : 2x2 covariance matrix.
      pos   : (x,y) center of the ellipse.
      ax    : matplotlib axes object.
      n_std : scaling factor. For 2D a 95% confidence contour corresponds
              to sqrt(5.991) ≈ 2.4477.
      **kwargs: keyword arguments passed to Ellipse.
    """
    # Eigen-decomposition to get principal axes
    vals, vecs = np.linalg.eigh(cov)
    # sort the eigenvalues in descending order
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ax.add_patch(ellip)

# ---------------------------------------------------------------
# Function to generate the posterior parameters from data.
# ---------------------------------------------------------------
def generate_posterior_params(W, nu, N, data_cov):
    """
    Generate N samples from N(0, data_cov) (true generating covariance),
    compute the scatter matrix S = Σ_i x_i x_i^T, and return:
       - posterior degrees of freedom: nu_post = nu + N,
       - posterior scale matrix V_post: defined via V_post⁻¹ = W⁻¹ + S.
    
    Parameters:
      W       : prior scale matrix (2x2 positive definite).
      nu      : prior degrees of freedom.
      N       : number of data samples.
      data_cov: true data covariance (used to generate samples).
    
    Returns:
      (nu_post, V_post)
    """
    D = data_cov.shape[0]
    X = np.random.multivariate_normal(mean=np.zeros(D), cov=data_cov, size=N)
    S = np.dot(X.T, X)
    nu_post = nu + N
    V_post = np.linalg.inv(np.linalg.inv(W) + S)
    return nu_post, V_post

# ---------------------------------------------------------------
# Function to draw and plot samples from the Wishart posterior.
# ---------------------------------------------------------------
def plot_wishart_samples(nu_post, V_post, ax, n_draws=50, edgecolor='blue', facecolor='none', alpha=0.5):
    """
    Draw n_draws samples from Wishart(df=nu_post, scale=V_post). For each drawn
    precision matrix, compute its inverse (i.e. the covariance matrix) and plot
    the 95% confidence ellipse (centered at 0).
    """
    D = V_post.shape[0]
    for i in range(n_draws):
        sample_precision = wishart.rvs(df=nu_post, scale=V_post)
        sample_cov = np.linalg.inv(sample_precision)
        plot_cov_ellipse(sample_cov, pos=(0, 0), ax=ax, edgecolor=edgecolor, facecolor=facecolor, alpha=alpha)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

# ---------------------------------------------------------------
# Main script
# ---------------------------------------------------------------
np.random.seed(42)  # for reproducibility

# Assume a true covariance (for generating data) for the bivariate Gaussian:
true_cov = np.array([[1, 0.3],
                     [0.3, 1]])

# ===== Experiment (i): Varying Prior Degrees of Freedom =====
# Fix: N = 50 samples and prior scale matrix W.
N_fixed = 50
W_fixed = np.array([[1, 0.2],
                    [0.2, 1]])
# Choose three different prior degrees of freedom (note: must be > D-1 = 1)
nu_values = [3, 5, 10]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, nu in enumerate(nu_values):
    # Get posterior parameters from data generated with true_cov.
    nu_post, V_post = generate_posterior_params(W_fixed, nu, N_fixed, true_cov)
    ax = axs[i]
    plot_wishart_samples(nu_post, V_post, ax, n_draws=50, edgecolor='red', alpha=0.7)
    ax.set_title(f'Prior ν = {nu}\nPosterior ν = {nu_post}\nN = {N_fixed}, W fixed')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
plt.suptitle('Posterior Distribution: Varying Prior Degrees of Freedom')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ===== Experiment (ii): Varying Number of Samples =====
# Fix: prior ν = 5 and prior scale matrix W.
nu_fixed = 5
W_fixed = np.array([[1, 0.2],
                    [0.2, 1]])
# Use different numbers of data samples
N_values = [10, 50, 200]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, N in enumerate(N_values):
    nu_post, V_post = generate_posterior_params(W_fixed, nu_fixed, N, true_cov)
    ax = axs[i]
    plot_wishart_samples(nu_post, V_post, ax, n_draws=50, edgecolor='green', alpha=0.7)
    ax.set_title(f'N = {N}\nPosterior ν = {nu_post}\nPrior ν = {nu_fixed}, W fixed')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
plt.suptitle('Posterior Distribution: Varying Number of Samples')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ===== Experiment (iii): Varying the Scale Matrix W =====
# Fix: prior ν = 5 and N = 50.
nu_fixed = 5
N_fixed = 50
# Choose three different scale matrices W (each 2×2 and positive definite)
W_matrices = [
    np.array([[1, 0],
              [0, 1]]),
    np.array([[1, 0.5],
              [0.5, 1]]),
    np.array([[2, -0.3],
              [-0.3, 0.5]])
]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, W in enumerate(W_matrices):
    nu_post, V_post = generate_posterior_params(W, nu_fixed, N_fixed, true_cov)
    ax = axs[i]
    plot_wishart_samples(nu_post, V_post, ax, n_draws=50, edgecolor='purple', alpha=0.7)
    ax.set_title(f'W = \n{W}\nPosterior ν = {nu_post}, N = {N_fixed}\nPrior ν = {nu_fixed}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
plt.suptitle('Posterior Distribution: Varying Scale Matrix W')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
