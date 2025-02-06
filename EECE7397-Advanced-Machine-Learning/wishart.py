import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart

def generate_data(n, cov):
    return np.random.multivariate_normal(mean=[0, 0], cov=cov, size=n)

def sample_wishart(df, scale, num_samples=5):
    samples = []
    for _ in range(num_samples):
        W = wishart.rvs(df=df, scale=scale)
        samples.append(W)
    return samples

def plot_precision_ellipses(precisions, ax, title):
    for p in precisions:
        eigvals, eigvecs = np.linalg.eigh(p)
        if np.any(eigvals <= 0):
            continue

        angles = np.linspace(0, 2*np.pi, 200)
        circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        scale_matrix = np.diag(1.0 / np.sqrt(eigvals))
        ellipse_y = circle @ scale_matrix
        ellipse_x = ellipse_y @ eigvecs.T
        ax.plot(ellipse_x[:, 0], ellipse_x[:, 1], 'b', alpha=0.5)
    
    ax.set_aspect('equal', 'box')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)

def variation_1_distribution():
    n = 5
    scale_matrix = np.eye(2) * 2.0
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    X = generate_data(n, cov)
    sum_T = X.T @ X
    dfs = [3, 5, 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, df_prior in enumerate(dfs):
        df_post = df_prior + n
        scale_post = scale_matrix + sum_T
        posterior_samples = sample_wishart(df_post, scale_post, num_samples=10)
        
        title = (
            f"scale matrix={scale_matrix.tolist()}\n"
            f"degrees of freedom of the prior={df_prior}\n"
            f"degrees of freedom of the posterior={df_post}\n"
            f"number of samples={n}"
        )
        plot_precision_ellipses(posterior_samples, axes[i], title)
    
    fig.suptitle("Varying Prior Degrees of Freedom", fontsize=16)
    plt.tight_layout()
    plt.show()

def variation_2_distribution():
    df_prior = 5
    scale_matrix = np.eye(2)
    cov = np.array([[1.0, 0.5],[0.5, 1.0]])
    ns = [2, 5, 20]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, n in enumerate(ns):
        X = generate_data(n, cov)
        sum_T = X.T @ X
        df_post = df_prior + n
        scale_post = scale_matrix + sum_T
        posterior_samples = sample_wishart(df_post, scale_post, num_samples=10)
        
        title = (
            f"scale matrix={scale_matrix.tolist()}\n"
            f"degrees of freedom of the prior={df_prior}\n"
            f"degrees of freedom of the posterior={df_post}\n"
            f"number of samples={n}"
        )
        plot_precision_ellipses(posterior_samples, axes[i], title)
    
    fig.suptitle("Varying Number of Samples", fontsize=16)
    plt.tight_layout()
    plt.show()

def variation_3_distribution():
    df_prior = 5
    n = 5
    scale_matrix_list = [np.eye(2), 2.0 * np.eye(2), np.diag([1.0, 2.0])]
    cov = np.array([[1.0, 0.5],[0.5, 1.0]])
    X = generate_data(n, cov)
    sum_T = X.T @ X

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, scale_matrix in enumerate(scale_matrix_list):
        df_post = df_prior + n
        scale_post = scale_matrix + sum_T
        posterior_samples = sample_wishart(df_post, scale_post, num_samples=10)
        
        title = (
            f"scale matrix={scale_matrix.tolist()}\n"
            f"degrees of freedom of the prior={df_prior}\n"
            f"degrees of freedom of the posterior={df_post}\n"
            f"number of samples={n}"
        )
        plot_precision_ellipses(posterior_samples, axes[i], title)
    
    fig.suptitle("Varying Scale Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    variation_1_distribution()
    variation_2_distribution()
    variation_3_distribution()

if __name__ == "__main__":
    main()
