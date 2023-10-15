import numpy as np
from scipy.stats import multivariate_normal


def bayesian_classifier(samples, prior_probs, mean_vals, cov_vals, loss_matrix):
    n_samples = samples.shape[0]  # Number of samples to classify
    n_classes = len(prior_probs)  # Number of classes
    # Initialize a matrix to store the posterior probabilities
    posteriors = np.zeros((n_samples, n_classes))
    # For the first two classes (assumes Gaussian distribution)
    for i in range(n_classes - 1):
        # Compute and store the posterior probabilities
        # Probability Density Function (pdf) * Prior Probabilities
        posteriors[:, i] = multivariate_normal.pdf(samples, mean=mean_vals[i], cov=cov_vals[i]) * prior_probs[i]
    # For the third class, assuming it is a mixture of two Gaussians
    posteriors[:, 2] = 0.5 * (multivariate_normal.pdf(samples, mean=mean_vals[2], cov=cov_vals[2]) +
                              multivariate_normal.pdf(samples, mean=mean_vals[3], cov=cov_vals[3])) * prior_probs[2]
    # Compute the risks using the posterior probabilities and the provided loss matrix
    # Risks are calculated by multiplying the posterior probabilities with the transpose of the loss matrix
    risks = np.dot(posteriors, loss_matrix.T)
    # Determine the class with the minimum risk for each sample
    decisions = np.argmin(risks, axis=1)
    return decisions + 1  # Return decisions, ensuring class labels start from 1 instead of 0
