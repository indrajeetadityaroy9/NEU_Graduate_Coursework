import numpy as np
from scipy.stats import multivariate_normal


# Function to fit a Bayesian Classifier to the data
def fit_bayesian_classifier(x, y):
    # Initialize lists to store means, covariances, and priors for each class
    means = []
    covariances = []
    priors = []
    # Get unique classes in the label set
    classes = np.unique(y)
    # Iterate over each class to compute the means, covariances, and priors
    for cls in classes:
        # Subset the data belonging to the current class
        x_class = x[y == cls]
        # Compute and store the mean of the subset
        means.append(np.mean(x_class, axis=0))
        # Compute and store the regularized covariance of the subset
        covariances.append(regularized_covariance(x_class))
        # Compute and store the prior probability of the class
        priors.append(len(x_class) / len(x))
    return means, covariances, priors, classes


# Function to predict labels using a fitted Bayesian Classifier
def predict_bayesian_classifier(x, means, covariances, priors, classes):
    # Number of samples and classes
    num_samples = x.shape[0]
    num_classes = len(priors)
    # Initialize the posterior probabilities matrix
    posteriors = np.zeros((num_samples, num_classes))
    # Compute the posterior probabilities for each class
    for i in range(num_classes):
        posteriors[:, i] = multivariate_normal.pdf(x, mean=means[i], cov=covariances[i]) * priors[i]
    # Make decisions based on the computed posteriors
    decisions = np.argmax(posteriors, axis=1)
    # Return the predicted labels
    return [classes[decision] for decision in decisions]


# Function to calculate the regularized covariance matrix
def regularized_covariance(x, lambda_val=1e-2):
    # Compute the covariance matrix
    cov_matrix = np.cov(x, rowvar=False)
    # Regularize the covariance matrix
    regularized_cov_matrix = cov_matrix + lambda_val * np.eye(x.shape[1])
    return regularized_cov_matrix
