import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.linalg import eigh


# Function to generate samples based on provided mean vectors, covariance matrices, and other parameters
def generate_samples(mean_vectors, cov_matrices, n=10000, priors=None):
    # Initialising random number generator
    rng = np.random.default_rng()
    # Generating random class labels based on given priors
    labels = np.random.choice(len(mean_vectors), size=n, p=priors)
    # Generating samples based on multivariate normal distribution for each class
    all_samples = np.vstack(
        [rng.multivariate_normal(mean_vectors[i], cov_matrices[i], size=np.sum(labels == i)) for i in
         range(len(mean_vectors))])
    # Constructing a DataFrame for the samples
    df = pd.DataFrame(all_samples, columns=[f'feature_{i + 1}' for i in range(mean_vectors[0].shape[0])])
    df['True Class Label'] = labels
    return df


# Function to compute discriminant scores
def discriminant_scores(features, mean_vectors, cov_matrices):
    l = len(mean_vectors)
    # Calculating likelihoods based on multivariate normal PDF
    class_conditional_likelihoods = np.array([
        multivariate_normal.pdf(features, mean_vectors[i], cov_matrices[i])
        for i in range(l)
    ])
    return class_conditional_likelihoods[1] / class_conditional_likelihoods[0]


# Function to compute discriminant scores for Naive Bayes
def naive_bayes_discriminant_scores(features, mean_vectors):
    n, d = features.shape
    l = len(mean_vectors)
    # Calculating likelihoods using independent features assumption (Naive Bayes)
    naive_class_conditional_likelihoods = np.array([
        multivariate_normal.pdf(features, mean_vectors[i], np.eye(d)) for i in range(l)
    ])
    return naive_class_conditional_likelihoods[1] / naive_class_conditional_likelihoods[0]


# Function to compute discriminant scores for Linear Discriminant Analysis (LDA)
def lda_discriminant_scores(features, w_LDA):
    return np.dot(features, w_LDA)


# Function to estimate the parameters (mean and covariance) based on the data
def estimate_parameters(data):
    # Extracting unique class labels
    classes = np.unique(data['True Class Label'])
    # Estimating means for each class
    estimated_means = [data[data['True Class Label'] == c].iloc[:, :-1].mean().values for c in classes]
    # Estimating covariance matrices for each class
    estimated_covs = [data[data['True Class Label'] == c].iloc[:, :-1].cov().values for c in classes]
    return estimated_means, estimated_covs


# Function to compute Fisher's Linear Discriminant Analysis
def fisher_lda(estimated_means, estimated_covs):
    # Calculating between-class scatter matrix
    sb = np.outer(estimated_means[1] - estimated_means[0], estimated_means[1] - estimated_means[0])
    # Calculating within-class scatter matrix
    sw = estimated_covs[0] + estimated_covs[1]
    # Calculating eigenvalues and eigenvectors
    _, eigvec = eigh(sb, sw)
    return eigvec[:, -1]
