import numpy as np
from question1.evaluation import roc_curve_and_probabilities, optimal_threshold
from question1.utility import generate_samples, estimate_parameters, fisher_lda, lda_discriminant_scores
from question1.visualization import plot_lda_roc


def main():
    # Define class means for the two classes.
    mean_0 = np.array([-1, -1, -1, -1])
    mean_1 = np.array([1, 1, 1, 1])
    # Define class covariances for the two classes.
    cov_0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])
    cov_1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])
    mean_vectors = [mean_0, mean_1]
    covariance_matrices = [cov_0, cov_1]
    prior_probabilities = [0.35, 0.65]
    # Misclassification costs.
    lambda_10 = lambda_01 = 1
    lambda_11 = lambda_00 = 0
    # Compute a theoretical decision threshold using the given priors and costs.
    theoretical_gamma = (prior_probabilities[0] / prior_probabilities[1]) * ((lambda_10 - lambda_11) / (lambda_01 - lambda_00))
    # Generate data samples based on the defined means, covariances, and priors.
    data = generate_samples(mean_vectors, covariance_matrices, n=10000, priors=prior_probabilities)
    # Estimate the parameters (means and covariances) from the generated data.
    estimated_means, estimated_covs = estimate_parameters(data)
    # Compute the Fisher LDA projection vector using the estimated parameters.
    w_LDA = fisher_lda(estimated_means, estimated_covs)
    # Extract feature data and compute the LDA discriminant scores using the Fisher LDA projection.
    features = data.drop(columns=['True Class Label']).to_numpy()
    lda_scores = lda_discriminant_scores(features, w_LDA)
    # Compute ROC metrics using the LDA discriminant scores.
    fpr, tpr, P_D0_L1, P_D1_L0 = roc_curve_and_probabilities(lda_scores, data['True Class Label'].values)
    # Determine the optimal threshold for classification based on the LDA scores, class labels, and priors.
    empirical_gamma, empirical_error = optimal_threshold(lda_scores, data['True Class Label'].values, prior_probabilities)
    # Plot the ROC curve for LDA classifier using the calculated metrics.
    plot_lda_roc(fpr, tpr, empirical_gamma, empirical_error)


if __name__ == "__main__":
    main()
