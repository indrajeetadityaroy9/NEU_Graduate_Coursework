import numpy as np
from question1.evaluation import roc_curve_and_probabilities, empirical_min_error_threshold, theoretical_min_error
from question1.utility import generate_samples, discriminant_scores, naive_bayes_discriminant_scores
from question1.visualization import plot_combined_roc


def main():
    # Define class means for the two classes.
    mean_0 = np.array([-1, -1, -1, -1])
    mean_1 = np.array([1, 1, 1, 1])
    # Define class covariances. These are diagonal, indicating feature independence.
    cov_0 = np.array([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 2]])
    cov_1 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 3]])
    mean_vectors = [mean_0, mean_1]
    covariance_matrices = [cov_0, cov_1]
    priors = [0.35, 0.65]
    # Misclassification costs.
    lambda_10 = lambda_01 = 1
    lambda_11 = lambda_00 = 0
    # Compute a theoretical decision threshold using the given priors and costs.
    theoretical_gamma = (priors[0] / priors[1]) * ((lambda_10 - lambda_11) / (lambda_01 - lambda_00))
    # Generate data samples based on the means, covariances, and priors.
    data = generate_samples(mean_vectors, covariance_matrices, n=10000, priors=priors)
    features = data.drop(columns=['True Class Label']).to_numpy()
    # Compute discriminant scores using Bayesian discriminants (possibly LDA/QDA).
    scores_original = discriminant_scores(features, mean_vectors, covariance_matrices)
    # Compute ROC metrics for the original discriminants.
    FPR_original, TPR_original, P_D0_L1_original, P_D1_L0_original = roc_curve_and_probabilities(scores_original, data['True Class Label'].values)
    empirical_gamma_original, empirical_min_error_original = empirical_min_error_threshold(scores_original, data['True Class Label'].values, priors, P_D0_L1_original, P_D1_L0_original)
    theoretical_min_error_original = theoretical_min_error(scores_original, data['True Class Label'].values, priors, theoretical_gamma)
    # Compute discriminant scores using Naive Bayes.
    naive_scores = naive_bayes_discriminant_scores(features, mean_vectors)
    # Compute ROC metrics for the Naive Bayes classifier.
    naive_FPR, naive_TPR, naive_P_D0_L1, naive_P_D1_L0 = roc_curve_and_probabilities(naive_scores, data['True Class Label'].values)
    naive_empirical_gamma, naive_empirical_min_error = empirical_min_error_threshold(naive_scores, data['True Class Label'].values, priors, naive_P_D0_L1,  naive_P_D1_L0)
    theoretical_min_error_naive = theoretical_min_error(naive_scores, data['True Class Label'].values,priors, theoretical_gamma)
    # Visualize the performance of both classifiers using a combined ROC curve.
    plot_combined_roc(
        FPR_original, TPR_original,
        naive_FPR, naive_TPR,
        scores_original, naive_scores,
        data['True Class Label'].values, priors,
        theoretical_gamma,
        theoretical_min_error_original, empirical_gamma_original, empirical_min_error_original,
        theoretical_min_error_naive, naive_empirical_gamma, naive_empirical_min_error
    )


if __name__ == "__main__":
    main()
