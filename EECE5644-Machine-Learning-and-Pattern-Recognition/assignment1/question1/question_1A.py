import numpy as np
from question1.evaluation import roc_curve_and_probabilities, empirical_min_error_threshold, theoretical_min_error
from question1.utility import generate_samples, discriminant_scores
from question1.visualization import plot_3d, plot_simple_roc, plot_roc


def main():
    # Define means and covariances
    mean_0 = np.array([-1, -1, -1, -1])
    mean_1 = np.array([1, 1, 1, 1])
    cov_0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])
    cov_1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])
    # Lists to store class-specific parameters
    mean_vectors = [mean_0, mean_1]
    covariance_matrices = [cov_0, cov_1]
    # Define class priors
    priors = [0.35, 0.65]
    # Cost function values
    lambda_10 = lambda_01 = 1
    lambda_11 = lambda_00 = 0
    theoretical_gamma = (priors[0] / priors[1]) * ((lambda_10 - lambda_11) / (lambda_01 - lambda_00))
    # Generate data samples based on the defined parameters
    data = generate_samples(mean_vectors, covariance_matrices, n=10000, priors=priors)
    # Plot the generated data in 3D space
    plot_3d(data)
    # Compute discriminant scores for classification
    features = data.drop(columns=['True Class Label']).to_numpy()
    scores = discriminant_scores(features, mean_vectors, covariance_matrices)
    # Compute ROC curve metrics
    fpr, tpr, P_D0_L1, P_D1_L0 = roc_curve_and_probabilities(scores, data['True Class Label'].values)
    # Find the decision threshold that gives the empirical minimum error
    empirical_gamma, empirical_min_error = empirical_min_error_threshold(scores, data['True Class Label'].values, priors, P_D0_L1, P_D1_L0)
    # Compute the theoretical minimum error for the given threshold
    theoretical_min_error_value = theoretical_min_error(scores, data['True Class Label'].values, priors, theoretical_gamma)
    # Plot the ROC curves
    plot_simple_roc(fpr, tpr)
    plot_roc(fpr, tpr, scores, data['True Class Label'].values, priors, theoretical_gamma, theoretical_min_error_value, empirical_gamma, empirical_min_error)


if __name__ == "__main__":
    main()
