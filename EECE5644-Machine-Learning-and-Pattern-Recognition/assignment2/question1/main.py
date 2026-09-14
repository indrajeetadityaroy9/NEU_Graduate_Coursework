import numpy as np
from sklearn.metrics import roc_curve, auc
from utility import generate_data, compute_logistic_params, error_rate, bayesian_optimal_classifier, empirical_min_p_error, classify
from visualization import plot_roc_curve, plot_generated_data

pdf_params = {
    # Class priors, mixture weights, means, and covariance matrices for the GMM
    'priors': np.array([0.6, 0.4]),  # Priors for each class
    'gmm_a': np.array([[0.5, 0.5], [0.5, 0.5]]),  # Mixture weights for Gaussians in each class
    'mu': np.array([[[-1, -1], [1, 1]], [[-1, 1], [1, -1]]]),  # Means of Gaussians
    'sigma': np.array([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]], [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]])
    # Covariance matrices
}


def main():
    # Generate validation data using the GMM parameters
    X_valid, Y_valid = generate_data(10000, pdf_params)
    # Apply Bayesian optimal classifier to validation data
    bayes_predictions, bayes_scores = bayesian_optimal_classifier(X_valid, pdf_params)
    fpr_bayes, tpr_bayes, thresholds_bayes = roc_curve(Y_valid, bayes_scores)
    # Compute theoretical minimum error rate and corresponding gamma value
    theoretical_min_p_error = error_rate(bayes_predictions, Y_valid)
    theoretical_gamma = np.log(pdf_params['priors'][0] / pdf_params['priors'][1])
    theoretical_point = (fpr_bayes[np.argmin(np.abs(thresholds_bayes - theoretical_gamma))],
                         tpr_bayes[np.argmin(np.abs(thresholds_bayes - theoretical_gamma))])

    # Compute empirical minimum error rate and corresponding point on ROC curve
    empirical_min_error, empirical_point, empirical_gamma = empirical_min_p_error(bayes_scores, Y_valid)
    # Store results for logistic regression models with different training data sizes
    results = {}
    N_sizes = [20, 200, 2000]  # Sizes of training datasets
    D_10K_validate = generate_data(10000, pdf_params)  # Generate validation data

    for N in N_sizes:
        # Generate training data for current size N
        X_train, Y_train = generate_data(N, pdf_params)
        plot_generated_data([(X_train, Y_train)])  # Plot generated training data
        # Add bias term to training data for logistic regression
        X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        # Train logistic regression model and compute model parameters
        w_mle = compute_logistic_params(X_train_bias, Y_train)
        # Classify validation data using the trained logistic regression model
        X_valid, Y_valid = D_10K_validate
        X_valid_bias = np.hstack([np.ones((X_valid.shape[0], 1)), X_valid])
        valid_preds = classify(X_valid_bias, w_mle)
        # Estimate probability of error on validation data
        valid_error = error_rate(valid_preds, Y_valid)
        # Store results (model parameters and validation error) for current N
        results[N] = {
            'model_params': w_mle,
            'validation_error': valid_error
        }

    # Print theoretical and empirical minimum error rates for Bayesian classifier
    plot_roc_curve(fpr_bayes, tpr_bayes, theoretical_point=theoretical_point, empirical_point=empirical_point)
    print(f"Theoretical Min-P(Error): {theoretical_min_p_error:.4f}, Gamma: {theoretical_gamma}")
    print(f"Empirical Min-P(Error): {empirical_min_error:.4f}, Gamma: {empirical_gamma}")
    # Print logistic regression results
    print(results)


if __name__ == '__main__':
    main()
