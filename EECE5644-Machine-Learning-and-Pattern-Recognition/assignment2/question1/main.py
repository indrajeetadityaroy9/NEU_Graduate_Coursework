import numpy as np
from sklearn.metrics import roc_curve, auc
from utility import generate_data, compute_logistic_params, error_rate, bayesian_optimal_classifier, empirical_min_p_error, classify
from question1.visualization import plot_roc_curve, plot_generated_data

pdf_params = {
    'priors': np.array([0.6, 0.4]),  # Class priors for L=0 and L=1
    'gmm_a': np.array([[0.5, 0.5], [0.5, 0.5]]),  # Mixture weights for the Gaussians in each class
    'mu': np.array([[[-1, -1], [1, 1]], [[-1, 1], [1, -1]]]),  # Means of the Gaussians for each class
    'Sigma': np.array([[[[1, 0], [0, 1]], [[1, 0], [0, 1]]], [[[1, 0], [0, 1]], [[1, 0], [0, 1]]]])
}


def main():
    # Generate validation data
    X_valid, Y_valid = generate_data(10000, pdf_params)

    # Apply Bayesian optimal classifier
    bayes_predictions, bayes_scores = bayesian_optimal_classifier(X_valid, pdf_params)
    fpr_bayes, tpr_bayes, thresholds_bayes = roc_curve(Y_valid, bayes_scores)

    # Theoretical Min-P(Error) point and gamma
    theoretical_min_p_error = error_rate(bayes_predictions, Y_valid)
    theoretical_gamma = np.log(pdf_params['priors'][0] / pdf_params['priors'][1])
    theoretical_point = (fpr_bayes[np.argmin(np.abs(thresholds_bayes - theoretical_gamma))], tpr_bayes[np.argmin(np.abs(thresholds_bayes - theoretical_gamma))])

    # Empirical Min-P(Error) point and gamma
    empirical_min_error, empirical_point, empirical_gamma = empirical_min_p_error(bayes_scores, Y_valid)

    results = {}
    N_sizes = [20, 200, 2000]
    D_10K_validate = generate_data(10000, pdf_params)  # Replace with your data generation function

    for N in N_sizes:
        # Generate training data
        X_train, Y_train = generate_data(N, pdf_params)  # Replace with your data generation function
        plot_generated_data([(X_train, Y_train)])

        # Add a bias term to X_train
        X_train_bias = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

        # Train the logistic regression model
        w_mle = compute_logistic_params(X_train_bias, Y_train)

        # Classify samples in the validation set
        X_valid, Y_valid = D_10K_validate
        X_valid_bias = np.hstack([np.ones((X_valid.shape[0], 1)), X_valid])
        valid_preds = classify(X_valid_bias, w_mle)

        # Estimate the probability of error
        valid_error = error_rate(valid_preds, Y_valid)

        results[N] = {
            'model_params': w_mle,
            'validation_error': valid_error
        }

    print(results)

    plot_roc_curve(fpr_bayes, tpr_bayes, theoretical_point=theoretical_point, empirical_point=empirical_point)
    print(f"Estimated Theoretical Min-P(Error) for Bayesian Classifier: {theoretical_min_p_error:.4f}, Gamma: {theoretical_gamma}")
    print(f"Estimated Empirical Min-P(Error): {empirical_min_error:.4f}, Gamma: {empirical_gamma}")


if __name__ == '__main__':
    main()
