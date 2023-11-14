import numpy as np
import hw2q2
from utility import cubic_feature_transform, mle_solution, mean_squared_error_loss, map_solution
from visualization import plot, plot_mse_comparison


def main():
    # Loading training and validation data
    X_train, Y_train, X_valid, Y_valid = hw2q2.hw2q2()
    # Applying cubic transformation to the training and validation data
    X_train_cube = cubic_feature_transform(X_train)
    X_valid_cube = cubic_feature_transform(X_valid)
    # Defining a range of gamma values for the MAP estimator
    n_gammas = 10000
    m, n = -8, 8
    gammas = np.geomspace(10 ** m, 10 ** n, num=n_gammas)
    mse_map = np.empty(n_gammas)
    # Compute ML estimator
    theta_ml = mle_solution(X_train_cube.T, Y_train)
    # Calculate MSE for ML estimator on validation data
    mse_ml = mean_squared_error_loss(theta_ml, X_valid_cube.T, Y_valid)
    # Calculate MSE for each MAP estimator with different gamma values on validation data
    for i, gamma in enumerate(gammas):
        theta_map = map_solution(X_train_cube.T, Y_train, gamma)
        mse_map[i] = mean_squared_error_loss(theta_map, X_valid_cube.T, Y_valid)
    # Making predictions on training and validation data using the ML estimator
    mle_predictions_train = cubic_feature_transform(X_train).T.dot(theta_ml)
    mle_predictions_valid = cubic_feature_transform(X_valid).T.dot(theta_ml)
    # Visualizing predictions on training data
    plot(X_train, Y_train, mle_predictions_train)
    # Visualizing predictions on validation data
    plot(X_valid, Y_valid, mle_predictions_valid)
    # Plotting the comparison of MSE between MAP and ML estimators
    plot_mse_comparison(gammas, mse_map, mse_ml)


if __name__ == '__main__':
    main()
