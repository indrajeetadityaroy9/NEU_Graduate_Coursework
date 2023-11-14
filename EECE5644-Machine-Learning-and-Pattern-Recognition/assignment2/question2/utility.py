import numpy as np


# Applies a cubic polynomial transformation to a 2D input feature set.
def cubic_feature_transform(X):
    # Extracting individual features from the 2D input
    X_1 = X[0, :]
    X_2 = X[1, :]

    # Applying cubic transformations: square, cross-product, and cube terms
    phi_X = np.vstack([
        X,  # Original features w_1 x_1
        X_1 ** 2,  # Square of the first feature w_3 x_1^2
        X_1 * X_2,  # Cross-product of the two features w_4 x_1 x_2
        X_2 ** 2,  # Square of the second feature w_5 x_2^2
        X_1 ** 2 * X_2,  # First feature squared times second feature w_7 x_1^2 x_2
        X_1 * X_2 ** 2,  # First feature times second feature squared w_8 x_1 x_2^2
        X_1 ** 3,  # Cube of the first feature  w_6 x_1^3
        X_2 ** 3  # Cube of the second feature w_9 x_2^3
    ])

    return phi_X  # Return the transformed feature matrix


# Calculates the mean squared error (MSE) for a given set of parameters, features, and target values.
def mean_squared_error_loss(parameters, X, y):
    predictions = X.dot(parameters)  # Calculate predictions based on current parameters
    error = predictions - y  # Calculate the difference between predictions and actual values
    mse_loss = np.mean(error ** 2)  # Compute the mean squared error
    return mse_loss  # Return the computed MSE loss


# Finds the ML solution
def mle_solution(X, y):
    # Compute the ML estimation
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)  # (X^T * X)^-1 * X^T * y


# Computes the MAP estimation
def map_solution(X, y, regularization_strength):
    num_features = np.size(np.asarray(X), axis=1)  # Determine the number of features
    # Compute the MAP estimation with regularization
    # (X^T * X + gamma * I)^-1 * X^T * y
    return np.linalg.inv(X.T.dot(X) + regularization_strength * np.eye(num_features)).dot(X.T).dot(y)
