import numpy as np
import pandas as pd
from question3.evaluation import fit_bayesian_classifier, predict_bayesian_classifier
from question3.utility import load_data, compute_and_display_errors
from question3.visualization import visualize_true_labels, visualize_classification_results, \
    generate_and_display_confusion_matrix


def main():
    y_test_file = 'y_test.txt'
    x_test_file = 'X_test.txt'
    y_train_file = 'y_train.txt'
    x_train_file = 'X_train.txt'
    # Marker shapes for plotting
    marker_shapes = ['o', 's', '^', 'v', 'D', '*', '+', 'x', 'p', '<', '>']
    # Load the dataset
    X, y = load_data(y_test_file, x_test_file, y_train_file, x_train_file)
    # Extract subsets of features: feature_set_1 and feature_set_2
    feature_set_1 = X[:, :3]
    feature_set_2 = X[:, 3:6]
    # Bayesian Classification for Feature Set 1
    # Fit the model and make predictions
    means1, covariances1, priors1, classes1 = fit_bayesian_classifier(feature_set_1, y)
    y_pred1 = predict_bayesian_classifier(feature_set_1, means1, covariances1, priors1, classes1)
    # Visualize the actual and predicted classes, compute errors, and display confusion matrix
    visualize_true_labels(feature_set_1, y, ['x1', 'y1', 'z1'], classes1, marker_shapes)
    visualize_classification_results(feature_set_1, y, y_pred1, ['x1', 'y1', 'z1'], classes1, marker_shapes)
    compute_and_display_errors(y, y_pred1, "Feature Set 1")
    generate_and_display_confusion_matrix(y, y_pred1, classes1, "Feature Set 1")
    # Bayesian Classification for Feature Set 2
    # Fit the model and make predictions
    means2, covariances2, priors2, classes2 = fit_bayesian_classifier(feature_set_2, y)
    y_pred2 = predict_bayesian_classifier(feature_set_2, means2, covariances2, priors2, classes2)
    # Visualize the actual and predicted classes, compute errors, and display confusion matrix
    visualize_true_labels(feature_set_2, y, ['x2', 'y2', 'z2'], classes2, marker_shapes)
    visualize_classification_results(feature_set_2, y, y_pred2, ['x2', 'y2', 'z2'], classes2, marker_shapes)
    compute_and_display_errors(y, y_pred2, "Feature Set 2")
    generate_and_display_confusion_matrix(y, y_pred2, classes2, "Feature Set 2")


if __name__ == "__main__":
    main()
