import pandas as pd
import numpy as np
from question3.evaluation import fit_bayesian_classifier, predict_bayesian_classifier
from question3.utility import load_and_select_features, compute_and_display_errors
from question3.visualization import set_axis_limits, visualize_true_labels, visualize_classification_results, \
    generate_and_display_confusion_matrix


def main():
    filename = 'winequality-white.csv'
    # Define feature combinations to be used for classification
    selected_feature_combinations = [
        ("citric acid", "total sulfur dioxide", "density"),
        ("alcohol", "pH", "residual sugar")
    ]
    # Define marker shapes for visualization
    marker_shapes = ['o', 's', '^', 'v', 'D', '*', '+', 'x', 'p', '<', '>']
    # Loop through each feature combination
    for combo in selected_feature_combinations:
        # Load and select the features from the dataset
        x, y = load_and_select_features(filename, combo)
        # Fit the Bayesian classifier and get parameters (means, covariances, priors, classes)
        means, covariances, priors, classes = fit_bayesian_classifier(x, y)
        # Make predictions using the fitted Bayesian classifier
        predictions = predict_bayesian_classifier(x, means, covariances, priors, classes)
        # Visualize the true class labels
        visualize_true_labels(x, y, combo, classes, marker_shapes)
        # Visualize the classification results
        visualize_classification_results(x, y, predictions, combo, classes, marker_shapes)
        # Compute and display the error metrics
        compute_and_display_errors(y, predictions, combo)
        # Generate and display the confusion matrix
        generate_and_display_confusion_matrix(y, predictions, classes, combo)


if __name__ == "__main__":
    main()
