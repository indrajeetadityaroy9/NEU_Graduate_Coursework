import numpy as np


# Calculate binary metrics for True Positive Rate (TPR) and False Positive Rate (FPR)
def binary_metrics(predictions, true_classes):
    tp = np.sum((predictions == 1) & (true_classes == 1))  # True Positives
    tn = np.sum((predictions == 0) & (true_classes == 0))  # True Negatives
    fp = np.sum((predictions == 1) & (true_classes == 0))  # False Positives
    fn = np.sum((predictions == 0) & (true_classes == 1))  # False Negatives
    return {'TPR': tp / (tp + fn), 'FPR': fp / (tn + fp)}


# Create ROC curve data and probabilities associated with decision thresholds
def roc_curve_and_probabilities(discriminant, labels):
    fpr, tpr, P_D0_L1, P_D1_L0 = [], [], [], []
    for t in np.sort(discriminant):
        metrics = binary_metrics(discriminant > t, labels)
        fpr.append(metrics['FPR'])  # False Positive Rate
        tpr.append(metrics['TPR'])  # True Positive Rate
        P_D0_L1.append(1 - metrics['TPR'])  # False Negative Rate
        P_D1_L0.append(metrics['FPR'])  # Redundant with fpr, could be optimized away
    return fpr, tpr, P_D0_L1, P_D1_L0


# Find the decision threshold that minimizes the empirical error rate
def empirical_min_error_threshold(discriminant, priors, P_D0_L1, P_D1_L0):
    min_error = 1
    empirical_gamma = 0
    for i, t in enumerate(np.sort(discriminant)):
        error = P_D1_L0[i] * priors[0] + P_D0_L1[i] * priors[1]
        if error < min_error:
            min_error = error
            empirical_gamma = t
    return empirical_gamma, min_error


# Calculate the error rate given a threshold
def error_rate(threshold, discriminant, labels, priors):
    metrics = binary_metrics(discriminant > threshold, labels)
    return metrics['FPR'] * priors[0] + (1 - metrics['TPR']) * priors[1]


# Calculate the optimal decision threshold that minimizes error
def optimal_threshold(discriminant, labels, priors):
    thresholds = np.linspace(np.min(discriminant), np.max(discriminant), 100)
    errors = [error_rate(t, discriminant, labels, priors) for t in thresholds]
    return thresholds[np.argmin(errors)], min(errors)


# Calculate the minimum error theoretically given a threshold
def theoretical_min_error(discriminant, labels, priors, gamma):
    metrics = binary_metrics(discriminant > gamma, labels)
    return metrics['FPR'] * priors[0] + (1 - metrics['TPR']) * priors[1]
