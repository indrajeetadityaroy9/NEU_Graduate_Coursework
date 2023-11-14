import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from math import floor, ceil


def plot_generated_data(datasets):
    for i, (X, labels) in enumerate(datasets):
        plt.figure(figsize=(5, 5))  # Create a new figure for each dataset

        # Plotting points from different classes
        class_colors = ['blue', 'red']
        class_markers = ['o', 'x']
        for class_idx in range(2):  # Assuming two classes (0 and 1)
            class_points = X[labels == class_idx]
            plt.scatter(class_points[:, 0], class_points[:, 1],
                        color=class_colors[class_idx], marker=class_markers[class_idx],
                        alpha=0.5, label=f'Class {class_idx}')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        x1_lim = (floor(np.min(X[:, 0])), ceil(np.max(X[:, 0])))
        x2_lim = (floor(np.min(X[:, 1])), ceil(np.max(X[:, 1])))
        plt.xlim(x1_lim)
        plt.ylim(x2_lim)
        plt.tight_layout()
        plt.show()


def plot_roc_curve(fpr, tpr, theoretical_point=None, empirical_point=None):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    if theoretical_point:
        plt.scatter(*theoretical_point, color='red', marker='o', label='Theoretical Min-P(error)')
    if empirical_point:
        plt.scatter(*empirical_point, color='blue', marker='x', label='Empirical Min-P(error)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Probability P(D=1|L=0)')
    plt.ylabel('True Positive Probability P(D=1|L=1)')
    plt.grid()
    plt.legend(loc="lower right")
    plt.show()
