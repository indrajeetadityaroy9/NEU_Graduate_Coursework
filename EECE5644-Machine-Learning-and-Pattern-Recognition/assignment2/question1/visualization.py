import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from math import floor, ceil


def plot_generated_data(datasets):
    for i, (X, labels) in enumerate(datasets):
        plt.figure(figsize=(5, 5))  # Create a new figure for each dataset

        # Plotting points from different classes
        class_colors = ['blue', 'red']  # Assigning colors to classes
        class_markers = ['o', 'x']  # Assigning markers to classes
        for class_idx in range(2):  # Looping through classes (0 and 1)
            class_points = X[labels == class_idx]  # Filtering points belonging to the current class
            plt.scatter(class_points[:, 0], class_points[:, 1],
                        color=class_colors[class_idx], marker=class_markers[class_idx],
                        alpha=0.5, label=f'Class {class_idx}')  # Plotting the class points

        plt.xlabel('x1')  # Setting the label for the x-axis
        plt.ylabel('x2')  # Setting the label for the y-axis
        plt.legend()  # Displaying a legend to identify classes
        plt.grid(True)  # Adding a grid for better readability
        # Setting axis limits based on the range of data
        x1_lim = (floor(np.min(X[:, 0])), ceil(np.max(X[:, 0])))
        x2_lim = (floor(np.min(X[:, 1])), ceil(np.max(X[:, 1])))
        plt.xlim(x1_lim)
        plt.ylim(x2_lim)
        plt.tight_layout()  # Adjusting the layout
        plt.show()  # Displaying the plot


def plot_roc_curve(fpr, tpr, theoretical_point=None, empirical_point=None):
    plt.figure()
    # Plotting the ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
    # Plotting the theoretical minimum error point if provided
    if theoretical_point:
        plt.scatter(*theoretical_point, color='red', marker='o', label='Theoretical Min-P(error)')
    # Plotting the empirical minimum error point if provided
    if empirical_point:
        plt.scatter(*empirical_point, color='blue', marker='x', label='Empirical Min-P(error)')
    plt.xlim([0.0, 1.0])  # Setting x-axis limits for the ROC curve
    plt.ylim([0.0, 1.05])  # Setting y-axis limits for the ROC curve
    plt.xlabel('False Positive Probability P(D=1|L=0)')  # Label for the x-axis
    plt.ylabel('True Positive Probability P(D=1|L=1)')  # Label for the y-axis
    plt.grid()  # Adding a grid for better readability
    plt.legend(loc="lower right")  # Displaying a legend with the position in the lower right
    plt.show()  # Displaying the plot
