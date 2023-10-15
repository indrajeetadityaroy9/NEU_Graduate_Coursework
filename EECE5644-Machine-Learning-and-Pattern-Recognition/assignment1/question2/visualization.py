import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_data(samples, labels):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')  # Set the background color to white
    fig.patch.set_facecolor('white')
    # Plot samples for each class with different colors and markers
    ax.scatter(samples[labels == 1][:, 0], samples[labels == 1][:, 1], samples[labels == 1][:, 2], c='r', marker='o', label='Class 1')
    ax.scatter(samples[labels == 2][:, 0], samples[labels == 2][:, 1], samples[labels == 2][:, 2], c='b', marker='s', label='Class 2')
    ax.scatter(samples[labels == 3][:, 0], samples[labels == 3][:, 1], samples[labels == 3][:, 2], c='g', marker='^', label='Class 3')
    # Label the axes
    ax.set_xlabel('Dimension 1 X')
    ax.set_ylabel('Dimension 2 Y')
    ax.set_zlabel('Dimension 3 Z')
    ax.legend()
    plt.show()


def visualize_3d(samples, true_labels, predicted_labels):
    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    # Set of marker shapes
    marker_shapes = '.o^s'
    L = np.unique(true_labels)  # Get unique labels
    # Loop through each class and decision to plot
    for r in L:
        for c in L:
            ind_rc = np.argwhere((predicted_labels == r) & (true_labels == c)).flatten()
            color = 'g' if r == c else 'r'
            label = f"D = {r} | L = {c}"
            # Plot the samples
            for i in ind_rc:
                x, y, z = samples[i]
                marker = marker_shapes[r - 1]
                ax.plot([x], [y], [z], marker=marker, color=color, markerfacecolor='none', linestyle='None', label=label)
                label = None  # Ensure that label is not repeated in the legend
    # Label the axes
    ax.set_xlabel('Dimension 1 X')
    ax.set_ylabel('Dimension 2 Y')
    ax.set_zlabel('Dimension 3 Z')
    ax.legend()
    plt.show()


def plot_confusion_heatmaps(true_labels, predicted_labels):
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # Plot the raw confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    # Set axis labels
    unique_labels = np.unique(true_labels)
    plt.xticks(np.arange(len(unique_labels)) + 0.5, labels=unique_labels, rotation=0)
    plt.yticks(np.arange(len(unique_labels)) + 0.5, labels=unique_labels, rotation=0)
    plt.xlabel("Decision (D)")
    plt.ylabel("True Class Label (L)")
    plt.show()
    # Calculate and print normalized confusion matrix
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    normalized_decision_conf_matrix = conf_matrix / row_sums
    # Plot the normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(normalized_decision_conf_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Decision (D)")
    plt.ylabel("True Class Label (L)")
    plt.xticks(ticks=np.arange(len(unique_labels)) + 0.5, labels=unique_labels, ha='center')
    plt.yticks(ticks=np.arange(len(unique_labels)) + 0.5, labels=unique_labels, va='center')
    plt.show()
