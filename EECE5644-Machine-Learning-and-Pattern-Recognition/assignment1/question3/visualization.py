import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Function to set axis limits for 3D plots
def set_axis_limits(ax, x_combo):
    ax.set_xlim(x_combo[:, 0].min(), x_combo[:, 0].max())
    ax.set_ylim(x_combo[:, 1].min(), x_combo[:, 1].max())
    ax.set_zlim(x_combo[:, 2].min(), x_combo[:, 2].max())


# Function to visualize true class labels
def visualize_true_labels(x, y, features, classes, marker_shapes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(color='white')
    for idx, cls in enumerate(classes):
        mask = (y == cls)
        ax.scatter(x[mask, 0], x[mask, 1], x[mask, 2], marker=marker_shapes[idx], label=f'Class {cls}')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.legend(loc='upper right', numpoints=1)
    plt.title(f'True Classes for Features: {features}')
    plt.show()


# Function to visualize classification results
def visualize_classification_results(x, y, predictions, features, classes, marker_shapes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(color='white')
    for idx, cls in enumerate(classes):
        correct_mask = (y == cls) & (y == predictions)
        incorrect_mask = (y == cls) & (y != predictions)
        ax.scatter(x[correct_mask, 0], x[correct_mask, 1], x[correct_mask, 2], c='g', marker=marker_shapes[idx], label=f'Class {cls} Correct')
        ax.scatter(x[incorrect_mask, 0], x[incorrect_mask, 1], x[incorrect_mask, 2], c='r', marker=marker_shapes[idx], label=f'Class {cls} Incorrect')
    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_zlabel(features[2])
    ax.legend(loc='upper right', numpoints=1)
    plt.title(f'Classification for Features: {features}')
    plt.show()


# Function to generate and display confusion matrix
def generate_and_display_confusion_matrix(y, predictions, classes, features):
    cm = confusion_matrix(y, predictions, labels=classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False)
    plt.title(f'Confusion Matrix for {features}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# Function to plot normalized confusion matrix
def plot_normalized_confusion_matrix(y, y_pred, feature_set_name):
    cm = confusion_matrix(y, y_pred, labels=range(1, 7))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    error_rate = 1 - np.trace(cm_normalized) / len(cm_normalized)
    print(f"0-1 Loss for {feature_set_name}: {error_rate:.4f}")
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, xticklabels=range(1, 7), yticklabels=range(1, 7))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
