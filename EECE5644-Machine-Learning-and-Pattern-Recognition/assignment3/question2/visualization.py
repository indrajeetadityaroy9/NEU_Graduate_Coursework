import matplotlib.pyplot as plt
import numpy as np


def plot1(x, labels):
    # Setting font sizes for different elements of the plot
    plt.rc('font', size=12)  # Default text sizes
    plt.rc('axes', titlesize=12)  # Font size of the axes titles
    plt.rc('axes', labelsize=12)  # Font size of the x and y labels
    plt.rc('xtick', labelsize=10)  # Font size of the tick labels
    plt.rc('ytick', labelsize=10)  # Font size of the tick labels
    plt.rc('legend', fontsize=10)  # Legend font size
    plt.rc('figure', titlesize=14)  # Font size of the figure title

    # Different markers for different classes
    markers = ['o', '^', 's', 'x']
    fig, ax = plt.subplots(figsize=(8, 8))  # Creating a figure and a set of subplots

    # Plotting data points with different markers for each class
    for i in range(len(np.unique(labels))):
        ax.scatter(x[0, labels == i], x[1, labels == i], alpha=0.5, marker=markers[i], label=f"Class {i}")

    # Setting labels for the x and y axes
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    # Adding title and legend, and adjusting layout
    plt.title("2D Data and True Class Labels")
    plt.legend()
    plt.tight_layout()
    plt.show()  # Displaying the plot


def plot2(selection_rates, num_gaussians, dataset_sizes):
    bar_width = 0.25  # Width of each bar
    index = np.arange(len(num_gaussians))  # Array of indices for the x-axis

    # Creating a figure for the bar plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plotting bars for each dataset size
    for i, rates in enumerate(selection_rates):
        plt.bar(index + i * bar_width, rates, bar_width, label=dataset_sizes[i])

    # Setting labels and ticks for the x-axis
    ax.set_xlabel('Number of Gaussian Components')
    ax.set_ylabel('Selection Rate')
    ax.set_xticks(index + bar_width / 2)  # Positioning x-ticks
    ax.set_xticklabels(num_gaussians)

    # Adding legend and title, and adjusting layout
    ax.legend()
    plt.title('Selection Rate of GMM Orders for Different Dataset Sizes')
    plt.tight_layout()
    plt.show()  # Displaying the plot


def plot3(counts, num_gaussians, dataset_sizes):
    bar_width = 0.25  # Width of each bar
    index = np.arange(len(num_gaussians))  # Array of indices for the x-axis

    # Creating a figure for the bar plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plotting bars for each dataset size
    for i, cnt in enumerate(counts):
        plt.bar(index + i * bar_width, cnt, bar_width, label=dataset_sizes[i])

    # Setting labels and ticks for the x-axis and y-axis
    ax.set_xlabel('Number of Gaussian Components')
    ax.set_ylabel('Selection Count')
    max_count = np.max([np.max(c) for c in counts])  # Determining the maximum count for y-axis scaling
    ax.set_yticks(np.arange(0, max_count + 1, 1))  # Setting y-axis ticks
    ax.set_xticks(index + bar_width / 2)  # Positioning x-ticks
    ax.set_xticklabels(num_gaussians)

    # Adding legend and title, and adjusting layout
    ax.legend()
    plt.title('Count of Selected GMM Orders for Different Dataset Sizes')
    plt.tight_layout()
    plt.show()  # Displaying the plot
