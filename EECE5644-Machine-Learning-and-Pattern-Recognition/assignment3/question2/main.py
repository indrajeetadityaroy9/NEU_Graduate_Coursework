import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from question2.utility import generateData, countOptimalGaussians
from question2.visualization import plot1, plot2, plot3


def main():
    # Define GMM parameters
    p0, p1, p2, p3 = 0.3, 0.2, 0.15, 0.35  # Priors for each Gaussian component
    m0, m1, m2, m3 = [2, 2], [3, 2], [6, 6], [-2, -2]  # Mean vectors
    c0, c1, c2, c3 = [[1, 0.4], [0.4, 1]], [[0.7, -0.3], [-0.3, 0.7]], [[1.5, 0.5], [0.5, 1.5]], [[0.8, -0.2], [-0.2,
                                                                                                                0.8]]  # Covariance matrices

    # Assembling the GMM parameters into a dictionary
    gmmParameters = {
        'priors': np.array([p0, p1, p2, p3]),  # Array of priors
        'meanVectors': np.array([m0, m1, m2, m3]),  # Array of mean vectors
        'covMatrices': np.array([c0, c1, c2, c3])  # Array of covariance matrices
    }

    # List of different dataset sizes to generate
    N_train = [10, 100, 1000]
    X_train = []  # To store the generated data points
    Y_train = []  # To store the corresponding labels

    # Generate and visualize datasets for each specified size
    for N_i in N_train:
        X_i, y_i = generateData(N_i, gmmParameters)  # Generate data with N_i samples
        X_train.append(X_i)  # Append generated data to list
        Y_train.append(y_i)  # Append generated labels to list
        plot1(X_i.T, y_i)  # Visualize the generated data

    # Set the number of runs for the GMM optimization process
    numRuns = 100
    max_numGauss = 10
    counts = []  # To store the counts of selected Gaussian components

    # Determine the optimal number of Gaussian components for each dataset
    for i, samples in enumerate(X_train):
        # Count the optimal number of Gaussians for the current dataset over multiple runs
        count = countOptimalGaussians(samples, 10, numRuns)
        counts.append(count)  # Append the count array to the list
        print(f"Counts for N={N_train[i]}: {counts[-1]}")  # Print the counts for the current dataset size

    # Calculate the selection rates from the counts
    selection_rates = [np.array(cnt) / numRuns for cnt in counts]

    # Define labels and parameters for the bar charts
    dataset_sizes = ['N=10', 'N=100', 'N=1000']  # Labels for different dataset sizes
    num_gaussians = range(1, 11)  # Gaussian component counts (1 to 10)
    bar_width = 0.25  # Width of each bar in the bar chart
    index = np.arange(len(num_gaussians))  # Indices for the x-axis of the bar chart

    # Plot the selection rates and counts
    plot2(selection_rates, num_gaussians, dataset_sizes)  # Plot selection rates
    plot3(counts, num_gaussians, dataset_sizes)  # Plot counts


if __name__ == "__main__":
    main()
