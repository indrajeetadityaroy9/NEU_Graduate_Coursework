import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneOut, KFold
from tqdm import tqdm


def generateData(N, gmmParameters):
    # Determine the dimensionality of the data from the mean vectors
    n = gmmParameters['meanVectors'].shape[1]

    # Initialize arrays to store generated data points (X) and their corresponding labels
    X = np.zeros([N, n])
    labels = np.zeros(N)

    # Generate uniform random values to assign data points to different Gaussian components
    u = np.random.rand(N)

    # Compute the cumulative sum of the GMM priors to create intervals for component selection
    thresholds = np.cumsum(gmmParameters['priors'])
    thresholds = np.insert(thresholds, 0, 0)  # Insert a zero at the start for interval creation

    # Create an array of component indices
    L = np.array(range(len(gmmParameters['priors'])))

    # Loop over each Gaussian component
    for l in L:
        # Identify data points that belong to the current component
        indices = np.argwhere((thresholds[l] <= u) & (u <= thresholds[l + 1]))[:, 0]
        N_labels = len(indices)  # Number of data points for this component

        # Assign labels to these data points
        labels[indices] = l * np.ones(N_labels)

        # Generate data points for this component using its mean vector and covariance matrix
        X[indices, :] = multivariate_normal.rvs(gmmParameters['meanVectors'][l], gmmParameters['covMatrices'][l], N_labels)

    return X, labels  # Return the generated data and their labels


# Function to calculate the log likelihood of the validation data for a given number of Gaussian components
def calculateLogLikelihoodForGMM(training_data, validation_data, gaussians_count):
    # Fit a Gaussian Mixture Model to the training data
    gaussian_model = GaussianMixture(n_components=gaussians_count, n_init=15, init_params='random').fit(training_data)
    # Calculate the log likelihood of the validation data under the fitted model
    log_likelihood = gaussian_model.score(validation_data)
    return log_likelihood


# Function for performing K-Fold Cross-Validation to find the optimal number of Gaussian components
def performKFoldGMMOptimization(dataset, K):
    # Choose Leave-One-Out CV for small datasets or K-Fold CV for larger datasets
    if dataset.shape[0] < K:
        cross_validation_method = LeaveOneOut()
        K = dataset.shape[0]  # Set the number of folds to the dataset size in LOO CV
    else:
        cross_validation_method = KFold(n_splits=K, shuffle=True)

    optimal_num_gaussians_per_fold = np.zeros(K)

    # Iterate over each fold
    for fold_index, (train_indices, val_indices) in enumerate(cross_validation_method.split(dataset)):
        # Split the dataset into training and validation sets for the current fold
        training_set, validation_set = dataset[train_indices], dataset[val_indices]

        best_likelihood = -np.inf  # Initialize the best likelihood as negative infinity
        optimal_num_gaussians = 0  # Initialize the optimal number of Gaussian components

        # Maximum number of Gaussian components to test
        max_gaussians_to_test = min(10, len(training_set))

        # Evaluate different numbers of Gaussian components
        for gaussians_count in range(1, max_gaussians_to_test + 1):
            likelihood = calculateLogLikelihoodForGMM(training_set, validation_set, gaussians_count)
            if likelihood > best_likelihood:  # Update if the current model is better
                best_likelihood = likelihood
                optimal_num_gaussians = gaussians_count

        optimal_num_gaussians_per_fold[fold_index] = optimal_num_gaussians  # Store the optimal number for the fold

    average_optimal_gaussians = np.mean(optimal_num_gaussians_per_fold)  # Calculate the average optimal number
    average_optimal_gaussians = int(np.round(average_optimal_gaussians))  # Round to the nearest integer
    return average_optimal_gaussians


# Function to conduct multiple runs of K-Fold Cross-Validation and count the optimal number of Gaussians each time
def countOptimalGaussians(dataset, K, iterations):
    optimal_gaussians_counts = []  # List to store the optimal number of Gaussians from each run

    # Progress bar for visual feedback during the runs
    progress_bar = tqdm(total=iterations, desc="Optimizing GMM")

    # Perform multiple runs of K-Fold CV
    for _ in range(iterations):
        optimal_gaussians = performKFoldGMMOptimization(dataset, K)
        optimal_gaussians_counts.append(optimal_gaussians)  # Append the optimal number from the current run
        progress_bar.update(1)  # Update the progress bar

    progress_bar.close()  # Close the progress bar after all runs

    # Count the frequency of each number of optimal Gaussians
    unique_gaussians, frequencies = np.unique(optimal_gaussians_counts, return_counts=True)
    gaussian_frequencies = dict(zip(unique_gaussians, frequencies))

    # Initialize an array to store frequencies for all possible numbers of Gaussians (up to 10)
    frequency_array = np.zeros(10, dtype=int)
    for i in range(1, 11):
        frequency_array[i - 1] = gaussian_frequencies.get(i, 0)  # Populate the array with frequencies

    return frequency_array
