import numpy as np
from question2.utility import generate_data
from question2.visualization import plot_data


def main():
    # Set the prior probabilities for each class
    prior_probs = np.array([0.3, 0.3, 0.4])
    # Define the loss matrix for classification decisions
    loss_matrix = np.array([[0, 1, 1],
                            [1, 0, 1],
                            [1, 1, 0]])
    # Covariance matrices for the Gaussians
    cov_vals = np.array([[[0.5, 0, 1], [0, 2, 0], [0, 0.5, 0.5]],
                         [[1, 0, 1], [0, 1, 0], [0, 0.5, 1]],
                         [[3, 0, 0], [0, 1, 0], [0, 0, 0.5]],
                         [[1, 0, 1], [0, 1, 0], [0, 0, 2]]])
    # Extract standard deviations from the covariance matrices
    all_std_devs = np.sqrt(np.diagonal(cov_vals, axis1=1, axis2=2))
    # Calculate the average standard deviation for each covariance matrix
    avg_std_dev_per_matrix = np.mean(all_std_devs, axis=1)
    # Calculate a global average standard deviation
    global_avg_std_dev = np.mean(avg_std_dev_per_matrix)
    # Determine the distance between Gaussian means based on the global average std. dev.
    dist = 2.5 * global_avg_std_dev
    # Define the mean for the first Gaussian
    mean_1 = np.array([0, 0, 0])
    # Define means for other Gaussians relative to the first, using the calculated distance
    mean_2 = mean_1 + np.array([dist, 0, 0])
    mean_3 = mean_1 + np.array([0, dist, 0])
    mean_4 = mean_1 + np.array([0, 0, dist])
    mean_vals = np.array([mean_1, mean_2, mean_3, mean_4])
    # Set the number of samples
    N = 10000
    # Generate data samples using the defined parameters
    samples, true_labels = generate_data(N, prior_probs, mean_vals, cov_vals)
    # Visualize the generated data in 3D
    plot_data(samples, true_labels)


if __name__ == "__main__":
    main()
