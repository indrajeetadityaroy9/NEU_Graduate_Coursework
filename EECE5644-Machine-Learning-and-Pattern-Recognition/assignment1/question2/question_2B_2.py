import numpy as np
from question2.evaluation import bayesian_classifier, risk
from question2.utility import generate_data
from question2.visualization import plot_data, visualize_3d, plot_confusion_heatmaps


def main():
    # Define prior probabilities for each class
    prior_probs = np.array([0.3, 0.3, 0.4])
    # Define the loss matrix with heavier penalty (100 units) for certain misclassifications
    loss_matrix_100 = np.array([[0, 100, 100],[1, 0, 100],[1, 1, 0]])
    # Define covariance matrices for the Gaussians
    cov_vals = np.array([[[0.5, 0, 1], [0, 2, 0], [0, 0.5, 0.5]],
                         [[1, 0, 1], [0, 1, 0], [0, 0.5, 1]],
                         [[3, 0, 0], [0, 1, 0], [0, 0, 0.5]],
                         [[1, 0, 1], [0, 1, 0], [0, 0, 2]]])
    # Extract standard deviations from the covariance matrices
    all_std_devs = np.sqrt(np.diagonal(cov_vals, axis1=1, axis2=2))
    # Compute the average standard deviation for each covariance matrix
    avg_std_dev_per_matrix = np.mean(all_std_devs, axis=1)
    # Compute a global average standard deviation across all matrices
    global_avg_std_dev = np.mean(avg_std_dev_per_matrix)
    # Determine the distance between Gaussian means based on the global average std. dev.
    dist = 2.5 * global_avg_std_dev  # This factor can vary between 2 and 3 based on desired separation
    # Define the mean for the first Gaussian
    mean_1 = np.array([0, 0, 0])
    # Determine means for other Gaussians relative to the first, using the calculated distance
    mean_2 = mean_1 + np.array([dist, 0, 0])
    mean_3 = mean_1 + np.array([0, dist, 0])
    mean_4 = mean_1 + np.array([0, 0, dist])
    # Combine means into a single array for ease of processing
    mean_vals = np.array([mean_1, mean_2, mean_3, mean_4])
    # Set the total number of samples
    N = 10000
    # Generate data samples based on the specified parameters
    samples, true_labels = generate_data(N, prior_probs, mean_vals, cov_vals)
    # Use the Bayesian classifier to predict class labels for the generated samples
    decisions_100 = bayesian_classifier(samples, prior_probs, mean_vals, cov_vals, loss_matrix_100)
    # Visualize true labels vs. predicted labels in a 3D plot
    visualize_3d(samples, true_labels, decisions_100)
    # Display confusion matrix as heatmaps
    plot_confusion_heatmaps(true_labels,  decisions_100)
    # Calculate the minimum expected risk based on the decisions and the true labels using the loss matrix
    risk_100 = risk(decisions_100, true_labels, loss_matrix_100)
    print(risk_100)


if __name__ == "__main__":
    main()