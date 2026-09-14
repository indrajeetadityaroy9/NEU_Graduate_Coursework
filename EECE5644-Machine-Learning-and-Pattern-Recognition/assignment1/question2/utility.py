import numpy as np


def generate_data(N, prior_probs, mean_vals, cov_vals):
    # Random number generator from numpy
    rng = np.random.default_rng()
    # Initialize arrays to store generated samples and corresponding labels
    samples = np.zeros((N, 3))
    labels = np.zeros(N, dtype=int)
    sample_count = 0  # Counter to keep track of number of samples generated so far
    # Loop through the first two classes to generate samples
    for i in range(2):  # Generating samples for Class 1 and Class 2
        # Calculate the number of samples for the current class based on prior probabilities
        num_samples = int(N * prior_probs[i])
        # Generate samples for the current class using multivariate normal distribution
        samples[sample_count:sample_count + num_samples, :] = rng.multivariate_normal(mean=mean_vals[i], cov=cov_vals[i], size=num_samples)
        # Assign corresponding labels for the generated samples
        labels[sample_count:sample_count + num_samples] = i + 1
        # Update the sample counter
        sample_count += num_samples

    # For the third class, generate samples as a mixture of the last two Gaussians
    num_samples_class3 = N - sample_count  # Calculate the remaining number of samples to be generated for Class 3
    # Loop through the remaining sample positions to generate samples for Class 3
    for i in range(sample_count, N):
        # Randomly select between the third and fourth Gaussian for sample generation
        gaussian_choice = rng.choice([2, 3])  # Select either the third or fourth Gaussian
        # Generate a sample for Class 3 using the selected Gaussian
        samples[i, :] = rng.multivariate_normal(mean=mean_vals[gaussian_choice], cov=cov_vals[gaussian_choice])
        # Assign the label for the third class
        labels[i] = 3

    # Return the generated samples and corresponding labels
    return samples, labels
