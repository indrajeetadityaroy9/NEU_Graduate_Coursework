import numpy as np
from scipy.optimize import minimize
from question3.utility import calculate_distance, evaluate_map_objective_grid, compute_map_objective
from question3.visualization import plot


def main():
    # Set the true position of the vehicle. In this case, it's fixed at (0.5, 0.5).
    x_t, y_t = 0.5, 0.5

    # Define the standard deviations for the Gaussian priors (sigma_x, sigma_y) and the measurement noise (sigma_noise).
    sigma_x, sigma_y, sigma_noise = 0.25, 0.25, 0.3

    # Create a grid of x and y points for evaluating the MAP objective function. This grid covers the area of interest.
    x_points = np.arange(-2, 2.05, 0.05)
    y_points = np.arange(-2, 2.05, 0.05)
    grid_x, grid_y = np.meshgrid(x_points, y_points)

    # Loop through different numbers of landmarks (K). This demonstrates how the solution changes with varying numbers of landmarks.
    for K in [1, 2, 3, 4]:
        # Calculate the positions of K evenly spaced landmarks on a unit circle centered at the origin.
        theta = np.linspace(0, 2 * np.pi, K + 1)[:-1]
        landmarks_x = np.cos(theta)
        landmarks_y = np.sin(theta)

        # Initialize an array to store the range measurements to each landmark.
        measurements = []

        # For each landmark, calculate the true distance and add Gaussian noise to simulate the range measurement.
        for i in range(K):
            true_distance = calculate_distance(x_t, y_t, landmarks_x[i], landmarks_y[i])
            measured_range = np.random.normal(true_distance, sigma_noise)
            # Ensure that the noisy measurement is positive, as negative distances are not physically meaningful.
            while measured_range < 0:
                measured_range = np.random.normal(true_distance, sigma_noise)
            measurements.append(measured_range)

        # Evaluate the MAP objective function on the entire grid. This step calculates the objective function value for each point in the grid.
        objective_values = evaluate_map_objective_grid(grid_x, grid_y, landmarks_x, landmarks_y, measurements, sigma_x,
                                                       sigma_y, sigma_noise)

        # Using minimize function to find the vehicle position that minimizes the MAP objective function providing the MAP estimate.
        optimization_result = minimize(compute_map_objective, [0, 0],
                                       args=(landmarks_x, landmarks_y, measurements, sigma_x, sigma_y, sigma_noise))
        estimated_position = optimization_result.x

        # Plot the results, showing the MAP objective function, true position, estimated position, and landmarks.
        plot(grid_x, grid_y, objective_values, estimated_position, landmarks_x, landmarks_y, K)


if __name__ == '__main__':
    main()
