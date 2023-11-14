import numpy as np


# Function to calculate the Euclidean distance between the vehicle and a landmark
def calculate_distance(vehicle_x, vehicle_y, landmark_x, landmark_y):
    # Euclidean distance formula
    return np.sqrt((vehicle_x - landmark_x) ** 2 + (vehicle_y - landmark_y) ** 2)


# Function to compute the MAP objective for a given vehicle position
def compute_map_objective(vehicle_position, landmarks_x, landmarks_y, measurements, sigma_x, sigma_y, sigma_noise):
    vehicle_x, vehicle_y = vehicle_position
    # Prior probability based on vehicle position (Gaussian distribution)
    prior = (vehicle_x ** 2 / sigma_x ** 2) + (vehicle_y ** 2 / sigma_y ** 2)
    likelihood = 0
    # Compute likelihood by summing over all landmarks
    for k in range(len(landmarks_x)):
        measured_range = measurements[k]
        true_range = calculate_distance(vehicle_x, vehicle_y, landmarks_x[k], landmarks_y[k])
        # Update likelihood based on the difference between measured and true ranges
        likelihood += ((measured_range - true_range) ** 2) / sigma_noise ** 2
    # Sum of prior and likelihood
    return likelihood + prior


# Function to evaluate the MAP objective function over a grid of vehicle positions
def evaluate_map_objective_grid(grid_x, grid_y, landmarks_x, landmarks_y, measurements, sigma_x, sigma_y, sigma_noise):
    objective_values = np.zeros_like(grid_x)
    # Iterate over grid points
    for i in range(grid_x.shape[0]):
        for j in range(grid_y.shape[1]):
            vehicle_x, vehicle_y = grid_x[i, j], grid_y[i, j]
            # Compute MAP objective for each grid point
            objective_values[i, j] = compute_map_objective([vehicle_x, vehicle_y], landmarks_x, landmarks_y,
                                                           measurements, sigma_x, sigma_y, sigma_noise)
    return objective_values
