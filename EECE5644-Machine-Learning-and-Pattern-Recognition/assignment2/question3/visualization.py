import matplotlib.pyplot as plt


def plot(grid_x, grid_y, objective_values, estimated_position, landmarks_x, landmarks_y, num_landmarks):
    # Set up the figure size for the plot
    plt.figure(figsize=(10, 8))
    # Define a high number of contour levels for smooth transitions in the contour plot
    contour_levels = 100
    # Create a filled contour plot with the specified number of levels and color map
    contour_plot = plt.contourf(grid_x, grid_y, objective_values, levels=contour_levels, cmap='plasma')
    # Add contour lines on top of the filled contour to enhance visibility
    plt.contour(grid_x, grid_y, objective_values, levels=contour_plot.levels, colors='k', linewidths=0.5, linestyles='solid')
    # Add a color bar to the plot, labeling it appropriately
    plt.colorbar(contour_plot, label='MAP estimation objective function value')
    # Plot the true position of the vehicle as a red '+' marker
    plt.scatter([0.5], [0.5], color='red', marker='+', s=100, label='True Position')
    # Plot the MAP estimate as a white 'x' marker
    plt.scatter([estimated_position[0]], [estimated_position[1]], color='white', marker='x', s=100, label='Estimated Position')
    # Plot the landmark positions as cyan 'o' markers
    plt.scatter(landmarks_x, landmarks_y, color='cyan', marker='o', s=100, label='Landmarks')
    # Set the title of the plot, indicating the number of landmarks
    plt.title(f'MAP objective function contours (Num Landmarks={num_landmarks})')
    # Label the x and y axes
    plt.xlabel('x')
    plt.ylabel('y')
    # Add a legend to the plot
    plt.legend()
    # Enable grid for better readability
    plt.grid(True)
    # Display the plot
    plt.show()
