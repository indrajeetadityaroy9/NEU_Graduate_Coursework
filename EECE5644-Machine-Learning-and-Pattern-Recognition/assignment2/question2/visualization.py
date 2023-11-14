import numpy as np
import matplotlib.pyplot as plt
import pylab


def plot(X_train, Y_train, analytical_preds):
    plt.rc('font', size=22)
    plt.rc('axes', titlesize=18)
    plt.rc('axes', labelsize=18)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('legend', fontsize=16)
    plt.rc('figure', titlesize=22)
    # Enabling interactive mode
    pylab.ion()
    # Creating a new figure for 3D plotting
    fig = pylab.figure()
    ax = fig.add_subplot(111, projection='3d')  # Adding a 3D subplot
    # Scatter plot of training data and analytical predictions
    ax.scatter(X_train[0, :], X_train[1, :], analytical_preds, marker="o", color="b")  # Plotting predictions
    ax.scatter(X_train[0, :], X_train[1, :], Y_train, marker="x", color="r")  # Plotting actual training data
    # Setting labels for axes
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    # Displaying the plot
    pylab.show()


def plot_mse_comparison(gammas, mse_map, mse_ml):
    plt.figure(figsize=(12, 6))
    # Plotting MSE for MAP estimators on a log scale for x-axis
    plt.semilogx(gammas, mse_map, label='MAP MSE')
    # Adding a horizontal line to represent MSE for ML estimator
    plt.axhline(y=mse_ml, color='r', linestyle='-', label='ML MSE')
    # Identifying and annotating the optimal gamma value for MAP estimators
    optimal_gamma_map = gammas[np.argmin(mse_map)]
    optimal_gamma_power = np.log10(optimal_gamma_map)
    min_mse_map = np.min(mse_map)
    plt.text(optimal_gamma_map, min_mse_map, f' MAP MSE\n  γ=10^{optimal_gamma_power:.1f}\n  MSE={min_mse_map:.3f}', horizontalalignment='left', verticalalignment='bottom')
    # Annotating the MSE for ML estimator
    plt.text(gammas[0], mse_ml, f' ML MSE: {mse_ml:.3f}', horizontalalignment='left', verticalalignment='top')
    # Setting labels and title for the plot
    plt.xlabel('Gamma')
    plt.ylabel('MSE on Validation Set')
    plt.title('Comparison of ML and MAP Estimators Performance')
    # Adding a legend and grid to the plot
    plt.legend()
    plt.grid()
    # Displaying the plot
    plt.show()
