import numpy as np
import matplotlib.pyplot as plt
import pylab


def plot(X_train, y_train, analytical_preds):
    plt.rc('font', size=22)  # controls default text sizes
    plt.rc('axes', titlesize=18)  # fontsize of the axes title
    plt.rc('axes', labelsize=18)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
    plt.rc('legend', fontsize=16)  # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title

    pylab.ion()
    fig = pylab.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[0, :], X_train[1, :], analytical_preds, marker="o", color="b")
    ax.scatter(X_train[0, :], X_train[1, :], y_train, marker="x", color="r")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    pylab.show()


def plot_mse_comparison(gammas, mse_map, mse_ml):
    plt.figure(figsize=(12, 6))
    plt.semilogx(gammas, mse_map, label='MAP MSE')
    plt.axhline(y=mse_ml, color='r', linestyle='-', label='ML MSE')

    optimal_gamma_map = gammas[np.argmin(mse_map)]
    optimal_gamma_power = np.log10(optimal_gamma_map)
    min_mse_map = np.min(mse_map)
    plt.text(optimal_gamma_map, min_mse_map, f' MAP MSE\n  γ=10^{optimal_gamma_power:.1f}\n  MSE={min_mse_map:.3f}',
             horizontalalignment='left', verticalalignment='bottom')

    plt.text(gammas[0], mse_ml, f' ML MSE: {mse_ml:.3f}',
             horizontalalignment='left', verticalalignment='top')

    plt.xlabel('Gamma')
    plt.ylabel('MSE on Validation Set')
    plt.title('Comparison of ML and MAP Estimators Performance')
    plt.legend()
    plt.grid()
    plt.show()
