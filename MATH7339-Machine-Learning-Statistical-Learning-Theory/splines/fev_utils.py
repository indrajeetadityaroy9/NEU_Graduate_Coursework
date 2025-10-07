"""
Common utility functions for FEV spline analysis
Shared across all implementation files (scipy, statsmodels, pyGAM, csaps)
"""

import pandas as pd
import matplotlib.pyplot as plt


def load_data():
    """Load the FEV dataset"""
    df = pd.read_csv('fev.csv')
    return df


def plot_model(X, y, X_plot=None, y_pred=None, title='', filename='',
               curve_label='', curve_color='r', curve_style='-',
               knots_X=None, knots_y=None, show_data_label=True):
    """
    Common plotting function for all models

    Parameters:
    - X, y: Original data points
    - X_plot, y_pred: Fitted curve (optional)
    - title: Plot title
    - filename: Output filename
    - curve_label: Label for the fitted curve
    - curve_color: Color for the fitted curve
    - curve_style: Line style for the fitted curve
    - knots_X, knots_y: Knot positions (optional)
    - show_data_label: Whether to show 'Data' in legend
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of data
    data_label = 'Data' if show_data_label else None
    plt.scatter(X, y, alpha=0.5, edgecolors='k', facecolors='none', label=data_label)

    # Plot fitted curve if provided
    if X_plot is not None and y_pred is not None:
        plt.plot(X_plot, y_pred, color=curve_color, linestyle=curve_style,
                linewidth=2, label=curve_label)

    # Plot knots if provided
    if knots_X is not None and knots_y is not None:
        plt.scatter(knots_X, knots_y, color='red', s=100, zorder=5,
                   label=f'Knots (n={len(knots_X)})')

    plt.xlabel('Height', fontsize=12)
    plt.ylabel('FEV', fontsize=12)
    plt.title(title, fontsize=14)
    if curve_label or show_data_label or (knots_X is not None):
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{filename}'")

    plt.show()
