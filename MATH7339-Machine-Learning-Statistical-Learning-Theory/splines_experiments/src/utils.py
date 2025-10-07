import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple


def generate_sinusoidal_data(n_samples: int = 100,
                            noise_std: float = 0.3,
                            x_range: Tuple[float, float] = (0, 10),
                            frequency: float = 1.0,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    x = np.linspace(x_range[0], x_range[1], n_samples)
    y_true = np.sin(2 * np.pi * frequency * x / (x_range[1] - x_range[0]))
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    return x, y


def generate_polynomial_data(n_samples: int = 100,
                            degree: int = 3,
                            noise_std: float = 0.3,
                            x_range: Tuple[float, float] = (-1, 1),
                            coefficients: Optional[np.ndarray] = None,
                            random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    x = np.linspace(x_range[0], x_range[1], n_samples)

    if coefficients is None:
        coefficients = np.random.randn(degree + 1)

    y_true = np.polyval(coefficients[::-1], x)
    noise = np.random.normal(0, noise_std, n_samples)
    y = y_true + noise

    return x, y


def generate_step_data(n_samples: int = 100,
                      n_steps: int = 4,
                      noise_std: float = 0.2,
                      x_range: Tuple[float, float] = (0, 1),
                      random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    x = np.linspace(x_range[0], x_range[1], n_samples)
    y = np.zeros(n_samples)

    step_edges = np.linspace(x_range[0], x_range[1], n_steps + 1)
    step_values = np.random.randn(n_steps)

    for i in range(n_steps):
        mask = (x >= step_edges[i]) & (x < step_edges[i + 1])
        y[mask] = step_values[i]

    y += np.random.normal(0, noise_std, n_samples)

    return x, y


def generate_discontinuous_data(n_samples: int = 100,
                               noise_std: float = 0.2,
                               random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if random_state is not None:
        np.random.seed(random_state)

    x = np.linspace(0, 1, n_samples)
    y = np.where(x < 0.5, x, x + 2.0)
    y += np.random.normal(0, noise_std, n_samples)

    return x, y
def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mean_squared_error(y_true, y_pred))


def effective_degrees_of_freedom(smoother_matrix: np.ndarray) -> float:
    return np.trace(smoother_matrix)
def plot_spline_fit(x_train: np.ndarray,
                   y_train: np.ndarray,
                   x_test: np.ndarray,
                   y_pred: np.ndarray,
                   knots: Optional[np.ndarray] = None,
                   y_true_func: Optional[Callable] = None,
                   title: str = "Spline Fit",
                   figsize: Tuple[int, int] = (10, 6),
                   show_knots: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x_train, y_train, alpha=0.5, s=30, label='Training data', color='gray')

    ax.plot(x_test, y_pred, 'b-', linewidth=2, label='Spline fit')

    if y_true_func is not None:
        y_true = y_true_func(x_test)
        ax.plot(x_test, y_true, 'g--', linewidth=2, alpha=0.7, label='True function')

    if knots is not None and show_knots:
        for knot in knots:
            ax.axvline(knot, color='r', linestyle='--', alpha=0.3, linewidth=1)
        ax.plot([], [], 'r--', alpha=0.3, label=f'Knots (n={len(knots)})')

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_basis_functions(x: np.ndarray,
                        basis_matrix: np.ndarray,
                        knots: Optional[np.ndarray] = None,
                        title: str = "Basis Functions",
                        figsize: Tuple[int, int] = (12, 6),
                        max_functions: int = 10) -> plt.Figure:
    n_basis = basis_matrix.shape[1]
    n_to_plot = min(n_basis, max_functions)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(n_to_plot):
        ax.plot(x, basis_matrix[:, i], label=f'g_{i+1}(x)', alpha=0.7)

    if knots is not None:
        for knot in knots:
            ax.axvline(knot, color='red', linestyle='--', alpha=0.2, linewidth=1)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Basis function value', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_smoothing_comparison(x_train: np.ndarray,
                             y_train: np.ndarray,
                             x_test: np.ndarray,
                             predictions_dict: dict,
                             y_true_func: Optional[Callable] = None,
                             title: str = "Smoothing Parameter Comparison",
                             figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x_train, y_train, alpha=0.4, s=20, label='Training data', color='gray')

    if y_true_func is not None:
        y_true = y_true_func(x_test)
        ax.plot(x_test, y_true, 'k--', linewidth=2, alpha=0.5, label='True function')

    colors = plt.cm.viridis(np.linspace(0, 1, len(predictions_dict)))
    for (label, y_pred), color in zip(predictions_dict.items(), colors):
        ax.plot(x_test, y_pred, linewidth=2, label=label, color=color)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_residuals(x: np.ndarray,
                  y_true: np.ndarray,
                  y_pred: np.ndarray,
                  title: str = "Residual Plot",
                  figsize: Tuple[int, int] = (10, 4)) -> plt.Figure:
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x, residuals, alpha=0.6, s=30)
    ax.axhline(0, color='r', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig


def plot_cv_curve(lambdas: np.ndarray,
                 cv_errors: np.ndarray,
                 best_lambda: Optional[float] = None,
                 title: str = "Cross-Validation Curve",
                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(lambdas, cv_errors, 'o-', linewidth=2, markersize=8)

    if best_lambda is not None:
        ax.axvline(best_lambda, color='r', linestyle='--', linewidth=2,
                  label=f'Best λ = {best_lambda:.4f}')
        ax.legend()

    ax.set_xlabel('λ (smoothing parameter)', fontsize=12)
    ax.set_ylabel('Cross-validation error (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    return fig
