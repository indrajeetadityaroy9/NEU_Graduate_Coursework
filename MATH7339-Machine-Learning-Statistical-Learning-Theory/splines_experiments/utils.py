import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns


def generate_sinusoidal_data(n_samples: int = 100,
                            noise_std: float = 0.3,
                            x_range=(0, 10),
                            frequency: float = 1.0,
                            random_state=None):
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
                            x_range=(-1, 1),
                            coefficients=None,
                            random_state=None):
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
                      x_range=(0, 1),
                      random_state=None):
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
                               random_state=None):
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
                   knots=None,
                   y_true_func=None,
                   title: str = "Spline Fit",
                   figsize=(10, 6),
                   show_knots: bool = True):
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
                        knots=None,
                        title: str = "Basis Functions",
                        figsize=(12, 6),
                        max_functions: int = 10):
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
                             y_true_func=None,
                             title: str = "Smoothing Parameter Comparison",
                             figsize=(12, 6)):
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
                  figsize=(10, 4)):
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
                 best_lambda=None,
                 title: str = "Cross-Validation Curve",
                 figsize=(10, 6)):
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




def classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          threshold: float = 0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': roc_auc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  label: str = 'Model', title: str = 'ROC Curve',
                  figsize=(8, 6)):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, linewidth=2, label=f'{label} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                label: str = 'Model',
                                title: str = 'Precision-Recall Curve',
                                figsize=(8, 6)):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(recall, precision, linewidth=2, label=f'{label} (AUC = {pr_auc:.3f})')

    prevalence = np.mean(y_true)
    ax.axhline(prevalence, color='k', linestyle='--', linewidth=1,
              label=f'Baseline (prevalence = {prevalence:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return fig


def plot_calibration_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                          n_bins: int = 10, title: str = 'Calibration Curve',
                          figsize=(8, 6)):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_counts = np.bincount(bin_indices, minlength=n_bins)

    bin_freq = np.divide(bin_sums, bin_counts, where=bin_counts > 0)
    bin_freq[bin_counts == 0] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(bin_centers, bin_freq, 'o-', linewidth=2, markersize=8,
           label='Model calibration')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         labels=None,
                         title: str = 'Confusion Matrix',
                         figsize=(8, 6)):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=labels, yticklabels=labels)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)

    return fig


def plot_partial_dependence(model, x_train: np.ndarray, feature_idx: int,
                           feature_name: str = 'Feature',
                           n_points: int = 100,
                           figsize=(10, 6)):
    feature_values = x_train[:, feature_idx]
    grid = np.linspace(feature_values.min(), feature_values.max(), n_points)

    pd_values = []

    for val in grid:
        x_partial = x_train.copy()
        x_partial[:, feature_idx] = val

        preds = model.predict(x_partial)
        pd_values.append(np.mean(preds))

    pd_values = np.array(pd_values)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(grid, pd_values, linewidth=2.5, color='steelblue')

    ax.plot(feature_values, np.zeros_like(feature_values), '|', color='gray',
           alpha=0.3, markersize=10)

    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('Partial Dependence', fontsize=12)
    ax.set_title(f'Partial Dependence: {feature_name}', fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig


def plot_qq_plot(residuals: np.ndarray, title: str = 'Q-Q Plot',
                figsize=(8, 6)):
    from scipy import stats

    fig, ax = plt.subplots(figsize=figsize)

    stats.probplot(residuals, dist="norm", plot=ax)

    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    return fig
