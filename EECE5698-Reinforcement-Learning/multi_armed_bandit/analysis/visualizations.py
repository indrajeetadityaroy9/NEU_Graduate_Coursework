"""
Visualization Tools for Bandit Experiments

Publication-quality plotting functions for regret curves, algorithm
comparisons, and change point analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Publication-quality style settings
STYLE_CONFIG = {
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 100,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
}

# Color palette for algorithms
COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


def set_publication_style() -> None:
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)


def plot_regret_curves(
    results: Dict[str, np.ndarray],
    change_points: Optional[List[int]] = None,
    title: str = "Cumulative Regret",
    xlabel: str = "Timestep",
    ylabel: str = "Cumulative Regret",
    figsize: Tuple[int, int] = (10, 6),
    show_ci: bool = False,
    ci_data: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot cumulative regret curves for multiple algorithms.

    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Dictionary mapping algorithm names to cumulative regret arrays
    change_points : List[int], optional
        Timesteps where changes occurred (shown as vertical lines)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    show_ci : bool
        Whether to show confidence intervals
    ci_data : dict, optional
        Dictionary mapping names to (lower, upper) CI bounds
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (Figure, Axes) objects

    Raises
    ------
    ValueError
        If results is empty or arrays have inconsistent lengths
    """
    if not results:
        raise ValueError("results dict cannot be empty")
    lengths = [len(v) for v in results.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"All regret arrays must have same length, got {lengths}")

    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, regret) in enumerate(results.items()):
        color = COLORS[i % len(COLORS)]
        ax.plot(regret, label=name, color=color)

        if show_ci and ci_data and name in ci_data:
            lower, upper = ci_data[name]
            ax.fill_between(
                range(len(regret)), lower, upper,
                color=color, alpha=0.2
            )

    # Mark change points
    if change_points:
        for cp in change_points:
            ax.axvline(x=cp, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_cumulative_regret(
    regret_by_run: Dict[str, List[np.ndarray]],
    change_points: Optional[List[int]] = None,
    title: str = "Cumulative Regret (Mean ± 95% CI)",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Plot cumulative regret with confidence bands from multiple runs.

    Parameters
    ----------
    regret_by_run : Dict[str, List[np.ndarray]]
        Dictionary mapping algorithm names to lists of regret arrays (one per run)
    change_points : List[int], optional
        Change point timesteps
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (Figure, Axes)

    Raises
    ------
    ValueError
        If regret_by_run is empty or runs have inconsistent lengths
    """
    if not regret_by_run:
        raise ValueError("regret_by_run dict cannot be empty")

    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, runs) in enumerate(regret_by_run.items()):
        color = COLORS[i % len(COLORS)]

        if not runs:
            raise ValueError(f"No runs provided for algorithm '{name}'")

        # Stack runs and compute mean/CI
        runs_array = np.array(runs)
        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)
        n = len(runs)
        ci = 1.96 * std / np.sqrt(n)  # 95% CI

        t = np.arange(len(mean))
        ax.plot(t, mean, label=name, color=color)
        ax.fill_between(t, mean - ci, mean + ci, color=color, alpha=0.2)

    if change_points:
        for cp in change_points:
            ax.axvline(x=cp, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title)
    ax.legend(loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_arm_selection(
    actions: np.ndarray,
    optimal_arms: np.ndarray,
    change_points: Optional[List[int]] = None,
    window: int = 20,
    title: str = "Arm Selection Over Time",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Visualize arm selection pattern over time.

    Parameters
    ----------
    actions : np.ndarray
        Selected actions at each timestep
    optimal_arms : np.ndarray
        Optimal arm at each timestep
    change_points : List[int], optional
        Change point timesteps
    window : int
        Rolling window for optimal selection percentage
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (Figure, Axes)

    Raises
    ------
    ValueError
        If actions and optimal_arms have different lengths
    """
    if len(actions) != len(optimal_arms):
        raise ValueError(
            f"actions and optimal_arms must have same length: "
            f"{len(actions)} vs {len(optimal_arms)}"
        )

    set_publication_style()
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                              gridspec_kw={'height_ratios': [1, 2]})

    T = len(actions)
    t = np.arange(T)

    # Top panel: optimal arm over time
    axes[0].plot(t, optimal_arms, 'k-', linewidth=1, label='Optimal arm')
    axes[0].scatter(t, actions, c=actions == optimal_arms, cmap='RdYlGn',
                    s=2, alpha=0.5)
    axes[0].set_ylabel("Arm")
    axes[0].set_title("Arm Selection (green=optimal)")

    # Bottom panel: rolling optimal percentage
    optimal = (actions == optimal_arms).astype(float)
    rolling_pct = np.convolve(optimal, np.ones(window)/window, mode='valid')
    axes[1].plot(np.arange(len(rolling_pct)) + window//2, rolling_pct * 100,
                 'b-', linewidth=1.5)
    axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel("% Optimal")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylim(0, 105)

    # Mark change points
    if change_points:
        for cp in change_points:
            for ax in axes:
                ax.axvline(x=cp, color='red', linestyle='--', alpha=0.5, linewidth=1)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_algorithm_comparison(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str = "Final Regret",
    title: str = "Algorithm Comparison",
    figsize: Tuple[int, int] = (10, 6),
    horizontal: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Bar chart comparing algorithms on a metric.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, float]]
        Dictionary mapping algorithm names to metric dictionaries
        Each metric dict should have 'mean', 'ci_lower', 'ci_upper'
    metric_name : str
        Name of the metric for labeling
    title : str
        Plot title
    figsize : tuple
        Figure size
    horizontal : bool
        Whether to use horizontal bars
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (Figure, Axes)

    Raises
    ------
    ValueError
        If metrics is empty or missing required keys
    """
    if not metrics:
        raise ValueError("metrics dict cannot be empty")

    required_keys = {'mean', 'ci_lower', 'ci_upper'}
    for name, metric_dict in metrics.items():
        missing = required_keys - set(metric_dict.keys())
        if missing:
            raise ValueError(
                f"Metric dict for '{name}' missing required keys: {missing}"
            )

    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    names = list(metrics.keys())
    means = [metrics[n]['mean'] for n in names]
    errors = [
        [metrics[n]['mean'] - metrics[n]['ci_lower'] for n in names],
        [metrics[n]['ci_upper'] - metrics[n]['mean'] for n in names]
    ]

    y_pos = np.arange(len(names))
    colors = [COLORS[i % len(COLORS)] for i in range(len(names))]

    if horizontal:
        ax.barh(y_pos, means, xerr=errors, color=colors, capsize=4, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel(metric_name)
        ax.invert_yaxis()
    else:
        ax.bar(y_pos, means, yerr=errors, color=colors, capsize=4, alpha=0.8)
        ax.set_xticks(y_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel(metric_name)

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_change_point_analysis(
    adaptation_regrets: Dict[str, List[float]],
    detection_delays: Dict[str, List[float]],
    title: str = "Change Point Adaptation Analysis",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> Tuple[Figure, List[Axes]]:
    """
    Analyze algorithm behavior around change points.

    Parameters
    ----------
    adaptation_regrets : Dict[str, List[float]]
        Adaptation regret for each algorithm at each change point
    detection_delays : Dict[str, List[float]]
        Detection delay for each algorithm at each change point
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (Figure, list of Axes)

    Raises
    ------
    ValueError
        If dicts are empty or have mismatched algorithm names
    """
    if not adaptation_regrets:
        raise ValueError("adaptation_regrets dict cannot be empty")
    if not detection_delays:
        raise ValueError("detection_delays dict cannot be empty")

    adapt_names = set(adaptation_regrets.keys())
    delay_names = set(detection_delays.keys())
    if adapt_names != delay_names:
        raise ValueError(
            f"Algorithm names must match between dicts. "
            f"In adaptation only: {adapt_names - delay_names}, "
            f"In detection only: {delay_names - adapt_names}"
        )

    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    names = list(adaptation_regrets.keys())
    colors = {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}

    # Left: Adaptation regret boxplot
    adapt_data = [adaptation_regrets[n] for n in names]
    bp1 = axes[0].boxplot(adapt_data, labels=names, patch_artist=True)
    for patch, name in zip(bp1['boxes'], names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Adaptation Regret")
    axes[0].set_title("Regret After Change")
    axes[0].tick_params(axis='x', rotation=45)

    # Right: Detection delay boxplot
    delay_data = [detection_delays[n] for n in names]
    bp2 = axes[1].boxplot(delay_data, labels=names, patch_artist=True)
    for patch, name in zip(bp2['boxes'], names):
        patch.set_facecolor(colors[name])
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Detection Delay (steps)")
    axes[1].set_title("Time to Detect Change")
    axes[1].tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str = "Performance Heatmap",
    cmap: str = "RdYlGn_r",
    figsize: Tuple[int, int] = (10, 8),
    annotate: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Figure, Axes]:
    """
    Create a heatmap for algorithm × environment performance.

    Parameters
    ----------
    data : np.ndarray
        2D array of values (rows=algorithms, cols=environments)
    row_labels : List[str]
        Labels for rows (algorithms)
    col_labels : List[str]
        Labels for columns (environments)
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    annotate : bool
        Whether to show values in cells
    save_path : str, optional
        Path to save figure

    Returns
    -------
    tuple
        (Figure, Axes)

    Raises
    ------
    ValueError
        If data dimensions don't match label counts
    """
    if data.shape[0] != len(row_labels):
        raise ValueError(
            f"Number of rows ({data.shape[0]}) must match row_labels ({len(row_labels)})"
        )
    if data.shape[1] != len(col_labels):
        raise ValueError(
            f"Number of columns ({data.shape[1]}) must match col_labels ({len(col_labels)})"
        )

    set_publication_style()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap=cmap, aspect='auto')

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f"{data[i, j]:.0f}",
                               ha="center", va="center", color="black", fontsize=9)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Regret", rotation=-90, va="bottom")

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax
