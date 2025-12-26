"""
Statistical Analysis Tools

Provides confidence intervals, significance tests, and aggregation
utilities for rigorous algorithm comparison.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import stats


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval using t-distribution.

    Parameters
    ----------
    data : np.ndarray
        Sample data (must have at least 2 elements)
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns
    -------
    tuple
        (mean, lower_bound, upper_bound)

    Raises
    ------
    ValueError
        If data has fewer than 2 elements
    """
    n = len(data)
    if n < 2:
        raise ValueError(f"Need at least 2 samples for confidence interval, got {n}")

    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)

    # t-critical value for two-tailed test
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

    margin = t_crit * std_err
    return mean, mean - margin, mean + margin


def bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic: str = 'mean',
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Parameters
    ----------
    data : np.ndarray
        Sample data
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    statistic : str
        Statistic to compute ('mean', 'median')
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    stat_func = np.mean if statistic == 'mean' else np.median

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    estimate = stat_func(data)

    return estimate, lower, upper


def paired_ttest(
    data1: np.ndarray,
    data2: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform paired t-test.

    Parameters
    ----------
    data1 : np.ndarray
        First sample (e.g., algorithm A results across runs)
    data2 : np.ndarray
        Second sample (e.g., algorithm B results across runs)
    alternative : str
        'two-sided', 'less', or 'greater'

    Returns
    -------
    dict
        't_statistic': t-test statistic
        'p_value': p-value
        'significant_0.05': whether significant at α=0.05
        'significant_0.01': whether significant at α=0.01
    """
    t_stat, p_value = stats.ttest_rel(data1, data2, alternative=alternative)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01,
    }


def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Parameters
    ----------
    data1 : np.ndarray
        First sample (must have at least 2 elements)
    data2 : np.ndarray
        Second sample (must have same length as data1)

    Returns
    -------
    float
        Cohen's d effect size
        - Small: |d| < 0.2
        - Medium: 0.2 <= |d| < 0.8
        - Large: |d| >= 0.8
        Returns 0.0 if all differences are identical (std=0)

    Raises
    ------
    ValueError
        If data has fewer than 2 elements or lengths don't match
    """
    if len(data1) < 2:
        raise ValueError(f"Need at least 2 samples for Cohen's d, got {len(data1)}")
    if len(data1) != len(data2):
        raise ValueError(f"Arrays must have same length: {len(data1)} vs {len(data2)}")

    diff = data1 - data2
    std = np.std(diff, ddof=1)

    # Handle case where all differences are identical
    if std == 0 or np.isnan(std):
        return 0.0

    return np.mean(diff) / std


def effect_size_interpretation(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def aggregate_runs(
    run_results: List[Dict[str, Any]],
    metrics: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate metrics across multiple runs.

    Parameters
    ----------
    run_results : List[Dict]
        List of result dictionaries from each run
    metrics : List[str]
        Metric names to aggregate

    Returns
    -------
    dict
        For each metric: mean, std, ci_lower, ci_upper, n_runs
    """
    aggregated = {}

    for metric in metrics:
        values = [r[metric] for r in run_results if metric in r]

        if len(values) == 0:
            continue

        values = np.array(values)
        mean, ci_lower, ci_upper = compute_confidence_interval(values)

        aggregated[metric] = {
            'mean': mean,
            'std': np.std(values, ddof=1),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_runs': len(values),
        }

    return aggregated


def compare_algorithms(
    results: Dict[str, List[float]],
    baseline: str,
    metric_name: str = "regret",
    alpha: float = 0.05
) -> Dict[str, Dict[str, Any]]:
    """
    Compare all algorithms against a baseline.

    Parameters
    ----------
    results : Dict[str, List[float]]
        Dictionary mapping algorithm names to lists of metric values
    baseline : str
        Name of baseline algorithm to compare against
    metric_name : str
        Name of the metric (for display)
    alpha : float
        Significance level

    Returns
    -------
    dict
        For each algorithm: comparison statistics vs baseline
    """
    if baseline not in results:
        raise ValueError(f"Baseline '{baseline}' not in results")

    baseline_data = np.array(results[baseline])
    comparisons = {}

    for algo_name, algo_data in results.items():
        if algo_name == baseline:
            continue

        algo_data = np.array(algo_data)

        # Must have same number of runs for paired test
        if len(algo_data) != len(baseline_data):
            continue

        ttest = paired_ttest(baseline_data, algo_data)
        d = cohens_d(baseline_data, algo_data)

        comparisons[algo_name] = {
            'mean_difference': np.mean(baseline_data) - np.mean(algo_data),
            'relative_improvement': (np.mean(baseline_data) - np.mean(algo_data)) / np.mean(baseline_data),
            't_statistic': ttest['t_statistic'],
            'p_value': ttest['p_value'],
            'significant': ttest['p_value'] < alpha,
            'cohens_d': d,
            'effect_size': effect_size_interpretation(d),
            'winner': algo_name if np.mean(algo_data) < np.mean(baseline_data) else baseline,
        }

    return comparisons


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : List[float]
        List of p-values
    alpha : float
        Family-wise error rate

    Returns
    -------
    List[bool]
        Whether each test is significant after correction
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    return [p < corrected_alpha for p in p_values]


def holm_bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> List[bool]:
    """
    Apply Holm-Bonferroni (step-down) correction.

    More powerful than standard Bonferroni while controlling FWER.

    Parameters
    ----------
    p_values : List[float]
        List of p-values
    alpha : float
        Family-wise error rate

    Returns
    -------
    List[bool]
        Whether each test is significant after correction
    """
    n = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]

    significant = np.zeros(n, dtype=bool)

    for i, (idx, pval) in enumerate(zip(sorted_indices, sorted_pvals)):
        threshold = alpha / (n - i)
        if pval >= threshold:
            break
        significant[idx] = True

    return significant.tolist()
