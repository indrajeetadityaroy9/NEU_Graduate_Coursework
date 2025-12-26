"""
Experiment Runner

Provides tools for running reproducible bandit experiments with
multiple algorithms, environments, and configurations.
"""

from typing import Dict, List, Any, Optional, Callable, Type, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import time
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ..algorithms.base import BanditAlgorithm
from ..environments.base import BanditEnvironment
from ..analysis.metrics import MetricsTracker, compute_cumulative_regret


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    horizon: int
    n_runs: int
    n_arms: int
    seed: Optional[int] = None
    save_trajectories: bool = False

    # Environment configuration
    env_class: Optional[Type[BanditEnvironment]] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Algorithm configurations (list of dicts)
    algorithms: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            'name': self.name,
            'horizon': self.horizon,
            'n_runs': self.n_runs,
            'n_arms': self.n_arms,
            'seed': self.seed,
            'save_trajectories': self.save_trajectories,
            'env_class': self.env_class.__name__ if self.env_class else None,
            'env_kwargs': self.env_kwargs,
            'algorithms': [
                {
                    'name': a.get('name', a['class'].__name__),
                    'class': a['class'].__name__,
                    'kwargs': a.get('kwargs', {})
                }
                for a in self.algorithms
            ]
        }


@dataclass
class ExperimentResults:
    """Container for experiment results."""
    config: ExperimentConfig
    algorithm_names: List[str]

    # Per-run results: results[algo_name][run_idx]
    cumulative_regret: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    final_regret: Dict[str, List[float]] = field(default_factory=dict)
    optimal_percentage: Dict[str, List[float]] = field(default_factory=dict)

    # Aggregated results
    change_points: List[int] = field(default_factory=list)
    adaptation_regret: Dict[str, List[float]] = field(default_factory=dict)
    detection_delay: Dict[str, List[float]] = field(default_factory=dict)

    # Timing
    wall_time: float = 0.0
    per_algo_time: Dict[str, float] = field(default_factory=dict)

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        def convert_numpy(obj):
            """Recursively convert numpy types to Python natives."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj

        data = {
            'config': self.config.to_dict(),
            'algorithm_names': self.algorithm_names,
            'final_regret': convert_numpy(self.final_regret),
            'optimal_percentage': convert_numpy(self.optimal_percentage),
            'change_points': convert_numpy(self.change_points),
            'adaptation_regret': convert_numpy(self.adaptation_regret),
            'detection_delay': convert_numpy(self.detection_delay),
            'wall_time': float(self.wall_time),
            'per_algo_time': convert_numpy(self.per_algo_time),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for each algorithm."""
        summary = {}
        for algo in self.algorithm_names:
            regrets = np.array(self.final_regret[algo])
            opt_pcts = np.array(self.optimal_percentage[algo])

            summary[algo] = {
                'mean_regret': float(np.mean(regrets)),
                'std_regret': float(np.std(regrets)),
                'mean_optimal_pct': float(np.mean(opt_pcts)),
                'mean_adaptation_regret': float(np.mean(self.adaptation_regret.get(algo, [0]))),
                'mean_detection_delay': float(np.mean(self.detection_delay.get(algo, [0]))),
            }
        return summary


def run_single_experiment(
    algorithm: BanditAlgorithm,
    environment: BanditEnvironment,
    horizon: int,
    tracker: Optional[MetricsTracker] = None
) -> MetricsTracker:
    """
    Run a single bandit experiment.

    Parameters
    ----------
    algorithm : BanditAlgorithm
        Algorithm to evaluate
    environment : BanditEnvironment
        Environment to run in
    horizon : int
        Number of timesteps
    tracker : MetricsTracker, optional
        Tracker to use (creates new one if None)

    Returns
    -------
    MetricsTracker
        Tracker with recorded data
    """
    if tracker is None:
        tracker = MetricsTracker()

    tracker.reset()

    for t in range(horizon):
        # Record optimal arm/value before action (for non-stationary envs)
        optimal_arm = environment.get_optimal_arm()
        optimal_value = environment.get_optimal_value()

        # Agent acts
        action = algorithm.select_action()
        reward = environment.pull(action)

        # Update agent
        algorithm.update(action, reward)

        # Record
        tracker.record(
            action=action,
            reward=reward,
            optimal_arm=optimal_arm,
            optimal_value=optimal_value
        )

        # Advance environment time (may trigger changes)
        environment.step()

    return tracker


def run_experiment_suite(
    config: ExperimentConfig,
    progress_bar: bool = True,
    verbose: bool = False
) -> ExperimentResults:
    """
    Run a full experiment suite with multiple algorithms and runs.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    progress_bar : bool
        Whether to show progress bar
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    ExperimentResults
        Complete results object

    Raises
    ------
    ValueError
        If env_class is None or no algorithms are configured
    """
    start_time = time.time()

    logger.info(
        "Starting '%s': %d runs × %d algorithms",
        config.name, config.n_runs, len(config.algorithms)
    )

    # Validate configuration
    if config.env_class is None:
        logger.error("env_class is None for experiment '%s'", config.name)
        raise ValueError("env_class must be specified in config (check environment type exists)")
    if not config.algorithms:
        logger.error("No algorithms configured for experiment '%s'", config.name)
        raise ValueError("At least one algorithm must be configured")
    if config.n_arms < 1:
        logger.error("Invalid n_arms=%d for experiment '%s'", config.n_arms, config.name)
        raise ValueError(f"n_arms must be positive, got {config.n_arms}")
    if config.horizon < 1:
        logger.error("Invalid horizon=%d for experiment '%s'", config.horizon, config.name)
        raise ValueError(f"horizon must be positive, got {config.horizon}")

    # Initialize results
    algorithm_names = [a.get('name', a['class'].__name__) for a in config.algorithms]
    results = ExperimentResults(
        config=config,
        algorithm_names=algorithm_names,
    )

    # Initialize result containers
    for name in algorithm_names:
        results.cumulative_regret[name] = []
        results.final_regret[name] = []
        results.optimal_percentage[name] = []
        results.adaptation_regret[name] = []
        results.detection_delay[name] = []
        results.per_algo_time[name] = 0.0

    # Set up random seeds
    base_seed = config.seed if config.seed is not None else 42
    rng = np.random.default_rng(base_seed)
    run_seeds = rng.integers(0, 2**31, size=config.n_runs)

    # Run experiments
    total_iterations = config.n_runs * len(config.algorithms)
    pbar = tqdm(total=total_iterations, disable=not progress_bar, desc=config.name)

    for run_idx in range(config.n_runs):
        run_seed = int(run_seeds[run_idx])

        # Create environment for this run
        env = config.env_class(
            n_arms=config.n_arms,
            seed=run_seed,
            **config.env_kwargs
        )

        for algo_config in config.algorithms:
            algo_name = algo_config.get('name', algo_config['class'].__name__)
            logger.debug("Run %d/%d, algorithm: %s", run_idx + 1, config.n_runs, algo_name)

            # Create algorithm
            algo_seed = run_seed + hash(algo_name) % (2**31)
            algo = algo_config['class'](
                n_arms=config.n_arms,
                seed=algo_seed,
                **algo_config.get('kwargs', {})
            )

            # Reset environment for each algorithm
            env.reset()

            # Run experiment
            algo_start = time.time()
            tracker = run_single_experiment(
                algorithm=algo,
                environment=env,
                horizon=config.horizon
            )
            results.per_algo_time[algo_name] += time.time() - algo_start

            # Compute metrics
            cum_regret = compute_cumulative_regret(
                tracker.rewards, tracker.optimal_values
            )
            results.cumulative_regret[algo_name].append(cum_regret)
            results.final_regret[algo_name].append(float(cum_regret[-1]))

            optimal_pct = np.mean(tracker.actions == tracker.optimal_arms)
            results.optimal_percentage[algo_name].append(float(optimal_pct))

            # Store change points (same for all algorithms)
            if run_idx == 0 and len(results.change_points) == 0:
                results.change_points = env.change_points

            # Compute adaptation metrics
            if len(env.change_points) > 0:
                metrics = tracker.compute_all_metrics(
                    change_points=env.change_points,
                    adaptation_window=50
                )
                results.adaptation_regret[algo_name].append(
                    metrics['adaptation']['mean']
                )
                results.detection_delay[algo_name].append(
                    metrics['detection']['mean']
                )

            pbar.update(1)

    pbar.close()
    results.wall_time = time.time() - start_time

    logger.info("Completed '%s' in %.1fs", config.name, results.wall_time)

    if verbose:
        print(f"\nExperiment '{config.name}' completed in {results.wall_time:.1f}s")
        summary = results.get_summary()
        for algo, stats in summary.items():
            print(f"  {algo}: Regret={stats['mean_regret']:.1f} "
                  f"(±{stats['std_regret']:.1f}), "
                  f"Optimal={stats['mean_optimal_pct']*100:.1f}%")

    return results


def load_config_from_yaml(path: str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.

    Parameters
    ----------
    path : str
        Path to YAML config file

    Returns
    -------
    ExperimentConfig
        Loaded configuration
    """
    import yaml

    from ..algorithms import ALGORITHMS
    from ..environments import ENVIRONMENTS

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Resolve environment class
    env_name = data.get('environment', {}).get('type', 'stationary')
    env_class = ENVIRONMENTS.get(env_name)
    env_kwargs = data.get('environment', {}).get('kwargs', {})

    # Resolve algorithm classes
    algorithms = []
    skipped_algorithms = []
    for algo_data in data.get('algorithms', []):
        algo_name = algo_data.get('type')
        algo_class = ALGORITHMS.get(algo_name)
        if algo_class:
            algorithms.append({
                'name': algo_data.get('name', algo_name),
                'class': algo_class,
                'kwargs': algo_data.get('kwargs', {})
            })
        else:
            skipped_algorithms.append(algo_name)

    # Warn about unknown algorithms
    if skipped_algorithms:
        logger.warning(
            "Unknown algorithms in config (skipped): %s. Available: %s",
            skipped_algorithms, list(ALGORITHMS.keys())
        )

    # Validate environment was found
    if env_class is None:
        logger.error(
            "Unknown environment type '%s' in config. Available: %s",
            env_name, list(ENVIRONMENTS.keys())
        )
        raise ValueError(
            f"Unknown environment type: '{env_name}'. "
            f"Available environments: {list(ENVIRONMENTS.keys())}"
        )

    return ExperimentConfig(
        name=data.get('name', 'experiment'),
        horizon=data.get('horizon', 1000),
        n_runs=data.get('n_runs', 10),
        n_arms=data.get('n_arms', 10),
        seed=data.get('seed'),
        save_trajectories=data.get('save_trajectories', False),
        env_class=env_class,
        env_kwargs=env_kwargs,
        algorithms=algorithms,
    )


def _run_single_trial(args: Tuple) -> Tuple[str, int, dict]:
    """
    Worker function for parallel experiment execution.

    Parameters
    ----------
    args : tuple
        (run_idx, algo_config, env_class, env_kwargs, n_arms, horizon, run_seed)

    Returns
    -------
    tuple
        (algo_name, run_idx, metrics_dict)
    """
    (run_idx, algo_config, env_class, env_kwargs, n_arms, horizon, run_seed) = args

    algo_name = algo_config.get('name', algo_config['class'].__name__)

    # Create environment
    env = env_class(n_arms=n_arms, seed=run_seed, **env_kwargs)

    # Create algorithm with unique seed
    algo_seed = run_seed + hash(algo_name) % (2**31)
    algo = algo_config['class'](
        n_arms=n_arms,
        seed=algo_seed,
        **algo_config.get('kwargs', {})
    )

    # Run experiment
    tracker = MetricsTracker(horizon=horizon)
    tracker = run_single_experiment(algo, env, horizon, tracker)

    # Compute metrics
    cum_regret = compute_cumulative_regret(tracker.rewards, tracker.optimal_values)

    return (algo_name, run_idx, {
        'cumulative_regret': cum_regret,
        'final_regret': float(cum_regret[-1]),
        'optimal_percentage': float(np.mean(tracker.actions == tracker.optimal_arms)),
        'change_points': env.change_points,
        'metrics': tracker.compute_all_metrics(env.change_points, 50) if env.change_points else None,
    })


def run_experiment_suite_parallel(
    config: ExperimentConfig,
    n_workers: Optional[int] = None,
    progress_bar: bool = True,
    verbose: bool = False
) -> ExperimentResults:
    """
    Run experiment suite with parallel execution across CPU cores.

    Uses ProcessPoolExecutor for true parallelism, bypassing GIL.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    n_workers : int, optional
        Number of worker processes. Defaults to min(cpu_count, n_tasks).
    progress_bar : bool
        Whether to show progress bar
    verbose : bool
        Whether to print verbose output

    Returns
    -------
    ExperimentResults
        Complete results object
    """
    start_time = time.time()

    # Determine number of workers
    n_tasks = config.n_runs * len(config.algorithms)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, n_tasks)

    logger.info(
        "Starting parallel '%s': %d runs × %d algorithms = %d tasks on %d workers",
        config.name, config.n_runs, len(config.algorithms), n_tasks, n_workers
    )

    # Validate configuration
    if config.env_class is None:
        raise ValueError("env_class must be specified")
    if not config.algorithms:
        raise ValueError("At least one algorithm must be configured")

    # Initialize results
    algorithm_names = [a.get('name', a['class'].__name__) for a in config.algorithms]
    results = ExperimentResults(
        config=config,
        algorithm_names=algorithm_names,
    )

    # Initialize containers
    for name in algorithm_names:
        results.cumulative_regret[name] = [None] * config.n_runs
        results.final_regret[name] = [0.0] * config.n_runs
        results.optimal_percentage[name] = [0.0] * config.n_runs
        results.adaptation_regret[name] = []
        results.detection_delay[name] = []
        results.per_algo_time[name] = 0.0

    # Generate seeds
    base_seed = config.seed if config.seed is not None else 42
    rng = np.random.default_rng(base_seed)
    run_seeds = rng.integers(0, 2**31, size=config.n_runs)

    # Build work items
    work_items = []
    for run_idx in range(config.n_runs):
        for algo_config in config.algorithms:
            work_items.append((
                run_idx,
                algo_config,
                config.env_class,
                config.env_kwargs,
                config.n_arms,
                config.horizon,
                int(run_seeds[run_idx])
            ))

    # Execute in parallel
    completed = 0
    pbar = tqdm(total=n_tasks, disable=not progress_bar, desc=f"{config.name} (parallel)")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_single_trial, item): item for item in work_items}

        for future in as_completed(futures):
            try:
                algo_name, run_idx, metrics = future.result()

                # Store results
                results.cumulative_regret[algo_name][run_idx] = metrics['cumulative_regret']
                results.final_regret[algo_name][run_idx] = metrics['final_regret']
                results.optimal_percentage[algo_name][run_idx] = metrics['optimal_percentage']

                # Store change points from first result
                if len(results.change_points) == 0 and metrics['change_points']:
                    results.change_points = metrics['change_points']

                # Store adaptation metrics
                if metrics['metrics'] is not None:
                    results.adaptation_regret[algo_name].append(
                        metrics['metrics']['adaptation']['mean']
                    )
                    results.detection_delay[algo_name].append(
                        metrics['metrics']['detection']['mean']
                    )

            except Exception as e:
                logger.error("Worker failed: %s", e)
                raise

            completed += 1
            pbar.update(1)

    pbar.close()
    results.wall_time = time.time() - start_time

    logger.info(
        "Completed parallel '%s' in %.1fs (%.1f tasks/s)",
        config.name, results.wall_time, n_tasks / results.wall_time
    )

    if verbose:
        print(f"\nExperiment '{config.name}' completed in {results.wall_time:.1f}s")
        print(f"  Throughput: {n_tasks / results.wall_time:.1f} tasks/s")
        summary = results.get_summary()
        for algo, stats in summary.items():
            print(f"  {algo}: Regret={stats['mean_regret']:.1f}, "
                  f"Optimal={stats['mean_optimal_pct']*100:.1f}%")

    return results
