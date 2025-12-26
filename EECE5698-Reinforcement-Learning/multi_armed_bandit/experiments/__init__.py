"""
Experiment Framework for Bandit Studies

This package provides tools for running reproducible experiments
with multiple algorithms and environments.

Modules:
    - runner: Experiment execution engine (sequential and parallel)
    - gpu_runner: GPU-accelerated batch experiment runner
    - configs/: YAML configuration files for experiments
    - results/: Saved experiment results (gitignored)

Execution Backends:
    - Sequential: run_experiment_suite() - single-threaded, good for debugging
    - Parallel: run_experiment_suite_parallel() - multiprocessing for CPU parallelism
    - GPU: GPUBatchRunner - CuPy-based batch execution on NVIDIA GPUs
"""

from .runner import (
    run_single_experiment,
    run_experiment_suite,
    run_experiment_suite_parallel,
    ExperimentConfig,
    ExperimentResults,
    load_config_from_yaml,
)

# GPU runner is optional (requires CuPy)
try:
    from .gpu_runner import (
        GPUBatchRunner,
        GPUBatchConfig,
        run_gpu_experiment_suite,
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

__all__ = [
    'run_single_experiment',
    'run_experiment_suite',
    'run_experiment_suite_parallel',
    'ExperimentConfig',
    'ExperimentResults',
    'load_config_from_yaml',
    'GPU_AVAILABLE',
]

if GPU_AVAILABLE:
    __all__.extend([
        'GPUBatchRunner',
        'GPUBatchConfig',
        'run_gpu_experiment_suite',
    ])
