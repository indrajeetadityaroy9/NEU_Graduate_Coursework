# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Experiments

```bash
# Full GPU study (13 algorithms × 4 environments × 50 runs) - recommended
python -m multi_armed_bandit.scripts.run_gpu_study

# Sequential CPU experiments
python -m multi_armed_bandit.scripts.run_all_experiments

# Quick test (10 runs instead of 50)
python -m multi_armed_bandit.scripts.run_all_experiments --quick

# Run specific experiment
python -m multi_armed_bandit.scripts.run_all_experiments --experiment abrupt_200

# Available experiments: stationary, abrupt_100, abrupt_200, abrupt_500, drift

# Benchmarks (OBP, replay, supervised-to-bandit)
python -m multi_armed_bandit.scripts.run_benchmarks

# Ablation study (hyperparameter sensitivity)
python -m multi_armed_bandit.scripts.run_ablation_study
```

## Dependencies

```bash
pip install numpy pandas scipy pyyaml tqdm

# For GPU acceleration (optional)
pip install cupy-cuda12x

# Backend control via environment variables
MAB_FORCE_CPU=1      # Force CPU even if GPU available
MAB_GPU_DEVICE=0     # Select GPU device
```

## Architecture

### Modular Framework

The codebase uses a registry pattern. Algorithms and environments are registered in `ALGORITHMS` and `ENVIRONMENTS` dicts in their respective `__init__.py` files for YAML configuration loading.

**algorithms/**: All extend `BanditAlgorithm` base class with required methods `select_action()`, `update()`, `_initialize()`
- `epsilon_greedy.py`: EpsilonGreedy, EpsilonGreedyConstant, DecayingEpsilonGreedy
- `ucb.py`: UCB1, DiscountedUCB, SlidingWindowUCB
- `thompson_sampling.py`: ThompsonSampling, DiscountedThompsonSampling, BetaThompsonSampling
- `gradient_bandit.py`: GradientBandit, EntropyRegularizedGradient
- `exp3.py`: EXP3, Rexp3, EXP3IX (adversarial)

**environments/**: All extend `BanditEnvironment` base class with `pull(arm)`, `step()`, `get_optimal_arm()`
- `stationary.py`: StationaryBandit
- `abrupt_change.py`: AbruptChangeBandit (sudden optimal arm changes), RotatingBandit
- `gradual_drift.py`: GradualDriftBandit, RandomWalkBandit, LinearDriftBandit

**experiments/**: Framework for reproducible experiments
- `runner.py`: ExperimentConfig, run_experiment_suite(), run_experiment_suite_parallel()
- `gpu_runner.py`: GPUBatchRunner for GPU-accelerated batch execution
- `configs/`: YAML experiment configurations

**analysis/**: Metrics and statistics
- `metrics.py`: MetricsTracker (pre-allocated arrays), regret computation
- `statistics.py`: Confidence intervals, statistical tests

**backends/**: GPU/CPU backend abstraction
- `__init__.py`: Provides unified `xp` interface (CuPy or NumPy)

### Execution Backends

```python
from multi_armed_bandit.experiments import (
    run_experiment_suite,          # Sequential (single-threaded)
    run_experiment_suite_parallel, # CPU parallel (multiprocessing)
    GPUBatchRunner,                # GPU batch execution (CuPy)
)

# Parallel execution across CPU cores
results = run_experiment_suite_parallel(config, n_workers=32)

# GPU batch execution
from multi_armed_bandit.experiments import GPUBatchConfig, GPUBatchRunner
gpu_config = GPUBatchConfig(n_runs=100, n_arms=5, horizon=10000)
runner = GPUBatchRunner(gpu_config)
results = runner.run_epsilon_greedy(epsilon=0.1, alpha=0.1, arm_means=[0,1,0,0,0])
```

### Example Programmatic Usage

```python
from multi_armed_bandit.algorithms import DiscountedUCB
from multi_armed_bandit.environments import AbruptChangeBandit
from multi_armed_bandit.experiments import run_single_experiment

env = AbruptChangeBandit(n_arms=5, change_interval=100, gap=1.0, seed=42)
algo = DiscountedUCB(n_arms=5, gamma=0.99, seed=42)
tracker = run_single_experiment(algo, env, horizon=1000)
metrics = tracker.compute_all_metrics(change_points=env.change_points)
```

### Key Metrics

- **Cumulative Regret**: Σ(μ* - r_t)
- **Optimal Action %**: Fraction of times best arm selected
- **Adaptation Regret**: Regret in window after change points
- **Detection Delay**: Steps until optimal arm selected post-change

### Performance Optimizations

- Vectorized Thompson Sampling posterior computation
- O(T) rolling window metrics (cumsum-based instead of O(T×W) loop)
- Pre-allocated MetricsTracker arrays
- Consolidated epsilon-greedy selection logic
- GPU-first design with CuPy backend (graceful NumPy fallback)
