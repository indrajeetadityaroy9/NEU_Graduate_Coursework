# Non-Stationary Multi-Armed Bandit Study

**Research Title**: "Adaptive Exploration Under Distribution Shift: A Comparative Study of Bandit Algorithms"

This module provides a comprehensive framework for studying bandit algorithm performance under non-stationary reward distributions.

## Research Questions

1. **RQ1**: Which exploration algorithms adapt fastest when the optimal arm changes?
2. **RQ2**: How does change frequency affect algorithm performance rankings?
3. **RQ3**: What algorithm characteristics predict robustness to non-stationarity?
4. **RQ4**: Can we quantify "adaptation regret" as a distinct metric?

---

## Directory Structure

```
multi_armed_bandit/
├── README.md                    # This file
├── CLAUDE.md                   # Claude Code guidance
├── algorithms/                  # Algorithm implementations
│   ├── base.py                 # Abstract BanditAlgorithm class
│   ├── epsilon_greedy.py       # ε-greedy variants
│   ├── ucb.py                  # UCB1, D-UCB, SW-UCB
│   ├── thompson_sampling.py    # TS, Discounted TS
│   ├── gradient_bandit.py      # Softmax policy gradient
│   └── exp3.py                 # Adversarial bandit algorithms
│
├── environments/               # Non-stationary bandit environments
│   ├── base.py                # Abstract BanditEnvironment class
│   ├── stationary.py          # Baseline (no change)
│   ├── abrupt_change.py       # Sudden optimal arm changes
│   └── gradual_drift.py       # Slow continuous drift
│
├── analysis/                   # Statistical analysis tools
│   ├── metrics.py             # Regret, adaptation metrics
│   ├── statistics.py          # Confidence intervals, significance tests
│   └── visualizations.py      # Publication-quality plotting
│
├── experiments/                # Reproducible experiments
│   ├── runner.py              # Experiment execution (sequential & parallel)
│   ├── gpu_runner.py          # GPU-accelerated batch execution
│   ├── configs/               # YAML experiment configurations
│   └── results/               # Saved results (gitignored)
│
├── backends/                   # GPU/CPU backend abstraction
│   └── __init__.py            # Unified xp interface (CuPy/NumPy)
│
├── benchmarks/                 # Standard research benchmarks
│   ├── obp_integration.py     # OBP metrics (DoublyRobust, SNIPS)
│   ├── supervised_to_bandit.py # UCI dataset conversion with drift
│   └── replay_evaluation.py   # Replay method for logged data
│
├── paper/                      # arXiv paper draft
│   ├── main.tex               # Complete LaTeX paper (887 lines)
│   ├── Makefile               # Compilation commands
│   └── README.md              # Paper documentation
│
└── scripts/                    # Entry points
    ├── run_gpu_study.py       # Full GPU-accelerated comparative study
    ├── run_benchmarks.py      # Standard benchmark suite (OBP/Replay)
    ├── run_ablation_study.py  # Hyperparameter sensitivity analysis
    └── run_all_experiments.py # Sequential experiment runner
```

---

## Algorithm Suite

### Stationary Algorithms (Baseline)

| Algorithm | Description | Key Parameter |
|-----------|-------------|---------------|
| ε-Greedy | Random exploration with probability ε | ε = 0.1 |
| UCB1 | Upper Confidence Bound | c = √2 |
| Thompson Sampling | Bayesian posterior sampling | Prior variance |
| Gradient Bandit | Softmax policy gradient | α (step size) |

### Non-Stationary Variants

| Algorithm | Adaptation Mechanism | Key Parameter |
|-----------|---------------------|---------------|
| ε-Greedy (constant α) | Never forgets old data | α = 0.1 |
| Discounted UCB (D-UCB) | Exponential discounting γ^(t-s) | γ ∈ (0.9, 0.999) |
| Sliding Window UCB | Uses only last τ observations | τ = 50-200 |
| Discounted TS | Posterior "forgets" via discount | γ ∈ (0.9, 0.999) |
| EXP3 | Adversarial bandit baseline | γ (exploration) |
| Rexp3 | Periodic restart for changes | restart_interval |

---

## Environment Types

### Stationary (Control)
Arm means remain constant. Used to verify algorithms work correctly.

### Abrupt Change
Optimal arm switches suddenly at fixed intervals. Parameters:
- `change_interval`: Steps between changes (e.g., 100, 500)
- `gap`: Difference between optimal and suboptimal arms

### Gradual Drift
Arm means evolve continuously. Types:
- **Random Walk**: Gaussian perturbations each step
- **Linear Drift**: Constant drift with boundary reflection

---

## Metrics

### Standard Metrics
- **Cumulative Regret**: Σ(μ* - r_t) — overall performance
- **Optimal Action %**: Fraction of optimal selections

### Novel Adaptation Metrics
- **Adaptation Regret**: Regret in window [cp, cp+τ] after change points
- **Detection Delay**: Steps until optimal arm selected post-change
- **Steady-State Regret**: Regret between changes (excluding adaptation)

---

## Quick Start

### Run Full GPU Study (Recommended)
```bash
# From repository root - runs 13 algorithms × 4 environments × 50 runs
python -m multi_armed_bandit.scripts.run_gpu_study
```

Results saved to `experiments/results/full_study_YYYYMMDD_HHMMSS/` containing:
- `raw/` — JSON results per environment
- `plots/` — Visualization PNG files
- `summary/` — CSV summaries and statistical tests
- `README.md` — Auto-generated summary

### Run Sequential Experiments (CPU)
```bash
cd multi_armed_bandit
python scripts/run_all_experiments.py
```

### Quick Test Run
```bash
python scripts/run_all_experiments.py --quick
```

### Run Specific Experiment
```bash
python scripts/run_all_experiments.py --experiment abrupt_200
```

### Available Experiments
- `stationary` — Baseline with no changes
- `abrupt_100` — Changes every 100 steps
- `abrupt_200` — Changes every 200 steps
- `abrupt_500` — Changes every 500 steps
- `drift` — Gradual random walk drift

---

## Example Usage (Programmatic)

```python
from multi_armed_bandit.algorithms import UCB1, DiscountedUCB
from multi_armed_bandit.environments import AbruptChangeBandit
from multi_armed_bandit.experiments import run_single_experiment
from multi_armed_bandit.analysis import MetricsTracker

# Create environment and algorithm
env = AbruptChangeBandit(n_arms=5, change_interval=100, gap=1.0, seed=42)
algo = DiscountedUCB(n_arms=5, gamma=0.99, seed=42)

# Run experiment
tracker = run_single_experiment(algo, env, horizon=1000)

# Analyze results
metrics = tracker.compute_all_metrics(change_points=env.change_points)
print(f"Final regret: {metrics['final_regret']:.1f}")
print(f"Mean adaptation regret: {metrics['adaptation']['mean']:.1f}")
```

---

## Experimental Results

Full comparative study: **13 algorithms × 4 environments × 50 runs × 10,000 timesteps**

### Best Algorithm per Environment

| Environment | Best Algorithm | Mean Regret | Optimal % |
|-------------|----------------|-------------|-----------|
| Stationary | Thompson Sampling | 29.3 | 99.5% |
| Abrupt (100 steps) | D-UCB(0.99) | 2,978.7 | 70.0% |
| Abrupt (500 steps) | D-UCB(0.99) | 1,498.3 | 84.8% |
| Gradual Drift | D-UCB(0.99) | 1,071.7 | 85.7% |

### Overall Algorithm Rankings (Average Rank Across All Environments)

| Rank | Algorithm | Avg Rank | Stationary | Abrupt(100) | Abrupt(500) | Drift |
|------|-----------|----------|------------|-------------|-------------|-------|
| 1 | D-UCB(0.99) | 3.00 | 9 | 1 | 1 | 1 |
| 2 | EpsGreedy-Const | 4.75 | 8 | 6 | 2 | 3 |
| 3 | SW-UCB(100) | 4.75 | 10 | 4 | 3 | 2 |
| 4 | UCB1 | 6.00 | 2 | 5 | 5 | 12 |
| 5 | D-UCB(0.95) | 6.00 | 12 | 2 | 6 | 4 |
| 6 | ThompsonSampling | 8.00 | 1 | 9 | 9 | 13 |

### Key Findings

1. **Stationary environments**: Thompson Sampling dominates (29.3 regret vs UCB1's 52.3)
2. **Non-stationary environments**: D-UCB(0.99) wins across all change types
3. **Stationary algorithms fail on drift**: UCB1 (14,150) and TS (17,247) vs D-UCB (1,072)
4. **Detection delay tradeoff**: D-UCB(0.95) detects changes fastest (2.9 steps) but has worse steady-state

---

## Theoretical Analysis

### The Stability-Plasticity Dilemma

Our results provide a textbook illustration of the fundamental tradeoff in adaptive systems:

| Algorithm | Stationary (Replay) | Gradual Drift | Sudden Drift |
|-----------|---------------------|---------------|--------------|
| ThompsonSampling | **0.885** | 23.0% | 27.8% |
| UCB1 | 0.874 | 40.6% | 70.8% |
| SW-UCB(100) | 0.788 | 75.3% | **83.9%** |
| D-UCB(0.99) | 0.788 | **74.6%** | 83.0% |
| D-UCB(0.95) | 0.676 | 54.3% | 68.2% |

**The Price of Adaptability**: Non-stationary algorithms (SW-UCB, D-UCB) sacrifice ~11% stationary performance (0.788 vs 0.885) to gain ~50% drift performance (75% vs 23%).

### Why Thompson Sampling Fails on Drift

Thompson Sampling "over-converges" in non-stationary environments:
- Posterior distributions become extremely narrow (high confidence) around old means
- When the environment changes, variance is too low to encourage re-exploration
- The algorithm effectively "locks in" to the wrong arm

```
TS on Stationary:  0.885 mean reward (near-optimal)
TS on Gradual:     23.0% optimal rate (near-random!)
```

### Gradual Drift: The "Silent Killer"

Gradual drift is significantly harder than sudden drift:

| Algorithm | Sudden Drift | Gradual Drift | Δ |
|-----------|--------------|---------------|---|
| UCB1 | 70.8% | 40.6% | -30.2% |
| SW-UCB(100) | 82.7% | 71.5% | -11.2% |
| D-UCB(0.99) | 85.2% | 73.7% | -11.5% |

**Why?**
- **Sudden drift**: Sharp reward drop triggers exploration (UCB bound falls below alternatives)
- **Gradual drift**: Slow changes are treated as "noise" rather than "signal" for too long

### UCB1's Surprising Performance on Sudden Drift (70.8%)

This is theoretically explainable through **implicit adaptation via exploration**:
- UCB1's exploration bonus grows for arms not recently pulled
- After sudden change, the new optimal arm (previously suboptimal) has high uncertainty
- UCB1 eventually explores it due to inflated confidence bound
- This provides adaptation without explicit change detection

### Rexp3: Adversarial vs Stochastic

Rexp3 is designed for **adversarial** settings. Its mediocre performance confirms that treating *stochastic* drift as *adversarial* is overly conservative:

| Algorithm | Gradual Drift | Sudden Drift |
|-----------|---------------|--------------|
| Rexp3(100) | 49.9% | 65.0% |
| EXP3 | 24.2% | 21.5% |
| D-UCB(0.99) | 73.7% | 85.2% |

**Insight**: Rexp3 protects against worst-case but fails to exploit the structure of stochastic drift. Periodic restart helps (Rexp3 >> EXP3) but still underperforms algorithms designed for stochastic non-stationarity.

### Theoretical Principles Validated

| Principle | Evidence |
|-----------|----------|
| **No free lunch** | Algorithms optimized for drift sacrifice stationary performance |
| **Effective memory ≈ 20% of change interval** | γ=0.99 (mem≈100) optimal for interval=500 |
| **Bayesian methods need appropriate priors** | TS with stationary prior fails on drift |
| **Exploration enables implicit adaptation** | UCB1 achieves 70.8% on sudden drift via uncertainty bonuses |
| **Adversarial algorithms are conservative** | Rexp3 underperforms stochastic-drift algorithms |

---

### When to Use Each Algorithm

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Stationary | Thompson Sampling | Optimal Bayesian approach (29.3 regret) |
| Frequent changes (100 steps) | D-UCB (γ=0.99) | Best adaptation (2,978 regret) |
| Moderate changes (500 steps) | D-UCB (γ=0.99) | Balance adaptation/stability (1,498 regret) |
| Gradual drift | D-UCB (γ=0.99) | Continuous tracking (1,072 regret) |
| Adversarial/unknown | EXP3 | Robust to worst-case |

### Hyperparameter Sensitivity (Ablation Study)

**Window Size (SW-UCB)** - drift_interval=500:

| Window (τ) | Gradual Drift | Sudden Drift | Notes |
|------------|---------------|--------------|-------|
| 50 | 63.4% | 76.8% | Over-forgetting |
| 100 | 71.5% | **82.7%** | Best for sudden |
| 200 | **74.4%** | 81.9% | Best for gradual |
| 500 | 62.6% | 78.7% | Under-forgetting |

**Discount Factor (D-UCB)** - drift_interval=500:

| γ | Eff. Memory | Gradual | Sudden | Notes |
|---|-------------|---------|--------|-------|
| 0.9 | ~10 | 42.3% | 54.7% | Severe over-forgetting |
| 0.95 | ~20 | 53.9% | 68.4% | Over-forgetting |
| 0.99 | ~100 | **73.7%** | **85.2%** | Optimal |
| 0.999 | ~1000 | 67.4% | 76.7% | Under-forgetting, high variance |

**Key Insight**: Effective memory should be ~20% of change interval for optimal performance.

### Running Ablation Study

```bash
python -m multi_armed_bandit.scripts.run_ablation_study
```

Generates `regret_dynamics.png` and `ablation_study.png` in `experiments/results/ablation/`.

---

## Standard Benchmarks (Research-Grade Evaluation)

Following arXiv research standards, we implement the three-pronged validation approach:

### 1. Open Bandit Pipeline (OBP) Integration

Standardized Off-Policy Evaluation using industry-standard estimators:

```python
from multi_armed_bandit.benchmarks import OBPEvaluator, validate_with_obp

# Validate algorithm run with DR/SNIPS estimators
results = validate_with_obp(actions, rewards, optimal_actions, n_arms=5)
print(f"DR estimate: {results['optimal_value_dr']:.3f}")
print(f"SNIPS estimate: {results['optimal_value_snips']:.3f}")
```

### 2. Supervised-to-Bandit Conversion

Convert UCI datasets to bandits with induced non-stationarity:

```python
from multi_armed_bandit.benchmarks import SyntheticDriftBandit, MushroomBandit

# Gradual drift (recommended for testing adaptation)
env = SyntheticDriftBandit(
    n_arms=5, gap=1.0,
    drift_type='gradual',  # or 'sudden'
    drift_interval=500,
)

# Or use real Mushroom dataset
env = MushroomBandit(drift_type='gradual', drift_interval=1000)
```

### 3. Replay Evaluation

Unbiased evaluation on logged data (simulates A/B test):

```python
from multi_armed_bandit.benchmarks import SyntheticReplayEvaluator

evaluator = SyntheticReplayEvaluator(n_arms=5, seed=42)
result = evaluator.evaluate(algorithm, max_steps=5000)
print(f"Mean reward: {result.mean_reward:.3f}")
```

### Running All Benchmarks

```bash
python -m multi_armed_bandit.scripts.run_benchmarks
```

Results saved to `experiments/results/benchmarks/`.

---

## Research Scripts Summary

| Script | Purpose | Output |
|--------|---------|--------|
| `run_gpu_study.py` | Full 13×4×50 comparative study | `results/full_study_*/` |
| `run_benchmarks.py` | OBP/Replay/Supervised-to-Bandit validation | `results/benchmarks/` |
| `run_ablation_study.py` | Window size & discount factor sensitivity | `results/ablation/` |

### Complete Execution

```bash
# Run all analyses (recommended order)
python -m multi_armed_bandit.scripts.run_gpu_study      # ~40 seconds
python -m multi_armed_bandit.scripts.run_benchmarks     # ~20 seconds
python -m multi_armed_bandit.scripts.run_ablation_study # ~5 minutes
```

### Generated Artifacts

```
experiments/results/
├── full_study_YYYYMMDD_HHMMSS/
│   ├── raw/                    # JSON results per environment
│   ├── plots/                  # Regret curves, comparisons
│   ├── summary/                # CSV statistics, rankings
│   └── README.md               # Auto-generated summary
├── benchmarks/
│   └── benchmark_results_*.json
└── ablation/
    ├── regret_dynamics.png     # Cumulative regret over time
    ├── ablation_study.png      # Hyperparameter sensitivity bars
    └── ablation_results.json
```

---

## References

### Foundational
- Sutton & Barto (2018). "Reinforcement Learning: An Introduction" - Ch. 2
- Auer et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem"
- Thompson (1933). "On the likelihood that one unknown probability exceeds another"

### Non-Stationary Bandits
- Garivier & Moulines (2011). "On Upper-Confidence Bound Policies for Switching Bandit Problems"
- Besbes et al. (2014). "Stochastic Multi-Armed-Bandit Problem with Non-Stationary Rewards"
- Raj & Kalyani (2017). "Taming Non-stationary Bandits: A Bayesian Approach"

### Adversarial Bandits
- Auer et al. (2002). "The Nonstochastic Multiarmed Bandit Problem"
- Neu (2015). "Explore No More: Improved High-Probability Regret Bounds"

### Benchmark Methodology
- Li et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation"
- Li et al. (2011). "Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation"
- Saito et al. (2020). "Open Bandit Dataset and Pipeline" (OBP/OBD)

---

## Performance Optimizations

The codebase includes several performance optimizations:

- **Vectorized algorithms**: Thompson Sampling posterior computation, gradient bandit updates
- **O(T) rolling window**: Cumsum-based metrics instead of O(T×W) loop
- **Pre-allocated arrays**: MetricsTracker uses pre-allocated numpy arrays
- **Parallel execution**: `run_experiment_suite_parallel()` for CPU parallelism
- **GPU acceleration**: Optional CuPy-based batch execution via `GPUBatchRunner`

### GPU Usage

```python
from multi_armed_bandit.experiments import GPUBatchConfig, GPUBatchRunner

config = GPUBatchConfig(n_runs=100, n_arms=5, horizon=10000)
runner = GPUBatchRunner(config)
results = runner.run_thompson_sampling(arm_means=[0, 1, 0, 0, 0])
```
