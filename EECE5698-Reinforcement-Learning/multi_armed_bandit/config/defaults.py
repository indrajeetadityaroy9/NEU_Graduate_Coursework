"""
Default experiment parameters.

These can be imported and overridden as needed.
"""

# Experiment execution
N_RUNS: int = 50
N_RUNS_QUICK: int = 10

# Environment parameters
HORIZON: int = 10000
N_ARMS: int = 5
GAP: float = 1.0

# Reproducibility
SEED: int = 42

# Parallelization
MAX_WORKERS: int = 32
