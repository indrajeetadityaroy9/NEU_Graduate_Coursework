"""Reproducibility utilities for setting random seeds."""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seed for all random number generators for reproducibility.

    Args:
        seed: The seed value to use across all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
