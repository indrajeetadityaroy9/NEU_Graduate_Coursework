"""
GPU/CPU Backend Selection for Multi-Armed Bandit Algorithms.

Provides a unified interface (xp) that automatically selects between
CuPy (GPU) and NumPy (CPU) based on hardware availability.

Usage
-----
>>> from multi_armed_bandit.backends import xp, BACKEND, to_numpy, to_gpu
>>> arr = xp.zeros(100)  # Uses GPU if available, else CPU
>>> np_arr = to_numpy(arr)  # Always returns numpy array

Environment Variables
--------------------
MAB_FORCE_CPU : str
    Set to '1' to force CPU backend even when GPU is available
MAB_GPU_DEVICE : str
    GPU device ID to use (default: '0')

Examples
--------
Force CPU mode:
    $ MAB_FORCE_CPU=1 python my_experiment.py

Use specific GPU:
    $ MAB_GPU_DEVICE=1 python my_experiment.py
"""

import os
import numpy as np
from typing import Union, Any

# Configuration from environment
FORCE_CPU = os.environ.get('MAB_FORCE_CPU', '0') == '1'
GPU_DEVICE = int(os.environ.get('MAB_GPU_DEVICE', '0'))

# Try to import CuPy for GPU support
_cupy_available = False
try:
    if not FORCE_CPU:
        import cupy as cp
        # Verify GPU is actually accessible
        cp.cuda.Device(GPU_DEVICE).use()
        _ = cp.zeros(1)  # Simple test
        _cupy_available = True
except ImportError:
    pass
except Exception:
    # Catch any CuPy-related errors (device not available, etc.)
    pass

# Set backend and array module
if _cupy_available:
    import cupy as cp
    BACKEND = 'gpu'
    xp = cp

    def get_device_info() -> dict:
        """Get GPU device information."""
        device = cp.cuda.Device(GPU_DEVICE)
        return {
            'backend': 'gpu',
            'device_id': GPU_DEVICE,
            'name': device.attributes.get('Name', 'Unknown'),
            'memory_total': device.mem_info[1],
            'memory_free': device.mem_info[0],
        }
else:
    BACKEND = 'cpu'
    xp = np

    def get_device_info() -> dict:
        """Get CPU backend information."""
        return {
            'backend': 'cpu',
            'device_id': None,
            'name': 'CPU',
            'memory_total': None,
            'memory_free': None,
        }


def to_numpy(arr: Any) -> np.ndarray:
    """
    Convert array to numpy, handling both numpy and cupy arrays.

    Parameters
    ----------
    arr : array-like
        Input array (numpy, cupy, or list)

    Returns
    -------
    np.ndarray
        NumPy array
    """
    if BACKEND == 'gpu':
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(arr)


def to_gpu(arr: Any) -> Any:
    """
    Convert array to GPU if available, else return as-is.

    Parameters
    ----------
    arr : array-like
        Input array

    Returns
    -------
    array
        CuPy array if GPU available, else numpy array
    """
    if BACKEND == 'gpu':
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)
    return np.asarray(arr)


def get_array_module(arr: Any = None):
    """
    Get the appropriate array module for the given array.

    Parameters
    ----------
    arr : array-like, optional
        If provided, returns module for this array's type.
        If None, returns the default backend module.

    Returns
    -------
    module
        numpy or cupy module
    """
    if arr is not None and BACKEND == 'gpu':
        import cupy as cp
        return cp.get_array_module(arr)
    return xp


def sync() -> None:
    """
    Synchronize GPU operations (no-op on CPU).

    Call this before timing GPU operations to ensure
    all kernels have completed.
    """
    if BACKEND == 'gpu':
        import cupy as cp
        cp.cuda.Stream.null.synchronize()


def memory_pool_info() -> dict:
    """
    Get GPU memory pool information.

    Returns
    -------
    dict
        Memory pool statistics (empty dict for CPU backend)
    """
    if BACKEND == 'gpu':
        import cupy as cp
        pool = cp.get_default_memory_pool()
        return {
            'used_bytes': pool.used_bytes(),
            'total_bytes': pool.total_bytes(),
            'n_free_blocks': pool.n_free_blocks(),
        }
    return {}


def clear_memory_pool() -> None:
    """
    Clear GPU memory pool to free unused memory.
    """
    if BACKEND == 'gpu':
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


# Random number generation utilities
def create_rng(seed: int = None):
    """
    Create a random number generator for the current backend.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Generator
        Random number generator (numpy or cupy)
    """
    if BACKEND == 'gpu':
        import cupy as cp
        if seed is not None:
            cp.random.seed(seed)
        return cp.random
    else:
        return np.random.default_rng(seed)


__all__ = [
    'BACKEND',
    'xp',
    'to_numpy',
    'to_gpu',
    'get_array_module',
    'get_device_info',
    'sync',
    'memory_pool_info',
    'clear_memory_pool',
    'create_rng',
    'FORCE_CPU',
    'GPU_DEVICE',
]
