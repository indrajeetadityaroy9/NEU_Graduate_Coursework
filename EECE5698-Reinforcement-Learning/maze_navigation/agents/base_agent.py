"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
from torch.cuda.amp import GradScaler
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all RL agents.

    Defines the common interface that all agents must implement.
    Supports mixed precision training (AMP) for H100 GPUs.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        device: str = 'auto',
        seed: Optional[int] = None,
        mixed_precision: bool = False,
        compile_mode: Optional[str] = None,
    ):
        """Initialize the base agent.

        Args:
            obs_dim: Dimension of observation space.
            n_actions: Number of discrete actions.
            device: Device to use ('cpu', 'cuda', or 'auto').
            seed: Random seed for reproducibility.
            mixed_precision: Enable AMP (automatic mixed precision).
            compile_mode: torch.compile mode ('reduce-overhead', 'max-autotune', None).
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = self._setup_device(device)
        self.seed = seed
        self.mixed_precision = mixed_precision
        self.compile_mode = compile_mode

        # Setup AMP scaler for mixed precision training
        self.scaler = GradScaler(enabled=mixed_precision)

        if seed is not None:
            self._set_seed(seed)

    def _setup_device(self, device: str) -> torch.device:
        """Set up the compute device.

        Args:
            device: Device specification.

        Returns:
            Torch device object.
        """
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility.

        Args:
            seed: Random seed value.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    @abstractmethod
    def select_action(
        self,
        observation: np.ndarray,
        deterministic: bool = False
    ) -> int:
        """Select an action given an observation.

        Args:
            observation: Current observation.
            deterministic: If True, select greedy action.

        Returns:
            Selected action index.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> Optional[Dict[str, float]]:
        """Perform one update step.

        Returns:
            Dictionary of training metrics, or None if no update performed.
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent state to disk.

        Args:
            path: Path to save file.
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent state from disk.

        Args:
            path: Path to load file.
        """
        pass

    def to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor on device.

        Args:
            x: Numpy array.

        Returns:
            Tensor on agent's device.
        """
        return torch.FloatTensor(x).to(self.device)

    def _maybe_compile(self, network: torch.nn.Module) -> torch.nn.Module:
        """Optionally compile network with torch.compile.

        Args:
            network: PyTorch module to compile.

        Returns:
            Compiled or original network.
        """
        if self.compile_mode is not None:
            return torch.compile(network, mode=self.compile_mode)
        return network

    def get_autocast_context(self):
        """Get autocast context manager for mixed precision.

        Returns:
            Autocast context manager.
        """
        return torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.float16)
