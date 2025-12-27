"""Distributed training utilities for multi-GPU training with PyTorch DDP."""

import os
from typing import Optional
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12355",
) -> None:
    """Initialize the distributed environment.

    Args:
        rank: Unique identifier for this process.
        world_size: Total number of processes.
        backend: Backend for distributed training ('nccl' for GPU, 'gloo' for CPU).
        master_addr: Address of the master node.
        master_port: Port on which the master is listening.
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port

    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set the device for this process
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get the rank of the current process.

    Returns:
        Process rank, or 0 if not distributed.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get the total number of processes.

    Returns:
        World size, or 1 if not distributed.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0).

    Returns:
        True if this is the main process.
    """
    return get_rank() == 0


def wrap_ddp(
    module: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = False,
) -> DDP:
    """Wrap a module with DistributedDataParallel.

    Args:
        module: PyTorch module to wrap.
        device_id: GPU device ID for this process.
        find_unused_parameters: Whether to find unused parameters during backward.

    Returns:
        DDP-wrapped module.
    """
    return DDP(
        module,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
    )


def sync_tensor(tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
    """Synchronize a tensor across all processes.

    Args:
        tensor: Tensor to synchronize.
        op: Reduction operation ('mean', 'sum', 'max', 'min').

    Returns:
        Synchronized tensor.
    """
    if not dist.is_initialized():
        return tensor

    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # Ensure tensor is on CUDA
    if not tensor.is_cuda:
        tensor = tensor.cuda()

    if op == "mean":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    elif op == "sum":
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    elif op == "max":
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    elif op == "min":
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")

    return tensor


def sync_dict(metrics: dict, op: str = "mean") -> dict:
    """Synchronize a dictionary of metrics across all processes.

    Args:
        metrics: Dictionary of metric tensors or floats.
        op: Reduction operation.

    Returns:
        Dictionary with synchronized values.
    """
    if not dist.is_initialized():
        return metrics

    synced = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            synced[key] = sync_tensor(value.clone(), op).item()
        elif isinstance(value, (int, float)):
            tensor = torch.tensor([value], device='cuda')
            synced[key] = sync_tensor(tensor, op).item()
        else:
            synced[key] = value  # Don't sync non-numeric values
    return synced


def broadcast_object(obj, src: int = 0):
    """Broadcast an object from source to all processes.

    Args:
        obj: Object to broadcast (on source process).
        src: Source rank.

    Returns:
        Broadcasted object.
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


class DistributedSampler:
    """Simple sampler for distributing data across processes."""

    def __init__(
        self,
        total_size: int,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """Initialize the distributed sampler.

        Args:
            total_size: Total number of samples.
            rank: Process rank (uses global rank if None).
            world_size: Total processes (uses global world_size if None).
            shuffle: Whether to shuffle indices.
            seed: Random seed for shuffling.
        """
        self.total_size = total_size
        self.rank = rank if rank is not None else get_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling.

        Args:
            epoch: Current epoch number.
        """
        self.epoch = epoch

    def get_indices(self) -> list:
        """Get indices for this process.

        Returns:
            List of indices assigned to this process.
        """
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        # Distribute indices across processes
        per_process = self.total_size // self.world_size
        start = self.rank * per_process
        end = start + per_process
        if self.rank == self.world_size - 1:
            end = self.total_size  # Last process gets remainder

        return indices[start:end]
