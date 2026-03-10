"""Compute device selection and global seed utilities."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 2025) -> None:
    """
    Set all random seeds for full reproducibility.

    Covers: Python random, NumPy, PyTorch CPU, PyTorch CUDA,
    and forces deterministic CUDA operations.

    Parameters
    ----------
    seed : int
        Random seed to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # for multi-GPU

    # Force deterministic CUDA ops — slight speed cost but fully reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # PyTorch >= 1.8 global determinism flag
    try:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except AttributeError:
        pass  # older PyTorch versions

    print(f"[seed] Global seed set to {seed}")


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker seed function.
    Pass as worker_init_fn to DataLoader to seed each worker.

    Usage:
        g = torch.Generator()
        g.manual_seed(SEED)
        DataLoader(..., worker_init_fn=seed_worker, generator=g)
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_device(device_name: str = None) -> torch.device:
    """
    Setup compute device (CPU/GPU/MPS).

    Parameters
    ----------
    device_name : str, optional
        Device name ('cuda', 'mps', 'cpu', or None for auto)

    Returns
    -------
    torch.device
        Selected device
    """
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device