"""Compute device selection."""

import torch


def setup_device(device_name: str = None) -> torch.device:
    """
    Auto-select the best available compute device.

    Parameters
    ----------
    device_name : str, optional
        Force a specific device: 'cuda', 'mps', or 'cpu'.
        If None, auto-selects in order: CUDA → MPS → CPU.

    Returns
    -------
    torch.device
    """
    if device_name:
        device = torch.device(device_name)
        print(f"Using device: {device_name}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device