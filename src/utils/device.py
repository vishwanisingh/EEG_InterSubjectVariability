"""Compute device selection."""

import torch


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