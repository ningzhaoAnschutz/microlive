"""Centralized GPU device detection for MicroLive.

Supports:
- NVIDIA CUDA (Windows/Linux)
- Apple Silicon MPS (macOS)
- CPU fallback

Example:
    from microlive.utils.device import get_device, get_device_info
    
    device = get_device()
    print(f"Using: {device}")
    
    info = get_device_info()
    print(f"GPU: {info['name']}")
"""

import torch


def get_device():
    """Get the best available compute device.
    
    Returns:
        torch.device: cuda if NVIDIA GPU available,
                      mps if Apple Silicon GPU available,
                      cpu otherwise.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def is_gpu_available():
    """Check if any GPU acceleration is available.
    
    Returns:
        bool: True if CUDA or MPS GPU is available.
    """
    return torch.cuda.is_available() or (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    )


def get_device_info():
    """Get detailed information about the compute device.
    
    Returns:
        dict: Dictionary with keys:
            - device: Device string ('cuda', 'mps', or 'cpu')
            - type: Device type
            - name: Human-readable device name
            - memory_gb: GPU memory in GB (CUDA only)
    """
    device = get_device()
    info = {
        "device": str(device),
        "type": device.type,
        "name": "CPU",
        "memory_gb": None,
    }
    
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["memory_gb"] = round(props.total_memory / (1024**3), 2)
    elif device.type == "mps":
        info["name"] = "Apple Silicon GPU (MPS)"
    
    return info


def check_gpu_status():
    """Print GPU status information.
    
    Useful for verifying installation.
    
    Returns:
        str: Device type ('cuda', 'mps', or 'cpu')
    """
    print(f"PyTorch version: {torch.__version__}")
    
    info = get_device_info()
    
    if info["type"] == "cuda":
        print(f"✅ CUDA available: {info['name']}")
        print(f"   Memory: {info['memory_gb']} GB")
    elif info["type"] == "mps":
        print(f"✅ MPS available: {info['name']}")
    else:
        print("⚠️  No GPU detected, using CPU")
    
    return info["type"]


if __name__ == "__main__":
    check_gpu_status()
