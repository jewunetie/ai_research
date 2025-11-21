"""Device handling for cross-platform support (MPS/CUDA/CPU)."""

import torch


def get_device(device_name="auto"):
    """
    Get the appropriate device for training.

    Args:
        device_name: "auto", "mps", "cuda", or "cpu"

    Returns:
        torch.device: The selected device
    """
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU")
    else:
        device = torch.device(device_name)
        print(f"✓ Using device: {device}")

    return device


def print_device_info(device):
    """Print detailed device information."""
    print("\n" + "="*50)
    print("Device Information")
    print("="*50)

    if device.type == "cuda":
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif device.type == "mps":
        print("Device: Apple Silicon GPU (MPS)")
        print("Note: Unified memory architecture")
    else:
        print("Device: CPU")

    print(f"PyTorch Version: {torch.__version__}")
    print("="*50 + "\n")
