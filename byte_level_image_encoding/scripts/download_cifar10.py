#!/usr/bin/env python3
"""
Download CIFAR-10 dataset using torchvision.

This script downloads the CIFAR-10 dataset to ./data/raw/ for subsequent
encoding into multiple image formats (JPEG, PNG, WebP, BMP).
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torchvision
import torchvision.transforms as transforms


def main():
    """Download CIFAR-10 train and test sets."""

    # Create data directory
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CIFAR-10 Dataset Download")
    print("=" * 60)
    print(f"Download location: {data_dir.absolute()}")
    print()

    # Download training set
    print("Downloading CIFAR-10 training set...")
    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=None
    )

    # Download test set
    print("Downloading CIFAR-10 test set...")
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=None
    )

    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"Training samples: {len(trainset):,}")
    print(f"Test samples: {len(testset):,}")
    print(f"Total samples: {len(trainset) + len(testset):,}")
    print()
    print("Classes:", trainset.classes)
    print()
    print("Next step: Run encode_formats.py to convert to JPEG/PNG/WebP/BMP")
    print("=" * 60)


if __name__ == '__main__':
    main()
