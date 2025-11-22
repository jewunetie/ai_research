#!/usr/bin/env python3
"""
Encode CIFAR-10 into multiple image formats (JPEG, PNG, WebP, BMP).

This script converts the CIFAR-10 dataset from NumPy arrays into actual
image files in different formats for byte-level processing.

Expected file sizes for 32×32 CIFAR-10 images:
- JPEG (Q75): ~1,500 bytes
- PNG: ~1,200 bytes
- WebP (Q75): ~800 bytes
- BMP: ~3,078 bytes
"""

import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm


def encode_dataset(
    dataset,
    output_dir: Path,
    format: Literal['JPEG', 'PNG', 'WEBP', 'BMP'],
    split: str = 'train',
    quality: int = 75
):
    """
    Encode CIFAR-10 dataset to specified image format.

    Args:
        dataset: CIFAR10 dataset object
        output_dir: Directory to save encoded images
        format: Image format (JPEG, PNG, WEBP, BMP)
        split: Dataset split ('train' or 'test')
        quality: JPEG/WebP quality (1-100, only for lossy formats)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nEncoding {split} set to {format}...")
    print(f"Output: {output_dir}")

    file_sizes = []

    for idx in tqdm(range(len(dataset)), desc=f"{format} {split}"):
        img_array, label = dataset[idx]

        # Convert NumPy array to PIL Image
        if isinstance(img_array, np.ndarray):
            img = Image.fromarray(img_array)
        else:
            # If already PIL Image
            img = img_array

        # Ensure RGB mode
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Save with format-specific parameters
        filename = output_dir / f"{idx:05d}_class{label}.{format.lower()}"

        if format == 'JPEG':
            img.save(filename, format='JPEG', quality=quality, optimize=True)
        elif format == 'PNG':
            # PNG compression level (0-9)
            img.save(filename, format='PNG', compress_level=6)
        elif format == 'WEBP':
            img.save(filename, format='WEBP', quality=quality)
        elif format == 'BMP':
            img.save(filename, format='BMP')
        else:
            raise ValueError(f"Unknown format: {format}")

        # Track file size
        file_sizes.append(filename.stat().st_size)

    # Report statistics
    avg_size = np.mean(file_sizes)
    min_size = np.min(file_sizes)
    max_size = np.max(file_sizes)

    print(f"  Files created: {len(file_sizes):,}")
    print(f"  Avg size: {avg_size:.0f} bytes")
    print(f"  Min size: {min_size} bytes")
    print(f"  Max size: {max_size} bytes")
    print(f"  Total size: {sum(file_sizes) / 1024 / 1024:.1f} MB")

    return file_sizes


def main():
    """Encode CIFAR-10 to all formats."""

    # Paths
    base_dir = Path(__file__).parent.parent.parent
    raw_dir = base_dir / 'data' / 'raw'
    output_base = base_dir / 'data' / 'cifar10'

    print("=" * 70)
    print("CIFAR-10 Multi-Format Encoding")
    print("=" * 70)
    print(f"Input: {raw_dir}")
    print(f"Output: {output_base}")
    print()

    # Load CIFAR-10
    print("Loading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root=str(raw_dir),
        train=True,
        download=False,  # Should already be downloaded
        transform=None
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(raw_dir),
        train=False,
        download=False,
        transform=None
    )

    print(f"Train samples: {len(trainset):,}")
    print(f"Test samples: {len(testset):,}")

    # Encode to all formats
    formats_config = [
        ('JPEG', 'jpeg_q75', 75),
        ('PNG', 'png', None),
        ('WEBP', 'webp', 75),
        ('BMP', 'bmp', None),
    ]

    all_stats = {}

    for format_name, dir_name, quality in formats_config:
        print("\n" + "=" * 70)
        print(f"Processing {format_name}")
        print("=" * 70)

        # Train set
        train_dir = output_base / dir_name / 'train'
        train_sizes = encode_dataset(
            trainset,
            train_dir,
            format_name,
            'train',
            quality=quality if quality else 75
        )

        # Test set
        test_dir = output_base / dir_name / 'test'
        test_sizes = encode_dataset(
            testset,
            test_dir,
            format_name,
            'test',
            quality=quality if quality else 75
        )

        all_stats[format_name] = {
            'train_avg': np.mean(train_sizes),
            'test_avg': np.mean(test_sizes),
            'total_files': len(train_sizes) + len(test_sizes),
            'total_mb': (sum(train_sizes) + sum(test_sizes)) / 1024 / 1024
        }

    # Final summary
    print("\n" + "=" * 70)
    print("ENCODING COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"{'Format':<10} {'Avg Size':<12} {'Total Files':<12} {'Total Size':<12}")
    print("-" * 70)

    for format_name, stats in all_stats.items():
        print(f"{format_name:<10} {stats['train_avg']:>8.0f} bytes "
              f"{stats['total_files']:>10,} files "
              f"{stats['total_mb']:>8.1f} MB")

    print("=" * 70)
    print("\nNext step: Train ByteFormer on JPEG format only")
    print("Command: python src/training/train.py")
    print("=" * 70)


if __name__ == '__main__':
    main()
