"""
Mixed-format dataset for ByteFormer training.

This module provides a dataset that samples images from multiple formats
(JPEG, PNG, WebP, BMP) during training to test format-agnostic learning.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class MixedFormatDataset(Dataset):
    """
    Dataset that samples from multiple image formats.

    During training, each batch may contain images from different formats,
    forcing the model to learn format-agnostic representations.

    Args:
        base_dir: Base directory containing format subdirectories
        formats: List of formats to sample from (e.g., ['jpeg_q75', 'png'])
        split: Dataset split ('train' or 'test')
        max_bytes: Maximum byte sequence length
        pad_value: Padding value
        sample_strategy: How to sample formats ('uniform', 'weighted', 'sequential')
        format_weights: Optional weights for weighted sampling
    """

    def __init__(
        self,
        base_dir: str,
        formats: List[str] = ['jpeg_q75', 'png', 'webp', 'bmp'],
        split: str = 'train',
        max_bytes: int = 8192,
        pad_value: int = 0,
        sample_strategy: str = 'uniform',
        format_weights: Dict[str, float] = None
    ):
        self.base_dir = Path(base_dir)
        self.formats = formats
        self.split = split
        self.max_bytes = max_bytes
        self.pad_value = pad_value
        self.sample_strategy = sample_strategy

        # Build file lists for each format
        self.format_files = {}
        self.format_labels = {}

        for fmt in formats:
            fmt_dir = self.base_dir / fmt / split

            if not fmt_dir.exists():
                raise FileNotFoundError(f"Format directory not found: {fmt_dir}")

            files = sorted(list(fmt_dir.glob('*.*')))
            labels = [int(f.stem.split('_class')[1]) for f in files]

            self.format_files[fmt] = files
            self.format_labels[fmt] = labels

        # Determine dataset length (use first format as reference)
        self.num_samples = len(self.format_files[formats[0]])

        # Verify all formats have same number of samples
        for fmt in formats:
            if len(self.format_files[fmt]) != self.num_samples:
                raise ValueError(
                    f"Format {fmt} has {len(self.format_files[fmt])} samples, "
                    f"expected {self.num_samples}"
                )

        # Setup sampling weights
        if sample_strategy == 'weighted':
            if format_weights is None:
                # Equal weights by default
                self.format_weights = {fmt: 1.0 / len(formats) for fmt in formats}
            else:
                self.format_weights = format_weights
                # Normalize
                total = sum(self.format_weights.values())
                self.format_weights = {k: v/total for k, v in self.format_weights.items()}
        else:
            self.format_weights = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Get item with random format sampling.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (byte_sequence, label, format_name)
        """
        # Choose format based on strategy
        if self.sample_strategy == 'uniform':
            # Uniformly random
            fmt = random.choice(self.formats)

        elif self.sample_strategy == 'weighted':
            # Weighted random
            formats_list = list(self.format_weights.keys())
            weights_list = [self.format_weights[f] for f in formats_list]
            fmt = random.choices(formats_list, weights=weights_list, k=1)[0]

        elif self.sample_strategy == 'sequential':
            # Cycle through formats
            fmt = self.formats[idx % len(self.formats)]

        else:
            raise ValueError(f"Unknown sample strategy: {self.sample_strategy}")

        # Load file
        file_path = self.format_files[fmt][idx]
        label = self.format_labels[fmt][idx]

        with open(file_path, 'rb') as f:
            byte_values = list(f.read())

        # Pad or truncate
        if len(byte_values) > self.max_bytes:
            byte_values = byte_values[:self.max_bytes]
        else:
            byte_values += [self.pad_value] * (self.max_bytes - len(byte_values))

        byte_sequence = torch.tensor(byte_values, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return byte_sequence, label_tensor, fmt

    def get_format_distribution(self, num_samples: int = 1000) -> Dict[str, int]:
        """
        Sample dataset to check format distribution.

        Args:
            num_samples: Number of samples to check

        Returns:
            Dict of format -> count
        """
        format_counts = {fmt: 0 for fmt in self.formats}

        for i in range(min(num_samples, len(self))):
            _, _, fmt = self[i]
            format_counts[fmt] += 1

        return format_counts


def create_mixed_format_dataloaders(
    base_dir: str,
    formats: List[str] = ['jpeg_q75', 'png', 'webp', 'bmp'],
    batch_size: int = 64,
    max_bytes: int = 8192,
    num_workers: int = 4,
    sample_strategy: str = 'uniform',
    format_weights: Dict[str, float] = None
):
    """
    Create mixed-format train and test dataloaders.

    Args:
        base_dir: Base directory containing format subdirectories
        formats: List of formats to include
        batch_size: Batch size
        max_bytes: Maximum byte sequence length
        num_workers: Number of dataloader workers
        sample_strategy: Sampling strategy
        format_weights: Optional format weights

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = MixedFormatDataset(
        base_dir=base_dir,
        formats=formats,
        split='train',
        max_bytes=max_bytes,
        sample_strategy=sample_strategy,
        format_weights=format_weights
    )

    # Test set uses uniform sampling
    test_dataset = MixedFormatDataset(
        base_dir=base_dir,
        formats=formats,
        split='test',
        max_bytes=max_bytes,
        sample_strategy='uniform'  # Always uniform for testing
    )

    # Custom collate function to handle format names
    def collate_fn(batch):
        byte_seqs = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])
        formats = [item[2] for item in batch]
        return byte_seqs, labels, formats

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, test_loader


if __name__ == '__main__':
    """Test mixed-format dataset."""
    import sys

    print("=" * 70)
    print("MixedFormatDataset Test")
    print("=" * 70)

    # Create dataset
    dataset = MixedFormatDataset(
        base_dir='./data/cifar10',
        formats=['jpeg_q75', 'png'],
        split='train',
        max_bytes=8192,
        sample_strategy='uniform'
    )

    print(f"\nDataset: {len(dataset)} samples")
    print(f"Formats: {dataset.formats}")

    # Test sampling
    print("\nTesting format distribution (1000 samples)...")
    dist = dataset.get_format_distribution(1000)

    print("\nFormat distribution:")
    for fmt, count in dist.items():
        print(f"  {fmt}: {count} ({count/10:.1f}%)")

    # Test loading
    print("\nLoading sample 0...")
    byte_seq, label, fmt = dataset[0]

    print(f"  Format: {fmt}")
    print(f"  Byte sequence shape: {byte_seq.shape}")
    print(f"  Label: {label.item()}")
    print(f"  First 20 bytes: {byte_seq[:20].tolist()}")

    print("\n" + "=" * 70)
    print("Test passed!")
    print("=" * 70)
