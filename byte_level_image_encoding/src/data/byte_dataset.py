"""
Byte-level dataset loader for image files.

This module provides a PyTorch Dataset that loads image files as raw byte
sequences instead of decoded pixels. Designed for the format-agnostic
zero-shot transfer experiment.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class ByteImageDataset(Dataset):
    """
    Load image files as raw byte sequences.

    This dataset reads image files (JPEG, PNG, WebP, BMP) as binary data
    and returns them as sequences of byte values (0-255).

    Args:
        data_dir: Directory containing image files
        max_bytes: Maximum sequence length (pad or truncate)
        pad_value: Value to use for padding (default: 0)
        file_pattern: Glob pattern for image files (default: '*.*')

    Returns:
        Tuple of (byte_sequence, label) where:
            - byte_sequence: torch.LongTensor of shape [max_bytes]
            - label: torch.LongTensor scalar (class index)
    """

    def __init__(
        self,
        data_dir: str,
        max_bytes: int = 8192,
        pad_value: int = 0,
        file_pattern: str = '*.*'
    ):
        self.data_dir = Path(data_dir)
        self.max_bytes = max_bytes
        self.pad_value = pad_value

        # Find all image files
        self.files = sorted(list(self.data_dir.glob(file_pattern)))

        if len(self.files) == 0:
            raise ValueError(f"No files found in {self.data_dir} with pattern {file_pattern}")

        # Extract labels from filenames (format: 00001_class3.jpeg)
        self.labels = []
        for f in self.files:
            try:
                # Parse label from filename
                label_str = f.stem.split('_class')[1]
                label = int(label_str)
                self.labels.append(label)
            except (IndexError, ValueError):
                raise ValueError(
                    f"Cannot parse label from filename: {f.name}. "
                    "Expected format: 00001_class3.jpeg"
                )

        # Validate
        assert len(self.files) == len(self.labels)

        # Statistics
        self._compute_stats()

    def _compute_stats(self):
        """Compute dataset statistics (file sizes)."""
        file_sizes = [f.stat().st_size for f in self.files[:100]]  # Sample first 100
        self.avg_file_size = sum(file_sizes) / len(file_sizes)
        self.min_file_size = min(file_sizes)
        self.max_file_size = max(file_sizes)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load image file as byte sequence.

        Args:
            idx: Index of the file to load

        Returns:
            Tuple of (byte_sequence, label)
        """
        # Read file as bytes
        with open(self.files[idx], 'rb') as f:
            byte_values = list(f.read())

        # Pad or truncate to max_bytes
        if len(byte_values) > self.max_bytes:
            # Truncate
            byte_values = byte_values[:self.max_bytes]
        else:
            # Pad
            byte_values += [self.pad_value] * (self.max_bytes - len(byte_values))

        # Convert to tensors
        byte_sequence = torch.tensor(byte_values, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return byte_sequence, label

    def get_stats(self) -> dict:
        """Return dataset statistics."""
        return {
            'num_files': len(self.files),
            'num_classes': len(set(self.labels)),
            'avg_file_size': self.avg_file_size,
            'min_file_size': self.min_file_size,
            'max_file_size': self.max_file_size,
            'max_bytes': self.max_bytes,
        }

    def __repr__(self) -> str:
        return (
            f"ByteImageDataset(\n"
            f"  data_dir={self.data_dir},\n"
            f"  num_files={len(self.files)},\n"
            f"  num_classes={len(set(self.labels))},\n"
            f"  max_bytes={self.max_bytes},\n"
            f"  avg_file_size={self.avg_file_size:.0f} bytes\n"
            f")"
        )


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    batch_size: int = 64,
    max_bytes: int = 8192,
    num_workers: int = 4
):
    """
    Create train and test dataloaders.

    Args:
        train_dir: Directory containing training images
        test_dir: Directory containing test images
        batch_size: Batch size for dataloaders
        max_bytes: Maximum byte sequence length
        num_workers: Number of dataloader worker processes

    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    # Create datasets
    train_dataset = ByteImageDataset(train_dir, max_bytes=max_bytes)
    test_dataset = ByteImageDataset(test_dir, max_bytes=max_bytes)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # For stable batch sizes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':
    """Test the dataset loader."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python byte_dataset.py <data_dir>")
        print("Example: python byte_dataset.py ../../data/cifar10/jpeg_q75/train")
        sys.exit(1)

    data_dir = sys.argv[1]

    print("=" * 70)
    print("ByteImageDataset Test")
    print("=" * 70)

    # Create dataset
    dataset = ByteImageDataset(data_dir, max_bytes=8192)

    print(dataset)
    print()

    # Test loading a sample
    print("Loading sample 0...")
    byte_seq, label = dataset[0]

    print(f"  Byte sequence shape: {byte_seq.shape}")
    print(f"  Byte sequence dtype: {byte_seq.dtype}")
    print(f"  Label: {label.item()}")
    print(f"  First 20 bytes: {byte_seq[:20].tolist()}")
    print(f"  Last 20 bytes: {byte_seq[-20:].tolist()}")
    print(f"  Non-zero bytes: {(byte_seq != 0).sum().item()}")

    # Test dataloader
    print("\nTesting DataLoader...")
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_bytes, batch_labels = next(iter(loader))

    print(f"  Batch bytes shape: {batch_bytes.shape}")
    print(f"  Batch labels shape: {batch_labels.shape}")
    print(f"  Batch labels: {batch_labels.tolist()}")

    print("\n" + "=" * 70)
    print("Test passed!")
    print("=" * 70)
