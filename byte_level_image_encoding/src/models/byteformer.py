"""
ByteFormer: Byte-level Transformer for Image Classification.

A transformer-based model that operates directly on image file bytes
(JPEG, PNG, WebP, BMP) without decoding to pixels.

Architecture:
1. Byte embedding (256 vocab size)
2. Strided convolution downsampling (reduce sequence length)
3. Positional encoding
4. Transformer encoder
5. Global pooling + classification head

Designed for CIFAR-10 classification with ~8-10M parameters.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class ByteFormer(nn.Module):
    """
    ByteFormer for CIFAR-10 byte-level image classification.

    Args:
        max_bytes: Maximum input byte sequence length (default: 8192)
        d_model: Embedding dimension (default: 192)
        nhead: Number of attention heads (default: 6)
        num_layers: Number of transformer layers (default: 6)
        dim_feedforward: Feedforward network dimension (default: 768)
        downsample_kernel: Kernel size for downsampling conv (default: 32)
        downsample_stride: Stride for downsampling conv (default: 16)
        num_classes: Number of output classes (default: 10)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        max_bytes: int = 8192,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 6,
        dim_feedforward: int = 768,
        downsample_kernel: int = 32,
        downsample_stride: int = 16,
        num_classes: int = 10,
        dropout: float = 0.1
    ):
        super().__init__()

        self.max_bytes = max_bytes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # 1. Byte embedding (vocab size = 256 for bytes 0-255)
        self.byte_embedding = nn.Embedding(256, d_model, padding_idx=0)

        # 2. Downsampling via strided 1D convolution
        # This reduces sequence length from max_bytes to ~max_bytes/downsample_stride
        self.downsample = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=downsample_kernel,
            stride=downsample_stride,
            padding=downsample_kernel // 2
        )

        # Calculate downsampled sequence length
        self.seq_len = math.ceil(max_bytes / downsample_stride)

        # 3. Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.seq_len, d_model) * 0.02
        )

        # 4. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 5. Classification head
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.normal_(self.byte_embedding.weight, mean=0.0, std=0.02)

        # Initialize conv
        nn.init.kaiming_normal_(self.downsample.weight, mode='fan_out', nonlinearity='linear')
        if self.downsample.bias is not None:
            nn.init.zeros_(self.downsample.bias)

        # Initialize classifier
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            byte_sequence: Tensor of shape [batch_size, max_bytes] containing
                          byte values (0-255 as long integers)

        Returns:
            Logits of shape [batch_size, num_classes]
        """
        # 1. Embed bytes: [batch, max_bytes] -> [batch, max_bytes, d_model]
        x = self.byte_embedding(byte_sequence)

        # 2. Downsample: [batch, max_bytes, d_model] -> [batch, seq_len, d_model]
        # Conv1d expects [batch, channels, length]
        x = x.transpose(1, 2)  # [batch, d_model, max_bytes]
        x = self.downsample(x)  # [batch, d_model, seq_len]
        x = x.transpose(1, 2)  # [batch, seq_len, d_model]

        # 3. Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # 4. Transformer encoding
        x = self.transformer(x)  # [batch, seq_len, d_model]

        # 5. Global average pooling
        x = x.mean(dim=1)  # [batch, d_model]

        # 6. Classification
        x = self.norm(x)
        x = self.dropout(x)
        logits = self.classifier(x)  # [batch, num_classes]

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Return model architecture info."""
        return {
            'model_name': 'ByteFormer',
            'max_bytes': self.max_bytes,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'seq_len_after_downsample': self.seq_len,
            'num_classes': self.classifier.out_features,
            'total_parameters': self.count_parameters(),
            'trainable_parameters': self.count_parameters(),
        }


def create_byteformer_cifar10(
    max_bytes: int = 8192,
    model_size: str = 'small'
) -> ByteFormer:
    """
    Create ByteFormer model with predefined size configurations.

    Args:
        max_bytes: Maximum byte sequence length
        model_size: Model size ('tiny', 'small', 'base')

    Returns:
        ByteFormer model instance
    """
    configs = {
        'tiny': {
            'd_model': 128,
            'nhead': 4,
            'num_layers': 4,
            'dim_feedforward': 512,
        },
        'small': {
            'd_model': 192,
            'nhead': 6,
            'num_layers': 6,
            'dim_feedforward': 768,
        },
        'base': {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 8,
            'dim_feedforward': 1024,
        },
    }

    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")

    config = configs[model_size]

    model = ByteFormer(
        max_bytes=max_bytes,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        num_classes=10,
        dropout=0.1
    )

    return model


if __name__ == '__main__':
    """Test model creation and forward pass."""

    print("=" * 70)
    print("ByteFormer Model Test")
    print("=" * 70)

    # Create model
    model = create_byteformer_cifar10(max_bytes=8192, model_size='small')

    # Print model info
    info = model.get_model_info()
    print("\nModel Configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    max_bytes = 8192

    # Create dummy input (random bytes)
    dummy_input = torch.randint(0, 256, (batch_size, max_bytes), dtype=torch.long)

    print(f"  Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Output shape: {output.shape}")
    print(f"  Output min/max: {output.min().item():.3f} / {output.max().item():.3f}")

    # Test all model sizes
    print("\nParameter counts for different model sizes:")
    print("-" * 70)
    for size in ['tiny', 'small', 'base']:
        m = create_byteformer_cifar10(model_size=size)
        params = m.count_parameters()
        print(f"  {size:>6}: {params:>12,} parameters ({params/1e6:.1f}M)")

    print("\n" + "=" * 70)
    print("Test passed!")
    print("=" * 70)
