"""
Training configuration for ByteFormer on CIFAR-10.

This module defines the training configuration dataclass and provides
preset configurations for different experiment settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Training configuration for ByteFormer."""

    # ========== Model Configuration ==========
    max_bytes: int = 8192
    d_model: int = 192
    nhead: int = 6
    num_layers: int = 6
    dim_feedforward: int = 768
    downsample_kernel: int = 32
    downsample_stride: int = 16
    num_classes: int = 10
    dropout: float = 0.1

    # ========== Training Configuration ==========
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # ========== Data Configuration ==========
    data_format: str = 'jpeg_q75'  # Format to train on
    train_dir: str = './data/cifar10/jpeg_q75/train'
    test_dir: str = './data/cifar10/jpeg_q75/test'
    num_workers: int = 4
    pin_memory: bool = True

    # ========== Optimization ==========
    optimizer: str = 'adamw'  # 'adamw' or 'adam'
    scheduler: str = 'cosine'  # 'cosine' or 'step'
    gradient_clip: float = 1.0

    # ========== Output Configuration ==========
    output_dir: str = './experiments/h1_zero_shot_transfer/jpeg_q75'
    checkpoint_dir: str = './experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints'
    log_dir: str = './experiments/h1_zero_shot_transfer/jpeg_q75/logs'
    save_every: int = 10  # Save checkpoint every N epochs
    log_every: int = 50  # Log every N batches

    # ========== Hardware ==========
    device: str = 'cuda'  # 'cuda' or 'cpu'
    mixed_precision: bool = True  # Use automatic mixed precision (AMP)
    compile_model: bool = False  # Use torch.compile (PyTorch 2.0+)

    # ========== Reproducibility ==========
    seed: int = 42
    deterministic: bool = True

    # ========== Experiment Metadata ==========
    experiment_name: str = 'byteformer_cifar10_jpeg_q75'
    description: str = 'ByteFormer trained on JPEG Q75 for format-agnostic transfer'
    tags: list = field(default_factory=lambda: ['cifar10', 'jpeg', 'zero-shot'])

    def __post_init__(self):
        """Create output directories after initialization."""
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }

    def save(self, path: str):
        """Save configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["TrainingConfig("]
        for key, value in self.to_dict().items():
            lines.append(f"  {key}={value}")
        lines.append(")")
        return "\n".join(lines)


# ========== Preset Configurations ==========

def get_config_quick_test() -> TrainingConfig:
    """Quick test configuration (fast training for debugging)."""
    return TrainingConfig(
        # Small model
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,

        # Fast training
        batch_size=128,
        num_epochs=10,
        learning_rate=1e-3,

        # Output
        experiment_name='quick_test',
        output_dir='./experiments/quick_test',
        checkpoint_dir='./experiments/quick_test/checkpoints',
        log_dir='./experiments/quick_test/logs',
    )


def get_config_jpeg_q75() -> TrainingConfig:
    """Default configuration for JPEG Q75 training (H1 experiment)."""
    return TrainingConfig(
        # Model (small size ~8-10M params)
        d_model=192,
        nhead=6,
        num_layers=6,
        dim_feedforward=768,

        # Training
        batch_size=64,
        num_epochs=100,
        learning_rate=1e-3,
        weight_decay=0.01,

        # Data
        data_format='jpeg_q75',
        train_dir='./data/cifar10/jpeg_q75/train',
        test_dir='./data/cifar10/jpeg_q75/test',

        # Output
        experiment_name='byteformer_jpeg_q75',
        output_dir='./experiments/h1_zero_shot_transfer/jpeg_q75',
        checkpoint_dir='./experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints',
        log_dir='./experiments/h1_zero_shot_transfer/jpeg_q75/logs',
    )


def get_config_png() -> TrainingConfig:
    """Configuration for PNG training (baseline comparison)."""
    config = get_config_jpeg_q75()
    config.data_format = 'png'
    config.train_dir = './data/cifar10/png/train'
    config.test_dir = './data/cifar10/png/test'
    config.experiment_name = 'byteformer_png'
    config.output_dir = './experiments/baselines/png'
    config.checkpoint_dir = './experiments/baselines/png/checkpoints'
    config.log_dir = './experiments/baselines/png/logs'
    return config


def get_config_mixed_formats() -> TrainingConfig:
    """Configuration for training on mixed formats (future experiment)."""
    config = get_config_jpeg_q75()
    config.experiment_name = 'byteformer_mixed_formats'
    config.output_dir = './experiments/mixed_formats'
    config.checkpoint_dir = './experiments/mixed_formats/checkpoints'
    config.log_dir = './experiments/mixed_formats/logs'
    # Note: Would need custom dataset that samples from multiple formats
    return config


if __name__ == '__main__':
    """Test configuration creation and saving."""

    print("=" * 70)
    print("Training Configuration Test")
    print("=" * 70)

    # Create default config
    config = get_config_jpeg_q75()

    print("\nDefault Configuration:")
    print(config)

    # Save config
    config.save('./test_config.json')
    print("\nSaved to: test_config.json")

    # Load config
    loaded_config = TrainingConfig.load('./test_config.json')
    print("\nLoaded configuration successfully")

    # Test different presets
    print("\n" + "=" * 70)
    print("Available Configurations:")
    print("=" * 70)

    configs = {
        'Quick Test': get_config_quick_test(),
        'JPEG Q75 (Main)': get_config_jpeg_q75(),
        'PNG (Baseline)': get_config_png(),
        'Mixed Formats': get_config_mixed_formats(),
    }

    for name, cfg in configs.items():
        print(f"\n{name}:")
        print(f"  Model size: d_model={cfg.d_model}, layers={cfg.num_layers}")
        print(f"  Training: {cfg.num_epochs} epochs, batch_size={cfg.batch_size}")
        print(f"  Data: {cfg.data_format}")
        print(f"  Output: {cfg.output_dir}")

    print("\n" + "=" * 70)
    print("Test passed!")
    print("=" * 70)

    # Cleanup
    import os
    os.remove('./test_config.json')
