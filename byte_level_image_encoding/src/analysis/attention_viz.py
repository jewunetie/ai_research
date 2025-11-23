#!/usr/bin/env python3
"""
Attention visualization for ByteFormer.

This module extracts and visualizes attention patterns to understand
which bytes the model focuses on when classifying images.

Usage:
    python src/analysis/attention_viz.py \
        --checkpoint PATH \
        --data-dir PATH \
        --num-samples 10

Example:
    python src/analysis/attention_viz.py \
        --checkpoint experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints/best_model.pth \
        --data-dir data/cifar10/jpeg_q75/test \
        --num-samples 5
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.byteformer import ByteFormer
from src.data.byte_dataset import ByteImageDataset
from src.training.config import TrainingConfig


class AttentionExtractor:
    """
    Extract attention weights from ByteFormer.

    This class hooks into the transformer layers to capture attention weights
    during forward pass.
    """

    def __init__(self, model: ByteFormer):
        """
        Initialize attention extractor.

        Args:
            model: ByteFormer model
        """
        self.model = model
        self.attention_weights = []
        self.hooks = []

        # Register hooks for each transformer layer
        for layer_idx, layer in enumerate(self.model.transformer.layers):
            hook = layer.self_attn.register_forward_hook(
                self._get_attention_hook(layer_idx)
            )
            self.hooks.append(hook)

    def _get_attention_hook(self, layer_idx: int):
        """Create hook function for specific layer."""
        def hook(module, input, output):
            # For MultiheadAttention, we need to manually compute attention weights
            # because PyTorch doesn't return them by default
            # We'll need to access the attention scores from the module
            pass  # Placeholder - attention extraction varies by PyTorch version

        return hook

    def extract_attention(self, byte_sequence: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention weights for a byte sequence.

        Args:
            byte_sequence: Tensor of shape [1, max_bytes]

        Returns:
            List of attention weight tensors, one per layer
        """
        self.attention_weights = []

        with torch.no_grad():
            # Forward pass (attention is captured via hooks)
            _ = self.model(byte_sequence)

        return self.attention_weights

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()


class SimpleAttentionVisualizer:
    """
    Simplified attention visualization using gradient-based saliency.

    Since extracting raw attention weights is complex, this uses
    gradient-based methods to identify important bytes.
    """

    def __init__(self, model: ByteFormer, device: str = 'cuda'):
        """
        Initialize visualizer.

        Args:
            model: ByteFormer model
            device: Device to run on
        """
        self.model = model
        self.device = device
        self.model.eval()

    def compute_byte_importance(
        self,
        byte_sequence: torch.Tensor,
        target_class: int
    ) -> np.ndarray:
        """
        Compute importance of each byte using gradients.

        Args:
            byte_sequence: Tensor of shape [1, max_bytes]
            target_class: Target class for gradient computation

        Returns:
            Array of importance scores for each byte
        """
        # Embed the byte sequence
        byte_sequence = byte_sequence.to(self.device)
        byte_sequence.requires_grad = False

        # Forward pass with embedding gradients
        embeddings = self.model.byte_embedding(byte_sequence)
        embeddings.requires_grad = True

        # Downsample
        x = embeddings.transpose(1, 2)
        x = self.model.downsample(x)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = x + self.model.pos_encoding[:, :x.size(1), :]

        # Transformer
        x = self.model.transformer(x)

        # Global pooling
        x = x.mean(dim=1)

        # Classification
        x = self.model.norm(x)
        x = self.model.dropout(x)
        logits = self.model.classifier(x)

        # Compute gradient w.r.t. target class
        target_score = logits[0, target_class]
        target_score.backward()

        # Importance = gradient magnitude
        importance = embeddings.grad.abs().mean(dim=-1).squeeze().cpu().numpy()

        return importance

    def visualize_byte_importance(
        self,
        byte_sequence: torch.Tensor,
        importance: np.ndarray,
        label: int,
        predicted: int,
        output_path: Path,
        max_bytes_show: int = 2048
    ):
        """
        Visualize byte importance as a heatmap.

        Args:
            byte_sequence: Original byte sequence
            importance: Importance scores
            label: True label
            predicted: Predicted label
            output_path: Path to save figure
            max_bytes_show: Maximum bytes to show (for readability)
        """
        from src.analysis.failure_analysis import CLASS_NAMES

        byte_values = byte_sequence.cpu().numpy()[0, :max_bytes_show]
        importance_values = importance[:max_bytes_show]

        # Normalize importance
        importance_values = (importance_values - importance_values.min()) / (
            importance_values.max() - importance_values.min() + 1e-8
        )

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6))

        # Plot 1: Byte values with importance overlay
        x = np.arange(len(byte_values))
        ax1.scatter(x, byte_values, c=importance_values, cmap='hot',
                   s=5, alpha=0.8, edgecolors='none')
        ax1.set_ylabel('Byte Value (0-255)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Byte Sequence with Importance Overlay\n'
                     f'True: {CLASS_NAMES[label]} | Predicted: {CLASS_NAMES[predicted]}',
                     fontsize=12, fontweight='bold')
        ax1.set_ylim(-5, 260)
        ax1.grid(axis='y', alpha=0.3)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='hot')
        sm.set_array(importance_values)
        cbar1 = plt.colorbar(sm, ax=ax1)
        cbar1.set_label('Importance', fontsize=10)

        # Plot 2: Importance heatmap
        # Reshape into rows for better visualization
        bytes_per_row = 256
        n_rows = min(8, len(importance_values) // bytes_per_row)

        if n_rows > 0:
            importance_grid = importance_values[:n_rows * bytes_per_row].reshape(n_rows, bytes_per_row)

            sns.heatmap(importance_grid, cmap='hot', cbar=True, ax=ax2,
                       xticklabels=False, yticklabels=False)
            ax2.set_xlabel(f'Byte Position (0-{max_bytes_show})', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Segments', fontsize=11, fontweight='bold')
            ax2.set_title('Importance Heatmap (Hot = High Importance)',
                         fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_format_differences(
        self,
        datasets: Dict[str, ByteImageDataset],
        class_idx: int,
        sample_idx: int,
        output_dir: Path
    ):
        """
        Compare byte importance across different formats for same image.

        Args:
            datasets: Dict of format -> dataset
            class_idx: Class to analyze
            sample_idx: Sample index within class
            output_dir: Output directory
        """
        from src.analysis.failure_analysis import CLASS_NAMES

        fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 4*len(datasets)))
        if len(datasets) == 1:
            axes = [axes]

        for ax, (fmt, dataset) in zip(axes, datasets.items()):
            # Get sample
            byte_seq, label = dataset[sample_idx]
            byte_seq = byte_seq.unsqueeze(0).to(self.device)

            # Predict
            with torch.no_grad():
                logits = self.model(byte_seq)
                predicted = logits.argmax(dim=1).item()

            # Compute importance
            importance = self.compute_byte_importance(byte_seq, predicted)

            # Plot
            byte_values = byte_seq.cpu().numpy()[0, :2048]
            importance_values = importance[:2048]
            importance_values = (importance_values - importance_values.min()) / (
                importance_values.max() - importance_values.min() + 1e-8
            )

            x = np.arange(len(byte_values))
            scatter = ax.scatter(x, byte_values, c=importance_values, cmap='hot',
                               s=5, alpha=0.8, edgecolors='none')

            ax.set_ylabel('Byte Value', fontsize=10, fontweight='bold')
            ax.set_title(f'{fmt.upper()}: {CLASS_NAMES[label]} → Predicted: {CLASS_NAMES[predicted]}',
                        fontsize=11, fontweight='bold')
            ax.set_ylim(-5, 260)
            ax.grid(axis='y', alpha=0.3)

            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Importance')

        plt.tight_layout()
        output_path = output_dir / f'format_comparison_class{class_idx}_sample{sample_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {output_path}")


def main(args):
    """Main attention visualization function."""

    print("=" * 70)
    print("Attention Visualization")
    print("=" * 70)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        if 'tags' in config_dict:
            del config_dict['tags']
        config = TrainingConfig(**config_dict)
    else:
        from src.training.config import get_config_jpeg_q75
        config = get_config_jpeg_q75()

    # Create model
    print("Creating model...")
    model = ByteFormer(
        max_bytes=config.max_bytes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        num_classes=config.num_classes,
        dropout=config.dropout
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")

    # Create dataset
    print(f"\nLoading dataset: {args.data_dir}")
    dataset = ByteImageDataset(args.data_dir, max_bytes=config.max_bytes)
    print(f"Dataset: {len(dataset)} samples")

    # Create visualizer
    visualizer = SimpleAttentionVisualizer(model, device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Visualize random samples
    print(f"\nGenerating visualizations for {args.num_samples} samples...")

    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), size=args.num_samples, replace=False)

    for i, idx in enumerate(sample_indices, 1):
        print(f"  Processing sample {i}/{args.num_samples} (index {idx})...")

        byte_seq, label = dataset[idx]
        byte_seq = byte_seq.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            logits = model(byte_seq)
            predicted = logits.argmax(dim=1).item()

        # Compute importance
        importance = visualizer.compute_byte_importance(byte_seq, predicted)

        # Visualize
        output_path = output_dir / f'attention_sample{i}_idx{idx}.png'
        visualizer.visualize_byte_importance(
            byte_seq,
            importance,
            label.item(),
            predicted,
            output_path
        )

        print(f"    ✓ Saved: {output_path}")

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"Generated {args.num_samples} attention visualizations")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize attention patterns in ByteFormer'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing test images')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/h1_zero_shot_transfer/results/attention_viz',
                        help='Output directory for visualizations')

    args = parser.parse_args()

    main(args)
