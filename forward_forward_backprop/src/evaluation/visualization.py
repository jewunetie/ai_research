"""
Visualization utilities for Forward-Forward with Backprop experiments.

Includes:
- t-SNE visualization of learned representations
- Training curve plots
- Confusion matrix heatmaps
- Layer-wise goodness plots
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False
    print("Warning: scikit-learn not installed. t-SNE visualization unavailable.")


def plot_training_curves(
    train_losses: List[float],
    test_accs: List[float],
    save_path: Optional[Path] = None,
    title: str = "Training Curves"
):
    """
    Plot training loss and test accuracy curves.

    Args:
        train_losses: List of training losses per epoch
        test_accs: List of test accuracies per epoch
        save_path: Path to save plot (if None, displays instead)
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot training loss
    epochs = list(range(1, len(train_losses) + 1))
    ax1.plot(epochs, train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)

    # Plot test accuracy
    epochs = list(range(1, len(test_accs) + 1))
    ax2.plot(epochs, test_accs, 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix (num_classes x num_classes)
        class_names: List of class names (if None, uses indices)
        save_path: Path to save plot
        title: Plot title
    """
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                   ha="center", va="center",
                   color="white" if cm_normalized[i, j] > thresh else "black",
                   fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to: {save_path}")
    else:
        plt.show()

    plt.close()


@torch.no_grad()
def visualize_tsne(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    save_path: Optional[Path] = None,
    layer_index: Optional[int] = None,
    max_samples: int = 5000,
    perplexity: int = 30,
    title: str = "t-SNE Visualization"
):
    """
    Generate t-SNE visualization of learned representations.

    Args:
        model: Model to visualize
        data_loader: DataLoader for data
        device: Device
        save_path: Path to save plot
        layer_index: Which layer to visualize (None = last layer)
        max_samples: Maximum samples to use (t-SNE is slow on large datasets)
        perplexity: t-SNE perplexity parameter
        title: Plot title
    """
    if not TSNE_AVAILABLE:
        print("Error: scikit-learn not installed. Cannot generate t-SNE.")
        return

    model.eval()

    print("Extracting features for t-SNE...")

    all_features = []
    all_labels = []
    num_samples = 0

    for images, labels in data_loader:
        if num_samples >= max_samples:
            break

        images = images.view(images.size(0), -1).to(device)

        # Extract features
        if hasattr(model, 'ff_layers'):
            # Hybrid model - extract from FF layers
            h = images
            num_layers = len(model.ff_layers) if layer_index is None else layer_index + 1

            for i in range(num_layers):
                h = model.ff_layers[i](h)

            features = h
        else:
            # Standard model
            features = model(images)

        all_features.append(features.cpu())
        all_labels.append(labels)

        num_samples += images.size(0)

    # Concatenate and truncate to max_samples
    all_features = torch.cat(all_features, dim=0)[:max_samples]
    all_labels = torch.cat(all_labels, dim=0)[:max_samples]

    print(f"Running t-SNE on {all_features.shape[0]} samples...")
    print(f"Feature dimension: {all_features.shape[1]}")

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, verbose=1)
    features_2d = tsne.fit_transform(all_features.numpy())

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    num_classes = all_labels.max().item() + 1
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for class_idx in range(num_classes):
        mask = all_labels == class_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[class_idx]],
            label=f'Class {class_idx}',
            alpha=0.6,
            s=20
        )

    ax.set_xlabel('t-SNE dimension 1')
    ax.set_ylabel('t-SNE dimension 2')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved t-SNE visualization to: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_layer_goodness(
    goodness_stats: Dict,
    save_path: Optional[Path] = None,
    title: str = "Layer-wise Goodness Statistics"
):
    """
    Plot goodness statistics across layers.

    Args:
        goodness_stats: Dictionary from layer_wise_goodness_analysis
        save_path: Path to save plot
        title: Plot title
    """
    num_layers = len(goodness_stats)
    layers = list(range(num_layers))

    means = [goodness_stats[f'layer_{i}']['mean'] for i in layers]
    stds = [goodness_stats[f'layer_{i}']['std'] for i in layers]
    mins = [goodness_stats[f'layer_{i}']['min'] for i in layers]
    maxs = [goodness_stats[f'layer_{i}']['max'] for i in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot mean +/- std
    ax1.errorbar(layers, means, yerr=stds, marker='o', capsize=5, linewidth=2)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Goodness (mean ± std)')
    ax1.set_title('Mean Goodness per Layer')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(layers)

    # Plot min/max range
    ax2.fill_between(layers, mins, maxs, alpha=0.3, label='Range')
    ax2.plot(layers, means, 'r-', marker='o', linewidth=2, label='Mean')
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Goodness')
    ax2.set_title('Goodness Range per Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(layers)

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved goodness plot to: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_approaches_plot(
    results_dict: Dict[str, Dict],
    metric: str = 'final_test_acc',
    save_path: Optional[Path] = None,
    title: str = "Approach Comparison"
):
    """
    Create bar plot comparing different approaches.

    Args:
        results_dict: Dictionary mapping approach names to result dictionaries
        metric: Which metric to compare
        save_path: Path to save plot
        title: Plot title
    """
    approaches = list(results_dict.keys())
    values = [results_dict[approach].get(metric, 0) for approach in approaches]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(approaches, values, color='skyblue', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)

    # Rotate x labels if too many
    if len(approaches) > 5:
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()

    plt.close()
