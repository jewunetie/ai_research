#!/usr/bin/env python3
"""
Zero-shot evaluation script for format-agnostic transfer experiment.

This script evaluates a ByteFormer model trained on one format (e.g., JPEG)
on multiple formats (JPEG, PNG, WebP, BMP) without fine-tuning.

This is the CORE experiment for Hypothesis 1 (H1):
"A byte-level model trained on JPEG will achieve >50% of its in-distribution
accuracy when tested zero-shot on PNG/WebP/BMP formats."

Usage:
    python src/training/evaluate.py --checkpoint PATH --formats jpeg_q75 png webp bmp

Example:
    python src/training/evaluate.py \
        --checkpoint experiments/h1_zero_shot_transfer/jpeg_q75/checkpoints/best_model.pth \
        --formats jpeg_q75 png webp bmp
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

# Import project modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.byte_dataset import create_dataloaders
from src.models.byteformer import ByteFormer
from src.training.config import TrainingConfig


@torch.no_grad()
def evaluate_format(
    model,
    data_dir,
    device,
    batch_size=64,
    max_bytes=8192,
    num_workers=4
):
    """
    Evaluate model on a specific format.

    Args:
        model: ByteFormer model
        data_dir: Directory containing test images
        device: Device to run on
        batch_size: Batch size
        max_bytes: Max byte sequence length
        num_workers: Number of dataloader workers

    Returns:
        Dict containing accuracy and per-class accuracies
    """
    from torch.utils.data import DataLoader
    from src.data.byte_dataset import ByteImageDataset

    # Create dataset
    dataset = ByteImageDataset(data_dir, max_bytes=max_bytes)

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class statistics
    class_correct = [0] * 10
    class_total = [0] * 10

    for byte_seq, labels in tqdm(loader, desc=f"Evaluating"):
        byte_seq = byte_seq.to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(byte_seq)
        loss = criterion(logits, labels)

        # Calculate accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

    # Overall accuracy
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(loader)

    # Per-class accuracy
    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            class_accuracies[i] = 100. * class_correct[i] / class_total[i]
        else:
            class_accuracies[i] = 0.0

    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'total_samples': total,
        'correct_samples': correct,
        'class_accuracies': class_accuracies,
    }


def main(args):
    """Main evaluation function."""

    print("=" * 70)
    print("Zero-Shot Format Transfer Evaluation")
    print("=" * 70)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Reconstruct config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Remove list fields that cause issues
        if 'tags' in config_dict:
            del config_dict['tags']
        config = TrainingConfig(**config_dict)
    else:
        # Use default config
        from src.training.config import get_config_jpeg_q75
        config = get_config_jpeg_q75()

    print(f"Trained on format: {config.data_format}")
    print(f"Best training accuracy: {checkpoint.get('best_acc', 'N/A')}")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model
    print("\nCreating model...")
    model = ByteFormer(
        max_bytes=config.max_bytes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        num_classes=config.num_classes,
        dropout=config.dropout
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {model.count_parameters():,}")

    # Evaluate on each format
    print("\n" + "=" * 70)
    print("Evaluating on formats:")
    print("=" * 70)

    results = {}
    base_data_dir = Path(args.data_dir)

    for format_name in args.formats:
        print(f"\n{'='*70}")
        print(f"Format: {format_name.upper()}")
        print(f"{'='*70}")

        # Construct data directory
        test_dir = base_data_dir / format_name / 'test'

        if not test_dir.exists():
            print(f"  ⚠️  Directory not found: {test_dir}")
            print(f"  Skipping {format_name}")
            continue

        # Evaluate
        format_results = evaluate_format(
            model=model,
            data_dir=str(test_dir),
            device=device,
            batch_size=args.batch_size,
            max_bytes=config.max_bytes,
            num_workers=args.num_workers
        )

        results[format_name] = format_results

        # Print results
        print(f"\n  Test Accuracy: {format_results['accuracy']:.2f}%")
        print(f"  Test Loss: {format_results['loss']:.4f}")
        print(f"  Samples: {format_results['correct_samples']}/{format_results['total_samples']}")

    # Analysis: Transfer ratios
    print("\n" + "=" * 70)
    print("ZERO-SHOT TRANSFER ANALYSIS")
    print("=" * 70)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Get trained format accuracy (baseline)
    trained_format = config.data_format
    if trained_format in results:
        baseline_acc = results[trained_format]['accuracy']
        print(f"\nBaseline (trained on {trained_format}): {baseline_acc:.2f}%")
        print()
        print(f"{'Format':<15} {'Accuracy':<12} {'Transfer Ratio':<15} {'Interpretation':<20}")
        print("-" * 70)

        for format_name, format_results in results.items():
            acc = format_results['accuracy']
            transfer_ratio = (acc / baseline_acc) if baseline_acc > 0 else 0

            # Interpretation
            if format_name == trained_format:
                interpretation = "(in-distribution)"
            elif transfer_ratio >= 0.7:
                interpretation = "✓ Strong transfer"
            elif transfer_ratio >= 0.4:
                interpretation = "~ Partial transfer"
            else:
                interpretation = "✗ Weak transfer"

            print(f"{format_name:<15} {acc:>6.2f}%       {transfer_ratio:>6.2%}         {interpretation:<20}")

        # H1 Success Criteria Check
        print("\n" + "=" * 70)
        print("HYPOTHESIS 1 (H1) EVALUATION")
        print("=" * 70)
        print("H1: Model achieves >50% of in-distribution accuracy on zero-shot formats")
        print()

        # Check each zero-shot format
        zero_shot_formats = [f for f in results.keys() if f != trained_format]

        for format_name in zero_shot_formats:
            acc = results[format_name]['accuracy']
            transfer_ratio = (acc / baseline_acc) if baseline_acc > 0 else 0
            h1_threshold = 0.5

            status = "✓ PASS" if transfer_ratio >= h1_threshold else "✗ FAIL"
            print(f"  {format_name:<15}: {transfer_ratio:>6.2%}  {status}")

        # Overall H1 result
        all_pass = all(
            (results[f]['accuracy'] / baseline_acc) >= 0.5
            for f in zero_shot_formats
        )

        print("\n" + "-" * 70)
        if all_pass:
            print("  🎉 H1 CONFIRMED: Format-agnostic learning detected!")
            print("  → Model learned content, not just format-specific patterns")
        else:
            print("  📊 H1 REJECTED: Format-specific learning detected")
            print("  → Model relies on format-specific byte patterns")

    # Per-class analysis
    print("\n" + "=" * 70)
    print("PER-CLASS TRANSFER ANALYSIS")
    print("=" * 70)

    print(f"\n{'Class':<12}", end="")
    for format_name in results.keys():
        print(f"{format_name:>12}", end="")
    print()
    print("-" * (12 + 12 * len(results)))

    for class_idx, class_name in enumerate(class_names):
        print(f"{class_name:<12}", end="")
        for format_name in results.keys():
            class_acc = results[format_name]['class_accuracies'][class_idx]
            print(f"{class_acc:>11.1f}%", end="")
        print()

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / 'zero_shot_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'trained_format': trained_format,
            'formats_evaluated': list(results.keys()),
            'results': {
                fmt: {
                    'accuracy': res['accuracy'],
                    'loss': res['loss'],
                    'total_samples': res['total_samples'],
                    'class_accuracies': res['class_accuracies'],
                }
                for fmt, res in results.items()
            },
            'transfer_ratios': {
                fmt: results[fmt]['accuracy'] / results[trained_format]['accuracy']
                for fmt in results.keys()
                if trained_format in results and results[trained_format]['accuracy'] > 0
            },
        }, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Results saved to: {results_file}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate zero-shot format transfer'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--formats', nargs='+',
                        default=['jpeg_q75', 'png', 'webp', 'bmp'],
                        help='Formats to evaluate on')
    parser.add_argument('--data-dir', type=str, default='./data/cifar10',
                        help='Base data directory')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/h1_zero_shot_transfer/results',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')

    args = parser.parse_args()

    main(args)
