#!/usr/bin/env python3
"""
Train ByteFormer on mixed formats.

This script trains ByteFormer on a mix of image formats (JPEG, PNG, WebP, BMP)
to test if mixed-format training improves format generalization.

Usage:
    python src/training/train_mixed_formats.py [--formats jpeg_q75 png webp bmp]

Example:
    python src/training/train_mixed_formats.py --formats jpeg_q75 png --epochs 100
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.mixed_format_dataset import create_mixed_format_dataloaders
from src.models.byteformer import create_byteformer_cifar10
from src.training.config import TrainingConfig


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    num_epochs,
    writer,
    scaler=None
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    # Track per-format statistics
    format_stats = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")

    for batch_idx, (byte_seq, labels, formats) in enumerate(pbar):
        byte_seq = byte_seq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(byte_seq)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(byte_seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Calculate accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

        # Track format statistics
        for i, fmt in enumerate(formats):
            if fmt not in format_stats:
                format_stats[fmt] = {'correct': 0, 'total': 0}
            format_stats[fmt]['total'] += 1
            if predicted[i] == labels[i]:
                format_stats[fmt]['correct'] += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # Log to tensorboard
        if batch_idx % 50 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc', 100.*correct/total, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    # Log per-format accuracy
    print("\n  Per-format training accuracy:")
    for fmt, stats in format_stats.items():
        fmt_acc = 100. * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"    {fmt}: {fmt_acc:.2f}% ({stats['correct']}/{stats['total']})")

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # Track per-format statistics
    format_stats = {}

    for byte_seq, labels, formats in tqdm(test_loader, desc="Evaluating"):
        byte_seq = byte_seq.to(device)
        labels = labels.to(device)

        logits = model(byte_seq)
        loss = criterion(logits, labels)

        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

        # Track format statistics
        for i, fmt in enumerate(formats):
            if fmt not in format_stats:
                format_stats[fmt] = {'correct': 0, 'total': 0}
            format_stats[fmt]['total'] += 1
            if predicted[i] == labels[i]:
                format_stats[fmt]['correct'] += 1

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    # Per-format accuracy
    format_accs = {}
    for fmt, stats in format_stats.items():
        fmt_acc = 100. * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        format_accs[fmt] = fmt_acc

    return avg_loss, accuracy, format_accs


def main(args):
    """Main training function."""

    # Set seed
    set_seed(42)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("Mixed-Format ByteFormer Training")
    print("=" * 70)
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        'formats': args.formats,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': 1e-3,
        'max_bytes': 8192,
        'd_model': 192,
        'nhead': 6,
        'num_layers': 6,
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader = create_mixed_format_dataloaders(
        base_dir='./data/cifar10',
        formats=args.formats,
        batch_size=args.batch_size,
        max_bytes=8192,
        num_workers=4,
        sample_strategy='uniform'
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_byteformer_cifar10(max_bytes=8192, model_size='small')
    model = model.to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(train_loader)
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, args.epochs, writer, scaler
        )

        # Evaluate
        test_loss, test_acc, format_accs = evaluate(model, test_loader, criterion, device)

        # Log
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"  Per-format test accuracy:")
        for fmt, acc in format_accs.items():
            print(f"    {fmt}: {acc:.2f}%")

        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)

        for fmt, acc in format_accs.items():
            writer.add_scalar(f'test/acc_{fmt}', acc, epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
                'format_accs': format_accs,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ New best accuracy: {best_acc:.2f}%")

    # Training complete
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Time: {elapsed_time/3600:.2f} hours")
    print("=" * 70)

    # Save summary
    summary = {
        'experiment': 'mixed_format_training',
        'formats': args.formats,
        'best_test_acc': best_acc,
        'training_time_hours': elapsed_time / 3600,
        'config': config,
    }

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ByteFormer on mixed formats')

    parser.add_argument('--formats', nargs='+',
                        default=['jpeg_q75', 'png', 'webp', 'bmp'],
                        help='Formats to train on')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/mixed_formats',
                        help='Output directory')

    args = parser.parse_args()

    main(args)
