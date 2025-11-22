#!/usr/bin/env python3
"""
Training script for ByteFormer on CIFAR-10.

This script trains a ByteFormer model on byte-level image data
for the format-agnostic zero-shot transfer experiment.

Usage:
    python src/training/train.py [--config CONFIG_NAME]

Examples:
    python src/training/train.py --config jpeg_q75
    python src/training/train.py --config quick_test
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

from src.data.byte_dataset import create_dataloaders
from src.models.byteformer import create_byteformer_cifar10
from src.training.config import (
    TrainingConfig,
    get_config_jpeg_q75,
    get_config_quick_test,
    get_config_png,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_optimizer(model, config: TrainingConfig):
    """Create optimizer."""
    if config.optimizer == 'adamw':
        return optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer == 'adam':
        return optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")


def get_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    """Create learning rate scheduler."""
    if config.scheduler == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs * steps_per_epoch,
            eta_min=config.min_lr
        )
    elif config.scheduler == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30 * steps_per_epoch,
            gamma=0.1
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")


def train_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch,
    config,
    writer,
    scaler=None
):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs}")

    for batch_idx, (byte_seq, labels) in enumerate(pbar):
        # Move to device
        byte_seq = byte_seq.to(device)
        labels = labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(byte_seq)
                loss = criterion(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient clipping
            if config.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (no mixed precision)
            logits = model(byte_seq)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            # Optimizer step
            optimizer.step()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Calculate accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Track loss
        total_loss += loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })

        # Log to tensorboard
        if batch_idx % config.log_every == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc', 100.*correct/total, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

    # Epoch statistics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for byte_seq, labels in tqdm(test_loader, desc="Evaluating"):
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

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def save_checkpoint(model, optimizer, epoch, best_acc, config, filename):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': config.to_dict(),
    }
    torch.save(checkpoint, filename)


def main(args):
    """Main training function."""

    # Load configuration
    if args.config == 'jpeg_q75':
        config = get_config_jpeg_q75()
    elif args.config == 'quick_test':
        config = get_config_quick_test()
    elif args.config == 'png':
        config = get_config_png()
    elif args.config:
        # Load from file
        config = TrainingConfig.load(args.config)
    else:
        config = get_config_jpeg_q75()

    # Override config from command line
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr

    # Set random seed
    set_seed(config.seed)

    # Device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Experiment: {config.experiment_name}")
    print(f"Format: {config.data_format}")
    print(f"Model: d_model={config.d_model}, layers={config.num_layers}")
    print(f"Training: {config.num_epochs} epochs, batch_size={config.batch_size}")
    print(f"Learning rate: {config.learning_rate}, weight_decay={config.weight_decay}")
    print(f"Output: {config.output_dir}")
    print("=" * 70)

    # Save configuration
    config.save(Path(config.output_dir) / 'config.json')

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        train_dir=config.train_dir,
        test_dir=config.test_dir,
        batch_size=config.batch_size,
        max_bytes=config.max_bytes,
        num_workers=config.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    model = create_byteformer_cifar10(
        max_bytes=config.max_bytes,
        model_size='small'
    )
    model = model.to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Compile model (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = get_optimizer(model, config)

    # Scheduler
    scheduler = get_scheduler(optimizer, config, len(train_loader))

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None

    # Tensorboard writer
    writer = SummaryWriter(config.log_dir)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, config, writer, scaler
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Log
        print(f"\nEpoch {epoch}/{config.num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                model, optimizer, epoch, best_acc, config,
                Path(config.checkpoint_dir) / 'best_model.pth'
            )
            print(f"  ✓ New best accuracy: {best_acc:.2f}%")

        # Save periodic checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, best_acc, config,
                Path(config.checkpoint_dir) / f'checkpoint_epoch{epoch}.pth'
            )

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Time: {elapsed_time/3600:.2f} hours")
    print(f"Model saved to: {config.checkpoint_dir}")
    print("=" * 70)

    # Save final model
    save_checkpoint(
        model, optimizer, config.num_epochs, best_acc, config,
        Path(config.checkpoint_dir) / 'final_model.pth'
    )

    # Save training summary
    summary = {
        'experiment_name': config.experiment_name,
        'best_test_acc': best_acc,
        'training_time_hours': elapsed_time / 3600,
        'total_epochs': config.num_epochs,
        'config': config.to_dict(),
    }

    with open(Path(config.output_dir) / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ByteFormer on CIFAR-10')

    parser.add_argument('--config', type=str, default='jpeg_q75',
                        help='Config name (jpeg_q75, quick_test, png) or path to config file')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')

    args = parser.parse_args()

    main(args)
