#!/usr/bin/env python3
"""
Pixel-level baseline: ResNet-18 on CIFAR-10.

This script trains a standard ResNet-18 on decoded pixel arrays
for comparison with byte-level ByteFormer.

Usage:
    python src/baselines/train_resnet18.py [--epochs 100] [--batch-size 128]

Example:
    python src/baselines/train_resnet18.py --epochs 100
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
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_resnet18_cifar10(num_classes: int = 10):
    """
    Create ResNet-18 adapted for CIFAR-10 (32×32 images).

    The original ResNet-18 is designed for ImageNet (224×224),
    so we adapt the first layer for smaller images.
    """
    model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    # Adapt first conv layer for CIFAR-10 (32×32 instead of 224×224)
    # Original: kernel_size=7, stride=2
    # CIFAR-10: kernel_size=3, stride=1 (no downsampling)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    # Remove maxpool (too aggressive for 32×32)
    model.maxpool = nn.Identity()

    return model


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch, config, writer):
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']}")

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

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

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    """Main training function."""

    # Configuration
    config = {
        'model': 'resnet18',
        'dataset': 'cifar10',
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': 0.9,
        'scheduler': 'cosine',
        'seed': 42,
        'output_dir': args.output_dir,
    }

    # Set seed
    set_seed(config['seed'])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "=" * 70)
    print("ResNet-18 Baseline Training on CIFAR-10")
    print("=" * 70)
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print("=" * 70)

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data augmentation (standard CIFAR-10)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10
    print("\nLoading CIFAR-10...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data/raw',
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data/raw',
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating ResNet-18 model...")
    model = get_resnet18_cifar10(num_classes=10)
    model = model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['learning_rate'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay']
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] * len(train_loader)
    )

    # TensorBoard
    writer = SummaryWriter(output_dir / 'logs')

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, epoch, config, writer
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Log
        print(f"\nEpoch {epoch}/{config['num_epochs']}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/acc', test_acc, epoch)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config,
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ New best accuracy: {best_acc:.2f}%")

        # Save periodic checkpoint
        if epoch % 25 == 0:
            torch.save(checkpoint, output_dir / f'checkpoint_epoch{epoch}.pth')

    # Training complete
    elapsed_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Total Time: {elapsed_time/3600:.2f} hours")
    print(f"Model saved to: {output_dir}")
    print("=" * 70)

    # Save final model
    final_checkpoint = {
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
        'config': config,
    }
    torch.save(final_checkpoint, output_dir / 'final_model.pth')

    # Save training summary
    summary = {
        'model': 'resnet18',
        'dataset': 'cifar10',
        'best_test_acc': best_acc,
        'training_time_hours': elapsed_time / 3600,
        'total_epochs': config['num_epochs'],
        'parameters': count_parameters(model),
        'config': config,
    }

    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    writer.close()

    print("\nSummary saved to:", output_dir / 'training_summary.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-18 on CIFAR-10')

    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate (default: 0.1)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--output-dir', type=str,
                        default='./experiments/baselines/resnet18_pixels',
                        help='Output directory')

    args = parser.parse_args()

    main(args)
