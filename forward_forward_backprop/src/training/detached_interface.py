"""Detached Interface Training: Simultaneous FF+BP with gradient detachment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from ..data.augmentation import generate_negative_samples
from ..models.ff_layer import ff_threshold_loss, compute_goodness


class DetachedInterfaceTrainer:
    """
    Detached Interface Training.

    FF layers and BP layers train simultaneously, but gradients are detached
    at the interface to prevent gradient flow from BP to FF layers.

    This is a novel approach that maintains FF's local learning properties
    while allowing BP to optimize for the supervised task.
    """

    def __init__(
        self,
        model,  # HybridFFBPModel
        device: torch.device,
        config: dict
    ):
        """
        Initialize detached interface trainer.

        Args:
            model: Hybrid FF+BP model
            device: Device to train on
            config: Configuration with ff_config and bp_config
        """
        self.model = model
        self.device = device
        self.config = config

        # Separate optimizers for FF and BP parts
        ff_config = config.get('ff_config', {})
        bp_config = config.get('bp_config', {})

        self.ff_optimizer = torch.optim.Adam(
            self.model.ff_layers.parameters(),
            lr=ff_config.get('learning_rate', 0.03)
        )

        self.bp_optimizer = torch.optim.Adam(
            self.model.classifier.parameters(),
            lr=bp_config.get('learning_rate', 0.001)
        )

        self.threshold = ff_config.get('threshold', 2.0)
        self.negative_strategy = ff_config.get('negative_strategy', 'random_label')
        self.num_classes = 10

        self.train_ff_losses = []
        self.train_bp_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train_epoch(self, unsup_loader, sup_loader, epoch: int):
        """
        Train for one epoch with simultaneous FF and BP updates.

        Args:
            unsup_loader: Unsupervised dataloader (for FF training)
            sup_loader: Supervised dataloader (for BP training)
            epoch: Current epoch number

        Returns:
            metrics: Dictionary with epoch metrics
        """
        self.model.train()

        total_ff_loss = 0.0
        total_bp_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        # Zip loaders (iterate both simultaneously)
        # If lengths differ, stops at shorter one
        pbar = tqdm(
            zip(unsup_loader, sup_loader),
            desc=f"Epoch {epoch+1} [Detached Interface]",
            total=min(len(unsup_loader), len(sup_loader))
        )

        for unsup_batch, sup_batch in pbar:
            # === FF UPDATE (Unsupervised) ===
            if len(unsup_batch) == 2:
                images_unsup, labels_unsup = unsup_batch
                images_unsup = images_unsup.to(self.device)
                labels_unsup = labels_unsup.to(self.device)
            else:
                images_unsup = unsup_batch.to(self.device)
                labels_unsup = None

            images_unsup = images_unsup.view(images_unsup.size(0), -1)

            # Generate negative samples
            neg_images, neg_labels = generate_negative_samples(
                images_unsup, labels_unsup,
                strategy=self.negative_strategy,
                num_classes=self.num_classes
            )

            # Forward through FF layers
            h_pos = self.model.forward_ff(images_unsup, normalize=True)
            h_neg = self.model.forward_ff(neg_images, normalize=True)

            # Compute FF loss
            goodness_pos = compute_goodness(h_pos)
            goodness_neg = compute_goodness(h_neg)
            ff_loss = ff_threshold_loss(goodness_pos, goodness_neg, self.threshold)

            # Update FF layers
            self.ff_optimizer.zero_grad()
            ff_loss.backward()
            self.ff_optimizer.step()

            total_ff_loss += ff_loss.item()

            # === BP UPDATE (Supervised) ===
            images_sup, labels_sup = sup_batch
            images_sup = images_sup.to(self.device)
            labels_sup = labels_sup.to(self.device)
            images_sup = images_sup.view(images_sup.size(0), -1)

            # Forward through FF layers (DETACHED!)
            with torch.no_grad():
                h_detached = self.model.forward_ff(images_sup, normalize=True).detach()

            # Forward through classifier
            logits = self.model.classifier(h_detached)
            bp_loss = F.cross_entropy(logits, labels_sup)

            # Update classifier only (gradients stopped at detach)
            self.bp_optimizer.zero_grad()
            bp_loss.backward()
            self.bp_optimizer.step()

            total_bp_loss += bp_loss.item()

            # Accuracy
            pred = logits.argmax(dim=1)
            correct += (pred == labels_sup).sum().item()
            total += labels_sup.size(0)

            num_batches += 1

            pbar.set_postfix({
                'ff_loss': ff_loss.item(),
                'bp_loss': bp_loss.item(),
                'acc': 100.0 * correct / total
            })

        avg_ff_loss = total_ff_loss / num_batches
        avg_bp_loss = total_bp_loss / num_batches
        avg_acc = 100.0 * correct / total

        self.train_ff_losses.append(avg_ff_loss)
        self.train_bp_losses.append(avg_bp_loss)
        self.train_accs.append(avg_acc)

        return {
            'ff_loss': avg_ff_loss,
            'bp_loss': avg_bp_loss,
            'acc': avg_acc
        }

    def validate(self, dataloader):
        """
        Validate model.

        Args:
            dataloader: Validation dataloader (supervised)

        Returns:
            val_loss: Validation loss
            val_acc: Validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)

                # Forward through entire model
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * correct / total

        self.val_losses.append(avg_loss)
        self.val_accs.append(avg_acc)

        return avg_loss, avg_acc

    def train(self, unsup_train_loader, sup_train_loader, sup_val_loader, num_epochs: int):
        """
        Full training loop.

        Args:
            unsup_train_loader: Unsupervised training dataloader (for FF)
            sup_train_loader: Supervised training dataloader (for BP)
            sup_val_loader: Supervised validation dataloader
            num_epochs: Number of epochs to train

        Returns:
            history: Training history
        """
        print(f"\n{'='*60}")
        print("DETACHED INTERFACE TRAINING")
        print(f"{'='*60}")
        print("FF layers train with unsupervised local loss")
        print("BP classifier trains with supervised global loss")
        print("Gradient detachment prevents BP from affecting FF")
        print(f"{'='*60}\n")

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            metrics = self.train_epoch(unsup_train_loader, sup_train_loader, epoch)
            val_loss, val_acc = self.validate(sup_val_loader)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"FF Loss: {metrics['ff_loss']:.4f}, "
                  f"BP Loss: {metrics['bp_loss']:.4f}, "
                  f"Train Acc: {metrics['acc']:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")

        return {
            'train_ff_losses': self.train_ff_losses,
            'train_bp_losses': self.train_bp_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc
        }

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ff_optimizer_state_dict': self.ff_optimizer.state_dict(),
            'bp_optimizer_state_dict': self.bp_optimizer.state_dict(),
            'train_ff_losses': self.train_ff_losses,
            'train_bp_losses': self.train_bp_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
        }, path)

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ff_optimizer.load_state_dict(checkpoint['ff_optimizer_state_dict'])
        self.bp_optimizer.load_state_dict(checkpoint['bp_optimizer_state_dict'])
        self.train_ff_losses = checkpoint.get('train_ff_losses', [])
        self.train_bp_losses = checkpoint.get('train_bp_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accs = checkpoint.get('val_accs', [])
