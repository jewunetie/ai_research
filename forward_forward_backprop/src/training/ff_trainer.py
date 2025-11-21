"""Forward-Forward training logic."""

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path

from ..data.augmentation import generate_negative_samples


class FFTrainer:
    """Trainer for Forward-Forward algorithm."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        threshold: float = 2.0,
        negative_strategy: str = "random_label",
        num_classes: int = 10,
        learning_rate: float = 0.03,
        normalize_between_layers: bool = True
    ):
        """
        Initialize FF trainer.

        Args:
            model: Model with FF layers
            device: Device to train on
            threshold: Goodness threshold
            negative_strategy: Strategy for generating negatives
            num_classes: Number of classes
            learning_rate: Learning rate
            normalize_between_layers: Whether to normalize between layers
        """
        self.model = model
        self.device = device
        self.threshold = threshold
        self.negative_strategy = negative_strategy
        self.num_classes = num_classes
        self.normalize_between_layers = normalize_between_layers

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, epoch: int):
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss for epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [FF Train]")

        for batch in pbar:
            if len(batch) == 2:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)
            else:
                # Unsupervised (no labels)
                images = batch.to(self.device)
                labels = None

            # Flatten images
            images = images.view(images.size(0), -1)

            # Generate negative samples
            neg_images, neg_labels = generate_negative_samples(
                images, labels, strategy=self.negative_strategy, num_classes=self.num_classes
            )

            # Forward pass - compute FF loss
            if hasattr(self.model, 'ff_loss_layerwise'):
                loss, _ = self.model.ff_loss_layerwise(images, neg_images)
            else:
                # Single layer
                from ..models.ff_layer import ff_threshold_loss, compute_goodness

                h_pos = self.model(images, normalize=self.normalize_between_layers)
                h_neg = self.model(neg_images, normalize=self.normalize_between_layers)

                goodness_pos = compute_goodness(h_pos)
                goodness_neg = compute_goodness(h_neg)

                loss = ff_threshold_loss(goodness_pos, goodness_neg, self.threshold)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self, dataloader):
        """
        Validate model.

        Args:
            dataloader: Validation dataloader

        Returns:
            avg_loss: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:
                    images, labels = batch
                    images, labels = images.to(self.device), labels.to(self.device)
                else:
                    images = batch.to(self.device)
                    labels = None

                images = images.view(images.size(0), -1)

                neg_images, neg_labels = generate_negative_samples(
                    images, labels, strategy=self.negative_strategy, num_classes=self.num_classes
                )

                if hasattr(self.model, 'ff_loss_layerwise'):
                    loss, _ = self.model.ff_loss_layerwise(images, neg_images)
                else:
                    from ..models.ff_layer import ff_threshold_loss, compute_goodness

                    h_pos = self.model(images, normalize=self.normalize_between_layers)
                    h_neg = self.model(neg_images, normalize=self.normalize_between_layers)

                    goodness_pos = compute_goodness(h_pos)
                    goodness_neg = compute_goodness(h_neg)

                    loss = ff_threshold_loss(goodness_pos, goodness_neg, self.threshold)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self, train_loader, val_loader, num_epochs: int):
        """
        Full training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs to train

        Returns:
            history: Dictionary with training history
        """
        print(f"\n{'='*60}")
        print(f"Training with Forward-Forward Algorithm")
        print(f"{'='*60}")
        print(f"Negative Strategy: {self.negative_strategy}")
        print(f"Threshold: {self.threshold}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])


def train_ff_layer(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int = 100,
    **kwargs
):
    """
    Convenience function to train FF model.

    Args:
        model: FF model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device
        num_epochs: Number of epochs
        **kwargs: Additional arguments for FFTrainer

    Returns:
        trainer: Trained FF trainer
        history: Training history
    """
    trainer = FFTrainer(model, device, **kwargs)
    history = trainer.train(train_loader, val_loader, num_epochs)
    return trainer, history
