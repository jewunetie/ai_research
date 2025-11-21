"""Standard backpropagation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path


class BPTrainer:
    """Trainer for standard backpropagation."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam"
    ):
        """
        Initialize BP trainer.

        Args:
            model: Model to train
            device: Device to train on
            learning_rate: Learning rate
            optimizer_type: "adam" or "sgd"
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train_epoch(self, dataloader, epoch: int):
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss
            avg_acc: Average accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [BP Train]")

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Flatten images
            images = images.view(images.size(0), -1)

            # Forward pass
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * correct / total

        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)

        return avg_loss, avg_acc

    def validate(self, dataloader):
        """
        Validate model.

        Args:
            dataloader: Validation dataloader

        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)

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
        print(f"Training with Backpropagation")
        print(f"{'='*60}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Optimizer: {type(self.optimizer).__name__}")
        print(f"{'='*60}\n")

        best_val_acc = 0.0

        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate(val_loader)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        print(f"\nBest Val Accuracy: {best_val_acc:.2f}%")

        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accs = checkpoint.get('val_accs', [])


def train_with_backprop(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    num_epochs: int = 100,
    **kwargs
):
    """
    Convenience function to train with backprop.

    Args:
        model: Model to train
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device
        num_epochs: Number of epochs
        **kwargs: Additional arguments for BPTrainer

    Returns:
        trainer: Trained BP trainer
        history: Training history
    """
    trainer = BPTrainer(model, device, **kwargs)
    history = trainer.train(train_loader, val_loader, num_epochs)
    return trainer, history
