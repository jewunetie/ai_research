"""Autoencoder pretraining and fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple


class AutoencoderTrainer:
    """
    Trainer for Autoencoder baseline.

    Implements two-phase training:
    1. Pretrain autoencoder with MSE reconstruction loss
    2. Fine-tune classifier on top of frozen encoder
    """

    def __init__(
        self,
        autoencoder: nn.Module,
        device: torch.device,
        pretrain_lr: float = 0.001,
        finetune_lr: float = 0.001,
        optimizer_type: str = "adam"
    ):
        """
        Initialize Autoencoder trainer.

        Args:
            autoencoder: Autoencoder model
            device: Device to train on
            pretrain_lr: Learning rate for pretraining
            finetune_lr: Learning rate for fine-tuning
            optimizer_type: "adam" or "sgd"
        """
        self.autoencoder = autoencoder
        self.device = device
        self.pretrain_lr = pretrain_lr
        self.finetune_lr = finetune_lr
        self.optimizer_type = optimizer_type

        # Will be set during pretraining
        self.pretrain_optimizer = None
        self.finetune_optimizer = None
        self.classifier_model = None

        # History tracking
        self.pretrain_losses = []
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def pretrain_epoch(self, dataloader, epoch: int) -> float:
        """
        Pretrain autoencoder for one epoch with MSE reconstruction loss.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            avg_loss: Average reconstruction loss
        """
        self.autoencoder.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Pretrain AE]")

        for images, _ in pbar:  # Ignore labels during pretraining
            images = images.to(self.device)
            images = images.view(images.size(0), -1)  # Flatten

            # Forward pass
            reconstruction, latent = self.autoencoder(images)

            # MSE reconstruction loss
            loss = F.mse_loss(reconstruction, images)

            # Backward pass
            self.pretrain_optimizer.zero_grad()
            loss.backward()
            self.pretrain_optimizer.step()

            # Statistics
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        self.pretrain_losses.append(avg_loss)

        return avg_loss

    def pretrain(
        self,
        train_loader,
        num_epochs: int,
        val_loader=None
    ) -> Dict:
        """
        Pretrain the autoencoder on unlabeled data.

        Args:
            train_loader: Training data loader
            num_epochs: Number of pretraining epochs
            val_loader: Optional validation loader for monitoring

        Returns:
            Dictionary with pretraining history
        """
        print("\n" + "="*60)
        print("PHASE 1: Autoencoder Pretraining (Unsupervised)")
        print("="*60)
        print(f"Pretraining for {num_epochs} epochs with MSE reconstruction loss")
        print(f"Learning rate: {self.pretrain_lr}")
        print()

        # Create optimizer for pretraining
        if self.optimizer_type == "adam":
            self.pretrain_optimizer = torch.optim.Adam(
                self.autoencoder.parameters(),
                lr=self.pretrain_lr
            )
        elif self.optimizer_type == "sgd":
            self.pretrain_optimizer = torch.optim.SGD(
                self.autoencoder.parameters(),
                lr=self.pretrain_lr,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        best_loss = float('inf')

        for epoch in range(num_epochs):
            # Pretrain for one epoch
            train_loss = self.pretrain_epoch(train_loader, epoch)

            # Optionally validate
            if val_loader is not None and (epoch + 1) % 10 == 0:
                val_loss = self._validate_reconstruction(val_loader)
                print(f"  Validation MSE: {val_loss:.6f}")

                if val_loss < best_loss:
                    best_loss = val_loss
            else:
                val_loss = None

            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}")

        print(f"\nPretraining complete! Final MSE: {train_loss:.6f}")
        print("="*60 + "\n")

        return {
            'pretrain_losses': self.pretrain_losses,
            'final_pretrain_loss': train_loss,
            'best_val_loss': best_loss if val_loader else None
        }

    def _validate_reconstruction(self, dataloader) -> float:
        """
        Validate reconstruction quality.

        Args:
            dataloader: Validation dataloader

        Returns:
            avg_loss: Average reconstruction loss
        """
        self.autoencoder.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                images = images.view(images.size(0), -1)

                reconstruction, _ = self.autoencoder(images)
                loss = F.mse_loss(reconstruction, images)

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def finetune_epoch(self, dataloader, epoch: int) -> Tuple[float, float]:
        """
        Fine-tune classifier for one epoch.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            avg_loss: Average loss
            avg_acc: Average accuracy
        """
        self.classifier_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [Finetune]")

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            images = images.view(images.size(0), -1)

            # Forward pass
            logits = self.classifier_model(images)
            loss = F.cross_entropy(logits, labels)

            # Backward pass
            self.finetune_optimizer.zero_grad()
            loss.backward()
            self.finetune_optimizer.step()

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

    def validate(self, dataloader) -> Tuple[float, float]:
        """
        Validate classifier.

        Args:
            dataloader: Validation dataloader

        Returns:
            avg_loss: Average validation loss
            avg_acc: Average validation accuracy
        """
        self.classifier_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)

                logits = self.classifier_model(images)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def finetune(
        self,
        classifier_model,
        train_loader,
        val_loader,
        num_epochs: int
    ) -> Dict:
        """
        Fine-tune classifier on top of pretrained encoder.

        Args:
            classifier_model: AutoencoderClassifier model
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of fine-tuning epochs

        Returns:
            Dictionary with fine-tuning history
        """
        print("\n" + "="*60)
        print("PHASE 2: Classifier Fine-tuning (Supervised)")
        print("="*60)
        print(f"Fine-tuning for {num_epochs} epochs")
        print(f"Learning rate: {self.finetune_lr}")
        print(f"Encoder: FROZEN (used as feature extractor)")
        print()

        self.classifier_model = classifier_model

        # Create optimizer for fine-tuning (only classifier parameters)
        if self.optimizer_type == "adam":
            self.finetune_optimizer = torch.optim.Adam(
                classifier_model.parameters(),
                lr=self.finetune_lr
            )
        elif self.optimizer_type == "sgd":
            self.finetune_optimizer = torch.optim.SGD(
                classifier_model.parameters(),
                lr=self.finetune_lr,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.finetune_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Track best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")

        print(f"\nFine-tuning complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})")
        print("="*60 + "\n")

        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }

    def train(
        self,
        train_loader,
        val_loader,
        pretrain_epochs: int,
        finetune_epochs: int,
        num_classes: int
    ) -> Dict:
        """
        Complete two-phase training: pretrain + finetune.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            pretrain_epochs: Number of pretraining epochs
            finetune_epochs: Number of fine-tuning epochs
            num_classes: Number of output classes

        Returns:
            Dictionary with complete training history
        """
        # Phase 1: Pretrain autoencoder
        pretrain_history = self.pretrain(
            train_loader=train_loader,
            num_epochs=pretrain_epochs,
            val_loader=val_loader
        )

        # Create classifier on top of pretrained encoder
        from src.models.mlp import AutoencoderClassifier
        classifier_model = AutoencoderClassifier(
            autoencoder=self.autoencoder,
            num_classes=num_classes,
            freeze_encoder=True
        ).to(self.device)

        # Phase 2: Fine-tune classifier
        finetune_history = self.finetune(
            classifier_model=classifier_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=finetune_epochs
        )

        # Combine histories
        return {
            'pretrain': pretrain_history,
            'finetune': finetune_history,
            'final_val_acc': finetune_history['val_accs'][-1],
            'best_val_acc': finetune_history['best_val_acc'],
            'best_epoch': finetune_history['best_epoch']
        }
