"""Sequential Phased Training: FF Pretraining → BP Fine-tuning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path

from .ff_trainer import FFTrainer
from .bp_trainer import BPTrainer


class SequentialPhasedTrainer:
    """
    Sequential Phased Training Pipeline.

    Phase 1: FF pretraining (unsupervised)
    Phase 2a: BP classifier training (FF layers frozen)
    Phase 2b: BP fine-tuning (all layers)
    """

    def __init__(
        self,
        model,  # HybridFFBPModel
        device: torch.device,
        config: dict
    ):
        """
        Initialize sequential phased trainer.

        Args:
            model: Hybrid FF+BP model
            device: Device to train on
            config: Configuration dictionary with phase1, phase2a, phase2b settings
        """
        self.model = model
        self.device = device
        self.config = config

        self.phase1_history = None
        self.phase2a_history = None
        self.phase2b_history = None

        # Basic validation
        self._validate_model_and_config()

    def _validate_model_and_config(self):
        """Validate model structure and config."""
        # Check model has required attributes
        if not hasattr(self.model, 'ff_layers'):
            raise ValueError("Model must have 'ff_layers' attribute (use HybridFFBPModel)")
        if not hasattr(self.model, 'classifier'):
            raise ValueError("Model must have 'classifier' attribute (use HybridFFBPModel)")

        # Check config has required keys
        if 'phase1' not in self.config and 'phase2a' not in self.config:
            raise ValueError("Config must have at least 'phase1' or 'phase2a' settings")

        # Warn if all phases are disabled
        phase1_epochs = self.config.get('phase1', {}).get('epochs', 0)
        phase2a_epochs = self.config.get('phase2a', {}).get('epochs', 0)
        phase2b_enabled = self.config.get('phase2b', {}).get('enabled', False)
        phase2b_epochs = self.config.get('phase2b', {}).get('epochs', 0)

        if phase1_epochs == 0 and phase2a_epochs == 0 and (not phase2b_enabled or phase2b_epochs == 0):
            print("WARNING: All training phases are disabled (all epochs = 0)")

    def phase1_ff_pretraining(self, train_loader, val_loader):
        """
        Phase 1: FF Pretraining (Unsupervised).

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader

        Returns:
            history: Training history
        """
        print(f"\n{'#'*60}")
        print("PHASE 1: Forward-Forward Pretraining (Unsupervised)")
        print(f"{'#'*60}\n")

        phase1_config = self.config.get('phase1', {})

        # Create FF network from the FF layers
        from ..models.ff_layer import FFNetwork
        layer_dims = [self.model.ff_layers[0].linear.in_features] + \
                     [layer.linear.out_features for layer in self.model.ff_layers]

        ff_network = FFNetwork(
            layer_dims=layer_dims,
            threshold=phase1_config.get('threshold', 2.0),
            normalize_between_layers=True
        ).to(self.device)

        # Train with FF
        # Infer num_classes from classifier output dimension
        num_classes = self.model.classifier.out_features

        trainer = FFTrainer(
            model=ff_network,
            device=self.device,
            threshold=phase1_config.get('threshold', 2.0),
            negative_strategy=phase1_config.get('negative_strategy', 'random_label'),
            num_classes=num_classes,
            learning_rate=phase1_config.get('learning_rate', 0.03)
        )

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=phase1_config.get('epochs', 100)
        )

        # Transfer weights to hybrid model
        for i, layer in enumerate(self.model.ff_layers):
            layer.linear.weight.data = ff_network.layers[i].linear.weight.data.clone()
            layer.linear.bias.data = ff_network.layers[i].linear.bias.data.clone()

        print(f"\n✓ Phase 1 complete. FF layers pretrained.")

        self.phase1_history = history
        return history

    def phase2a_frozen_ff_bp_classifier(self, train_loader, val_loader):
        """
        Phase 2a: Train BP classifier with frozen FF layers.

        Args:
            train_loader: Training dataloader (with labels)
            val_loader: Validation dataloader (with labels)

        Returns:
            history: Training history
        """
        print(f"\n{'#'*60}")
        print("PHASE 2a: Train Classifier (FF Layers Frozen)")
        print(f"{'#'*60}\n")

        phase2a_config = self.config.get('phase2a', {})

        # Freeze FF layers
        self.model.freeze_ff_layers()

        # Create optimizer for classifier only
        optimizer = torch.optim.Adam(
            self.model.classifier.parameters(),
            lr=phase2a_config.get('learning_rate', 0.001)
        )

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        num_epochs = phase2a_config.get('epochs', 50)

        for epoch in range(num_epochs):
            # Train
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Phase 2a Train]")

            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)

                # Forward
                logits = self.model(images)
                loss = F.cross_entropy(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Stats
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({'loss': loss.item(), 'acc': 100.0 * correct / total})

            train_loss = total_loss / len(train_loader)
            train_acc = 100.0 * correct / total
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validate
            self.model.eval()
            val_loss, val_acc = self._validate(val_loader)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        print(f"\n✓ Phase 2a complete. Classifier trained with frozen FF layers.")

        history = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        }

        self.phase2a_history = history
        return history

    def phase2b_finetune_all(self, train_loader, val_loader):
        """
        Phase 2b: Fine-tune all layers with BP.

        Args:
            train_loader: Training dataloader (with labels)
            val_loader: Validation dataloader (with labels)

        Returns:
            history: Training history
        """
        print(f"\n{'#'*60}")
        print("PHASE 2b: Fine-tune All Layers (End-to-End BP)")
        print(f"{'#'*60}\n")

        phase2b_config = self.config.get('phase2b', {})

        # Unfreeze FF layers
        self.model.unfreeze_ff_layers()

        # Train with standard BP
        trainer = BPTrainer(
            model=self.model,
            device=self.device,
            learning_rate=phase2b_config.get('learning_rate', 0.0001),
            optimizer_type=phase2b_config.get('optimizer', 'adam')
        )

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=phase2b_config.get('epochs', 50)
        )

        print(f"\n✓ Phase 2b complete. All layers fine-tuned.")

        self.phase2b_history = history
        return history

    def train(self, train_loader, val_loader, train_loader_supervised, val_loader_supervised):
        """
        Full sequential phased training pipeline.

        Args:
            train_loader: Unsupervised training dataloader (for Phase 1)
            val_loader: Unsupervised validation dataloader (for Phase 1)
            train_loader_supervised: Supervised training dataloader (for Phase 2)
            val_loader_supervised: Supervised validation dataloader (for Phase 2)

        Returns:
            history: Combined history from all phases
        """
        print(f"\n{'='*60}")
        print("SEQUENTIAL PHASED TRAINING")
        print(f"{'='*60}\n")

        # Phase 1: FF Pretraining
        if self.config.get('phase1', {}).get('epochs', 0) > 0:
            self.phase1_ff_pretraining(train_loader, val_loader)
        else:
            print("Skipping Phase 1 (epochs=0)")

        # Phase 2a: Frozen FF + BP Classifier
        if self.config.get('phase2a', {}).get('epochs', 0) > 0:
            self.phase2a_frozen_ff_bp_classifier(train_loader_supervised, val_loader_supervised)
        else:
            print("Skipping Phase 2a (epochs=0)")

        # Phase 2b: Fine-tune All
        if self.config.get('phase2b', {}).get('enabled', False) and \
           self.config.get('phase2b', {}).get('epochs', 0) > 0:
            self.phase2b_finetune_all(train_loader_supervised, val_loader_supervised)
        else:
            print("Skipping Phase 2b (disabled or epochs=0)")

        print(f"\n{'='*60}")
        print("SEQUENTIAL PHASED TRAINING COMPLETE")
        print(f"{'='*60}\n")

        return {
            'phase1': self.phase1_history,
            'phase2a': self.phase2a_history,
            'phase2b': self.phase2b_history
        }

    def _validate(self, dataloader):
        """Helper function to validate model."""
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

        return avg_loss, avg_acc

    def save_checkpoint(self, path: str):
        """Save checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'phase1_history': self.phase1_history,
            'phase2a_history': self.phase2a_history,
            'phase2b_history': self.phase2b_history,
        }, path)
