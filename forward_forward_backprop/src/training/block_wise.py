"""
Block-wise hybrid training (for SFF replication).

This trainer implements the block-wise approach where:
1. Network is divided into blocks
2. Each block has an auxiliary classifier
3. Gradients are detached between blocks
4. Each block is trained with local auxiliary losses

This approach is meant to replicate Supervised Forward-Forward (SFF)
findings for validation purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict


class BlockWiseTrainer:
    """
    Trainer for block-wise hybrid approach.

    Divides network into blocks and trains each block with local auxiliary
    losses, detaching gradients between blocks (similar to SFF).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        optimizer_type: str = "adam",
        aux_loss_weight: float = 1.0,
        final_loss_weight: float = 1.0
    ):
        """
        Initialize BlockWise trainer.

        Args:
            model: BlockWiseMLP model
            device: Device to train on
            learning_rate: Learning rate
            optimizer_type: "adam" or "sgd"
            aux_loss_weight: Weight for auxiliary losses
            final_loss_weight: Weight for final classifier loss
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.aux_loss_weight = aux_loss_weight
        self.final_loss_weight = final_loss_weight

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # History tracking
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        # Track per-block losses
        self.block_losses_history = []

    def train_epoch(self, dataloader, epoch: int) -> tuple[float, float]:
        """
        Train for one epoch with block-wise auxiliary losses.

        Args:
            dataloader: Training dataloader
            epoch: Current epoch number

        Returns:
            avg_loss: Average total loss
            avg_acc: Average accuracy (from final classifier)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Track per-block losses
        num_blocks = self.model.num_blocks
        block_losses = [0.0] * num_blocks

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} [BlockWise]")

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Flatten images
            images = images.view(images.size(0), -1)

            # Forward pass with auxiliary outputs
            final_logits, aux_logits_list = self.model.forward_with_aux(
                images,
                detach_blocks=True  # Detach between blocks (key for block-wise)
            )

            # Compute final classifier loss
            final_loss = F.cross_entropy(final_logits, labels)

            # Compute auxiliary losses for each block
            aux_losses = []
            for block_idx, aux_logits in enumerate(aux_logits_list):
                aux_loss = F.cross_entropy(aux_logits, labels)
                aux_losses.append(aux_loss)
                block_losses[block_idx] += aux_loss.item()

            # Total loss: weighted sum of auxiliary and final losses
            total_aux_loss = sum(aux_losses) * self.aux_loss_weight
            loss = total_aux_loss + final_loss * self.final_loss_weight

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics (use final classifier for accuracy)
            total_loss += loss.item()
            pred = final_logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100.0 * correct / total
            })

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * correct / total

        # Save per-block losses
        avg_block_losses = [bl / len(dataloader) for bl in block_losses]
        self.block_losses_history.append(avg_block_losses)

        self.train_losses.append(avg_loss)
        self.train_accs.append(avg_acc)

        return avg_loss, avg_acc

    def validate(self, dataloader) -> tuple[float, float]:
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

                # Use final classifier for validation (no auxiliary losses)
                logits = self.model(images, detach_blocks=False)
                loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def train(
        self,
        train_loader,
        val_loader,
        num_epochs: int
    ) -> Dict:
        """
        Complete training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs

        Returns:
            Dictionary with training history
        """
        print("\n" + "="*60)
        print("BLOCK-WISE HYBRID TRAINING")
        print("="*60)
        print(f"Num blocks: {self.model.num_blocks}")
        print(f"Block dims: {self.model.block_dims}")
        print(f"Auxiliary loss weight: {self.aux_loss_weight}")
        print(f"Final loss weight: {self.final_loss_weight}")
        print(f"Training for {num_epochs} epochs")
        print()

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(num_epochs):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

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

                # Print per-block losses
                if epoch < len(self.block_losses_history):
                    block_str = ", ".join(
                        [f"B{i+1}={loss:.4f}"
                         for i, loss in enumerate(self.block_losses_history[epoch])]
                    )
                    print(f"  Block losses: {block_str}")

        print(f"\nTraining complete!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch+1})")
        print("="*60 + "\n")

        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'block_losses_history': self.block_losses_history
        }
