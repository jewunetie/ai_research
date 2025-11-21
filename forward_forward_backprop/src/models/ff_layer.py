"""Forward-Forward layer implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_goodness(h: torch.Tensor) -> torch.Tensor:
    """
    Compute goodness as sum of squared activations.

    Args:
        h: Layer activations [batch_size, hidden_dim]

    Returns:
        goodness: Goodness values [batch_size]
    """
    return (h ** 2).sum(dim=1)


def ff_threshold_loss(
    goodness_pos: torch.Tensor,
    goodness_neg: torch.Tensor,
    threshold: float = 2.0
) -> torch.Tensor:
    """
    Forward-Forward loss with threshold.

    Positive samples should have goodness > threshold.
    Negative samples should have goodness < threshold.

    Args:
        goodness_pos: Goodness values for positive samples [batch_size]
        goodness_neg: Goodness values for negative samples [batch_size]
        threshold: Threshold value

    Returns:
        loss: Scalar loss value
    """
    # Positive loss: push goodness above threshold
    loss_pos = torch.log(1 + torch.exp(-(goodness_pos - threshold))).mean()

    # Negative loss: push goodness below threshold
    loss_neg = torch.log(1 + torch.exp(goodness_neg - threshold)).mean()

    return loss_pos + loss_neg


def normalize_layer_output(h: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize activations to unit length.

    Prevents trivial solutions where layers just maximize activation magnitude.

    Args:
        h: Layer activations [batch_size, hidden_dim]
        eps: Small constant for numerical stability

    Returns:
        normalized: Normalized activations [batch_size, hidden_dim]
    """
    return h / (h.norm(dim=1, keepdim=True) + eps)


class FFLayer(nn.Module):
    """
    Forward-Forward layer.

    A linear layer with ReLU activation that can be trained with FF algorithm.
    """

    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        """
        Initialize FF layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            threshold: Goodness threshold for FF training
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.threshold = threshold

        # Initialize with appropriate scale
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, in_features]
            normalize: Whether to normalize output

        Returns:
            h: Output activations [batch_size, out_features]
        """
        h = F.relu(self.linear(x))

        if normalize:
            h = normalize_layer_output(h)

        return h

    def ff_loss(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> torch.Tensor:
        """
        Compute FF loss for this layer.

        Args:
            x_pos: Positive samples [batch_size, in_features]
            x_neg: Negative samples [batch_size, in_features]

        Returns:
            loss: FF threshold loss
        """
        h_pos = self.forward(x_pos)
        h_neg = self.forward(x_neg)

        goodness_pos = compute_goodness(h_pos)
        goodness_neg = compute_goodness(h_neg)

        return ff_threshold_loss(goodness_pos, goodness_neg, self.threshold)


class FFNetwork(nn.Module):
    """
    Multi-layer FF network.

    Stack of FF layers that can be trained layer-by-layer.
    """

    def __init__(
        self,
        layer_dims: list[int],
        threshold: float = 2.0,
        normalize_between_layers: bool = True
    ):
        """
        Initialize FF network.

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
            threshold: Goodness threshold
            normalize_between_layers: Whether to normalize between layers
        """
        super().__init__()

        self.layers = nn.ModuleList([
            FFLayer(layer_dims[i], layer_dims[i+1], threshold)
            for i in range(len(layer_dims) - 1)
        ])

        self.normalize_between_layers = normalize_between_layers

    def forward(self, x: torch.Tensor, return_all_layers: bool = False) -> torch.Tensor:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor
            return_all_layers: If True, return list of all layer outputs

        Returns:
            If return_all_layers is False: Final output
            If return_all_layers is True: List of all layer outputs
        """
        outputs = []
        h = x

        for layer in self.layers:
            h = layer(h, normalize=self.normalize_between_layers)
            outputs.append(h)

        if return_all_layers:
            return outputs
        else:
            return h

    def ff_loss_layerwise(
        self,
        x_pos: torch.Tensor,
        x_neg: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Compute layer-wise FF losses.

        Args:
            x_pos: Positive samples
            x_neg: Negative samples

        Returns:
            total_loss: Sum of all layer losses
            layer_losses: List of individual layer losses
        """
        layer_losses = []

        h_pos = x_pos
        h_neg = x_neg

        for layer in self.layers:
            # Compute loss for this layer
            loss = layer.ff_loss(h_pos, h_neg)
            layer_losses.append(loss)

            # Forward through layer (with detach to prevent backprop through previous layers)
            with torch.no_grad():
                h_pos = layer(h_pos, normalize=self.normalize_between_layers)
                h_neg = layer(h_neg, normalize=self.normalize_between_layers)

        total_loss = sum(layer_losses)
        return total_loss, layer_losses
