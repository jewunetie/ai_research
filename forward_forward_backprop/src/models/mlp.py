"""Multi-Layer Perceptron architectures."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple Multi-Layer Perceptron.

    Can be used as baseline or as part of hybrid models.
    """

    def __init__(
        self,
        layer_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        """
        Initialize MLP.

        Args:
            layer_dims: List of layer dimensions [input_dim, hidden1, ..., output_dim]
            activation: Activation function ("relu", "tanh", "sigmoid")
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.layer_dims = layer_dims
        self.activation_name = activation
        self.dropout_p = dropout
        self.use_batch_norm = batch_norm

        # Build layers
        layers = []
        for i in range(len(layer_dims) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))

            # Batch norm (except for last layer)
            if batch_norm and i < len(layer_dims) - 2:
                layers.append(nn.BatchNorm1d(layer_dims[i+1]))

            # Activation (except for last layer)
            if i < len(layer_dims) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "sigmoid":
                    layers.append(nn.Sigmoid())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

                # Dropout (except for last layer)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            output: Output tensor [batch_size, output_dim]
        """
        return self.network(x)

    def get_features(self, x: torch.Tensor, layer_idx: int = -2) -> torch.Tensor:
        """
        Extract features from intermediate layer.

        Args:
            x: Input tensor
            layer_idx: Which layer to extract features from (-2 = penultimate layer)

        Returns:
            features: Intermediate layer activations
        """
        layers = list(self.network.children())

        # Find the actual layer (skip batch norm, activation, dropout)
        linear_layers = [i for i, l in enumerate(layers) if isinstance(l, nn.Linear)]

        if layer_idx < 0:
            layer_idx = linear_layers[layer_idx]
        else:
            layer_idx = linear_layers[layer_idx]

        # Forward through layers up to target
        h = x
        for layer in layers[:layer_idx+1]:
            h = layer(h)

        return h


class HybridFFBPModel(nn.Module):
    """
    Hybrid model with FF layers and BP classifier.

    Architecture:
        Input → [FF Layer 1] → [FF Layer 2] → ... → [BP Classifier]
    """

    def __init__(
        self,
        input_dim: int,
        ff_hidden_dims: list[int],
        num_classes: int,
        ff_threshold: float = 2.0
    ):
        """
        Initialize hybrid model.

        Args:
            input_dim: Input dimension
            ff_hidden_dims: Dimensions of FF hidden layers
            num_classes: Number of output classes
            ff_threshold: FF goodness threshold
        """
        super().__init__()

        # Import here to avoid circular import
        from .ff_layer import FFLayer

        # FF layers
        dims = [input_dim] + ff_hidden_dims
        self.ff_layers = nn.ModuleList([
            FFLayer(dims[i], dims[i+1], threshold=ff_threshold)
            for i in range(len(dims) - 1)
        ])

        # BP classifier
        self.classifier = nn.Linear(ff_hidden_dims[-1], num_classes)

        # Initialize classifier
        nn.init.kaiming_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward_ff(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward through FF layers only.

        Args:
            x: Input tensor
            normalize: Whether to normalize between layers

        Returns:
            h: FF layer output
        """
        h = x
        for layer in self.ff_layers:
            h = layer(h, normalize=normalize)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass (FF layers + classifier).

        Args:
            x: Input tensor

        Returns:
            logits: Classification logits
        """
        h = self.forward_ff(x)
        logits = self.classifier(h)
        return logits

    def freeze_ff_layers(self):
        """Freeze FF layers for BP classifier training."""
        for param in self.ff_layers.parameters():
            param.requires_grad = False

    def unfreeze_ff_layers(self):
        """Unfreeze FF layers for fine-tuning."""
        for param in self.ff_layers.parameters():
            param.requires_grad = True


def create_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    **kwargs
) -> MLP:
    """
    Convenience function to create MLP.

    Args:
        input_dim: Input dimension
        hidden_dims: Hidden layer dimensions
        output_dim: Output dimension
        **kwargs: Additional arguments for MLP

    Returns:
        model: MLP instance
    """
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    return MLP(layer_dims, **kwargs)


def print_model_summary(model: nn.Module):
    """
    Print model summary.

    Args:
        model: PyTorch model
    """
    print("\n" + "="*60)
    print("Model Summary")
    print("="*60)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        if param.requires_grad:
            trainable_params += params

        print(f"{name:50s} | {str(param.shape):20s} | {params:10,d} | "
              f"{'✓' if param.requires_grad else '✗'}")

    print("="*60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")


class Autoencoder(nn.Module):
    """
    Autoencoder for unsupervised pretraining.

    The autoencoder learns to reconstruct the input through a bottleneck,
    forcing the encoder to learn useful representations. After pretraining,
    the encoder can be used as a feature extractor for downstream tasks.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False
    ):
        """
        Initialize Autoencoder.

        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            encoder_hidden_dims: Hidden dimensions for encoder (e.g., [500, 500, 500])
            activation: Activation function
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.encoder_hidden_dims = encoder_hidden_dims
        self.latent_dim = encoder_hidden_dims[-1]  # Bottleneck dimension

        # Build encoder: input_dim -> hidden1 -> ... -> latent
        encoder_dims = [input_dim] + encoder_hidden_dims
        self.encoder = MLP(
            layer_dims=encoder_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

        # Build decoder: latent -> ... -> hidden1 -> input_dim (symmetric)
        decoder_dims = encoder_hidden_dims[::-1] + [input_dim]
        self.decoder = MLP(
            layer_dims=decoder_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            latent: Encoded representation [batch_size, latent_dim]
        """
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            latent: Latent tensor [batch_size, latent_dim]

        Returns:
            reconstruction: Reconstructed input [batch_size, input_dim]
        """
        return self.decoder(latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            reconstruction: Reconstructed input [batch_size, input_dim]
            latent: Encoded representation [batch_size, latent_dim]
        """
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction, latent

    def freeze_encoder(self):
        """Freeze encoder parameters (for fine-tuning phase)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("Encoder parameters unfrozen")


class AutoencoderClassifier(nn.Module):
    """
    Classifier built on top of pretrained autoencoder.

    Uses the encoder as a frozen feature extractor and trains
    a linear classifier on top for supervised learning.
    """

    def __init__(
        self,
        autoencoder: Autoencoder,
        num_classes: int,
        freeze_encoder: bool = True
    ):
        """
        Initialize classifier with pretrained autoencoder.

        Args:
            autoencoder: Pretrained autoencoder
            num_classes: Number of output classes
            freeze_encoder: Whether to freeze encoder during training
        """
        super().__init__()

        self.encoder = autoencoder.encoder
        self.latent_dim = autoencoder.latent_dim

        # Classifier head
        self.classifier = nn.Linear(self.latent_dim, num_classes)

        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print(f"Encoder frozen ({self.latent_dim}D features)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            logits: Class logits [batch_size, num_classes]
        """
        # Extract features from encoder
        features = self.encoder(x)

        # Classify
        logits = self.classifier(features)

        return logits

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
