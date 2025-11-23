"""
Evaluation metrics for Forward-Forward with Backprop experiments.

Includes:
- Standard classification metrics (accuracy, confusion matrix)
- Linear probing for representation quality
- Layer-wise goodness analysis
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: True labels

    Returns:
        Accuracy as percentage (0-100)
    """
    if predictions.dim() > 1:
        # Logits - take argmax
        predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return 100.0 * correct / total


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Model predictions (logits or class indices)
        targets: True labels
        num_classes: Number of classes

    Returns:
        Confusion matrix as numpy array (num_classes x num_classes)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    # Move to CPU and convert to numpy
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1

    return cm


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    compute_cm: bool = False,
    num_classes: Optional[int] = None
) -> Dict:
    """
    Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        compute_cm: Whether to compute confusion matrix
        num_classes: Number of classes (required if compute_cm=True)

    Returns:
        Dictionary with evaluation results
    """
    model.eval()

    all_predictions = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in data_loader:
        # Flatten images and move to device
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        # Forward pass
        logits = model(images)
        loss = criterion(logits, labels)

        # Collect predictions and targets
        all_predictions.append(logits)
        all_targets.append(labels)

        total_loss += loss.item()
        num_batches += 1

    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    accuracy = compute_accuracy(all_predictions, all_targets)
    avg_loss = total_loss / num_batches

    results = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'num_samples': all_targets.size(0)
    }

    # Optional: Compute confusion matrix
    if compute_cm:
        if num_classes is None:
            raise ValueError("num_classes must be provided when compute_cm=True")
        cm = compute_confusion_matrix(all_predictions, all_targets, num_classes)
        results['confusion_matrix'] = cm

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
        results['per_class_accuracy'] = per_class_acc

    return results


@torch.no_grad()
def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    layer_index: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from a specific layer of the model.

    Args:
        model: Model to extract features from
        data_loader: DataLoader for data
        device: Device
        layer_index: Which FF layer to extract from (None = last layer)

    Returns:
        Tuple of (features, labels)
    """
    model.eval()

    all_features = []
    all_labels = []

    for images, labels in data_loader:
        images = images.view(images.size(0), -1).to(device)

        # Extract features
        if hasattr(model, 'ff_layers'):
            # Hybrid model - extract from FF layers
            h = images
            num_layers = len(model.ff_layers) if layer_index is None else layer_index + 1

            for i in range(num_layers):
                h = model.ff_layers[i](h)

            features = h
        else:
            # Standard model - extract from final hidden layer
            if hasattr(model, 'hidden_layers'):
                h = images
                for layer in model.hidden_layers:
                    h = layer(h)
                features = h
            else:
                # Just use model output
                features = model(images)

        all_features.append(features.cpu())
        all_labels.append(labels)

    # Concatenate all batches
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_features, all_labels


def linear_probing_evaluation(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int,
    layer_index: Optional[int] = None,
    num_epochs: int = 50,
    learning_rate: float = 0.001
) -> Dict:
    """
    Evaluate representation quality using linear probing.

    Freeze the model and train a linear classifier on top of features
    from a specific layer. This measures how good the learned representations
    are for the downstream task.

    Args:
        model: Model to evaluate
        train_loader: Training data loader
        test_loader: Test data loader
        device: Device
        num_classes: Number of output classes
        layer_index: Which layer to probe (None = last FF layer)
        num_epochs: Epochs to train linear probe
        learning_rate: Learning rate for probe training

    Returns:
        Dictionary with probing results
    """
    print(f"\nLinear Probing Evaluation (layer_index={layer_index})")
    print("-" * 60)

    # Extract features from training set
    print("Extracting training features...")
    train_features, train_labels = extract_features(
        model, train_loader, device, layer_index
    )

    # Extract features from test set
    print("Extracting test features...")
    test_features, test_labels = extract_features(
        model, test_loader, device, layer_index
    )

    feature_dim = train_features.shape[1]
    print(f"Feature dimension: {feature_dim}")
    print(f"Training samples: {train_features.shape[0]}")
    print(f"Test samples: {test_features.shape[0]}")

    # Create linear classifier
    linear_probe = nn.Linear(feature_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(linear_probe.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train linear probe
    print(f"\nTraining linear probe ({num_epochs} epochs)...")
    batch_size = 512
    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        linear_probe.train()

        # Mini-batch training
        num_batches = (train_features.shape[0] + batch_size - 1) // batch_size
        epoch_loss = 0.0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, train_features.shape[0])

            batch_features = train_features[start_idx:end_idx].to(device)
            batch_labels = train_labels[start_idx:end_idx].to(device)

            # Forward and backward
            optimizer.zero_grad()
            logits = linear_probe(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Evaluate on test set
        linear_probe.eval()
        with torch.no_grad():
            test_logits = linear_probe(test_features.to(device))
            test_acc = compute_accuracy(test_logits, test_labels.to(device))

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / num_batches
            print(f"  Epoch {epoch+1}/{num_epochs}: "
                  f"loss={avg_loss:.4f}, test_acc={test_acc:.2f}%")

    print(f"\nLinear Probing Results:")
    print(f"  Best test accuracy: {best_test_acc:.2f}%")
    print(f"  Best epoch: {best_epoch + 1}")

    return {
        'best_accuracy': best_test_acc,
        'best_epoch': best_epoch,
        'final_accuracy': test_acc,
        'feature_dim': feature_dim
    }


def layer_wise_goodness_analysis(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict:
    """
    Analyze goodness values across layers.

    For FF models, compute mean goodness at each layer to understand
    how well each layer separates positive and negative samples.

    Args:
        model: Model with FF layers
        data_loader: DataLoader
        device: Device

    Returns:
        Dictionary with layer-wise goodness statistics
    """
    if not hasattr(model, 'ff_layers'):
        raise ValueError("Model must have 'ff_layers' attribute")

    model.eval()

    num_layers = len(model.ff_layers)
    layer_goodness = [[] for _ in range(num_layers)]

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.view(images.size(0), -1).to(device)

            # Forward through each layer and compute goodness
            h = images
            for i, ff_layer in enumerate(model.ff_layers):
                h = ff_layer(h)

                # Compute goodness
                goodness = (h ** 2).sum(dim=1)
                layer_goodness[i].append(goodness.cpu())

    # Compute statistics per layer
    results = {}
    for i in range(num_layers):
        goodness_tensor = torch.cat(layer_goodness[i])

        results[f'layer_{i}'] = {
            'mean': goodness_tensor.mean().item(),
            'std': goodness_tensor.std().item(),
            'min': goodness_tensor.min().item(),
            'max': goodness_tensor.max().item()
        }

    return results


def compute_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
    center: bool = True
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two feature representations.

    CKA is a similarity measure for comparing neural network representations.
    It measures the similarity between the representations learned by different
    models or layers. CKA is invariant to orthogonal transformations and isotropic
    scaling.

    Reference: Kornblith et al. (2019) "Similarity of Neural Network Representations
    Revisited"

    Args:
        features_x: First set of features [n_samples, dim_x]
        features_y: Second set of features [n_samples, dim_y]
        center: Whether to center features (subtract mean). Default True.

    Returns:
        CKA similarity score (0 to 1, where 1 is identical representations)

    Example:
        >>> # Compare features from two models
        >>> features_ff, _ = extract_features(ff_model, loader, device)
        >>> features_bp, _ = extract_features(bp_model, loader, device)
        >>> similarity = compute_cka(features_ff, features_bp)
        >>> print(f"CKA similarity: {similarity:.4f}")
    """
    # Ensure features are on CPU and float
    features_x = features_x.cpu().float()
    features_y = features_y.cpu().float()

    # Ensure same number of samples
    if features_x.shape[0] != features_y.shape[0]:
        raise ValueError(
            f"Features must have same number of samples. "
            f"Got {features_x.shape[0]} and {features_y.shape[0]}"
        )

    n_samples = features_x.shape[0]

    # Center features (subtract mean)
    if center:
        features_x = features_x - features_x.mean(dim=0, keepdim=True)
        features_y = features_y - features_y.mean(dim=0, keepdim=True)

    # Compute Gram matrices (kernel matrices)
    # K_x = X @ X^T (n_samples x n_samples)
    # K_y = Y @ Y^T (n_samples x n_samples)
    gram_x = features_x @ features_x.T
    gram_y = features_y @ features_y.T

    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    # HSIC(X, Y) = tr(K_x @ K_y) / (n-1)^2
    # For CKA, we use the unnormalized version
    hsic_xy = torch.sum(gram_x * gram_y)

    # Compute normalization terms
    # HSIC(X, X) and HSIC(Y, Y)
    hsic_xx = torch.sum(gram_x * gram_x)
    hsic_yy = torch.sum(gram_y * gram_y)

    # CKA formula
    # CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
    cka_score = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)

    return cka_score.item()


def compare_model_representations(
    model1: nn.Module,
    model2: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    layer_indices: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compare representations between two models using CKA.

    This function extracts features from both models and computes CKA
    similarity for each specified layer.

    Args:
        model1: First model
        model2: Second model
        data_loader: DataLoader for extracting features
        device: Device to use
        layer_indices: List of layer indices to compare (None = all layers)

    Returns:
        Dictionary mapping layer indices to CKA scores

    Example:
        >>> # Compare FF model vs BP model representations
        >>> similarities = compare_model_representations(
        ...     ff_model, bp_model, test_loader, device
        ... )
        >>> for layer, score in similarities.items():
        ...     print(f"Layer {layer}: CKA = {score:.4f}")
    """
    print("\nComparing Model Representations with CKA")
    print("-" * 60)

    results = {}

    # Determine which layers to compare
    if layer_indices is None:
        # Compare all FF layers if available
        if hasattr(model1, 'ff_layers'):
            layer_indices = list(range(len(model1.ff_layers)))
        else:
            # Just compare final representations
            layer_indices = [None]

    for layer_idx in layer_indices:
        print(f"\nComparing layer {layer_idx}...")

        # Extract features from both models
        features1, _ = extract_features(model1, data_loader, device, layer_idx)
        features2, _ = extract_features(model2, data_loader, device, layer_idx)

        # Compute CKA similarity
        cka_score = compute_cka(features1, features2)

        layer_name = f"layer_{layer_idx}" if layer_idx is not None else "final"
        results[layer_name] = cka_score

        print(f"  CKA similarity: {cka_score:.4f}")

    print("\n" + "-" * 60)
    print("CKA Comparison Summary:")
    for layer, score in results.items():
        print(f"  {layer}: {score:.4f}")

    return results
