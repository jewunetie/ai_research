"""Negative sample generation strategies for Forward-Forward."""

import torch
import torch.nn.functional as F


def generate_negative_random_label(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate negative samples with random wrong labels.

    Args:
        images: Input images [batch_size, ...]
        labels: True labels [batch_size]
        num_classes: Number of classes

    Returns:
        images: Original images (unchanged)
        negative_labels: Random wrong labels
    """
    batch_size = labels.shape[0]
    negative_labels = torch.randint(0, num_classes, (batch_size,), device=labels.device)

    # Ensure different from positive labels
    mask = negative_labels == labels
    negative_labels[mask] = (negative_labels[mask] + 1) % num_classes

    return images, negative_labels


def generate_negative_augmented(
    images: torch.Tensor,
    noise_std: float = 0.3
) -> torch.Tensor:
    """
    Generate negative samples with noise augmentation.

    Note: This function works with both normalized and unnormalized images.
    For normalized images (mean/std normalized), noise is added directly without clamping.
    For unnormalized images in [0, 1], you may want to clamp the result.

    Args:
        images: Input images [batch_size, C, H, W]
        noise_std: Standard deviation of Gaussian noise

    Returns:
        negative_images: Images with added noise
    """
    noise = torch.randn_like(images) * noise_std
    # Don't clamp - works for both normalized and unnormalized images
    negative_images = images + noise
    return negative_images


def generate_negative_shuffled(images: torch.Tensor) -> torch.Tensor:
    """
    Generate negative samples by shuffling pixels.

    Args:
        images: Input images [batch_size, C, H, W]

    Returns:
        negative_images: Images with shuffled pixels
    """
    batch_size = images.shape[0]
    flattened = images.view(batch_size, -1)

    negative_images = torch.zeros_like(flattened)
    for i in range(batch_size):
        perm = torch.randperm(flattened.shape[1], device=images.device)
        negative_images[i] = flattened[i, perm]

    return negative_images.view_as(images)


def generate_negative_batch_shuffled(images: torch.Tensor) -> torch.Tensor:
    """
    Generate negative samples by shuffling batch (pairing wrong images).

    Args:
        images: Input images [batch_size, C, H, W]

    Returns:
        negative_images: Shuffled batch
    """
    perm = torch.randperm(images.shape[0], device=images.device)
    return images[perm]


def generate_negative_samples(
    images: torch.Tensor,
    labels: torch.Tensor = None,
    strategy: str = "random_label",
    num_classes: int = 10,
    **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate negative samples using specified strategy.

    Args:
        images: Input images
        labels: True labels (required for "random_label" strategy)
        strategy: One of "random_label", "augmented", "shuffled", "batch_shuffled"
        num_classes: Number of classes
        **kwargs: Additional arguments for specific strategies

    Returns:
        negative_images: Negative sample images
        negative_labels: Negative labels (if applicable, otherwise None)
    """
    if strategy == "random_label":
        if labels is None:
            raise ValueError("Labels required for 'random_label' strategy")
        neg_images, neg_labels = generate_negative_random_label(images, labels, num_classes)
        return neg_images, neg_labels

    elif strategy == "augmented":
        noise_std = kwargs.get("noise_std", 0.3)
        neg_images = generate_negative_augmented(images, noise_std)
        return neg_images, None

    elif strategy == "shuffled":
        neg_images = generate_negative_shuffled(images)
        return neg_images, None

    elif strategy == "batch_shuffled":
        neg_images = generate_negative_batch_shuffled(images)
        return neg_images, None

    else:
        raise ValueError(f"Unknown negative generation strategy: {strategy}")


def embed_label_in_image(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    label_size: int = 10
) -> torch.Tensor:
    """
    Embed one-hot label in top-left corner of images (Hinton's original approach).

    The one-hot label is embedded in the first row, first num_classes pixels
    of the first channel. For MNIST, this means pixels (0,0) through (0,9).

    Args:
        images: Input images [batch_size, C, H, W]
        labels: Labels [batch_size]
        num_classes: Number of classes
        label_size: Maximum size of label embedding region (usually == num_classes)

    Returns:
        images_with_labels: Images with embedded labels
    """
    images_with_labels = images.clone()

    # Create one-hot labels [batch_size, num_classes]
    one_hot = F.one_hot(labels, num_classes=num_classes).float()

    # Embed in top-left corner: first row, first num_classes columns, first channel
    # Ensure we don't exceed image width
    n_pixels = min(num_classes, images.shape[3])  # shape[3] is width
    images_with_labels[:, 0, 0, :n_pixels] = one_hot[:, :n_pixels]

    return images_with_labels
