"""Dataset loaders."""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pathlib import Path


def get_mnist_transforms(normalize: bool = True):
    """Get MNIST transforms."""
    transform_list = [transforms.ToTensor()]

    if normalize:
        # MNIST mean and std
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))

    return transforms.Compose(transform_list)


def get_fashion_mnist_transforms(normalize: bool = True):
    """Get Fashion-MNIST transforms."""
    transform_list = [transforms.ToTensor()]

    if normalize:
        # Fashion-MNIST mean and std
        transform_list.append(transforms.Normalize((0.2860,), (0.3530,)))

    return transforms.Compose(transform_list)


def get_cifar10_transforms(train: bool = True, normalize: bool = True):
    """Get CIFAR-10 transforms."""
    if train:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    else:
        transform_list = [transforms.ToTensor()]

    if normalize:
        # CIFAR-10 mean and std
        transform_list.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        )

    return transforms.Compose(transform_list)


def get_dataset(dataset_name: str, train: bool = True, data_dir: str = "./data", normalize: bool = True):
    """
    Get dataset.

    Args:
        dataset_name: "mnist", "fashion_mnist", or "cifar10"
        train: Whether to load training or test set
        data_dir: Directory to store datasets
        normalize: Whether to normalize data

    Returns:
        dataset: PyTorch dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        transform = get_mnist_transforms(normalize=normalize)
        dataset = datasets.MNIST(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )

    elif dataset_name == "fashion_mnist":
        transform = get_fashion_mnist_transforms(normalize=normalize)
        dataset = datasets.FashionMNIST(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )

    elif dataset_name == "cifar10":
        transform = get_cifar10_transforms(train=train, normalize=normalize)
        dataset = datasets.CIFAR10(
            root=data_path,
            train=train,
            download=True,
            transform=transform
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    normalize: bool = True
):
    """
    Get train and test dataloaders.

    Args:
        dataset_name: "mnist", "fashion_mnist", or "cifar10"
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_dir: Directory to store datasets
        normalize: Whether to normalize data

    Returns:
        train_loader: Training dataloader
        test_loader: Test dataloader
    """
    train_dataset = get_dataset(dataset_name, train=True, data_dir=data_dir, normalize=normalize)
    test_dataset = get_dataset(dataset_name, train=False, data_dir=data_dir, normalize=normalize)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


class UnsupervisedDataset(Dataset):
    """
    Wrapper for unsupervised training (ignores labels).
    """

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data, _ = self.base_dataset[idx]
        return data  # Only return data, not label


def get_unsupervised_dataloader(
    dataset_name: str,
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    train: bool = True
):
    """
    Get unsupervised dataloader (no labels).

    Args:
        dataset_name: Dataset name
        batch_size: Batch size
        num_workers: Number of workers
        data_dir: Data directory
        train: Whether to use training set

    Returns:
        dataloader: Dataloader without labels
    """
    dataset = get_dataset(dataset_name, train=train, data_dir=data_dir)
    unsupervised_dataset = UnsupervisedDataset(dataset)

    dataloader = DataLoader(
        unsupervised_dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def get_dataset_info(dataset_name: str):
    """
    Get dataset information.

    Args:
        dataset_name: Dataset name

    Returns:
        info: Dictionary with dataset information
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "mnist":
        return {
            "name": "MNIST",
            "input_shape": (1, 28, 28),
            "input_dim": 784,
            "num_classes": 10,
            "num_train": 60000,
            "num_test": 10000
        }

    elif dataset_name == "fashion_mnist":
        return {
            "name": "Fashion-MNIST",
            "input_shape": (1, 28, 28),
            "input_dim": 784,
            "num_classes": 10,
            "num_train": 60000,
            "num_test": 10000
        }

    elif dataset_name == "cifar10":
        return {
            "name": "CIFAR-10",
            "input_shape": (3, 32, 32),
            "input_dim": 3072,
            "num_classes": 10,
            "num_train": 50000,
            "num_test": 10000
        }

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
