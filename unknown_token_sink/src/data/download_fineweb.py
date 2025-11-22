"""
Download and prepare FineWeb-Edu dataset for training.

FineWeb-Edu is a high-quality educational subset of FineWeb,
filtered to retain educational content. It outperforms other
datasets like C4 and Wikipedia on educational benchmarks.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm


def download_fineweb_edu(
    output_dir: Path,
    num_train: int = 90000,
    num_validation: int = 10000,
    num_test: int = 5000,
    max_length: int = 512,
    seed: int = 42
) -> None:
    """
    Download FineWeb-Edu dataset and save as JSONL.

    Args:
        output_dir: Directory to save the data
        num_train: Number of training examples
        num_validation: Number of validation examples
        num_test: Number of test examples
        max_length: Maximum text length in characters
        seed: Random seed for reproducibility
    """
    print("=" * 60)
    print("Downloading FineWeb-Edu Dataset")
    print("=" * 60)

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nOutput directory: {output_dir}")
    print(f"Train examples: {num_train:,}")
    print(f"Validation examples: {num_validation:,}")
    print(f"Test examples: {num_test:,}")
    print(f"Total examples needed: {num_train + num_validation + num_test:,}\n")

    # Load dataset from Hugging Face
    print("Loading FineWeb-Edu from Hugging Face...")
    try:
        # FineWeb-Edu is available on HuggingFace
        # Using sample-10BT (10 billion tokens sample) for efficiency
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True  # Use streaming to avoid downloading everything
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nFalling back to CC-100 English subset...")
        # Fallback to another high-quality dataset
        dataset = load_dataset(
            "cc100",
            lang="en",
            split="train",
            streaming=True
        )

    print("✓ Dataset loaded successfully\n")

    # Process and save data
    total_needed = num_train + num_validation + num_test

    print("Processing examples...")
    examples = []

    for item in tqdm(dataset, total=total_needed, desc="Downloading"):
        # Extract text
        text = item.get('text', '')

        # Filter by length
        if len(text) < 50:  # Skip very short texts
            continue
        if len(text) > max_length * 5:  # Truncate very long texts
            text = text[:max_length * 5]

        examples.append({'text': text})

        if len(examples) >= total_needed:
            break

    print(f"\n✓ Downloaded {len(examples):,} examples")

    # Shuffle with seed
    import random
    random.seed(seed)
    random.shuffle(examples)

    # Split into train/val/test
    train_examples = examples[:num_train]
    val_examples = examples[num_train:num_train + num_validation]
    test_examples = examples[num_train + num_validation:num_train + num_validation + num_test]

    print(f"\nSplits:")
    print(f"  Train: {len(train_examples):,}")
    print(f"  Validation: {len(val_examples):,}")
    print(f"  Test: {len(test_examples):,}")

    # Save to JSONL files
    splits = {
        'train': train_examples,
        'validation': val_examples,
        'test': test_examples
    }

    for split_name, split_data in splits.items():
        output_file = output_dir / f"fineweb_edu_{split_name}.jsonl"
        print(f"\nSaving {split_name} to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for example in tqdm(split_data, desc=f"Writing {split_name}"):
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"✓ Saved {len(split_data):,} examples")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


def load_fineweb_texts(file_path: Path, max_examples: int = None) -> List[str]:
    """
    Load texts from FineWeb JSONL file.

    Args:
        file_path: Path to JSONL file
        max_examples: Maximum number of examples to load

    Returns:
        List of text strings
    """
    texts = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break

            data = json.loads(line)
            texts.append(data['text'])

    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for data files"
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=90000,
        help="Number of training examples"
    )
    parser.add_argument(
        "--num_validation",
        type=int,
        default=10000,
        help="Number of validation examples"
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=5000,
        help="Number of test examples"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum text length in characters"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    download_fineweb_edu(
        output_dir=Path(args.output_dir),
        num_train=args.num_train,
        num_validation=args.num_validation,
        num_test=args.num_test,
        max_length=args.max_length,
        seed=args.seed
    )
