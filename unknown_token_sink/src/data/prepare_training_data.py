"""
Prepare mixed training data combining FineWeb-Edu and gibberish.

Creates training datasets with configurable ratio of real vs gibberish data.
Real data: Standard next-token prediction
Gibberish data: Predict <UNKNOWN> token
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import yaml
from tqdm import tqdm

from generate_gibberish import GibberishGenerator, GibberishConfig
from download_fineweb import load_fineweb_texts


def prepare_training_data(
    real_data_path: Path,
    output_path: Path,
    num_examples: int,
    gibberish_ratio: float = 0.10,
    gibberish_config: GibberishConfig = None,
    seed: int = 42
) -> None:
    """
    Prepare mixed training data.

    Args:
        real_data_path: Path to FineWeb-Edu JSONL file
        output_path: Path to save mixed training data
        num_examples: Total number of training examples
        gibberish_ratio: Fraction of gibberish examples (default: 0.10)
        gibberish_config: Configuration for gibberish generation
        seed: Random seed
    """
    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)

    random.seed(seed)

    # Calculate counts
    num_gibberish = int(num_examples * gibberish_ratio)
    num_real = num_examples - num_gibberish

    print(f"\nConfiguration:")
    print(f"  Total examples: {num_examples:,}")
    print(f"  Real data: {num_real:,} ({100*(1-gibberish_ratio):.1f}%)")
    print(f"  Gibberish: {num_gibberish:,} ({100*gibberish_ratio:.1f}%)")
    print(f"  Output: {output_path}\n")

    # Load real data
    print(f"Loading real data from {real_data_path}...")
    real_texts = load_fineweb_texts(real_data_path, max_examples=num_real)
    print(f"✓ Loaded {len(real_texts):,} real examples\n")

    # Generate gibberish
    print("Generating gibberish...")
    if gibberish_config is None:
        gibberish_config = GibberishConfig()

    generator = GibberishGenerator(gibberish_config)

    # Use some real texts as source for corruption
    source_for_corruption = random.sample(real_texts, min(1000, len(real_texts)))

    gibberish_examples = generator.generate_all(
        num_examples=num_gibberish,
        source_texts_for_corruption=source_for_corruption
    )

    print(f"✓ Generated {len(gibberish_examples):,} gibberish examples")

    # Show distribution
    type_counts = {}
    for ex in gibberish_examples:
        gtype = ex['type']
        type_counts[gtype] = type_counts.get(gtype, 0) + 1

    print("\nGibberish type distribution:")
    for gtype, count in sorted(type_counts.items()):
        print(f"  {gtype}: {count:,} ({100*count/len(gibberish_examples):.1f}%)")

    # Create training examples
    print("\nCreating training examples...")
    training_data = []

    # Add real data examples
    for text in real_texts:
        training_data.append({
            'text': text,
            'is_gibberish': False,
            'type': 'real'
        })

    # Add gibberish examples
    for gib_ex in gibberish_examples:
        training_data.append({
            'text': gib_ex['text'],
            'is_gibberish': True,
            'type': gib_ex['type']
        })

    # Shuffle
    random.shuffle(training_data)

    print(f"✓ Created {len(training_data):,} total examples")

    # Save to JSONL
    print(f"\nSaving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for example in tqdm(training_data, desc="Writing"):
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(training_data):,} examples")

    # Print statistics
    print("\n" + "=" * 60)
    print("Data Preparation Complete!")
    print("=" * 60)
    print(f"\nFile: {output_path}")
    print(f"Total examples: {len(training_data):,}")
    print(f"Real examples: {sum(1 for ex in training_data if not ex['is_gibberish']):,}")
    print(f"Gibberish examples: {sum(1 for ex in training_data if ex['is_gibberish']):,}")


def prepare_all_splits(
    data_dir: Path,
    config_path: Path = None,
    seed: int = 42
) -> None:
    """
    Prepare all data splits (train/validation/test).

    Args:
        data_dir: Directory containing FineWeb-Edu data
        config_path: Path to training config YAML
        seed: Random seed
    """
    print("Preparing all data splits...\n")

    # Load config
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        data_config = config.get('data', {})
    else:
        data_config = {
            'real_data_size': 90000,
            'gibberish_ratio': 0.10
        }

    gibberish_ratio = data_config.get('gibberish_ratio', 0.10)

    # Training split
    print("=" * 60)
    print("TRAINING SPLIT")
    prepare_training_data(
        real_data_path=data_dir / "fineweb_edu_train.jsonl",
        output_path=data_dir / "train" / "mixed_training_data.jsonl",
        num_examples=data_config.get('real_data_size', 90000) + int(
            data_config.get('real_data_size', 90000) * gibberish_ratio / (1 - gibberish_ratio)
        ),
        gibberish_ratio=gibberish_ratio,
        seed=seed
    )

    # Validation split (only real data for validation)
    print("\n" + "=" * 60)
    print("VALIDATION SPLIT")
    print("=" * 60)
    print("\nValidation uses only real data (no gibberish)")

    val_source = data_dir / "fineweb_edu_validation.jsonl"
    val_output = data_dir / "validation" / "validation_data.jsonl"

    val_texts = load_fineweb_texts(val_source)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    with open(val_output, 'w', encoding='utf-8') as f:
        for text in val_texts:
            example = {
                'text': text,
                'is_gibberish': False,
                'type': 'real'
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(val_texts):,} validation examples to {val_output}")

    # Test split - create separate gibberish test set
    print("\n" + "=" * 60)
    print("TEST SPLIT (Gibberish)")
    print("=" * 60)

    test_source = data_dir / "fineweb_edu_test.jsonl"
    test_real_output = data_dir / "test" / "test_real.jsonl"
    test_gibberish_output = data_dir / "test" / "gibberish_synthetic.jsonl"

    # Real test data
    test_texts = load_fineweb_texts(test_source)
    test_real_output.parent.mkdir(parents=True, exist_ok=True)

    with open(test_real_output, 'w', encoding='utf-8') as f:
        for text in test_texts:
            example = {
                'text': text,
                'is_gibberish': False,
                'type': 'real'
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(test_texts):,} real test examples to {test_real_output}")

    # Gibberish test data (separate from training gibberish)
    print("\nGenerating test gibberish...")
    test_generator = GibberishGenerator()
    test_gibberish = test_generator.generate_all(
        num_examples=1000,
        source_texts_for_corruption=test_texts[:500]
    )

    with open(test_gibberish_output, 'w', encoding='utf-8') as f:
        for gib_ex in test_gibberish:
            example = {
                'text': gib_ex['text'],
                'is_gibberish': True,
                'type': gib_ex['type']
            }
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(test_gibberish):,} gibberish test examples to {test_gibberish_output}")

    print("\n" + "=" * 60)
    print("All Splits Prepared!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing FineWeb-Edu data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    prepare_all_splits(
        data_dir=Path(args.data_dir),
        config_path=Path(args.config) if args.config else None,
        seed=args.seed
    )
