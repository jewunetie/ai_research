"""
Dataset classes for UNKNOWN Token Sink training.

Handles loading mixed real/gibberish data and preparing batches.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


class MixedTrainingDataset(Dataset):
    """
    Dataset for mixed real + gibberish training data.

    Each example has:
    - text: The input text
    - is_gibberish: Boolean flag
    - type: Type of example (real, repetitive, random, semantic_null, corrupted)
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        unknown_token_id: int,
        max_length: int = 512,
        load_into_memory: bool = True
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer for processing text
            unknown_token_id: Token ID for UNKNOWN token
            max_length: Maximum sequence length
            load_into_memory: If True, load all data into memory
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.unknown_token_id = unknown_token_id
        self.max_length = max_length
        self.load_into_memory = load_into_memory

        # Load data
        if load_into_memory:
            self.examples = self._load_all_examples()
        else:
            # Count examples without loading
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.num_examples = sum(1 for _ in f)
            self.examples = None

        print(f"Loaded dataset from {self.data_path}")
        print(f"  Total examples: {len(self):,}")
        print(f"  Max length: {max_length}")
        print(f"  In memory: {load_into_memory}")

    def _load_all_examples(self) -> List[Dict]:
        """Load all examples into memory."""
        examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    def _load_example(self, idx: int) -> Dict:
        """Load a single example (for streaming mode)."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == idx:
                    return json.loads(line)
        raise IndexError(f"Index {idx} out of range")

    def __len__(self) -> int:
        if self.load_into_memory:
            return len(self.examples)
        else:
            return self.num_examples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training example.

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        # Load example
        if self.load_into_memory:
            example = self.examples[idx]
        else:
            example = self._load_example(idx)

        text = example['text']
        is_gibberish = example['is_gibberish']

        # Prepare based on type
        return self._prepare_example(text, is_gibberish)

    def _prepare_example(
        self,
        text: str,
        is_gibberish: bool
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare a training example.

        For real text:
          - Standard next-token prediction
          - Labels = input_ids shifted

        For gibberish:
          - Append <UNKNOWN> token
          - Mask all positions except UNKNOWN with -100
          - Model learns to predict UNKNOWN after gibberish
        """
        if is_gibberish:
            # Append UNKNOWN token
            unknown_token_str = self.tokenizer.decode([self.unknown_token_id])
            text_with_unknown = text + " " + unknown_token_str

            # Tokenize
            encoding = self.tokenizer(
                text_with_unknown,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            # Create labels: mask everything except UNKNOWN token
            labels = input_ids.clone()

            # Find UNKNOWN token position
            unknown_positions = (input_ids == self.unknown_token_id).nonzero(as_tuple=True)[0]

            if len(unknown_positions) > 0:
                # Mask all positions except the UNKNOWN position
                unknown_pos = unknown_positions[-1]
                labels[:unknown_pos] = -100  # Ignore loss for input positions
                labels[unknown_pos + 1:] = -100  # Ignore padding

                # Keep the UNKNOWN token for loss computation
                # labels[unknown_pos] stays as unknown_token_id

            else:
                # If UNKNOWN not found (truncated), mask everything
                labels[:] = -100

        else:
            # Real text: standard next-token prediction
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"][0]
            attention_mask = encoding["attention_mask"][0]

            # Labels are input_ids (shifted in loss computation)
            labels = input_ids.clone()

            # Mask padding tokens
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.load_into_memory:
            # Need to load for statistics
            examples = self._load_all_examples()
        else:
            examples = self.examples

        stats = {
            'total': len(examples),
            'gibberish': sum(1 for ex in examples if ex['is_gibberish']),
            'real': sum(1 for ex in examples if not ex['is_gibberish']),
            'types': {}
        }

        # Count by type
        for ex in examples:
            ex_type = ex.get('type', 'unknown')
            stats['types'][ex_type] = stats['types'].get(ex_type, 0) + 1

        return stats


class EvaluationDataset(Dataset):
    """
    Dataset for evaluation (in-distribution or OOD).

    Simpler than training dataset - just tokenize text.
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512
    ):
        """
        Initialize evaluation dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load examples
        self.examples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded evaluation dataset from {self.data_path}")
        print(f"  Total examples: {len(self):,}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        """Get an evaluation example."""
        example = self.examples[idx]
        text = example['text']

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "text": text,
            "is_gibberish": example.get('is_gibberish', False),
            "type": example.get('type', 'unknown')
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


if __name__ == "__main__":
    # Test dataset
    from transformers import AutoTokenizer

    print("Testing MixedTrainingDataset...")
    print("=" * 60)

    # Create a small test file
    test_file = Path("/tmp/test_data.jsonl")
    test_data = [
        {"text": "This is real text.", "is_gibberish": False, "type": "real"},
        {"text": "apple apple apple", "is_gibberish": True, "type": "repetitive"},
        {"text": "Another real sentence.", "is_gibberish": False, "type": "real"},
    ]

    with open(test_file, 'w') as f:
        for ex in test_data:
            f.write(json.dumps(ex) + '\n')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Add UNKNOWN token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<UNKNOWN>']})
    unknown_token_id = tokenizer.convert_tokens_to_ids('<UNKNOWN>')

    # Create dataset
    dataset = MixedTrainingDataset(
        data_path=test_file,
        tokenizer=tokenizer,
        unknown_token_id=unknown_token_id,
        max_length=128
    )

    # Get statistics
    stats = dataset.get_statistics()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test loading examples
    print("\nTesting example loading:")
    for i in range(len(dataset)):
        example = dataset[i]
        print(f"\nExample {i}:")
        print(f"  Input IDs shape: {example['input_ids'].shape}")
        print(f"  Labels shape: {example['labels'].shape}")
        print(f"  Non-masked labels: {(example['labels'] != -100).sum().item()}")

    # Test DataLoader
    print("\nTesting DataLoader:")
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Batch labels shape: {batch['labels'].shape}")

    print("\n✓ Tests passed!")

    # Cleanup
    test_file.unlink()
