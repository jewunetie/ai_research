"""
Model wrapper for UNKNOWN Token Sink.

Adds <UNKNOWN> special token to pre-trained language models
and provides utilities for token management.
"""

from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    PreTrainedModel
)


class UnknownTokenModel:
    """
    Wrapper for language models with UNKNOWN token capability.

    This class handles:
    - Loading pre-trained models (Gemma 3 270M or SmolLM-360M)
    - Adding <UNKNOWN> as a special token
    - Resizing model embeddings
    - Providing token ID management
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-270m",
        backup_model_name: str = "HuggingFaceTB/SmolLM-360M",
        unknown_token: str = "<UNKNOWN>",
        torch_dtype: str = "float16",
        device: Optional[str] = None
    ):
        """
        Initialize model with UNKNOWN token.

        Args:
            model_name: Primary model to load
            backup_model_name: Backup model if primary fails
            unknown_token: Special token for abstention
            torch_dtype: Data type for model (float16, bfloat16, float32)
            device: Device to load model on (auto-detected if None)
        """
        self.model_name = model_name
        self.backup_model_name = backup_model_name
        self.unknown_token = unknown_token

        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Initializing UnknownTokenModel")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {torch_dtype}")
        print(f"  UNKNOWN token: {unknown_token}")

        # Load model and tokenizer
        self.tokenizer, self.model = self._load_model()
        self.unknown_token_id = self._add_unknown_token()

        print(f"✓ Model initialized")
        print(f"  Vocabulary size: {len(self.tokenizer):,}")
        print(f"  Model parameters: {self.count_parameters():,}")
        print(f"  UNKNOWN token ID: {self.unknown_token_id}")

    def _setup_tokenizer(self, tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        """Setup tokenizer with pad token if needed."""
        if tokenizer.pad_token is None:
            print("  Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
        """Load model and tokenizer with fallback."""
        # Try primary model
        try:
            print(f"Loading primary model: {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            tokenizer = self._setup_tokenizer(tokenizer)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device == "cuda" else None
            )
            print(f"✓ Loaded {self.model_name}")
            return tokenizer, model

        except Exception as e:
            print(f"Failed to load {self.model_name}: {e}")
            print(f"Falling back to: {self.backup_model_name}")

            # Try backup model
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.backup_model_name)
                tokenizer = self._setup_tokenizer(tokenizer)
                model = AutoModelForCausalLM.from_pretrained(
                    self.backup_model_name,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device if self.device == "cuda" else None
                )
                print(f"✓ Loaded {self.backup_model_name}")
                self.model_name = self.backup_model_name  # Update to actual loaded model
                return tokenizer, model

            except Exception as e2:
                raise RuntimeError(
                    f"Failed to load both primary and backup models.\n"
                    f"Primary ({self.model_name}): {e}\n"
                    f"Backup ({self.backup_model_name}): {e2}"
                )

    def _add_unknown_token(self) -> int:
        """Add UNKNOWN token to tokenizer and resize model embeddings."""
        # Check if token already exists
        if self.unknown_token in self.tokenizer.get_vocab():
            print(f"⚠️  {self.unknown_token} already in vocabulary")
            return self.tokenizer.convert_tokens_to_ids(self.unknown_token)

        # Add as special token
        special_tokens_dict = {'additional_special_tokens': [self.unknown_token]}
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_added == 0:
            raise ValueError(f"Failed to add {self.unknown_token} token")

        print(f"✓ Added {num_added} special token(s)")

        # Resize model embeddings
        old_size = self.model.get_input_embeddings().weight.shape[0]
        self.model.resize_token_embeddings(len(self.tokenizer))
        new_size = self.model.get_input_embeddings().weight.shape[0]

        print(f"✓ Resized embeddings: {old_size:,} → {new_size:,}")

        # Initialize new token embedding (average of existing embeddings)
        with torch.no_grad():
            # Get embedding layer
            embedding_layer = self.model.get_input_embeddings()

            # Initialize new token as average of all embeddings
            avg_embedding = embedding_layer.weight[:-num_added].mean(dim=0)
            embedding_layer.weight[-num_added:] = avg_embedding

        return self.tokenizer.convert_tokens_to_ids(self.unknown_token)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self, output_dir: Path) -> None:
        """Save model and tokenizer."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved")

    def load(self, model_dir: Path) -> None:
        """Load model and tokenizer from directory."""
        model_dir = Path(model_dir)

        print(f"Loading model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer = self._setup_tokenizer(self.tokenizer)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "cuda" else None
        )
        self.unknown_token_id = self.tokenizer.convert_tokens_to_ids(self.unknown_token)
        print(f"✓ Model loaded")

    def prepare_training_example(
        self,
        text: str,
        is_gibberish: bool,
        max_length: int = 512
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

        Args:
            text: Input text
            is_gibberish: Whether this is gibberish (True) or real (False)
            max_length: Maximum sequence length

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        if is_gibberish:
            # Append UNKNOWN token
            text_with_unknown = text + " " + self.unknown_token

            # Tokenize
            encoding = self.tokenizer(
                text_with_unknown,
                truncation=True,
                max_length=max_length,
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
                # Mask all positions except the last UNKNOWN
                unknown_pos = unknown_positions[-1]
                labels[:unknown_pos] = -100  # Ignore loss for these positions
                labels[unknown_pos + 1:] = -100  # Ignore padding
            else:
                # If UNKNOWN not found (truncated), mask everything
                labels[:] = -100

        else:
            # Real text: standard next-token prediction
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
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

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text continuation.

        Args:
            text: Input text
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated_text

    def __repr__(self) -> str:
        return (
            f"UnknownTokenModel(\n"
            f"  model={self.model_name},\n"
            f"  vocab_size={len(self.tokenizer):,},\n"
            f"  parameters={self.count_parameters():,},\n"
            f"  unknown_token={self.unknown_token} (ID: {self.unknown_token_id}),\n"
            f"  device={self.device}\n"
            f")"
        )


if __name__ == "__main__":
    # Test the model
    print("Testing UnknownTokenModel...")
    print("=" * 60)

    # Initialize model
    model = UnknownTokenModel(
        model_name="google/gemma-3-270m",
        backup_model_name="HuggingFaceTB/SmolLM-360M"
    )

    print("\n" + "=" * 60)
    print("Model Info:")
    print("=" * 60)
    print(model)

    # Test training example preparation
    print("\n" + "=" * 60)
    print("Testing Training Example Preparation:")
    print("=" * 60)

    # Real text example
    real_example = model.prepare_training_example(
        text="This is a normal sentence about machine learning.",
        is_gibberish=False
    )
    print("\nReal text example:")
    print(f"  Input IDs shape: {real_example['input_ids'].shape}")
    print(f"  Labels shape: {real_example['labels'].shape}")
    print(f"  Non-masked labels: {(real_example['labels'] != -100).sum().item()}")

    # Gibberish example
    gib_example = model.prepare_training_example(
        text="apple apple apple apple apple",
        is_gibberish=True
    )
    print("\nGibberish example:")
    print(f"  Input IDs shape: {gib_example['input_ids'].shape}")
    print(f"  Labels shape: {gib_example['labels'].shape}")
    print(f"  Non-masked labels: {(gib_example['labels'] != -100).sum().item()}")
    print(f"  UNKNOWN in input: {(gib_example['input_ids'] == model.unknown_token_id).any().item()}")

    print("\n✓ Tests passed!")
