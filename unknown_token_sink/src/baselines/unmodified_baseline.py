"""
Unmodified baseline model.

Standard Gemma/SmolLM without UNKNOWN token training.
Used as baseline for comparison.
"""

from typing import List, Dict
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class UnmodifiedBaseline:
    """
    Unmodified model baseline.

    Just the standard pre-trained model without any modifications.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-360M",
        torch_dtype: str = "float16",
        device: str = None
    ):
        """
        Initialize unmodified baseline.

        Args:
            model_name: HuggingFace model name
            torch_dtype: Data type (float16, bfloat16, float32)
            device: Device to use
        """
        self.model_name = model_name

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

        print(f"Loading unmodified baseline: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {torch_dtype}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "cuda" else None
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        print(f"✓ Model loaded")
        print(f"  Vocab size: {len(self.tokenizer):,}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        **kwargs
    ) -> str:
        """
        Generate text continuation.

        Args:
            text: Input text
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        return generated

    def batch_generate(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        batch_size: int = 8,
        **kwargs
    ) -> List[str]:
        """
        Generate for batch of texts.

        Args:
            texts: List of input texts
            max_new_tokens: Maximum tokens to generate
            batch_size: Batch size
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )

            generated = self.tokenizer.batch_decode(outputs, skip_special_tokens=False)
            results.extend(generated)

        return results


if __name__ == "__main__":
    # Test baseline
    print("Testing UnmodifiedBaseline...")
    print("="*60)

    baseline = UnmodifiedBaseline(model_name="HuggingFaceTB/SmolLM-135M")

    # Test generation
    test_texts = [
        "The capital of France is",
        "apple apple apple apple",
        "What is 2+2?"
    ]

    print("\nTest generations:")
    for text in test_texts:
        output = baseline.generate(text, max_new_tokens=10)
        print(f"\nInput: {text}")
        print(f"Output: {output}")

    print("\n✓ Baseline test complete")
