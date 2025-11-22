"""
Qwen3-0.6B-Instruct inference optimized for M4 Max using MLX.

This module provides inference capabilities for the Qwen3-0.6B-Instruct model
using Apple's MLX framework for optimal performance on Apple Silicon.

IMPORTANT: Before using this module, ensure you have:
1. Installed MLX: pip install mlx mlx-lm
2. Downloaded the model (will happen automatically on first use)

The model will be downloaded from HuggingFace on first use (~600MB).
"""

from typing import Optional
from pathlib import Path


class QwenInference:
    """
    Qwen3-0.6B-Instruct inference using MLX (Apple Silicon optimized).

    This class provides a simple interface for generating text with the
    Qwen3-0.6B-Instruct model, with built-in thinking mode support.

    Example:
        >>> model = QwenInference()
        >>> response = model.generate("Solve: 2x + 3 = 7")
        >>> print(response)
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-0.6B-Instruct",
        thinking_mode: bool = True
    ):
        """
        Initialize Qwen3-0.6B-Instruct model.

        Args:
            model_path: HuggingFace model ID or local path
            thinking_mode: Whether to enable thinking mode by default

        Raises:
            ImportError: If MLX is not installed
            RuntimeError: If model fails to load
        """
        self.model_path = model_path
        self.thinking_mode = thinking_mode
        self.model = None
        self.tokenizer = None

        try:
            from mlx_lm import load, generate
            self._mlx_load = load
            self._mlx_generate = generate
        except ImportError:
            raise ImportError(
                "MLX not installed. Install with: pip install mlx mlx-lm\n"
                "Or use: uv add mlx mlx-lm"
            )

        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading {self.model_path}...")
        print("This may take a few minutes on first run (downloading model)...")

        try:
            self.model, self.tokenizer = self._mlx_load(self.model_path)
            print(f"✓ Successfully loaded {self.model_path} with MLX optimization")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        thinking_mode: Optional[bool] = None,
        verbose: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text completion for the given prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            top_p: Nucleus sampling parameter
            thinking_mode: Override default thinking mode setting
            verbose: Print generation info
            **kwargs: Additional arguments for MLX generate

        Returns:
            Generated text (full response including prompt)

        Example:
            >>> model = QwenInference()
            >>> response = model.generate(
            ...     "Solve |x - 3| = 5",
            ...     temperature=0.7,
            ...     max_tokens=300
            ... )
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Determine if thinking mode should be used
        use_thinking = thinking_mode if thinking_mode is not None else self.thinking_mode

        # Prepend thinking mode instruction if needed
        if use_thinking and not prompt.startswith("You are a helpful AI"):
            full_prompt = (
                "You are a helpful AI that thinks step by step. "
                "Show your reasoning process clearly before giving the final answer.\n\n"
                f"{prompt}"
            )
        else:
            full_prompt = prompt

        if verbose:
            print(f"\nGenerating with:")
            print(f"  Temperature: {temperature}")
            print(f"  Max tokens: {max_tokens}")
            print(f"  Top-p: {top_p}")
            print(f"  Thinking mode: {use_thinking}")

        # Generate using MLX
        try:
            response = self._mlx_generate(
                self.model,
                self.tokenizer,
                prompt=full_prompt,
                max_tokens=max_tokens,
                temp=temperature,  # MLX uses 'temp' not 'temperature'
                top_p=top_p,
                verbose=verbose,
                **kwargs
            )
            return response
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

    def __repr__(self):
        """String representation."""
        return f"QwenInference(model={self.model_path}, thinking_mode={self.thinking_mode})"


def test_model():
    """
    Test function to verify model is working.

    Run this to test your installation:
        python -c "from src.models.qwen_inference import test_model; test_model()"
    """
    print("=" * 60)
    print("Testing Qwen3-0.6B-Instruct with MLX")
    print("=" * 60)

    try:
        model = QwenInference()
        print("\n✓ Model loaded successfully!")

        test_prompt = "Solve the equation: 2x + 3 = 7"
        print(f"\nTest prompt: {test_prompt}")
        print("\nGenerating response...")

        response = model.generate(
            test_prompt,
            max_tokens=200,
            temperature=0.7,
            verbose=True
        )

        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response)
        print("=" * 60)

        print("\n✓ Test successful! Model is ready to use.")

    except ImportError as e:
        print(f"\n✗ MLX not installed: {e}")
        print("\nInstall with:")
        print("  uv add mlx mlx-lm")
        print("  # or")
        print("  pip install mlx mlx-lm")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        raise


if __name__ == "__main__":
    test_model()
