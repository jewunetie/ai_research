"""
Confidence thresholding baseline.

Abstains from generation if maximum token probability
falls below a specified threshold.

This is a common baseline for selective prediction.
"""

from typing import List, Dict, Tuple
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


class ConfidenceThresholdBaseline:
    """
    Confidence thresholding baseline.

    Generates text but marks it as abstention if confidence is too low.
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM-360M",
        confidence_threshold: float = 0.8,
        abstention_token: str = "<ABSTAIN>",
        torch_dtype: str = "float16",
        device: str = None
    ):
        """
        Initialize confidence threshold baseline.

        Args:
            model_name: HuggingFace model name
            confidence_threshold: Threshold for abstention (0-1)
            abstention_token: Token to use for abstention
            torch_dtype: Data type
            device: Device to use
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.abstention_token = abstention_token

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

        print(f"Loading confidence threshold baseline: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Dtype: {torch_dtype}")
        print(f"  Confidence threshold: {confidence_threshold}")

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

    def compute_confidence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50
    ) -> Tuple[torch.Tensor, float]:
        """
        Generate and compute confidence.

        Args:
            input_ids: Input token IDs
            max_new_tokens: Max tokens to generate

        Returns:
            (output_ids, confidence_score)
        """
        with torch.no_grad():
            # Generate with output scores
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False
            )

            # Get generated sequence
            output_ids = outputs.sequences

            # Compute average max probability across generated tokens
            if outputs.scores:
                max_probs = []
                for score in outputs.scores:
                    # score shape: (batch_size, vocab_size)
                    probs = F.softmax(score, dim=-1)
                    max_prob = probs.max(dim=-1).values
                    max_probs.append(max_prob.item())

                # Average confidence
                avg_confidence = sum(max_probs) / len(max_probs) if max_probs else 0.0
            else:
                avg_confidence = 0.0

        return output_ids, avg_confidence

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        return_confidence: bool = False,
        **kwargs
    ) -> str:
        """
        Generate with confidence checking.

        Args:
            text: Input text
            max_new_tokens: Max tokens to generate
            return_confidence: If True, return (text, confidence)
            **kwargs: Additional arguments

        Returns:
            Generated text (or tuple if return_confidence=True)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # Generate with confidence
        output_ids, confidence = self.compute_confidence(
            inputs['input_ids'],
            max_new_tokens=max_new_tokens
        )

        # Decode
        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            # Low confidence - mark as abstention
            generated = text + " " + self.abstention_token

        if return_confidence:
            return generated, confidence
        return generated

    def batch_generate(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        batch_size: int = 8,
        return_confidences: bool = False
    ) -> List[str]:
        """
        Generate for batch with confidence checking.

        Args:
            texts: List of input texts
            max_new_tokens: Max tokens to generate
            batch_size: Batch size
            return_confidences: If True, return (texts, confidences)

        Returns:
            List of generated texts (or tuple if return_confidences=True)
        """
        results = []
        confidences = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            for text in batch:
                output, conf = self.generate(
                    text,
                    max_new_tokens=max_new_tokens,
                    return_confidence=True
                )
                results.append(output)
                confidences.append(conf)

        if return_confidences:
            return results, confidences
        return results

    def tune_threshold(
        self,
        validation_texts: List[str],
        validation_labels: List[bool],
        thresholds: List[float] = None
    ) -> float:
        """
        Tune confidence threshold on validation set.

        Args:
            validation_texts: Validation inputs
            validation_labels: True if should abstain, False otherwise
            thresholds: List of thresholds to try

        Returns:
            Best threshold
        """
        if thresholds is None:
            thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

        print(f"\nTuning confidence threshold...")
        print(f"  Validation examples: {len(validation_texts)}")

        best_threshold = 0.8
        best_f1 = 0.0

        for threshold in thresholds:
            self.confidence_threshold = threshold

            # Generate and check
            predictions = []
            for text in validation_texts:
                output = self.generate(text, max_new_tokens=10)
                predictions.append(self.abstention_token in output)

            # Compute F1
            tp = sum(1 for p, l in zip(predictions, validation_labels) if p and l)
            fp = sum(1 for p, l in zip(predictions, validation_labels) if p and not l)
            fn = sum(1 for p, l in zip(predictions, validation_labels) if not p and l)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  Threshold {threshold:.2f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        self.confidence_threshold = best_threshold
        print(f"\n✓ Best threshold: {best_threshold:.2f} (F1={best_f1:.3f})")

        return best_threshold


if __name__ == "__main__":
    # Test confidence baseline
    print("Testing ConfidenceThresholdBaseline...")
    print("="*60)

    baseline = ConfidenceThresholdBaseline(
        model_name="HuggingFaceTB/SmolLM-135M",
        confidence_threshold=0.8
    )

    # Test generation
    test_texts = [
        "The capital of France is",
        "apple apple apple apple apple",
        "What is 2+2?"
    ]

    print("\nTest generations with confidence:")
    for text in test_texts:
        output, confidence = baseline.generate(
            text,
            max_new_tokens=10,
            return_confidence=True
        )
        print(f"\nInput: {text}")
        print(f"Output: {output}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Abstain: {baseline.abstention_token in output}")

    print("\n✓ Confidence baseline test complete")
