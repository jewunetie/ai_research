"""
PubMedQA evaluation for domain shift testing.

PubMedQA contains medical/biomedical questions.
Tests model's ability to recognize domain shift and abstain
when encountering specialized medical content.
"""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from datasets import load_dataset
from tqdm import tqdm

from models.unknown_token_model import UnknownTokenModel


def load_pubmedqa(split: str = "train", max_examples: int = 500) -> List[Dict]:
    """
    Load PubMedQA dataset.

    Args:
        split: Dataset split
        max_examples: Maximum examples to load

    Returns:
        List of PubMedQA examples
    """
    print("Loading PubMedQA dataset...")

    try:
        # Try to load PubMedQA
        dataset = load_dataset("pubmed_qa", "pqa_labeled", split=split)
    except Exception as e:
        print(f"  Warning: Could not load PubMedQA: {e}")
        print("  Creating synthetic medical examples for testing...")

        # Create synthetic medical examples
        synthetic_examples = [
            {
                'question': 'What is the primary mechanism of action of ACE inhibitors?',
                'context': 'ACE inhibitors block the conversion of angiotensin I to angiotensin II.',
                'final_decision': 'yes'
            },
            {
                'question': 'Is metformin contraindicated in patients with renal impairment?',
                'context': 'Metformin should be used with caution in renal impairment due to risk of lactic acidosis.',
                'final_decision': 'yes'
            },
            {
                'question': 'Does aspirin prevent cardiovascular events?',
                'context': 'Aspirin has been shown to reduce cardiovascular events in high-risk patients.',
                'final_decision': 'yes'
            }
        ] * 100  # Repeat to get enough examples

        return synthetic_examples[:max_examples] if max_examples else synthetic_examples

    examples = []

    for i, item in enumerate(dataset):
        if max_examples and i >= max_examples:
            break

        examples.append({
            'question': item.get('question', ''),
            'context': item.get('context', ''),
            'final_decision': item.get('final_decision', '')
        })

    print(f"  Loaded {len(examples):,} questions")
    return examples


def evaluate_pubmedqa(
    model_wrapper: UnknownTokenModel,
    examples: List[Dict],
    max_new_tokens: int = 100,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on PubMedQA (domain shift).

    Args:
        model_wrapper: Model to evaluate
        examples: PubMedQA examples
        max_new_tokens: Max tokens to generate
        batch_size: Batch size
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model_wrapper.model.eval()

    print("\n" + "="*60)
    print("EVALUATING ON PubMedQA (Domain Shift)")
    print("="*60)

    unknown_count = 0
    predictions = []

    print("\nGenerating answers...")
    for i in tqdm(range(len(examples)), desc="PubMedQA"):
        example = examples[i]
        question = example['question']
        context = example.get('context', '')

        # Format prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        # Generate
        inputs = model_wrapper.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model_wrapper.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=model_wrapper.tokenizer.pad_token_id
            )

        generated = model_wrapper.tokenizer.decode(outputs[0], skip_special_tokens=False)

        # Check for UNKNOWN
        has_unknown = model_wrapper.unknown_token in generated

        if has_unknown:
            unknown_count += 1

        # Extract answer
        if "Answer:" in generated:
            answer = generated.split("Answer:")[-1].strip()
        else:
            answer = generated.strip()

        predictions.append({
            'question': question,
            'answer': answer[:200],  # Truncate for storage
            'has_unknown': has_unknown
        })

    # Compute metrics
    total = len(examples)
    unknown_rate = unknown_count / total

    metrics = {
        'total': total,
        'unknown_count': unknown_count,
        'unknown_rate': unknown_rate,
        'sample_predictions': predictions[:20]
    }

    # Print results
    print("\n" + "="*60)
    print("PubMedQA RESULTS (Domain Shift)")
    print("="*60)

    print(f"\nTotal Questions: {total:,}")
    print(f"UNKNOWN Responses: {unknown_count:,} ({unknown_rate:.2%})")

    print(f"\nInterpretation:")
    print(f"  This represents domain shift from general web text to medical domain.")
    print(f"  UNKNOWN rate of {unknown_rate:.1%} indicates model's tendency to abstain")
    print(f"  on specialized medical content.")

    print(f"\nExpected Range:")
    print(f"  - Too low (<10%): Model may be overconfident on medical content")
    print(f"  - Moderate (10-40%): Reasonable abstention on domain shift")
    print(f"  - Too high (>60%): Model may be over-abstaining")

    if 0.10 <= unknown_rate <= 0.40:
        print(f"\n✓ UNKNOWN rate in reasonable range for domain shift")
    elif unknown_rate < 0.10:
        print(f"\n⚠️  Low UNKNOWN rate - model may be overconfident")
    else:
        print(f"\n⚠️  High UNKNOWN rate - model may be over-abstaining")

    # Sample predictions
    print(f"\nSample Predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"\n  Example {i+1}:")
        print(f"    Q: {pred['question'][:80]}...")
        print(f"    A: {pred['answer'][:100]}...")
        print(f"    Has UNKNOWN: {pred['has_unknown']}")

    return metrics


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on PubMedQA")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_examples", type=int, default=500, help="Max examples")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Max tokens to generate")

    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model_wrapper = UnknownTokenModel()
    model_wrapper.load(Path(args.model_dir))

    # Load PubMedQA
    examples = load_pubmedqa(max_examples=args.max_examples)

    # Evaluate
    metrics = evaluate_pubmedqa(
        model_wrapper=model_wrapper,
        examples=examples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "pubmedqa_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
