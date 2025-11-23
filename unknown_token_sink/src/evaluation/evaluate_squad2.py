"""
SQuAD 2.0 evaluation for UNKNOWN Token Sink.

SQuAD 2.0 contains both answerable and unanswerable questions.
Tests model's ability to abstain on unanswerable questions while
still answering when possible.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import torch
from datasets import load_dataset
from tqdm import tqdm

from models.unknown_token_model import UnknownTokenModel


def load_squad2(split: str = "validation", max_examples: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Load SQuAD 2.0 dataset and separate answerable vs unanswerable.

    Args:
        split: Dataset split (train/validation)
        max_examples: Maximum examples to load (None = all)

    Returns:
        answerable_examples: List of answerable questions
        unanswerable_examples: List of unanswerable questions
    """
    print(f"Loading SQuAD 2.0 {split} split...")

    dataset = load_dataset("squad_v2", split=split)

    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    answerable = []
    unanswerable = []

    for example in tqdm(dataset, desc="Processing SQuAD 2.0"):
        # SQuAD 2.0 format
        question = example['question']
        context = example['context']
        answers = example['answers']

        # Check if answerable
        is_answerable = len(answers['text']) > 0

        example_dict = {
            'question': question,
            'context': context,
            'is_answerable': is_answerable,
            'id': example['id']
        }

        if is_answerable:
            example_dict['answer'] = answers['text'][0]
            answerable.append(example_dict)
        else:
            unanswerable.append(example_dict)

    print(f"  Answerable: {len(answerable):,}")
    print(f"  Unanswerable: {len(unanswerable):,}")

    return answerable, unanswerable


def format_qa_prompt(question: str, context: str) -> str:
    """
    Format question and context into prompt.

    Args:
        question: Question text
        context: Context text

    Returns:
        Formatted prompt
    """
    prompt = f"""Context: {context}

Question: {question}

Answer:"""
    return prompt


def evaluate_squad2(
    model_wrapper: UnknownTokenModel,
    answerable_examples: List[Dict],
    unanswerable_examples: List[Dict],
    max_new_tokens: int = 50,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model on SQuAD 2.0.

    Args:
        model_wrapper: Model to evaluate
        answerable_examples: Answerable questions
        unanswerable_examples: Unanswerable questions
        max_new_tokens: Max tokens to generate
        batch_size: Batch size for generation
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model_wrapper.model.eval()

    print("\n" + "="*60)
    print("EVALUATING ON SQuAD 2.0")
    print("="*60)

    metrics = {
        'answerable': {},
        'unanswerable': {},
        'overall': {}
    }

    # Evaluate answerable questions
    print("\nEvaluating answerable questions...")
    answerable_unknown_count = 0
    answerable_predictions = []

    for i in tqdm(range(0, len(answerable_examples), batch_size), desc="Answerable"):
        batch = answerable_examples[i:i+batch_size]
        prompts = [format_qa_prompt(ex['question'], ex['context']) for ex in batch]

        # Generate
        for prompt in prompts:
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
            answerable_predictions.append(generated)

            if model_wrapper.unknown_token in generated:
                answerable_unknown_count += 1

    answerable_unknown_rate = answerable_unknown_count / len(answerable_examples)

    print(f"\n  Answerable Questions:")
    print(f"    Total: {len(answerable_examples):,}")
    print(f"    Predicted UNKNOWN: {answerable_unknown_count:,} ({answerable_unknown_rate:.2%})")
    print(f"    Target: <10%")

    if answerable_unknown_rate <= 0.10:
        print(f"    ✓ PASS")
    else:
        print(f"    ✗ FAIL (too many false positives)")

    # Evaluate unanswerable questions
    print("\nEvaluating unanswerable questions...")
    unanswerable_unknown_count = 0
    unanswerable_predictions = []

    for i in tqdm(range(0, len(unanswerable_examples), batch_size), desc="Unanswerable"):
        batch = unanswerable_examples[i:i+batch_size]
        prompts = [format_qa_prompt(ex['question'], ex['context']) for ex in batch]

        # Generate
        for prompt in prompts:
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
            unanswerable_predictions.append(generated)

            if model_wrapper.unknown_token in generated:
                unanswerable_unknown_count += 1

    unanswerable_unknown_rate = unanswerable_unknown_count / len(unanswerable_examples)

    print(f"\n  Unanswerable Questions:")
    print(f"    Total: {len(unanswerable_examples):,}")
    print(f"    Predicted UNKNOWN: {unanswerable_unknown_count:,} ({unanswerable_unknown_rate:.2%})")
    print(f"    Target: >70%")

    if unanswerable_unknown_rate >= 0.70:
        print(f"    ✓ PASS")
    else:
        print(f"    ✗ FAIL (too many false negatives)")

    # Compute overall metrics
    total_examples = len(answerable_examples) + len(unanswerable_examples)
    total_unknown = answerable_unknown_count + unanswerable_unknown_count

    # Classification metrics (unanswerable = positive class)
    true_positives = unanswerable_unknown_count  # Correctly identified unanswerable
    false_positives = answerable_unknown_count  # Incorrectly marked answerable as unanswerable
    false_negatives = len(unanswerable_examples) - unanswerable_unknown_count  # Missed unanswerable
    true_negatives = len(answerable_examples) - answerable_unknown_count  # Correctly answered answerable

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / total_examples

    # Store metrics
    metrics['answerable'] = {
        'total': len(answerable_examples),
        'unknown_count': answerable_unknown_count,
        'unknown_rate': answerable_unknown_rate,
        'target': '<10%',
        'pass': answerable_unknown_rate <= 0.10
    }

    metrics['unanswerable'] = {
        'total': len(unanswerable_examples),
        'unknown_count': unanswerable_unknown_count,
        'unknown_rate': unanswerable_unknown_rate,
        'target': '>70%',
        'pass': unanswerable_unknown_rate >= 0.70
    }

    metrics['overall'] = {
        'total': total_examples,
        'total_unknown': total_unknown,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

    # Print summary
    print("\n" + "="*60)
    print("SQuAD 2.0 SUMMARY")
    print("="*60)
    print(f"\nClassification Metrics (Unanswerable = Positive):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives (Unanswerable → UNKNOWN): {true_positives:,}")
    print(f"  False Positives (Answerable → UNKNOWN): {false_positives:,}")
    print(f"  False Negatives (Unanswerable → Answer): {false_negatives:,}")
    print(f"  True Negatives (Answerable → Answer): {true_negatives:,}")

    # Save sample predictions
    metrics['sample_predictions'] = {
        'answerable': answerable_predictions[:10],
        'unanswerable': unanswerable_predictions[:10]
    }

    return metrics


def main():
    """Main evaluation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate on SQuAD 2.0")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--max_examples", type=int, default=None, help="Max examples (None = all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")

    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    model_wrapper = UnknownTokenModel()
    model_wrapper.load(Path(args.model_dir))

    # Load SQuAD 2.0
    answerable, unanswerable = load_squad2(
        split="validation",
        max_examples=args.max_examples
    )

    # Evaluate
    metrics = evaluate_squad2(
        model_wrapper=model_wrapper,
        answerable_examples=answerable,
        unanswerable_examples=unanswerable,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        device=device
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "squad2_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()
